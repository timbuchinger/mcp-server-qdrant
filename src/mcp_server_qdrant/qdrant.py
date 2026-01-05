import logging
import uuid
from typing import Any
import re
import math
from collections import Counter, defaultdict

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.settings import METADATA_PATH

logger = logging.getLogger(__name__)

Metadata = dict[str, Any]
ArbitraryFilter = dict[str, Any]


class Entry(BaseModel):
    """
    A single entry in the Qdrant collection.
    """

    content: str
    metadata: Metadata | None = None
    id: str | None = None




class BM25Indexer:
    """
    In-memory BM25 index used to compute sparse vectors locally.
    This is a lightweight implementation intended to generate per-document
    sparse vectors (term ids + BM25 weights) to send to Qdrant when the
    server-side BM25 is unavailable.
    """

    def __init__(self, max_vocab: int = 32768, k1: float = 1.5, b: float = 0.75):
        self.max_vocab = max_vocab
        self.k1 = k1
        self.b = b
        self.vocab: dict[str, int] = {}
        self.df: Counter = Counter()
        self.N: int = 0
        self.doc_lens_total: int = 0
        # map doc_id -> (Counter(tokens), doc_len)
        self._docs: dict[str, tuple[Counter, int]] = {}

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    @property
    def avgdl(self) -> float:
        return (self.doc_lens_total / self.N) if self.N > 0 else 1.0

    def _ensure_term(self, term: str) -> int | None:
        """Ensure the term is in the vocabulary and return its index, or None if capacity exceeded."""
        if term in self.vocab:
            return self.vocab[term]
        if len(self.vocab) >= self.max_vocab:
            return None
        idx = len(self.vocab)
        self.vocab[term] = idx
        return idx

    def add_or_update(self, doc_id: str, text: str) -> tuple[list[int], list[float]]:
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        doc_len = len(tokens)

        # If updating existing, remove old counts
        if doc_id in self._docs:
            old_tf, old_len = self._docs[doc_id]
            # decrement df for terms present in old doc
            for term in old_tf.keys():
                self.df[term] -= 1
                if self.df[term] <= 0:
                    del self.df[term]
            self.doc_lens_total -= old_len
        else:
            # new document increases N
            self.N += 1

        # add new counts
        for term in tf.keys():
            self.df[term] += 1
            self._ensure_term(term)
        self.doc_lens_total += doc_len
        self._docs[doc_id] = (tf, doc_len)

        # compute BM25 scores for the document
        ids = []
        values = []
        for term, freq in tf.items():
            idx = self.vocab.get(term)
            if idx is None:
                continue
            df = self.df.get(term, 0)
            # idf with add-one smoothing to avoid negative/zero values
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            denom = freq + self.k1 * (1 - self.b + self.b * (doc_len / max(1.0, self.avgdl)))
            score = idf * ((freq * (self.k1 + 1)) / denom)
            ids.append(idx)
            values.append(float(score))

        return ids, values

    def transform(self, text: str) -> tuple[list[int], list[float]]:
        """Compute BM25 scores for a query text against the current index without modifying counts."""
        tokens = self._tokenize(text)
        tf = Counter(tokens)
        doc_len = len(tokens)
        ids = []
        values = []
        if self.N == 0:
            return ids, values
        for term, freq in tf.items():
            idx = self.vocab.get(term)
            if idx is None:
                continue
            df = self.df.get(term, 0)
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            denom = freq + self.k1 * (1 - self.b + self.b * (doc_len / max(1.0, self.avgdl)))
            score = idf * ((freq * (self.k1 + 1)) / denom)
            ids.append(idx)
            values.append(float(score))
        return ids, values

    def remove(self, doc_id: str) -> None:
        if doc_id not in self._docs:
            return
        old_tf, old_len = self._docs.pop(doc_id)
        for term in old_tf.keys():
            self.df[term] -= 1
            if self.df[term] <= 0:
                del self.df[term]
        self.doc_lens_total -= old_len
        self.N = max(0, self.N - 1)


class QdrantConnector:
    """
    Encapsulates the connection to a Qdrant server and all the methods to interact with it.
    :param qdrant_url: The URL of the Qdrant server.
    :param qdrant_api_key: The API key to use for the Qdrant server.
    :param collection_name: The name of the default collection to use. If not provided, each tool will require
                            the collection name to be provided.
    :param embedding_provider: The embedding provider to use.
    :param qdrant_local_path: The path to the storage directory for the Qdrant client, if local mode is used.
    """

    def __init__(
        self,
        qdrant_url: str | None,
        qdrant_api_key: str | None,
        collection_name: str | None,
        embedding_provider: EmbeddingProvider,
        qdrant_local_path: str | None = None,
        field_indexes: dict[str, models.PayloadSchemaType] | None = None,
    ):
        self._qdrant_url = qdrant_url.rstrip("/") if qdrant_url else None
        self._qdrant_api_key = qdrant_api_key
        self._default_collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._client = AsyncQdrantClient(
            location=qdrant_url, api_key=qdrant_api_key, path=qdrant_local_path
        )
        self._field_indexes = field_indexes
        # BM25 index used to compute sparse vectors locally when server-side BM25 isn't available
        self._bm25 = BM25Indexer()

    async def get_collection_names(self) -> list[str]:
        """
        Get the names of all collections in the Qdrant server.
        :return: A list of collection names.
        """
        response = await self._client.get_collections()
        return [collection.name for collection in response.collections]

    async def store(self, entry: Entry, *, collection_name: str | None = None):
        """
        Store some information in the Qdrant collection, along with the specified metadata.
        :param entry: The entry to store in the Qdrant collection.
        :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                the default collection is used.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None
        await self._ensure_collection_exists(collection_name)

        # Embed the document (dense)
        embeddings = await self._embedding_provider.embed_documents([entry.content])

        # Generate an id and build a local BM25 sparse vector
        point_id = uuid.uuid4().hex
        sparse_ids, sparse_values = self._bm25.add_or_update(point_id, entry.content)

        # Add to Qdrant
        vector_name = self._embedding_provider.get_vector_name()
        payload = {"document": entry.content, METADATA_PATH: entry.metadata}
        vector_payload = {vector_name: embeddings[0]}
        if sparse_ids:
            vector_payload["sparse"] = models.SparseVector(ids=sparse_ids, values=sparse_values)

        await self._client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector_payload,
                    payload=payload,
                )
            ],
        )

    async def update(
        self, point_id: str, entry: Entry, *, collection_name: str | None = None
    ):
        """
        Update an existing entry in the Qdrant collection.
        :param point_id: The ID of the point to update.
        :param entry: The updated entry data.
        :param collection_name: The name of the collection, optional. If not provided,
                                the default collection is used.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None

        # Check if the point exists
        point = await self._client.retrieve(
            collection_name=collection_name, ids=[point_id]
        )
        if not point:
            raise ValueError(f"Point with ID {point_id} not found")

        # Embed the updated document (dense)
        embeddings = await self._embedding_provider.embed_documents([entry.content])

        # Update BM25 index and build sparse vector
        sparse_ids, sparse_values = self._bm25.add_or_update(point_id, entry.content)

        # Update in Qdrant
        vector_name = self._embedding_provider.get_vector_name()
        payload = {"document": entry.content, METADATA_PATH: entry.metadata}
        vector_payload = {vector_name: embeddings[0]}
        if sparse_ids:
            vector_payload["sparse"] = models.SparseVector(ids=sparse_ids, values=sparse_values)

        await self._client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector_payload,
                    payload=payload,
                )
            ],
        )

    async def delete(self, point_id: str, *, collection_name: str | None = None):
        """
        Delete an entry from the Qdrant collection.
        :param point_id: The ID of the point to delete.
        :param collection_name: The name of the collection, optional. If not provided,
                                the default collection is used.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None

        # Check if the point exists
        point = await self._client.retrieve(
            collection_name=collection_name, ids=[point_id]
        )
        if not point:
            raise ValueError(f"Point with ID {point_id} not found")

        # Delete from Qdrant
        await self._client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=[point_id]),
        )

    async def search(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        limit: int = 10,
        query_filter: models.Filter | None = None,
    ) -> list[Entry]:
        """
        Find points in the Qdrant collection. If there are no entries found, an empty list is returned.
        :param query: The query to use for the search.
        :param collection_name: The name of the collection to search in, optional. If not provided,
                                the default collection is used.
        :param limit: The maximum number of entries to return.
        :param query_filter: The filter to apply to the query, if any.

        :return: A list of entries found.
        """
        collection_name = collection_name or self._default_collection_name
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return []

        # Embed the query
        # ToDo: instead of embedding text explicitly, use `models.Document`,
        # it should unlock usage of server-side inference.

        query_vector = await self._embedding_provider.embed_query(query)
        vector_name = self._embedding_provider.get_vector_name()

        # Search in Qdrant
        search_results = await self._client.query_points(
            collection_name=collection_name,
            query=query_vector,
            using=vector_name,
            limit=limit,
            query_filter=query_filter,
        )

        return [
            Entry(
                content=result.payload["document"],
                metadata=result.payload.get("metadata"),
                id=str(result.id),
            )
            for result in search_results.points
        ]

    async def find_hybrid(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        fusion_method: str = "rrf",
        dense_limit: int = 20,
        sparse_limit: int = 20,
        final_limit: int = 10,
        query_filter: models.Filter | None = None,
    ) -> list[Entry]:
        """
        Hybrid search combining dense and sparse vectors using Qdrant's Query API.

        :param query: The text query to search for.
        :param collection_name: The name of the collection to search in.
        :param fusion_method: Fusion method - "rrf" (Reciprocal Rank Fusion) or "dbsf" (Distribution-Based Score Fusion).
        :param dense_limit: Maximum results from dense vector search.
        :param sparse_limit: Maximum results from sparse vector search.
        :param final_limit: Maximum final results after fusion.
        :param query_filter: Optional filter to apply to the search.
        :return: A list of entries found, fused from both dense and sparse search.
        """
        collection_name = collection_name or self._default_collection_name
        if collection_name is None:
            return []

        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return []

        # Check if collection has both dense and sparse vectors
        # For now, we'll assume dense vector exists and fallback gracefully if sparse doesn't
        vector_name = self._embedding_provider.get_vector_name()

        try:
            # Build prefetch queries for hybrid search
            prefetch_queries = []

            # Dense vector search (semantic similarity)
            query_vector = await self._embedding_provider.embed_query(query)
            prefetch_queries.append(
                models.Prefetch(
                    query=query_vector,
                    using=vector_name,
                    limit=dense_limit,
                )
            )

            # Sparse vector search (keyword matching)
            # Note: This assumes sparse vectors are configured in the collection
            # In practice, you'd want to check collection config first
            try:
                # Build a sparse query vector locally using BM25
                sparse_ids, sparse_values = self._bm25.transform(query)
                if sparse_ids:
                    prefetch_queries.append(
                        models.Prefetch(
                            query=models.SparseVector(ids=sparse_ids, values=sparse_values),
                            using="sparse",
                            limit=sparse_limit,
                        )
                    )
                else:
                    # No sparse query vector available; skip sparse prefetch
                    pass
            except Exception:
                # If sparse vectors aren't available, fallback to dense-only search
                logger.warning(
                    f"Sparse vectors not available in collection {collection_name}, using dense-only search"
                )
                return await self.search(
                    query,
                    collection_name=collection_name,
                    limit=final_limit,
                    query_filter=query_filter,
                )

            # Execute hybrid search with fusion
            fusion_type = (
                models.Fusion.RRF
                if fusion_method.lower() == "rrf"
                else models.Fusion.DBSF
            )

            search_results = await self._client.query_points(
                collection_name=collection_name,
                prefetch=prefetch_queries,
                query=models.FusionQuery(fusion=fusion_type),
                limit=final_limit,
                query_filter=query_filter,
            )

            return [
                Entry(
                    content=result.payload["document"] if result.payload else "",
                    metadata=result.payload.get("metadata") if result.payload else None,
                    id=str(result.id),
                )
                for result in search_results.points
            ]

        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # Fallback to regular dense search
            logger.info(f"Falling back to dense vector search for query: {query}")
            return await self.search(
                query,
                collection_name=collection_name,
                limit=final_limit,
                query_filter=query_filter,
            )

    async def _ensure_collection_exists(self, collection_name: str):
        """
        Ensure that the collection exists, creating it if necessary.
        :param collection_name: The name of the collection to ensure exists.
        """
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            # Create the collection with the appropriate vector size
            vector_size = self._embedding_provider.get_vector_size()

            # Use the vector name as defined in the embedding provider
            vector_name = self._embedding_provider.get_vector_name()
            vectors_config = {
                vector_name: models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                )
            }
            # Reserve a sparse vector space for BM25 with the configured vocabulary cap
            vectors_config["sparse"] = models.VectorParams(
                size=self._bm25.max_vocab, distance=models.Distance.COSINE
            )
            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
            )

            # Create payload indexes if configured

            if self._field_indexes:
                for field_name, field_type in self._field_indexes.items():
                    await self._client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_type,
                    )
