import uuid

import pytest

from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.settings import ToolSettings, QdrantSettings
from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.qdrant import Entry


class DummyEmbedding(EmbeddingProvider):
    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        # return a small fixed dense vector for tests
        return [[0.01] * 8 for _ in documents]

    async def embed_query(self, query: str) -> list[float]:
        return [0.01] * 8

    def get_vector_name(self) -> str:
        return "test-vector"

    def get_vector_size(self) -> int:
        return 8


@pytest.mark.asyncio
async def test_add_and_search_tool_like():
    """Basic integration test: store a note and search for it using in-memory Qdrant."""
    embedding_provider = DummyEmbedding()
    collection_name = f"test_collection_{uuid.uuid4().hex}"

    qdrant_settings = QdrantSettings()
    server = QdrantMCPServer(tool_settings=ToolSettings(), qdrant_settings=qdrant_settings, embedding_provider=embedding_provider)

    # Replace the connector with an in-memory Qdrant client to isolate tests
    from mcp_server_qdrant.qdrant import QdrantConnector

    server.qdrant_connector = QdrantConnector(
        qdrant_url=":memory:",
        qdrant_api_key=None,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
    )

    # Store a note (simulating qdrant-add-note behavior)
    entry = Entry(content="Hello MCP world", metadata={"source": "test"})
    await server.qdrant_connector.store(entry)

    # Search for the stored note
    results = await server.qdrant_connector.search("Hello")

    assert len(results) >= 1
    assert any("Hello MCP world" in r.content for r in results)
