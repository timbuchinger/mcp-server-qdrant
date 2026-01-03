import json
import logging
from typing import Annotated, Any, Optional

from fastmcp import Context, FastMCP
from pydantic import Field
from qdrant_client import models

from mcp_server_qdrant.common.filters import make_indexes
from mcp_server_qdrant.common.func_tools import make_partial_function
from mcp_server_qdrant.common.wrap_filters import wrap_filters
from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.qdrant import ArbitraryFilter, Entry, Metadata, QdrantConnector
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)

logger = logging.getLogger(__name__)


# FastMCP is an alternative interface for declaring the capabilities
# of the server. Its API is based on FastAPI.
class QdrantMCPServer(FastMCP):
    """
    A MCP server for Qdrant.
    """

    def __init__(
        self,
        tool_settings: ToolSettings,
        qdrant_settings: QdrantSettings,
        embedding_provider_settings: Optional[EmbeddingProviderSettings] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        name: str = "mcp-server-qdrant",
        instructions: str | None = None,
        **settings: Any,
    ):
        self.tool_settings = tool_settings
        self.qdrant_settings = qdrant_settings

        if embedding_provider_settings and embedding_provider:
            raise ValueError(
                "Cannot provide both embedding_provider_settings and embedding_provider"
            )

        if not embedding_provider_settings and not embedding_provider:
            raise ValueError(
                "Must provide either embedding_provider_settings or embedding_provider"
            )

        self.embedding_provider_settings: Optional[EmbeddingProviderSettings] = None
        self.embedding_provider: Optional[EmbeddingProvider] = None

        if embedding_provider_settings:
            self.embedding_provider_settings = embedding_provider_settings
            self.embedding_provider = create_embedding_provider(
                embedding_provider_settings
            )
        else:
            self.embedding_provider_settings = None
            self.embedding_provider = embedding_provider

        assert self.embedding_provider is not None, "Embedding provider is required"

        self.qdrant_connector = QdrantConnector(
            qdrant_settings.location,
            qdrant_settings.api_key,
            qdrant_settings.collection_name,
            self.embedding_provider,
            qdrant_settings.local_path,
            make_indexes(qdrant_settings.filterable_fields_dict()),
        )

        super().__init__(name=name, instructions=instructions, **settings)

        self.setup_tools()

    def format_entry(self, entry: Entry) -> str:
        """
        Feel free to override this method in your subclass to customize the format of the entry.
        """
        entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
        entry_id = entry.id if entry.id else ""
        return f"<entry><id>{entry_id}</id><content>{entry.content}</content><metadata>{entry_metadata}</metadata></entry>"

    def setup_tools(self):
        """
        Register the tools in the server.
        """

        async def store(
            ctx: Context,
            information: Annotated[str, Field(description="Text to store")],
            collection_name: Annotated[
                str, Field(description="The collection to store the information in")
            ],
            # The `metadata` parameter is defined as non-optional, but it can be None.
            # If we set it to be optional, some of the MCP clients, like Cursor, cannot
            # handle the optional parameter correctly.
            metadata: Annotated[
                Metadata | None,
                Field(
                    description="Extra metadata stored along with memorised information. Any json is accepted."
                ),
            ] = None,
        ) -> str:
            """
            Store some information in Qdrant.
            :param ctx: The context for the request.
            :param information: The information to store.
            :param metadata: JSON metadata to store with the information, optional.
            :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                    the default collection is used.
            :return: A message indicating that the information was stored.
            """
            await ctx.debug(f"Storing information {information} in Qdrant")

            entry = Entry(content=information, metadata=metadata)

            await self.qdrant_connector.store(entry, collection_name=collection_name)
            if collection_name:
                return f"Remembered: {information} in collection {collection_name}"
            return f"Remembered: {information}"

        async def add_note(
            ctx: Context,
            text: Annotated[str, Field(description="The primary knowledge content")],
            context: Annotated[
                str,
                Field(description="Explains when / why / how the text is useful"),
            ],
            type: Annotated[
                str,
                Field(
                    description="Type of note: cli | api | learning | snippet | pattern"
                ),
            ],
            created_at: Annotated[
                str,
                Field(
                    description="ISO-8601 formatted timestamp of when the knowledge was recorded"
                ),
            ],
            tool: Annotated[
                str | None, Field(description="Tool or command name (optional)")
            ] = None,
            tags: Annotated[
                list[str] | None,
                Field(description="List of tags for categorization (optional)"),
            ] = None,
            language: Annotated[
                str | None,
                Field(description="Programming language if applicable (optional)"),
            ] = None,
            source: Annotated[
                str | None, Field(description="Source or reference URL (optional)")
            ] = None,
        ) -> str:
            """
            Add a structured note to Qdrant with specific metadata.
            Uses the default collection specified via environment variable.
            :param ctx: The context for the request.
            :param text: The main text content of the note.
            :param context: Context or description about when/why this note is useful.
            :param type: Type of note (cli, api, learning, snippet, or pattern).
            :param created_at: ISO-8601 formatted timestamp.
            :param tool: Optional tool or command name.
            :param tags: Optional list of tags.
            :param language: Optional programming language.
            :param source: Optional source or reference URL.
            :return: A message indicating that the note was stored.
            """
            await ctx.debug(f"Adding note: {text[:50]}... with type {type}")

            # Build the metadata dictionary
            metadata: Metadata = {
                "context": context,
                "type": type,
                "created_at": created_at,
            }

            # Add optional fields if provided
            if tool is not None:
                metadata["tool"] = tool
            if tags is not None:
                metadata["tags"] = tags
            if language is not None:
                metadata["language"] = language
            if source is not None:
                metadata["source"] = source

            entry = Entry(content=text, metadata=metadata)
            await self.qdrant_connector.store(entry, collection_name=None)

            return f"Note added: {text[:50]}... (type: {type})"

        async def update_note(
            ctx: Context,
            note_id: Annotated[
                str, Field(description="The unique identifier of the note to update")
            ],
            text: Annotated[str, Field(description="The primary knowledge content")],
            context: Annotated[
                str,
                Field(description="Explains when / why / how the text is useful"),
            ],
            type: Annotated[
                str,
                Field(
                    description="Type of note: cli | api | learning | snippet | pattern"
                ),
            ],
            created_at: Annotated[
                str,
                Field(
                    description="ISO-8601 formatted timestamp of when the knowledge was recorded"
                ),
            ],
            tool: Annotated[
                str | None, Field(description="Tool or command name (optional)")
            ] = None,
            tags: Annotated[
                list[str] | None,
                Field(description="List of tags for categorization (optional)"),
            ] = None,
            language: Annotated[
                str | None,
                Field(description="Programming language if applicable (optional)"),
            ] = None,
            source: Annotated[
                str | None, Field(description="Source or reference URL (optional)")
            ] = None,
        ) -> str:
            """
            Update an existing structured note in Qdrant.
            Uses the default collection specified via environment variable.
            :param ctx: The context for the request.
            :param note_id: The unique identifier of the note to update.
            :param text: The main text content of the note.
            :param context: Context or description about when/why this note is useful.
            :param type: Type of note (cli, api, learning, snippet, or pattern).
            :param created_at: ISO-8601 formatted timestamp.
            :param tool: Optional tool or command name.
            :param tags: Optional list of tags.
            :param language: Optional programming language.
            :param source: Optional source or reference URL.
            :return: A message indicating that the note was updated.
            """
            await ctx.debug(f"Updating note {note_id}: {text[:50]}...")

            # Build the metadata dictionary
            metadata: Metadata = {
                "context": context,
                "type": type,
                "created_at": created_at,
            }

            # Add optional fields if provided
            if tool is not None:
                metadata["tool"] = tool
            if tags is not None:
                metadata["tags"] = tags
            if language is not None:
                metadata["language"] = language
            if source is not None:
                metadata["source"] = source

            entry = Entry(content=text, metadata=metadata)
            await self.qdrant_connector.update(note_id, entry, collection_name=None)

            return f"Note updated: {text[:50]}... (type: {type}, id: {note_id})"

        async def delete_note(
            ctx: Context,
            note_id: Annotated[
                str, Field(description="The unique identifier of the note to delete")
            ],
        ) -> str:
            """
            Delete a note from Qdrant.
            Uses the default collection specified via environment variable.
            :param ctx: The context for the request.
            :param note_id: The unique identifier of the note to delete.
            :return: A message indicating that the note was deleted.
            """
            await ctx.debug(f"Deleting note {note_id}")

            await self.qdrant_connector.delete(note_id, collection_name=None)

            return f"Note deleted: {note_id}"

        async def find(
            ctx: Context,
            query: Annotated[str, Field(description="What to search for")],
            collection_name: Annotated[
                str, Field(description="The collection to search in")
            ],
            query_filter: ArbitraryFilter | None = None,
        ) -> list[str] | None:
            """
            Find memories in Qdrant.
            :param ctx: The context for the request.
            :param query: The query to use for the search.
            :param collection_name: The name of the collection to search in, optional. If not provided,
                                    the default collection is used.
            :param query_filter: The filter to apply to the query.
            :return: A list of entries found or None.
            """

            # Log query_filter
            await ctx.debug(f"Query filter: {query_filter}")

            query_filter = models.Filter(**query_filter) if query_filter else None

            await ctx.debug(f"Finding results for query {query}")

            entries = await self.qdrant_connector.search(
                query,
                collection_name=collection_name,
                limit=self.qdrant_settings.search_limit,
                query_filter=query_filter,
            )
            if not entries:
                return None
            content = [
                f"Results for the query '{query}'",
            ]
            for entry in entries:
                content.append(self.format_entry(entry))
            return content

        async def hybrid_find(
            ctx: Context,
            query: Annotated[str, Field(description="What to search for")],
            collection_name: Annotated[
                str, Field(description="The collection to search in")
            ],
            fusion_method: Annotated[
                str, Field(description="Fusion method: 'rrf' or 'dbsf'")
            ] = "rrf",
            dense_limit: Annotated[
                int, Field(description="Max results from semantic search")
            ] = 20,
            sparse_limit: Annotated[
                int, Field(description="Max results from keyword search")
            ] = 20,
            final_limit: Annotated[
                int, Field(description="Final number of results after fusion")
            ] = 10,
            query_filter: ArbitraryFilter | None = None,
        ) -> list[str] | None:
            """
            Hybrid search combining semantic similarity and keyword matching.
            Uses Qdrant's RRF/DBSF fusion for optimal search results.

            :param ctx: The context for the request.
            :param query: The query to use for the search.
            :param collection_name: The name of the collection to search in.
            :param fusion_method: Fusion method - 'rrf' (Reciprocal Rank Fusion) or 'dbsf' (Distribution-Based Score Fusion).
            :param dense_limit: Maximum results from dense vector search.
            :param sparse_limit: Maximum results from sparse vector search.
            :param final_limit: Maximum final results after fusion.
            :param query_filter: The filter to apply to the query.
            :return: A list of entries found or None.
            """
            await ctx.debug(
                f"Hybrid search for query '{query}' using fusion method '{fusion_method}'"
            )

            parsed_query_filter = (
                models.Filter(**query_filter) if query_filter else None
            )

            entries = await self.qdrant_connector.find_hybrid(
                query,
                collection_name=collection_name,
                fusion_method=fusion_method,
                dense_limit=dense_limit,
                sparse_limit=sparse_limit,
                final_limit=final_limit,
                query_filter=parsed_query_filter,
            )

            if not entries:
                return None

            content = [
                f"Hybrid search results for '{query}' (fusion: {fusion_method})",
            ]
            for entry in entries:
                content.append(self.format_entry(entry))
            return content

        find_foo = find
        store_foo = store
        add_note_foo = add_note
        update_note_foo = update_note
        delete_note_foo = delete_note
        hybrid_find_foo = hybrid_find

        filterable_conditions = (
            self.qdrant_settings.filterable_fields_dict_with_conditions()
        )

        if len(filterable_conditions) > 0:
            find_foo = wrap_filters(find_foo, filterable_conditions)
            hybrid_find_foo = wrap_filters(hybrid_find_foo, filterable_conditions)
        elif not self.qdrant_settings.allow_arbitrary_filter:
            find_foo = make_partial_function(find_foo, {"query_filter": None})
            hybrid_find_foo = make_partial_function(
                hybrid_find_foo, {"query_filter": None}
            )

        if self.qdrant_settings.collection_name:
            find_foo = make_partial_function(
                find_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            store_foo = make_partial_function(
                store_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            hybrid_find_foo = make_partial_function(
                hybrid_find_foo,
                {"collection_name": self.qdrant_settings.collection_name},
            )

        self.tool(
            hybrid_find_foo,
            name="qdrant-search-notes",
            description=self.tool_settings.tool_hybrid_find_description,
        )

        if not self.qdrant_settings.read_only:
            # Those methods can modify the database
            # Note: we intentionally do not register `qdrant-store` here.
            self.tool(
                add_note_foo,
                name="qdrant-add-note",
                description=self.tool_settings.tool_add_note_description,
            )
            self.tool(
                update_note_foo,
                name="qdrant-update-note",
                description=self.tool_settings.tool_update_note_description,
            )
            self.tool(
                delete_note_foo,
                name="qdrant-delete-note",
                description=self.tool_settings.tool_delete_note_description,
            )
