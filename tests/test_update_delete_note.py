"""
Test update and delete note functionality.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_server_qdrant.qdrant import Entry, QdrantConnector


@pytest.mark.asyncio
async def test_update_note():
    """Test updating a note."""
    # Mock the Qdrant client
    mock_client = AsyncMock()
    mock_client.retrieve.return_value = [MagicMock(id="test-id")]

    # Mock the embedding provider
    mock_provider = MagicMock()
    mock_provider.embed_documents = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    mock_provider.get_vector_name.return_value = "default"

    # Create connector
    connector = QdrantConnector(
        qdrant_url="http://localhost:6333",
        qdrant_api_key=None,
        collection_name="test-collection",
        embedding_provider=mock_provider,
    )
    connector._client = mock_client

    # Create an entry to update
    entry = Entry(
        content="Updated content",
        metadata={
            "type": "cli",
            "context": "Updated context",
            "created_at": "2024-01-01",
        },
    )

    # Update the note
    await connector.update("test-id", entry)

    # Verify retrieve was called to check if point exists
    mock_client.retrieve.assert_called_once()

    # Verify upsert was called with the updated entry
    mock_client.upsert.assert_called_once()
    call_args = mock_client.upsert.call_args
    assert call_args.kwargs["collection_name"] == "test-collection"
    assert call_args.kwargs["points"][0].id == "test-id"


@pytest.mark.asyncio
async def test_delete_note():
    """Test deleting a note."""
    # Mock the Qdrant client
    mock_client = AsyncMock()
    mock_client.retrieve.return_value = [MagicMock(id="test-id")]

    # Mock the embedding provider
    mock_provider = MagicMock()

    # Create connector
    connector = QdrantConnector(
        qdrant_url="http://localhost:6333",
        qdrant_api_key=None,
        collection_name="test-collection",
        embedding_provider=mock_provider,
    )
    connector._client = mock_client

    # Delete the note
    await connector.delete("test-id")

    # Verify retrieve was called to check if point exists
    mock_client.retrieve.assert_called_once()

    # Verify delete was called
    mock_client.delete.assert_called_once()
    call_args = mock_client.delete.call_args
    assert call_args.kwargs["collection_name"] == "test-collection"


@pytest.mark.asyncio
async def test_update_note_not_found():
    """Test updating a note that doesn't exist."""
    # Mock the Qdrant client to return no points
    mock_client = AsyncMock()
    mock_client.retrieve.return_value = []

    # Mock the embedding provider
    mock_provider = MagicMock()

    # Create connector
    connector = QdrantConnector(
        qdrant_url="http://localhost:6333",
        qdrant_api_key=None,
        collection_name="test-collection",
        embedding_provider=mock_provider,
    )
    connector._client = mock_client

    # Create an entry to update
    entry = Entry(content="Content", metadata={})

    # Try to update a non-existent note
    with pytest.raises(ValueError, match="Point with ID test-id not found"):
        await connector.update("test-id", entry)


@pytest.mark.asyncio
async def test_delete_note_not_found():
    """Test deleting a note that doesn't exist."""
    # Mock the Qdrant client to return no points
    mock_client = AsyncMock()
    mock_client.retrieve.return_value = []

    # Mock the embedding provider
    mock_provider = MagicMock()

    # Create connector
    connector = QdrantConnector(
        qdrant_url="http://localhost:6333",
        qdrant_api_key=None,
        collection_name="test-collection",
        embedding_provider=mock_provider,
    )
    connector._client = mock_client

    # Try to delete a non-existent note
    with pytest.raises(ValueError, match="Point with ID test-id not found"):
        await connector.delete("test-id")


def test_entry_with_id():
    """Test that Entry model supports optional id field."""
    # Create entry without ID
    entry1 = Entry(content="Test content", metadata={"key": "value"})
    assert entry1.id is None

    # Create entry with ID
    entry2 = Entry(content="Test content", metadata={"key": "value"}, id="test-id")
    assert entry2.id == "test-id"
