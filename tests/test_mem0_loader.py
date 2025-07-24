"""Tests for Mem0 loader focusing on public interface."""

from unittest.mock import Mock, patch

from src.extractors.ollama_extractor import ExtractedMemory
from src.loaders.mem0_loader import Mem0Loader


class TestMem0Loader:
    """Test cases for Mem0Loader."""

    @patch("src.loaders.mem0_loader.MemoryClient")
    def test_init_basic(self, mock_memory_client_class):
        """Test basic initialization."""
        mock_client = Mock()
        mock_memory_client_class.return_value = mock_client

        loader = Mem0Loader(api_key="test_key", user_id="test_user")
        assert loader.api_key == "test_key"
        assert loader.user_id == "test_user"
        assert loader.client == mock_client

    @patch("src.loaders.mem0_loader.MemoryClient")
    def test_prepare_memories_for_upload(self, mock_memory_client_class):
        """Test preparing memories for upload."""
        mock_client = Mock()
        mock_client.get_all.return_value = []  # Mock empty existing memories
        mock_memory_client_class.return_value = mock_client

        loader = Mem0Loader(api_key="test_key", user_id="test_user")

        memories = [
            ExtractedMemory(
                "User likes Python", "preference", 0.9, "context1", {"source": "conv1"}
            ),
            ExtractedMemory(
                "User is engineer", "fact", 0.95, "context2", {"source": "conv2"}
            ),
        ]

        prepared = loader.prepare_memories_for_upload(memories)

        assert len(prepared) == 2

        # Check first memory preparation - it should return ExtractedMemory objects
        mem1 = prepared[0]
        assert mem1.content == "User likes Python"
        assert mem1.category == "preference"
        assert mem1.confidence == 0.9
        assert mem1.context == "context1"
        assert mem1.metadata["source"] == "conv1"

    @patch("src.loaders.mem0_loader.MemoryClient")
    def test_prepare_memories_empty_list(self, mock_memory_client_class):
        """Test preparing empty memory list."""
        mock_client = Mock()
        mock_client.get_all.return_value = []  # Mock empty existing memories
        mock_memory_client_class.return_value = mock_client

        loader = Mem0Loader(api_key="test_key", user_id="test_user")

        prepared = loader.prepare_memories_for_upload([])
        assert prepared == []

    @patch("src.loaders.mem0_loader.MemoryClient")
    def test_load_memories_success(self, mock_memory_client_class):
        """Test successful memory loading."""
        mock_client = Mock()
        mock_client.add.return_value = {"id": "mem_123", "status": "success"}
        mock_client.get_all.return_value = []  # No existing memories
        mock_memory_client_class.return_value = mock_client

        loader = Mem0Loader(api_key="test_key", user_id="test_user")

        memories = [
            ExtractedMemory(
                content="User likes Python",
                category="preference",
                confidence=0.9,
                context="test context",
                metadata={"source": "test"},
            ),
            ExtractedMemory(
                content="User is engineer",
                category="fact",
                confidence=0.95,
                context="test context",
                metadata={"source": "test"},
            ),
        ]

        stats = loader.load_memories(memories, batch_size=10)

        assert stats["total_processed"] == 2
        assert stats["uploaded"] == 2
        assert stats["failed"] == 0
        assert stats["success_rate"] == 1.0

        # Verify client was called for each memory
        assert mock_client.add.call_count == 2

    @patch("src.loaders.mem0_loader.MemoryClient")
    def test_load_memories_with_upload_errors(self, mock_memory_client_class):
        """Test memory loading with some upload failures."""
        mock_client = Mock()
        mock_client.get_all.return_value = []  # No existing memories
        # First call succeeds, second fails
        mock_client.add.side_effect = [
            {"id": "mem_123", "status": "success"},
            Exception("Upload failed"),
        ]
        mock_memory_client_class.return_value = mock_client

        loader = Mem0Loader(api_key="test_key", user_id="test_user")

        memories = [
            ExtractedMemory(
                "User likes Python", "preference", 0.9, "context1", {"source": "test"}
            ),
            ExtractedMemory(
                "User is engineer", "fact", 0.95, "context2", {"source": "test"}
            ),
        ]

        stats = loader.load_memories(memories, batch_size=10)

        assert stats["total_processed"] == 2
        assert stats["uploaded"] == 1
        assert stats["failed"] == 1
        assert stats["success_rate"] == 0.5

    @patch("src.loaders.mem0_loader.MemoryClient")
    def test_load_memories_batching(self, mock_memory_client_class):
        """Test memory loading with batching."""
        mock_client = Mock()
        mock_client.add.return_value = {"id": "mem_123", "status": "success"}
        mock_client.get_all.return_value = []  # No existing memories
        mock_memory_client_class.return_value = mock_client

        loader = Mem0Loader(api_key="test_key", user_id="test_user")

        # Create 5 memories to test batching with batch_size=2
        memories = [
            ExtractedMemory(
                f"Memory {i}", "fact", 0.8, f"context{i}", {"source": "test"}
            )
            for i in range(5)
        ]

        stats = loader.load_memories(memories, batch_size=2)

        assert stats["total_processed"] == 5
        assert stats["uploaded"] == 5
        assert stats["failed"] == 0
        assert stats["success_rate"] == 1.0

        # Should be called once for each memory
        assert mock_client.add.call_count == 5

    @patch("src.loaders.mem0_loader.MemoryClient")
    def test_get_existing_memories_success(self, mock_memory_client_class):
        """Test successful retrieval of existing memories."""
        mock_client = Mock()
        mock_client.get_all.return_value = [
            {"id": "1", "content": "Existing memory 1"},
            {"id": "2", "content": "Existing memory 2"},
        ]
        mock_memory_client_class.return_value = mock_client

        loader = Mem0Loader(api_key="test_key", user_id="test_user")

        memories = loader.get_existing_memories()

        assert len(memories) == 2
        assert memories[0]["content"] == "Existing memory 1"
        mock_client.get_all.assert_called_once_with(user_id="test_user")

    @patch("src.loaders.mem0_loader.MemoryClient")
    def test_get_existing_memories_api_error(self, mock_memory_client_class):
        """Test handling of API errors when getting existing memories."""
        mock_client = Mock()
        mock_client.get_all.side_effect = Exception("API Error")
        mock_memory_client_class.return_value = mock_client

        loader = Mem0Loader(api_key="test_key", user_id="test_user")

        memories = loader.get_existing_memories()

        # Should return empty list on error
        assert memories == []

    @patch("src.loaders.mem0_loader.MemoryClient")
    def test_validate_memory_valid(self, mock_memory_client_class):
        """Test memory validation with valid memory."""
        mock_client = Mock()
        mock_memory_client_class.return_value = mock_client

        loader = Mem0Loader(api_key="test_key", user_id="test_user")

        memory = ExtractedMemory(
            "Valid content", "preference", 0.8, "context", {"source": "test"}
        )

        assert loader.validate_memory(memory)

    @patch("src.loaders.mem0_loader.MemoryClient")
    def test_validate_memory_low_confidence(self, mock_memory_client_class):
        """Test memory validation with low confidence memory."""
        mock_client = Mock()
        mock_memory_client_class.return_value = mock_client

        loader = Mem0Loader(api_key="test_key", user_id="test_user")

        memory = ExtractedMemory(
            "Valid content", "preference", 0.3, "context", {"source": "test"}
        )

        # Should be invalid due to low confidence (below default 0.7 threshold)
        assert not loader.validate_memory(memory)
