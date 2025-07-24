"""Tests for Mem0 loader."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

from src.loaders.mem0_loader import Mem0Loader
from src.extractors.ollama_extractor import ExtractedMemory


class TestMem0Loader:
    """Test cases for Mem0Loader."""

    @patch('src.loaders.mem0_loader.MemoryClient')
    def test_init_basic(self, mock_memory_client_class):
        """Test basic initialization."""
        mock_client = Mock()
        mock_memory_client_class.return_value = mock_client
        
        loader = Mem0Loader(api_key="test_key", user_id="test_user")
        assert loader.api_key == "test_key"
        assert loader.user_id == "test_user"
        assert loader.client == mock_client

    @patch('src.loaders.mem0_loader.MemoryClient')
    def test_prepare_memories_for_upload(self, mock_memory_client_class):
        """Test preparing memories for upload."""
        mock_client = Mock()
        mock_client.get_all.return_value = []  # Mock empty existing memories
        mock_memory_client_class.return_value = mock_client
        
        loader = Mem0Loader(api_key="test_key", user_id="test_user")
        
        memories = [
            ExtractedMemory("User likes Python", "preference", 0.9, "context1", {"source": "conv1"}),
            ExtractedMemory("User is engineer", "fact", 0.95, "context2", {"source": "conv2"}),
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

    @patch('src.loaders.mem0_loader.MemoryClient')
    def test_prepare_memories_empty_list(self, mock_memory_client_class):
        """Test preparing empty memory list."""
        mock_client = Mock()
        mock_client.get_all.return_value = []  # Mock empty existing memories
        mock_memory_client_class.return_value = mock_client
        
        loader = Mem0Loader(api_key="test_key", user_id="test_user")
        
        prepared = loader.prepare_memories_for_upload([])
        assert prepared == []

    @patch('src.loaders.mem0_loader.MemoryClient')
    def test_load_memories_success(self, mock_memory_client_class):
        """Test successful memory loading."""
        mock_client = Mock()
        mock_client.add.return_value = {"id": "mem_123", "status": "success"}
        mock_memory_client_class.return_value = mock_client
        
        loader = Mem0Loader(api_key="test_key", user_id="test_user")
        
        prepared_memories = [
            {
                "text": "User likes Python",
                "user_id": "test_user",
                "metadata": {"category": "preference", "confidence": 0.9}
            },
            {
                "text": "User is engineer", 
                "user_id": "test_user",
                "metadata": {"category": "fact", "confidence": 0.95}
            }
        ]
        
        stats = loader.load_memories(prepared_memories, batch_size=10)
        
        assert stats["total_processed"] == 2
        assert stats["uploaded"] == 2
        assert stats["failed"] == 0
        assert stats["success_rate"] == 1.0
        
        # Verify client was called for each memory
        assert mock_client.add.call_count == 2

    @patch('src.loaders.mem0_loader.MemoryClient')
    def test_load_memories_partial_failure(self, mock_memory_client_class):
        """Test memory loading with partial failures."""
        mock_client = Mock()
        
        # First call succeeds, second fails
        mock_client.add.side_effect = [
            {"id": "mem_123", "status": "success"},
            Exception("Upload failed")
        ]
        mock_memory_client_class.return_value = mock_client
        
        loader = Mem0Loader(api_key="test_key", user_id="test_user")
        
        prepared_memories = [
            {
                "text": "User likes Python",
                "user_id": "test_user", 
                "metadata": {"category": "preference"}
            },
            {
                "text": "User is engineer",
                "user_id": "test_user",
                "metadata": {"category": "fact"}
            }
        ]
        
        stats = loader.load_memories(prepared_memories, batch_size=10)
        
        assert stats["total_processed"] == 2
        assert stats["uploaded"] == 1
        assert stats["failed"] == 1
        assert stats["success_rate"] == 0.5

    @patch('src.loaders.mem0_loader.MemoryClient')
    def test_load_memories_empty_list(self, mock_memory_client_class):
        """Test loading empty memory list."""
        mock_client = Mock()
        mock_memory_client_class.return_value = mock_client
        
        loader = Mem0Loader(api_key="test_key", user_id="test_user")
        
        stats = loader.load_memories([], batch_size=10)
        
        assert stats["total_processed"] == 0
        assert stats["uploaded"] == 0
        assert stats["failed"] == 0
        assert stats["success_rate"] == 0.0

    @patch('src.loaders.mem0_loader.MemoryClient')
    def test_load_memories_batching(self, mock_memory_client_class):
        """Test memory loading with batching."""
        mock_client = Mock()
        mock_client.add.return_value = {"id": "mem_123", "status": "success"}
        mock_memory_client_class.return_value = mock_client
        
        loader = Mem0Loader(api_key="test_key", user_id="test_user")
        
        # Create 5 memories with batch size of 2
        prepared_memories = [
            {
                "text": f"Memory {i}",
                "user_id": "test_user",
                "metadata": {"category": "fact"}
            }
            for i in range(5)
        ]
        
        stats = loader.load_memories(prepared_memories, batch_size=2)
        
        assert stats["total_processed"] == 5
        assert stats["uploaded"] == 5
        assert stats["failed"] == 0
        assert stats["success_rate"] == 1.0

    @patch('src.loaders.mem0_loader.MemoryClient')
    def test_get_existing_memories_success(self, mock_memory_client_class):
        """Test getting existing memories."""
        mock_client = Mock()
        mock_client.get_all.return_value = [
            {"id": "mem_1", "text": "Memory 1"},
            {"id": "mem_2", "text": "Memory 2"}
        ]
        mock_memory_client_class.return_value = mock_client
        
        loader = Mem0Loader(api_key="test_key", user_id="test_user")
        
        existing = loader.get_existing_memories()
        
        assert len(existing) == 2
        assert existing[0]["id"] == "mem_1"
        assert existing[1]["id"] == "mem_2"
        
        mock_client.get_all.assert_called_once_with(user_id="test_user")

    @patch('src.loaders.mem0_loader.MemoryClient')
    def test_get_existing_memories_api_error(self, mock_memory_client_class):
        """Test getting existing memories with API error."""
        mock_client = Mock()
        mock_client.get_all.side_effect = Exception("API Error")
        mock_memory_client_class.return_value = mock_client
        
        loader = Mem0Loader(api_key="test_key", user_id="test_user")
        
        existing = loader.get_existing_memories()
        assert existing == []

    @patch('src.loaders.mem0_loader.MemoryClient')
    def test_delete_existing_memories_success(self, mock_memory_client_class):
        """Test deleting existing memories."""
        mock_client = Mock()
        mock_client.delete.return_value = {"status": "success"}
        mock_memory_client_class.return_value = mock_client
        
        loader = Mem0Loader(api_key="test_key", user_id="test_user")
        
        memory_ids = ["mem_1", "mem_2", "mem_3"]
        deleted_count = loader.delete_existing_memories(memory_ids)
        
        assert deleted_count == 3
        assert mock_client.delete.call_count == 3
        
        # Verify each memory was deleted
        for memory_id in memory_ids:
            mock_client.delete.assert_any_call(memory_id=memory_id)

    @patch('src.loaders.mem0_loader.MemoryClient')
    def test_delete_existing_memories_partial_failure(self, mock_memory_client_class):
        """Test deleting memories with partial failures."""
        mock_client = Mock()
        
        # First two succeed, third fails
        mock_client.delete.side_effect = [
            {"status": "success"},
            {"status": "success"},
            Exception("Delete failed")
        ]
        mock_memory_client_class.return_value = mock_client
        
        loader = Mem0Loader(api_key="test_key", user_id="test_user")
        
        memory_ids = ["mem_1", "mem_2", "mem_3"]
        deleted_count = loader.delete_existing_memories(memory_ids)
        
        assert deleted_count == 2  # Only first two succeeded

    @patch('src.loaders.mem0_loader.MemoryClient')
    def test_delete_existing_memories_empty_list(self, mock_memory_client_class):
        """Test deleting empty memory list."""
        mock_client = Mock()
        mock_memory_client_class.return_value = mock_client
        
        loader = Mem0Loader(api_key="test_key", user_id="test_user")
        
        deleted_count = loader.delete_existing_memories([])
        assert deleted_count == 0
        mock_client.delete.assert_not_called()

    @patch('src.loaders.mem0_loader.MemoryClient')
    def test_upload_memory_success(self, mock_memory_client_class):
        """Test successful single memory upload."""
        mock_client = Mock()
        mock_client.add.return_value = {"id": "mem_123", "status": "success"}
        mock_memory_client_class.return_value = mock_client
        
        loader = Mem0Loader(api_key="test_key", user_id="test_user")
        
        memory_data = {
            "text": "User likes Python",
            "user_id": "test_user",
            "metadata": {"category": "preference"}
        }
        
        success = loader._upload_memory(memory_data)
        assert success == True
        
        mock_client.add.assert_called_once_with(**memory_data)

    @patch('src.loaders.mem0_loader.MemoryClient')
    def test_upload_memory_failure(self, mock_memory_client_class):
        """Test failed single memory upload."""
        mock_client = Mock()
        mock_client.add.side_effect = Exception("Upload failed")
        mock_memory_client_class.return_value = mock_client
        
        loader = Mem0Loader(api_key="test_key", user_id="test_user")
        
        memory_data = {
            "text": "User likes Python",
            "user_id": "test_user", 
            "metadata": {"category": "preference"}
        }
        
        success = loader._upload_memory(memory_data)
        assert success == False

    @patch('src.loaders.mem0_loader.MemoryClient')
    def test_build_memory_metadata(self, mock_memory_client_class):
        """Test building memory metadata."""
        mock_client = Mock()
        mock_memory_client_class.return_value = mock_client
        
        loader = Mem0Loader(api_key="test_key", user_id="test_user")
        
        memory = ExtractedMemory(
            "User likes Python",
            "preference", 
            0.9,
            "conversation context",
            {"source": "conv1", "timestamp": "2024-01-01"}
        )
        
        metadata = loader._build_memory_metadata(memory)
        
        assert metadata["category"] == "preference"
        assert metadata["confidence"] == 0.9
        assert metadata["context"] == "conversation context"
        assert metadata["source"] == "conv1"
        assert metadata["timestamp"] == "2024-01-01"

    @patch('src.loaders.mem0_loader.MemoryClient')
    def test_build_memory_metadata_minimal(self, mock_memory_client_class):
        """Test building memory metadata with minimal data."""
        mock_client = Mock()
        mock_memory_client_class.return_value = mock_client
        
        loader = Mem0Loader(api_key="test_key", user_id="test_user")
        
        memory = ExtractedMemory("User likes Python", "preference", 0.9, "")
        
        metadata = loader._build_memory_metadata(memory)
        
        assert metadata["category"] == "preference"
        assert metadata["confidence"] == 0.9
        assert metadata["context"] == ""
        assert len(metadata) == 3  # Only basic fields