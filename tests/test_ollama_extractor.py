"""Tests for Ollama extractor."""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
import requests

from src.extractors.ollama_extractor import OllamaExtractor, ExtractedMemory
from src.config.settings import OllamaModel


class TestExtractedMemory:
    """Test cases for ExtractedMemory dataclass."""

    def test_init_basic(self):
        """Test basic initialization."""
        memory = ExtractedMemory("Test content", "fact", 0.9, "context")
        assert memory.content == "Test content"
        assert memory.category == "fact"
        assert memory.confidence == 0.9
        assert memory.context == "context"
        assert memory.metadata == {}

    def test_init_with_metadata(self):
        """Test initialization with metadata."""
        metadata = {"source": "test"}
        memory = ExtractedMemory("Test content", "fact", 0.9, "context", metadata)
        assert memory.metadata == metadata

    def test_post_init_none_metadata(self):
        """Test that None metadata is converted to empty dict."""
        memory = ExtractedMemory("Test content", "fact", 0.9, "context", None)
        assert memory.metadata == {}


class TestOllamaExtractor:
    """Test cases for OllamaExtractor."""

    @patch('src.extractors.ollama_extractor.requests.get')
    def test_init_model_available(self, mock_get):
        """Test initialization when model is available."""
        # Mock successful model check
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "models": [{"name": "nuextract"}, {"name": "llama3.2:1b"}]
        }
        mock_get.return_value = mock_response

        extractor = OllamaExtractor(model=OllamaModel.NUEXTRACT)
        assert extractor.model == OllamaModel.NUEXTRACT
        assert extractor.base_url == "http://localhost:11434"

    @patch('src.extractors.ollama_extractor.requests.get')
    @patch('src.extractors.ollama_extractor.requests.post')
    def test_init_model_not_available(self, mock_post, mock_get):
        """Test initialization when model needs to be pulled."""
        # Mock model not available
        mock_get_response = Mock()
        mock_get_response.raise_for_status.return_value = None
        mock_get_response.json.return_value = {"models": []}
        mock_get.return_value = mock_get_response
        
        # Mock pull model success
        mock_post_response = Mock()
        mock_post_response.raise_for_status.return_value = None
        mock_post.return_value = mock_post_response

        extractor = OllamaExtractor(model=OllamaModel.NUEXTRACT)
        
        # Should have attempted to pull the model
        mock_post.assert_called_once()

    @patch('src.extractors.ollama_extractor.requests.get')
    def test_init_connection_error(self, mock_get):
        """Test initialization with connection error."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
        
        with pytest.raises(Exception):  # Should raise connection error
            OllamaExtractor(model=OllamaModel.NUEXTRACT)

    @patch('src.extractors.ollama_extractor.OllamaExtractor._ensure_model_available')
    @patch('src.extractors.ollama_extractor.requests.post')
    def test_extract_memories_success(self, mock_post, mock_ensure):
        """Test successful memory extraction."""
        # Mock model availability check
        mock_ensure.return_value = None
        
        # Create extractor
        extractor = OllamaExtractor(model=OllamaModel.NUEXTRACT)
        
        # Mock API response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "response": json.dumps({
                "memories": [
                    {
                        "content": "User is a software engineer",
                        "category": "fact",
                        "confidence": 0.95
                    },
                    {
                        "content": "User prefers Python",
                        "category": "preference",
                        "confidence": 0.90
                    }
                ]
            })
        }
        mock_post.return_value = mock_response
        
        # Test extraction
        conversation_text = "I am a software engineer and I love Python programming"
        memories = extractor.extract_memories(conversation_text, "Test conversation")
        
        assert len(memories) == 2
        assert memories[0].content == "User is a software engineer"
        assert memories[0].category == "fact"
        assert memories[0].confidence == 0.95
        assert memories[1].content == "User prefers Python"
        assert memories[1].category == "preference"
        assert memories[1].confidence == 0.90

    @patch('src.extractors.ollama_extractor.OllamaExtractor._ensure_model_available')
    @patch('src.extractors.ollama_extractor.requests.post')
    def test_extract_memories_empty_response(self, mock_post, mock_ensure):
        """Test extraction with empty response."""
        mock_ensure.return_value = None
        extractor = OllamaExtractor(model=OllamaModel.NUEXTRACT)
        
        # Mock empty response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"response": json.dumps({"memories": []})}
        mock_post.return_value = mock_response
        
        memories = extractor.extract_memories("No interesting content", "Test")
        assert memories == []

    @patch('src.extractors.ollama_extractor.OllamaExtractor._ensure_model_available')
    @patch('src.extractors.ollama_extractor.requests.post')
    def test_extract_memories_invalid_json_response(self, mock_post, mock_ensure):
        """Test extraction with invalid JSON in response."""
        mock_ensure.return_value = None
        extractor = OllamaExtractor(model=OllamaModel.NUEXTRACT)
        
        # Mock invalid JSON response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"response": "invalid json"}
        mock_post.return_value = mock_response
        
        memories = extractor.extract_memories("Some text", "Test")
        assert memories == []

    @patch('src.extractors.ollama_extractor.OllamaExtractor._ensure_model_available')
    @patch('src.extractors.ollama_extractor.requests.post')
    def test_extract_memories_api_error(self, mock_post, mock_ensure):
        """Test extraction with API error."""
        mock_ensure.return_value = None
        extractor = OllamaExtractor(model=OllamaModel.NUEXTRACT)
        
        # Mock API error
        mock_post.side_effect = requests.exceptions.RequestException("API Error")
        
        memories = extractor.extract_memories("Some text", "Test")
        assert memories == []

    @patch('src.extractors.ollama_extractor.OllamaExtractor._ensure_model_available')
    @patch('src.extractors.ollama_extractor.requests.post')
    def test_extract_memories_timeout(self, mock_post, mock_ensure):
        """Test extraction with timeout."""
        mock_ensure.return_value = None
        extractor = OllamaExtractor(model=OllamaModel.NUEXTRACT)
        
        # Mock timeout
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")
        
        memories = extractor.extract_memories("Some text", "Test")
        assert memories == []

    @patch('src.extractors.ollama_extractor.OllamaExtractor._ensure_model_available')
    def test_build_extraction_prompt(self, mock_ensure):
        """Test prompt building."""
        mock_ensure.return_value = None
        extractor = OllamaExtractor(model=OllamaModel.NUEXTRACT)
        
        prompt = extractor._build_extraction_prompt("Test conversation", "Test title")
        
        assert "Test conversation" in prompt
        assert "Test title" in prompt
        assert "JSON" in prompt
        assert "memories" in prompt
        
        # Check that all categories are mentioned
        categories = ["preference", "fact", "pattern", "goal", "skill", "relationship", "context", "decision_criteria"]
        for category in categories:
            assert category in prompt

    @patch('src.extractors.ollama_extractor.OllamaExtractor._ensure_model_available')
    @patch('src.extractors.ollama_extractor.requests.post')
    def test_parse_ollama_response_success(self, mock_post, mock_ensure):
        """Test successful response parsing."""
        mock_ensure.return_value = None
        extractor = OllamaExtractor(model=OllamaModel.NUEXTRACT)
        
        response_data = {
            "response": json.dumps({
                "memories": [
                    {
                        "content": "Test memory",
                        "category": "fact",
                        "confidence": 0.85
                    }
                ]
            })
        }
        
        memories = extractor._parse_ollama_response(response_data, "Test context")
        
        assert len(memories) == 1
        assert memories[0].content == "Test memory"
        assert memories[0].category == "fact"
        assert memories[0].confidence == 0.85
        assert memories[0].context == "Test context"

    @patch('src.extractors.ollama_extractor.OllamaExtractor._ensure_model_available')
    def test_parse_ollama_response_invalid_json(self, mock_ensure):
        """Test response parsing with invalid JSON."""
        mock_ensure.return_value = None
        extractor = OllamaExtractor(model=OllamaModel.NUEXTRACT)
        
        response_data = {"response": "invalid json"}
        memories = extractor._parse_ollama_response(response_data, "Test context")
        
        assert memories == []

    @patch('src.extractors.ollama_extractor.OllamaExtractor._ensure_model_available')
    def test_parse_ollama_response_missing_fields(self, mock_ensure):
        """Test response parsing with missing fields."""
        mock_ensure.return_value = None
        extractor = OllamaExtractor(model=OllamaModel.NUEXTRACT)
        
        response_data = {
            "response": json.dumps({
                "memories": [
                    {
                        "content": "Test memory",
                        # Missing category and confidence
                    }
                ]
            })
        }
        
        memories = extractor._parse_ollama_response(response_data, "Test context")
        assert memories == []

    @patch('src.extractors.ollama_extractor.OllamaExtractor._ensure_model_available')
    def test_validate_memory_valid(self, mock_ensure):
        """Test memory validation with valid memory."""
        mock_ensure.return_value = None
        extractor = OllamaExtractor(model=OllamaModel.NUEXTRACT)
        
        memory_data = {
            "content": "User likes Python",
            "category": "preference",
            "confidence": 0.9
        }
        
        assert extractor._validate_memory(memory_data) == True

    @patch('src.extractors.ollama_extractor.OllamaExtractor._ensure_model_available')
    def test_validate_memory_invalid_category(self, mock_ensure):
        """Test memory validation with invalid category."""
        mock_ensure.return_value = None
        extractor = OllamaExtractor(model=OllamaModel.NUEXTRACT)
        
        memory_data = {
            "content": "User likes Python",
            "category": "invalid_category",
            "confidence": 0.9
        }
        
        assert extractor._validate_memory(memory_data) == False

    @patch('src.extractors.ollama_extractor.OllamaExtractor._ensure_model_available')
    def test_validate_memory_invalid_confidence(self, mock_ensure):
        """Test memory validation with invalid confidence."""
        mock_ensure.return_value = None
        extractor = OllamaExtractor(model=OllamaModel.NUEXTRACT)
        
        memory_data = {
            "content": "User likes Python",
            "category": "preference",
            "confidence": 1.5  # Invalid confidence > 1.0
        }
        
        assert extractor._validate_memory(memory_data) == False

    @patch('src.extractors.ollama_extractor.OllamaExtractor._ensure_model_available')
    def test_validate_memory_missing_content(self, mock_ensure):
        """Test memory validation with missing content."""
        mock_ensure.return_value = None
        extractor = OllamaExtractor(model=OllamaModel.NUEXTRACT)
        
        memory_data = {
            "category": "preference",
            "confidence": 0.9
            # Missing content
        }
        
        assert extractor._validate_memory(memory_data) == False

    @patch('src.extractors.ollama_extractor.OllamaExtractor._ensure_model_available')
    def test_validate_memory_empty_content(self, mock_ensure):
        """Test memory validation with empty content."""
        mock_ensure.return_value = None
        extractor = OllamaExtractor(model=OllamaModel.NUEXTRACT)
        
        memory_data = {
            "content": "",
            "category": "preference",
            "confidence": 0.9
        }
        
        assert extractor._validate_memory(memory_data) == False