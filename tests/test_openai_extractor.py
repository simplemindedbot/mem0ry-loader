"""Fixed tests for OpenAI extractor focusing on public interface."""

import json
from unittest.mock import Mock, patch

from src.config.settings import OpenAIModel
from src.extractors.openai_extractor import OpenAIExtractor


class TestOpenAIExtractor:
    """Test cases for OpenAIExtractor focusing on public interface."""

    @patch("src.extractors.openai_extractor.OpenAI")
    def test_init_basic(self, mock_openai_class):
        """Test basic initialization."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        extractor = OpenAIExtractor(model=OpenAIModel.GPT_4O_MINI, api_key="test-key")
        assert extractor.model == OpenAIModel.GPT_4O_MINI
        assert not extractor.use_batch
        assert extractor.client == mock_client

    @patch("src.extractors.openai_extractor.OpenAI")
    def test_init_with_batch(self, mock_openai_class):
        """Test initialization with batch processing."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        extractor = OpenAIExtractor(
            model=OpenAIModel.GPT_4O_MINI, api_key="test-key", use_batch=True
        )
        assert extractor.use_batch
        assert extractor.batch_requests == []

    @patch("src.extractors.openai_extractor.OpenAI")
    def test_extract_memories_success(self, mock_openai_class):
        """Test successful memory extraction."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock the OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "memories": [
                    {
                        "content": "User prefers Python programming",
                        "category": "preference",
                        "confidence": 0.9,
                        "reasoning": "Direct statement",
                    }
                ]
            }
        )
        mock_client.chat.completions.create.return_value = mock_response

        extractor = OpenAIExtractor(model=OpenAIModel.GPT_4O_MINI, api_key="test-key")

        memories = extractor.extract_memories(
            "I love Python programming", "Programming discussion"
        )

        assert len(memories) == 1
        assert memories[0].content == "User prefers Python programming"
        assert memories[0].category == "preference"
        assert memories[0].confidence == 0.9

    @patch("src.extractors.openai_extractor.OpenAI")
    def test_extract_memories_api_error(self, mock_openai_class):
        """Test handling of OpenAI API errors."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock API error
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        extractor = OpenAIExtractor(model=OpenAIModel.GPT_4O_MINI, api_key="test-key")

        memories = extractor.extract_memories("Test text", "Test context")
        assert memories == []

    @patch("src.extractors.openai_extractor.OpenAI")
    def test_extract_memories_invalid_json_response(self, mock_openai_class):
        """Test handling of invalid JSON responses."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock response with invalid JSON
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Invalid JSON response"
        mock_client.chat.completions.create.return_value = mock_response

        extractor = OpenAIExtractor(model=OpenAIModel.GPT_4O_MINI, api_key="test-key")

        memories = extractor.extract_memories("Test text", "Test context")
        assert memories == []

    @patch("src.extractors.openai_extractor.OpenAI")
    def test_extract_memories_empty_response(self, mock_openai_class):
        """Test handling of empty memory responses."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mock response with no memories
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({"memories": []})
        mock_client.chat.completions.create.return_value = mock_response

        extractor = OpenAIExtractor(model=OpenAIModel.GPT_4O_MINI, api_key="test-key")

        memories = extractor.extract_memories("Boring text", "Test context")
        assert memories == []

    @patch("src.extractors.openai_extractor.OpenAI")
    def test_batch_processing_integration(self, mock_openai_class):
        """Test batch processing functionality."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        extractor = OpenAIExtractor(
            model=OpenAIModel.GPT_4O_MINI, api_key="test-key", use_batch=True
        )

        # Test that batch requests are collected but not processed immediately
        # This is a simplified test of the batch functionality
        assert extractor.use_batch
        assert len(extractor.batch_requests) == 0

        # In batch mode, extract_memories would typically collect requests
        # The actual batch processing is complex and would require more setup

    @patch("src.extractors.openai_extractor.OpenAI")
    def test_parse_openai_response_success(self, mock_openai_class):
        """Test successful OpenAI response parsing."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        extractor = OpenAIExtractor(model=OpenAIModel.GPT_4O_MINI, api_key="test-key")

        response_content = json.dumps(
            {
                "memories": [
                    {
                        "content": "Test memory",
                        "category": "fact",
                        "confidence": 0.85,
                        "reasoning": "Test reasoning",
                    }
                ]
            }
        )

        memories = extractor._parse_openai_response(response_content, "Test context")

        assert len(memories) == 1
        assert memories[0].content == "Test memory"
        assert memories[0].category == "fact"
        assert memories[0].confidence == 0.85
        assert memories[0].context == "Test context"

    @patch("src.extractors.openai_extractor.OpenAI")
    def test_parse_openai_response_invalid_json(self, mock_openai_class):
        """Test parsing invalid JSON response."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        extractor = OpenAIExtractor(model=OpenAIModel.GPT_4O_MINI, api_key="test-key")

        memories = extractor._parse_openai_response("Invalid JSON", "Test context")
        assert memories == []

    @patch("src.extractors.openai_extractor.OpenAI")
    def test_parse_openai_response_missing_memories_key(self, mock_openai_class):
        """Test parsing response missing memories key."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        extractor = OpenAIExtractor(model=OpenAIModel.GPT_4O_MINI, api_key="test-key")

        response_content = json.dumps({"data": "no memories key"})
        memories = extractor._parse_openai_response(response_content, "Test context")
        assert memories == []
