"""Tests for OpenAI extractor."""

import json
from unittest.mock import Mock, patch

from src.extractors.openai_extractor import OpenAIExtractor


class TestOpenAIExtractor:
    """Test cases for OpenAIExtractor."""

    @patch("src.extractors.openai_extractor.OpenAI")
    def test_init_basic(self, mock_openai_class):
        """Test basic initialization."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        extractor = OpenAIExtractor(model="gpt-4o-mini", api_key="test-key")
        assert extractor.model == "gpt-4o-mini"
        assert not extractor.use_batch
        assert extractor.client == mock_client

    @patch("src.extractors.openai_extractor.OpenAI")
    def test_init_with_batch(self, mock_openai_class):
        """Test initialization with batch processing."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        extractor = OpenAIExtractor(
            model="gpt-4o-mini", api_key="test-key", use_batch=True
        )
        assert extractor.use_batch
        assert extractor.batch_requests == []

    @patch("src.extractors.openai_extractor.OpenAI")
    def test_extract_memories_success(self, mock_openai_class):
        """Test successful memory extraction."""
        # Mock OpenAI client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps(
            {
                "memories": [
                    {
                        "content": "User is a software engineer",
                        "category": "fact",
                        "confidence": 0.95,
                    },
                    {
                        "content": "User prefers Python",
                        "category": "preference",
                        "confidence": 0.90,
                    },
                ]
            }
        )
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        extractor = OpenAIExtractor(model="gpt-4o-mini", api_key="test-key")

        conversation_text = "I am a software engineer and I love Python programming"
        memories = extractor.extract_memories(conversation_text, "Test conversation")

        assert len(memories) == 2
        assert memories[0].content == "User is a software engineer"
        assert memories[0].category == "fact"
        assert memories[0].confidence == 0.95
        assert memories[1].content == "User prefers Python"
        assert memories[1].category == "preference"
        assert memories[1].confidence == 0.90

    @patch("src.extractors.openai_extractor.OpenAI")
    def test_extract_memories_batch_mode(self, mock_openai_class):
        """Test memory extraction in batch mode."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        extractor = OpenAIExtractor(
            model="gpt-4o-mini", api_key="test-key", use_batch=True
        )

        conversation_text = "I am a software engineer"
        memories = extractor.extract_memories(conversation_text, "Test conversation")

        # In batch mode, should return empty list and add to batch_requests
        assert memories == []
        assert len(extractor.batch_requests) == 1

        # Check that request was added to batch
        batch_request = extractor.batch_requests[0]
        assert batch_request["custom_id"] is not None
        assert "messages" in batch_request["body"]

    @patch("src.extractors.openai_extractor.OpenAI")
    def test_extract_memories_api_error(self, mock_openai_class):
        """Test extraction with API error."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai_class.return_value = mock_client

        extractor = OpenAIExtractor(model="gpt-4o-mini", api_key="test-key")

        memories = extractor.extract_memories("Some text", "Test")
        assert memories == []

    @patch("src.extractors.openai_extractor.OpenAI")
    def test_extract_memories_invalid_json_response(self, mock_openai_class):
        """Test extraction with invalid JSON response."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "invalid json"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        extractor = OpenAIExtractor(model="gpt-4o-mini", api_key="test-key")

        memories = extractor.extract_memories("Some text", "Test")
        assert memories == []

    @patch("src.extractors.openai_extractor.OpenAI")
    def test_extract_memories_empty_response(self, mock_openai_class):
        """Test extraction with empty memories response."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({"memories": []})
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        extractor = OpenAIExtractor(model="gpt-4o-mini", api_key="test-key")

        memories = extractor.extract_memories("No interesting content", "Test")
        assert memories == []

    @patch("src.extractors.openai_extractor.OpenAI")
    def test_build_extraction_messages(self, mock_openai_class):
        """Test building extraction messages."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        extractor = OpenAIExtractor(model="gpt-4o-mini", api_key="test-key")

        messages = extractor._build_extraction_messages(
            "Test conversation", "Test title"
        )

        assert len(messages) == 2  # System and user messages
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Test conversation" in messages[1]["content"]
        assert "Test title" in messages[1]["content"]

        # Check that all categories are mentioned in system message
        categories = [
            "preference",
            "fact",
            "pattern",
            "goal",
            "skill",
            "relationship",
            "context",
            "decision_criteria",
        ]
        system_content = messages[0]["content"]
        for category in categories:
            assert category in system_content

    @patch("src.extractors.openai_extractor.OpenAI")
    def test_parse_openai_response_success(self, mock_openai_class):
        """Test successful response parsing."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        extractor = OpenAIExtractor(model="gpt-4o-mini", api_key="test-key")

        response_content = json.dumps(
            {
                "memories": [
                    {"content": "Test memory", "category": "fact", "confidence": 0.85}
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
        """Test response parsing with invalid JSON."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        extractor = OpenAIExtractor(model="gpt-4o-mini", api_key="test-key")

        memories = extractor._parse_openai_response("invalid json", "Test context")
        assert memories == []

    @patch("src.extractors.openai_extractor.OpenAI")
    def test_parse_openai_response_missing_memories_key(self, mock_openai_class):
        """Test response parsing with missing memories key."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        extractor = OpenAIExtractor(model="gpt-4o-mini", api_key="test-key")

        response_content = json.dumps({"other_key": "value"})
        memories = extractor._parse_openai_response(response_content, "Test context")
        assert memories == []

    @patch("src.extractors.openai_extractor.OpenAI")
    @patch("src.extractors.openai_extractor.time.sleep")
    def test_process_batch_success(self, mock_sleep, mock_openai_class):
        """Test successful batch processing."""
        mock_client = Mock()

        # Mock batch creation
        mock_batch_response = Mock()
        mock_batch_response.id = "batch_123"
        mock_client.batches.create.return_value = mock_batch_response

        # Mock batch status checking (completed)
        mock_status_response = Mock()
        mock_status_response.status = "completed"
        mock_status_response.output_file_id = "file_123"
        mock_client.batches.retrieve.return_value = mock_status_response

        # Mock file content retrieval
        mock_file_content = Mock()
        mock_file_content.content = b'{"custom_id": "req_1", "response": {"body": {"choices": [{"message": {"content": "{\\"memories\\": [{\\"content\\": \\"Test memory\\", \\"category\\": \\"fact\\", \\"confidence\\": 0.9}]}"}}]}}}\n'
        mock_client.files.content.return_value = mock_file_content

        mock_openai_class.return_value = mock_client

        extractor = OpenAIExtractor(
            model="gpt-4o-mini", api_key="test-key", use_batch=True
        )

        # Add a request to batch
        extractor.batch_requests = [
            {
                "custom_id": "req_1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "test"}],
                },
            }
        ]

        memories = extractor.process_batch()

        assert len(memories) == 1
        assert memories[0].content == "Test memory"
        assert memories[0].category == "fact"
        assert memories[0].confidence == 0.9

    @patch("src.extractors.openai_extractor.OpenAI")
    def test_process_batch_no_requests(self, mock_openai_class):
        """Test batch processing with no requests."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        extractor = OpenAIExtractor(
            model="gpt-4o-mini", api_key="test-key", use_batch=True
        )

        memories = extractor.process_batch()
        assert memories == []

    @patch("src.extractors.openai_extractor.OpenAI")
    @patch("src.extractors.openai_extractor.time.sleep")
    def test_process_batch_failure(self, mock_sleep, mock_openai_class):
        """Test batch processing with failure."""
        mock_client = Mock()

        # Mock batch creation failure
        mock_client.batches.create.side_effect = Exception("Batch creation failed")

        mock_openai_class.return_value = mock_client

        extractor = OpenAIExtractor(
            model="gpt-4o-mini", api_key="test-key", use_batch=True
        )
        extractor.batch_requests = [
            {
                "custom_id": "req_1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {},
            }
        ]

        memories = extractor.process_batch()
        assert memories == []

    @patch("src.extractors.openai_extractor.OpenAI")
    def test_validate_memory_valid(self, mock_openai_class):
        """Test memory validation with valid memory."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        extractor = OpenAIExtractor(model="gpt-4o-mini", api_key="test-key")

        memory_data = {
            "content": "User likes Python",
            "category": "preference",
            "confidence": 0.9,
        }

        assert extractor._validate_memory(memory_data)

    @patch("src.extractors.openai_extractor.OpenAI")
    def test_validate_memory_invalid_category(self, mock_openai_class):
        """Test memory validation with invalid category."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        extractor = OpenAIExtractor(model="gpt-4o-mini", api_key="test-key")

        memory_data = {
            "content": "User likes Python",
            "category": "invalid_category",
            "confidence": 0.9,
        }

        assert not extractor._validate_memory(memory_data)

    @patch("src.extractors.openai_extractor.OpenAI")
    def test_validate_memory_invalid_confidence(self, mock_openai_class):
        """Test memory validation with invalid confidence."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        extractor = OpenAIExtractor(model="gpt-4o-mini", api_key="test-key")

        memory_data = {
            "content": "User likes Python",
            "category": "preference",
            "confidence": 1.5,  # Invalid confidence > 1.0
        }

        assert not extractor._validate_memory(memory_data)

    @patch("src.extractors.openai_extractor.OpenAI")
    def test_validate_memory_missing_content(self, mock_openai_class):
        """Test memory validation with missing content."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        extractor = OpenAIExtractor(model="gpt-4o-mini", api_key="test-key")

        memory_data = {
            "category": "preference",
            "confidence": 0.9,
            # Missing content
        }

        assert not extractor._validate_memory(memory_data)

    @patch("src.extractors.openai_extractor.OpenAI")
    def test_validate_memory_empty_content(self, mock_openai_class):
        """Test memory validation with empty content."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        extractor = OpenAIExtractor(model="gpt-4o-mini", api_key="test-key")

        memory_data = {"content": "", "category": "preference", "confidence": 0.9}

        assert not extractor._validate_memory(memory_data)
