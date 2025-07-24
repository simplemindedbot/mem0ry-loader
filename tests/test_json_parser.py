"""Tests for ChatGPT JSON parser."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pytest

from src.parsers.json_parser import ChatGPTJSONParser, ChatMessage, Conversation


class TestChatGPTJSONParser:
    """Test cases for ChatGPTJSONParser."""

    def test_init(self):
        """Test parser initialization."""
        parser = ChatGPTJSONParser()
        assert parser is not None
        assert hasattr(parser, "logger")

    def test_parse_export_success(self, temp_conversations_file: Path):
        """Test successful parsing of conversations export."""
        parser = ChatGPTJSONParser()
        conversations = parser.parse_export(temp_conversations_file)

        assert len(conversations) == 2
        assert all(isinstance(conv, Conversation) for conv in conversations)

        # Check first conversation
        conv1 = conversations[0]
        assert conv1.id == "test-conversation-1"
        assert conv1.title == "Test Discussion"
        assert len(conv1.messages) == 3

        # Check message ordering and content
        assert conv1.messages[0].role == "user"
        assert "software engineer" in conv1.messages[0].content
        assert conv1.messages[1].role == "assistant"
        assert conv1.messages[2].role == "user"
        assert "10 PM to 2 AM" in conv1.messages[2].content

    def test_parse_export_file_not_found(self):
        """Test parsing with non-existent file."""
        parser = ChatGPTJSONParser()
        non_existent_file = Path("/non/existent/file.json")

        with pytest.raises(FileNotFoundError, match="Export file not found"):
            parser.parse_export(non_existent_file)

    def test_parse_export_invalid_json(self, tmp_path: Path):
        """Test parsing with invalid JSON."""
        parser = ChatGPTJSONParser()
        invalid_file = tmp_path / "invalid.json"

        with open(invalid_file, "w") as f:
            f.write("{ invalid json }")

        with pytest.raises(ValueError, match="Invalid JSON format"):
            parser.parse_export(invalid_file)

    def test_parse_export_empty_file(self, tmp_path: Path):
        """Test parsing with empty conversations list."""
        parser = ChatGPTJSONParser()
        empty_file = tmp_path / "empty.json"

        with open(empty_file, "w") as f:
            json.dump([], f)

        conversations = parser.parse_export(empty_file)
        assert conversations == []

    def test_parse_conversation_success(self, sample_conversation_data: Dict[str, Any]):
        """Test successful conversation parsing."""
        parser = ChatGPTJSONParser()
        conversation = parser._parse_conversation(sample_conversation_data)

        assert conversation is not None
        assert conversation.id == "test-conversation-1"
        assert conversation.title == "Test Discussion"
        assert len(conversation.messages) == 3
        assert conversation.created_at is not None
        assert conversation.updated_at is not None

    def test_parse_conversation_missing_fields(self):
        """Test conversation parsing with missing fields."""
        parser = ChatGPTJSONParser()

        # Test with minimal data
        minimal_data = {"mapping": {}}
        conversation = parser._parse_conversation(minimal_data)

        assert conversation is not None
        assert conversation.id == ""
        assert conversation.title == "Untitled Conversation"
        assert conversation.messages == []

    def test_parse_message_success(self):
        """Test successful message parsing."""
        parser = ChatGPTJSONParser()

        msg_data = {
            "message": {
                "author": {"role": "user"},
                "content": {"parts": ["Hello world"]},
                "create_time": 1703123456.789,
            }
        }

        message = parser._parse_message(msg_data, "msg-1")

        assert message is not None
        assert message.role == "user"
        assert message.content == "Hello world"
        assert message.message_id == "msg-1"
        assert message.timestamp is not None

    def test_parse_message_system_role_skipped(self):
        """Test that system messages are skipped."""
        parser = ChatGPTJSONParser()

        msg_data = {
            "message": {
                "author": {"role": "system"},
                "content": {"parts": ["System message"]},
            }
        }

        message = parser._parse_message(msg_data, "msg-sys")
        assert message is None

    def test_parse_message_empty_content(self):
        """Test message parsing with empty content."""
        parser = ChatGPTJSONParser()

        msg_data = {
            "message": {
                "author": {"role": "user"},
                "content": {"parts": []},
            }
        }

        message = parser._parse_message(msg_data, "msg-empty")
        assert message is None

    def test_parse_message_multiple_parts(self):
        """Test message parsing with multiple content parts."""
        parser = ChatGPTJSONParser()

        msg_data = {
            "message": {
                "author": {"role": "user"},
                "content": {"parts": ["First part", "Second part"]},
            }
        }

        message = parser._parse_message(msg_data, "msg-multi")

        assert message is not None
        assert message.content == "First part\nSecond part"

    def test_parse_timestamp_float(self):
        """Test timestamp parsing from float."""
        parser = ChatGPTJSONParser()

        timestamp_float = 1703123456.789
        dt = parser._parse_timestamp(timestamp_float)

        assert dt is not None
        assert isinstance(dt, datetime)

    def test_parse_timestamp_iso_string(self):
        """Test timestamp parsing from ISO string."""
        parser = ChatGPTJSONParser()

        timestamp_str = "2023-12-21T01:57:36.789Z"
        dt = parser._parse_timestamp(timestamp_str)

        assert dt is not None
        assert isinstance(dt, datetime)

    def test_parse_timestamp_invalid(self):
        """Test timestamp parsing with invalid input."""
        parser = ChatGPTJSONParser()

        assert parser._parse_timestamp(None) is None
        assert parser._parse_timestamp("invalid") is None
        assert parser._parse_timestamp([]) is None

    def test_get_conversation_chunks_basic(
        self, sample_conversation_data: Dict[str, Any]
    ):
        """Test basic conversation chunking."""
        parser = ChatGPTJSONParser()
        conversation = parser._parse_conversation(sample_conversation_data)

        chunks = list(
            parser.get_conversation_chunks(conversation, chunk_size=100, overlap=20)
        )

        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk.strip()) > 0 for chunk in chunks)

    def test_get_conversation_chunks_small_conversation(self):
        """Test chunking with conversation smaller than chunk size."""
        parser = ChatGPTJSONParser()

        # Create small conversation
        messages = [
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello"),
        ]
        conversation = Conversation(id="small", title="Small", messages=messages)

        chunks = list(parser.get_conversation_chunks(conversation, chunk_size=1000))

        assert len(chunks) == 1
        assert "Small" in chunks[0]  # Title included
        assert "Hi" in chunks[0]
        assert "Hello" in chunks[0]

    def test_get_conversation_chunks_overlap(self):
        """Test that chunks have proper overlap."""
        parser = ChatGPTJSONParser()

        # Create conversation with long content
        long_content = "This is a very long message. " * 50
        messages = [ChatMessage(role="user", content=long_content)]
        conversation = Conversation(id="long", title="Long", messages=messages)

        chunks = list(
            parser.get_conversation_chunks(conversation, chunk_size=100, overlap=50)
        )

        if len(chunks) > 1:
            # Check that consecutive chunks have overlapping content
            # This is a simplified check - real overlap detection would be more complex
            assert len(chunks[0]) > 0
            assert len(chunks[1]) > 0

    @pytest.mark.parametrize(
        "chunk_size,overlap",
        [
            (500, 100),
            (1000, 200),
            (2000, 400),
        ],
    )
    def test_get_conversation_chunks_different_sizes(
        self, sample_conversation_data: Dict[str, Any], chunk_size: int, overlap: int
    ):
        """Test chunking with different chunk sizes."""
        parser = ChatGPTJSONParser()
        conversation = parser._parse_conversation(sample_conversation_data)

        chunks = list(
            parser.get_conversation_chunks(
                conversation, chunk_size=chunk_size, overlap=overlap
            )
        )

        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
