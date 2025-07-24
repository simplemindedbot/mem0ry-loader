"""Pytest configuration and fixtures for memloader tests."""

import json
import pytest
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock

from src.config.settings import Settings, LLMProvider
from src.parsers.json_parser import Conversation


@pytest.fixture
def sample_conversation_data() -> Dict[str, Any]:
    """Sample ChatGPT conversation data for testing."""
    return {
        "id": "test-conversation-1",
        "title": "Test Discussion",
        "create_time": 1703123456.789,
        "update_time": 1703123500.123,
        "mapping": {
            "msg-1": {
                "id": "msg-1",
                "message": {
                    "id": "msg-1",
                    "author": {"role": "user", "name": None, "metadata": {}},
                    "create_time": 1703123456.789,
                    "update_time": None,
                    "content": {
                        "content_type": "text",
                        "parts": ["Hello, I'm a software engineer working on Python projects."]
                    },
                    "status": "finished_successfully",
                    "end_turn": None,
                    "weight": 1.0,
                    "metadata": {},
                    "recipient": "all"
                },
                "parent": None,
                "children": ["msg-2"]
            },
            "msg-2": {
                "id": "msg-2",
                "message": {
                    "id": "msg-2",
                    "author": {"role": "assistant", "name": None, "metadata": {}},
                    "create_time": 1703123470.123,
                    "update_time": None,
                    "content": {
                        "content_type": "text",
                        "parts": ["That's great! Python is an excellent language for many applications. What kind of projects are you working on?"]
                    },
                    "status": "finished_successfully",
                    "end_turn": True,
                    "weight": 1.0,
                    "metadata": {},
                    "recipient": "all"
                },
                "parent": "msg-1",
                "children": ["msg-3"]
            },
            "msg-3": {
                "id": "msg-3",
                "message": {
                    "id": "msg-3",
                    "author": {"role": "user", "name": None, "metadata": {}},
                    "create_time": 1703123480.456,
                    "update_time": None,
                    "content": {
                        "content_type": "text",
                        "parts": ["I prefer working late at night, usually from 10 PM to 2 AM when it's quiet. I'm currently building web applications using Django and FastAPI."]
                    },
                    "status": "finished_successfully",
                    "end_turn": None,
                    "weight": 1.0,
                    "metadata": {},
                    "recipient": "all"
                },
                "parent": "msg-2",
                "children": []
            }
        },
        "moderation_results": [],
        "current_node": "msg-3"
    }


@pytest.fixture
def sample_conversations_export(sample_conversation_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Sample ChatGPT export data with multiple conversations."""
    conversation2 = sample_conversation_data.copy()
    conversation2["id"] = "test-conversation-2"
    conversation2["title"] = "Learning Goals Discussion"
    
    # Update mapping for second conversation
    conversation2["mapping"] = {
        "msg-a": {
            "id": "msg-a",
            "message": {
                "id": "msg-a",
                "author": {"role": "user", "name": None, "metadata": {}},
                "create_time": 1703200000.0,
                "update_time": None,
                "content": {
                    "content_type": "text",
                    "parts": ["I want to learn machine learning to transition into AI/ML engineering."]
                },
                "status": "finished_successfully",
                "end_turn": None,
                "weight": 1.0,
                "metadata": {},
                "recipient": "all"
            },
            "parent": None,
            "children": []
        }
    }
    conversation2["current_node"] = "msg-a"
    
    return [sample_conversation_data, conversation2]


@pytest.fixture
def temp_conversations_file(tmp_path: Path, sample_conversations_export: List[Dict[str, Any]]) -> Path:
    """Create a temporary conversations.json file for testing."""
    conversations_file = tmp_path / "conversations.json"
    with open(conversations_file, 'w') as f:
        json.dump(sample_conversations_export, f)
    return conversations_file


@pytest.fixture
def mock_settings() -> Settings:
    """Mock settings for testing."""
    return Settings(
        llm_provider=LLMProvider.OLLAMA,
        ollama_model="nuextract",
        openai_model="gpt-4o-mini",
        confidence_threshold=0.7,
        batch_size=100,
        chunk_size=1500,
        chunk_overlap=200
    )


@pytest.fixture
def sample_memories() -> List[Dict[str, Any]]:
    """Sample extracted memories for testing."""
    return [
        {
            "content": "User is a software engineer with Python expertise",
            "category": "fact",
            "confidence": 0.95,
            "conversation_id": "test-conversation-1",
            "source": "Test Discussion"
        },
        {
            "content": "User prefers working late at night (10 PM - 2 AM)",
            "category": "preference", 
            "confidence": 0.90,
            "conversation_id": "test-conversation-1",
            "source": "Test Discussion"
        },
        {
            "content": "User wants to learn machine learning for career transition",
            "category": "goal",
            "confidence": 0.85,
            "conversation_id": "test-conversation-2", 
            "source": "Learning Goals Discussion"
        },
        {
            "content": "User is a software engineer with Python expertise",  # Duplicate
            "category": "fact",
            "confidence": 0.93,
            "conversation_id": "test-conversation-2",
            "source": "Learning Goals Discussion"
        }
    ]


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama API response."""
    return {
        "model": "nuextract",
        "created_at": "2024-01-01T00:00:00Z",
        "response": json.dumps({
            "memories": [
                {
                    "content": "User is a software engineer",
                    "category": "fact",
                    "confidence": 0.95
                }
            ]
        }),
        "done": True
    }


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "memories": [
                            {
                                "content": "User prefers Python programming",
                                "category": "preference", 
                                "confidence": 0.90
                            }
                        ]
                    })
                }
            }
        ]
    }


@pytest.fixture
def mock_mem0_client():
    """Mock Mem0 client for testing."""
    mock_client = Mock()
    mock_client.add.return_value = {"id": "mem-123", "status": "success"}
    mock_client.get_all.return_value = []
    mock_client.delete.return_value = {"status": "success"}
    return mock_client