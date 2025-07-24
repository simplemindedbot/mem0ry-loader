"""Fixed tests for Ollama extractor focusing on public interface."""

import json
from unittest.mock import patch

from src.config.settings import OllamaModel
from src.extractors.ollama_extractor import OllamaExtractor


class TestOllamaExtractorFixed:
    """Test cases for OllamaExtractor focusing on public interface."""

    @patch("src.extractors.ollama_extractor.OllamaExtractor._ensure_model_available")
    def test_init_basic(self, mock_ensure):
        """Test basic initialization."""
        mock_ensure.return_value = None
        extractor = OllamaExtractor(model=OllamaModel.NUEXTRACT)
        assert extractor.model == OllamaModel.NUEXTRACT
        assert "localhost" in extractor.base_url

    @patch("src.extractors.ollama_extractor.OllamaExtractor._ensure_model_available")
    def test_extract_memories_success(self, mock_ensure):
        """Test successful memory extraction."""
        mock_ensure.return_value = None
        extractor = OllamaExtractor(model=OllamaModel.NUEXTRACT)

        # Mock the request to return valid JSON
        with patch.object(extractor, "_make_ollama_request") as mock_request:
            mock_request.return_value = json.dumps(
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

            memories = extractor.extract_memories(
                "I love Python programming", "Programming discussion"
            )

            assert len(memories) == 1
            assert memories[0].content == "User prefers Python programming"
            assert memories[0].category == "preference"
            assert memories[0].confidence == 0.9
            assert memories[0].context == "I love Python programming"

    @patch("src.extractors.ollama_extractor.OllamaExtractor._ensure_model_available")
    def test_extract_memories_error_handling(self, mock_ensure):
        """Test error handling in memory extraction."""
        mock_ensure.return_value = None
        extractor = OllamaExtractor(model=OllamaModel.NUEXTRACT)

        # Mock request to raise exception
        with patch.object(extractor, "_make_ollama_request") as mock_request:
            mock_request.side_effect = Exception("Network error")

            memories = extractor.extract_memories("Test text", "Test context")
            assert memories == []

    @patch("src.extractors.ollama_extractor.OllamaExtractor._ensure_model_available")
    def test_extract_memories_invalid_json(self, mock_ensure):
        """Test handling of invalid JSON responses."""
        mock_ensure.return_value = None
        extractor = OllamaExtractor(model=OllamaModel.NUEXTRACT)

        # Mock request to return invalid JSON
        with patch.object(extractor, "_make_ollama_request") as mock_request:
            mock_request.return_value = "Invalid JSON"

            memories = extractor.extract_memories("Test text", "Test context")
            assert memories == []

    @patch("src.extractors.ollama_extractor.OllamaExtractor._ensure_model_available")
    def test_extract_memories_empty_memories(self, mock_ensure):
        """Test handling when no memories are extracted."""
        mock_ensure.return_value = None
        extractor = OllamaExtractor(model=OllamaModel.NUEXTRACT)

        # Mock request to return empty memories
        with patch.object(extractor, "_make_ollama_request") as mock_request:
            mock_request.return_value = json.dumps({"memories": []})

            memories = extractor.extract_memories("Boring text", "Test context")
            assert memories == []

    @patch("src.extractors.ollama_extractor.OllamaExtractor._ensure_model_available")
    def test_parse_nuextract_response_success(self, mock_ensure):
        """Test successful nuextract response parsing."""
        mock_ensure.return_value = None
        extractor = OllamaExtractor(model=OllamaModel.NUEXTRACT)

        response_text = json.dumps(
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

        memories = extractor._parse_nuextract_response(response_text, "Test context")

        assert len(memories) == 1
        assert memories[0].content == "Test memory"
        assert memories[0].category == "fact"
        assert memories[0].confidence == 0.85
        assert memories[0].context == "Test context"

    @patch("src.extractors.ollama_extractor.OllamaExtractor._ensure_model_available")
    def test_general_model_extraction(self, mock_ensure):
        """Test extraction with general model."""
        mock_ensure.return_value = None
        extractor = OllamaExtractor(model=OllamaModel.LLAMA_3_2_1B)

        # Mock the request to return valid response
        with patch.object(extractor, "_make_ollama_request") as mock_request:
            mock_request.return_value = json.dumps(
                [
                    {
                        "content": "User is a software engineer",
                        "category": "fact",
                        "confidence": 0.8,
                        "reasoning": "Mentioned profession",
                    }
                ]
            )

            memories = extractor.extract_memories(
                "I work as a software engineer", "Career discussion"
            )

            assert len(memories) == 1
            assert memories[0].content == "User is a software engineer"
            assert memories[0].category == "fact"
