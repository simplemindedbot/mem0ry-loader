"""Memory extraction using Ollama models."""

import json
import logging
from dataclasses import dataclass
from typing import Dict, List

import requests
from requests.exceptions import RequestException, Timeout

from ..config.settings import OllamaModel, settings


@dataclass
class ExtractedMemory:
    """Represents an extracted memory."""

    content: str
    category: str
    confidence: float
    context: str
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class OllamaExtractor:
    """Memory extractor using Ollama models."""

    def __init__(self, model: OllamaModel = None, base_url: str = None):
        self.model = model or settings.ollama_model
        self.base_url = base_url or settings.ollama_base_url
        self.logger = logging.getLogger(__name__)

        # Ensure model is available
        self._ensure_model_available()

    def _ensure_model_available(self):
        """Check if the model is available and pull if necessary."""
        try:
            # Check if model is already available
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            response.raise_for_status()

            available_models = {model["name"] for model in response.json()["models"]}

            if self.model not in available_models:
                self.logger.info(f"Model {self.model} not found. Pulling model...")
                self._pull_model()
            else:
                self.logger.info(f"Model {self.model} is available")

        except RequestException as e:
            self.logger.error(f"Failed to check model availability: {e}")
            raise

    def _pull_model(self):
        """Pull the model from Ollama."""
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model},
                timeout=300,  # 5 minutes timeout for model pulling
            )
            response.raise_for_status()
            self.logger.info(f"Successfully pulled model {self.model}")

        except RequestException as e:
            self.logger.error(f"Failed to pull model {self.model}: {e}")
            raise

    def extract_memories(
        self, text_chunk: str, conversation_title: str = ""
    ) -> List[ExtractedMemory]:
        """Extract memories from a text chunk.

        Args:
            text_chunk: Text to extract memories from
            conversation_title: Title of the conversation for context

        Returns:
            List of extracted memories
        """
        if self.model == OllamaModel.NUEXTRACT:
            return self._extract_with_nuextract(text_chunk, conversation_title)
        else:
            return self._extract_with_general_model(text_chunk, conversation_title)

    def _extract_with_nuextract(
        self, text_chunk: str, conversation_title: str
    ) -> List[ExtractedMemory]:
        """Extract memories using NuExtract model with JSON template."""

        # JSON template for NuExtract
        extraction_template = {
            "memories": [
                {
                    "content": "The extracted memory content",
                    "category": "One of: preference, fact, pattern, goal, skill, relationship, context, decision_criteria",
                    "confidence": "Float between 0 and 1",
                    "reasoning": "Why this is considered a memory",
                }
            ]
        }

        prompt = f"""Extract personal memories, preferences, and contextual information from this conversation.

Context: {conversation_title}

Text:
{text_chunk}

Focus on:
- User preferences and habits
- Important personal details
- Recurring topics/interests
- Problem-solving patterns
- Decision-making criteria
- Skills and expertise areas
- Goal statements
- Relationship information

Return only memories that are:
1. Personal to the user (not general facts)
2. Likely to be useful for future conversations
3. Specific and actionable

Use this JSON template:
{json.dumps(extraction_template, indent=2)}

Extract memories:"""

        try:
            response = self._make_ollama_request(prompt)
            return self._parse_nuextract_response(response, text_chunk)

        except Exception as e:
            self.logger.error(f"Failed to extract memories with NuExtract: {e}")
            return []

    def _extract_with_general_model(
        self, text_chunk: str, conversation_title: str
    ) -> List[ExtractedMemory]:
        """Extract memories using general-purpose models."""

        prompt = f"""Extract personal memories from this conversation text.

Context: {conversation_title}

Text:
{text_chunk}

Extract memories that are:
- Personal preferences or habits
- Important facts about the user
- Behavioral patterns
- Goals and aspirations
- Skills and expertise
- Relationship information
- Decision-making criteria

For each memory, provide:
1. Memory content (what to remember)
2. Category (preference/fact/pattern/goal/skill/relationship/context/decision_criteria)
3. Confidence (0-1, how confident are you this is worth remembering)
4. Brief reasoning

Format as JSON array:
[
  {{
    "content": "memory content here",
    "category": "category here",
    "confidence": 0.9,
    "reasoning": "why this is important"
  }}
]

Memories:"""

        try:
            response = self._make_ollama_request(prompt)
            return self._parse_general_response(response, text_chunk)

        except Exception as e:
            self.logger.error(f"Failed to extract memories with {self.model}: {e}")
            return []

    def _make_ollama_request(self, prompt: str) -> str:
        """Make a request to the Ollama API.

        Args:
            prompt: The prompt to send

        Returns:
            The model's response text

        Raises:
            RequestException: If the request fails
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower temperature for more consistent extraction
                        "top_p": 0.9,
                        "num_predict": 2000,  # Limit response length
                    },
                },
                timeout=settings.ollama_timeout,
            )
            response.raise_for_status()

            result = response.json()
            return result.get("response", "")

        except Timeout:
            self.logger.error(
                f"Request to Ollama timed out after {settings.ollama_timeout} seconds"
            )
            self.logger.info(
                "Consider reducing chunk size or increasing timeout for large conversations"
            )
            raise
        except RequestException as e:
            self.logger.error(f"Ollama request failed: {e}")
            raise

    def _parse_nuextract_response(
        self, response: str, context: str
    ) -> List[ExtractedMemory]:
        """Parse NuExtract model response."""
        try:
            # Try to find JSON in the response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                self.logger.warning("No JSON found in NuExtract response")
                return []

            json_str = response[json_start:json_end]
            data = json.loads(json_str)

            memories = []
            for memory_data in data.get("memories", []):
                try:
                    memory = ExtractedMemory(
                        content=memory_data.get("content", ""),
                        category=memory_data.get("category", "context"),
                        confidence=float(memory_data.get("confidence", 0.5)),
                        context=context,
                        metadata={"reasoning": memory_data.get("reasoning", "")},
                    )

                    # Filter by confidence threshold
                    if memory.confidence >= settings.confidence_threshold:
                        memories.append(memory)

                except (ValueError, KeyError) as e:
                    self.logger.warning(f"Failed to parse memory: {e}")
                    continue

            return memories

        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON from NuExtract: {e}")
            return []

    def _parse_general_response(
        self, response: str, context: str
    ) -> List[ExtractedMemory]:
        """Parse response from general-purpose models."""
        try:
            # Try to find JSON array in the response
            json_start = response.find("[")
            json_end = response.rfind("]") + 1

            if json_start == -1 or json_end == 0:
                self.logger.warning("No JSON array found in response")
                return []

            json_str = response[json_start:json_end]
            data = json.loads(json_str)

            memories = []
            for memory_data in data:
                try:
                    memory = ExtractedMemory(
                        content=memory_data.get("content", ""),
                        category=memory_data.get("category", "context"),
                        confidence=float(memory_data.get("confidence", 0.5)),
                        context=context,
                        metadata={"reasoning": memory_data.get("reasoning", "")},
                    )

                    # Filter by confidence threshold
                    if memory.confidence >= settings.confidence_threshold:
                        memories.append(memory)

                except (ValueError, KeyError) as e:
                    self.logger.warning(f"Failed to parse memory: {e}")
                    continue

            return memories

        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse JSON from response: {e}")
            return []
