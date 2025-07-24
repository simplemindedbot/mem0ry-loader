"""Memory extraction using OpenAI models."""

import json
import logging
import os
import time
import uuid
from typing import List

from openai import OpenAI

from ..config.settings import OpenAIModel, settings
from .ollama_extractor import ExtractedMemory


class OpenAIExtractor:
    """Memory extractor using OpenAI models."""

    def __init__(
        self, model: OpenAIModel = None, api_key: str = None, use_batch: bool = False
    ):
        self.model = model or settings.openai_model
        self.api_key = (
            api_key or settings.openai_api_key or os.getenv("MEMLOADER_OPENAI_API_KEY")
        )
        self.use_batch = use_batch
        self.logger = logging.getLogger(__name__)

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set MEMLOADER_OPENAI_API_KEY environment variable."
            )

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

        # Batch processing state
        self.batch_requests = []
        self.batch_id = None

        self.logger.info(
            f"Initialized OpenAI extractor with model: {self.model}, batch: {self.use_batch}"
        )

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
        if self.use_batch:
            return self._add_to_batch(text_chunk, conversation_title)
        else:
            return self._extract_with_openai(text_chunk, conversation_title)

    def _add_to_batch(
        self, text_chunk: str, conversation_title: str
    ) -> List[ExtractedMemory]:
        """Add request to batch processing queue."""
        request_id = str(uuid.uuid4())

        # Create the batch request
        batch_request = {
            "custom_id": request_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert at extracting personal memories and preferences from conversations. Always return valid JSON.",
                    },
                    {
                        "role": "user",
                        "content": self._create_extraction_prompt(
                            text_chunk, conversation_title
                        ),
                    },
                ],
                "temperature": 0.3,
                "max_tokens": 2000,
                "top_p": 0.9,
            },
        }

        # Store request with context for later processing
        self.batch_requests.append(
            {
                "request": batch_request,
                "context": text_chunk,
                "title": conversation_title,
            }
        )

        # Return empty list - memories will be processed later
        return []

    def process_batch(self) -> List[ExtractedMemory]:
        """Process all batch requests and return extracted memories."""
        if not self.batch_requests:
            return []

        try:
            # Create batch file
            batch_file_path = f"/tmp/batch_requests_{uuid.uuid4()}.jsonl"
            with open(batch_file_path, "w") as f:
                for item in self.batch_requests:
                    f.write(json.dumps(item["request"]) + "\n")

            # Upload batch file
            self.logger.info(
                f"Uploading batch file with {len(self.batch_requests)} requests..."
            )
            with open(batch_file_path, "rb") as f:
                batch_file = self.client.files.create(file=f, purpose="batch")

            # Create batch job
            self.logger.info("Creating batch job...")
            batch = self.client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": "Memory extraction batch"},
            )

            self.batch_id = batch.id
            self.logger.info(f"Batch job created: {batch.id}")

            # Wait for completion
            return self._wait_for_batch_completion()

        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            return []
        finally:
            # Clean up temp file
            if os.path.exists(batch_file_path):
                os.remove(batch_file_path)

    def _wait_for_batch_completion(self) -> List[ExtractedMemory]:
        """Wait for batch completion and process results."""
        if not self.batch_id:
            return []

        self.logger.info(f"Waiting for batch {self.batch_id} completion...")

        while True:
            batch = self.client.batches.retrieve(self.batch_id)

            if batch.status == "completed":
                self.logger.info("Batch completed successfully!")
                return self._process_batch_results(batch)
            elif batch.status == "failed":
                self.logger.error("Batch processing failed")
                return []
            elif batch.status in ["cancelled", "expired"]:
                self.logger.error(f"Batch {batch.status}")
                return []
            else:
                self.logger.info(f"Batch status: {batch.status}")
                time.sleep(10)  # Check every 10 seconds

    def _process_batch_results(self, batch) -> List[ExtractedMemory]:
        """Process batch results and extract memories."""
        if not batch.output_file_id:
            self.logger.error("No output file in batch results")
            return []

        # Download results
        result_file = self.client.files.content(batch.output_file_id)
        results = result_file.read().decode("utf-8")

        # Parse results
        all_memories = []
        request_map = {req["request"]["custom_id"]: req for req in self.batch_requests}

        for line in results.strip().split("\n"):
            if not line:
                continue

            try:
                result = json.loads(line)
                custom_id = result["custom_id"]

                if custom_id in request_map:
                    context_info = request_map[custom_id]
                    response_content = result["response"]["body"]["choices"][0][
                        "message"
                    ]["content"]

                    # Parse memories from response
                    memories = self._parse_openai_response(
                        response_content, context_info["context"]
                    )
                    all_memories.extend(memories)

            except Exception as e:
                self.logger.warning(f"Failed to process batch result: {e}")
                continue

        self.logger.info(f"Extracted {len(all_memories)} memories from batch")
        return all_memories

    def _create_extraction_prompt(
        self, text_chunk: str, conversation_title: str
    ) -> str:
        """Create the extraction prompt."""
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

        return f"""Extract personal memories, preferences, and contextual information from this conversation.

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

    def _extract_with_openai(
        self, text_chunk: str, conversation_title: str
    ) -> List[ExtractedMemory]:
        """Extract memories using OpenAI model."""
        try:
            prompt = self._create_extraction_prompt(text_chunk, conversation_title)
            response = self._make_openai_request(prompt)
            return self._parse_openai_response(response, text_chunk)

        except Exception as e:
            self.logger.error(f"Failed to extract memories with OpenAI: {e}")
            return []

    def _make_openai_request(self, prompt: str) -> str:
        """Make a request to the OpenAI API.

        Args:
            prompt: The prompt to send

        Returns:
            The model's response text

        Raises:
            Exception: If the request fails
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at extracting personal memories and preferences from conversations. Always return valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,  # Lower temperature for more consistent extraction
                max_tokens=2000,
                top_p=0.9,
            )

            return response.choices[0].message.content

        except Exception as e:
            self.logger.error(f"OpenAI request failed: {e}")
            raise

    def _parse_openai_response(
        self, response: str, context: str
    ) -> List[ExtractedMemory]:
        """Parse OpenAI model response."""
        try:
            # Try to find JSON in the response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                self.logger.warning("No JSON found in OpenAI response")
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
            self.logger.warning(f"Failed to parse JSON from OpenAI: {e}")
            return []
