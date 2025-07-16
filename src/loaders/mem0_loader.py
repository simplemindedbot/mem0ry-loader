"""Memory loader for Mem0 platform."""

import logging
import time
from typing import List, Dict, Optional
from dataclasses import asdict
from mem0 import MemoryClient
from mem0.client.main import APIError

from ..extractors.ollama_extractor import ExtractedMemory
from ..config.settings import settings


class Mem0Loader:
    """Loader for uploading memories to Mem0 platform."""
    
    def __init__(self, api_key: str = None, user_id: str = "chatgpt_import"):
        self.api_key = api_key or settings.mem0_api_key
        self.user_id = user_id
        self.logger = logging.getLogger(__name__)
        
        if not self.api_key:
            raise ValueError("Mem0 API key is required")
        
        self.client = MemoryClient(api_key=self.api_key)
        self.uploaded_count = 0
        self.failed_count = 0
    
    def load_memories(self, memories: List[ExtractedMemory], 
                     batch_size: int = None) -> Dict[str, int]:
        """Load memories to Mem0 platform.
        
        Args:
            memories: List of memories to upload
            batch_size: Number of memories to upload per batch
            
        Returns:
            Dictionary with upload statistics
        """
        batch_size = batch_size or settings.batch_size
        
        self.logger.info(f"Starting upload of {len(memories)} memories to Mem0")
        
        # Process in batches
        for i in range(0, len(memories), batch_size):
            batch = memories[i:i + batch_size]
            self._upload_batch(batch, i // batch_size + 1)
            
            # Rate limiting
            time.sleep(1)  # Small delay between batches
        
        stats = {
            "total_processed": len(memories),
            "uploaded": self.uploaded_count,
            "failed": self.failed_count,
            "success_rate": self.uploaded_count / len(memories) if memories else 0
        }
        
        self.logger.info(f"Upload complete. Stats: {stats}")
        return stats
    
    def _upload_batch(self, batch: List[ExtractedMemory], batch_num: int):
        """Upload a batch of memories.
        
        Args:
            batch: List of memories to upload
            batch_num: Batch number for logging
        """
        self.logger.info(f"Uploading batch {batch_num} ({len(batch)} memories)")
        
        for memory in batch:
            try:
                self._upload_single_memory(memory)
                self.uploaded_count += 1
                
            except Exception as e:
                self.logger.error(f"Failed to upload memory: {e}")
                self.failed_count += 1
    
    def _upload_single_memory(self, memory: ExtractedMemory):
        """Upload a single memory to Mem0.
        
        Args:
            memory: Memory to upload
            
        Raises:
            APIError: If the upload fails
        """
        # Prepare the message format for Mem0
        message_content = f"Remember: {memory.content}"
        
        # Add category context if available
        if memory.category and memory.category != "context":
            message_content = f"[{memory.category.upper()}] {message_content}"
        
        messages = [
            {
                "role": "user",
                "content": message_content
            }
        ]
        
        # Prepare metadata
        metadata = {
            "source": "chatgpt_export",
            "category": memory.category,
            "confidence": memory.confidence,
            "original_context": memory.context[:500],  # Truncate long context
            **memory.metadata
        }
        
        # Upload to Mem0
        try:
            result = self.client.add(
                messages=messages,
                user_id=self.user_id,
                metadata=metadata
            )
            
            self.logger.debug(f"Successfully uploaded memory: {memory.content[:50]}...")
            return result
            
        except APIError as e:
            self.logger.error(f"Mem0 API error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error uploading memory: {e}")
            raise
    
    def get_existing_memories(self) -> List[Dict]:
        """Get existing memories from Mem0 for the user.
        
        Returns:
            List of existing memories
        """
        try:
            memories = self.client.get_all(user_id=self.user_id)
            self.logger.info(f"Found {len(memories)} existing memories for user {self.user_id}")
            return memories
            
        except APIError as e:
            self.logger.error(f"Failed to fetch existing memories: {e}")
            return []
    
    def delete_existing_memories(self, memory_ids: List[str]) -> int:
        """Delete existing memories from Mem0.
        
        Args:
            memory_ids: List of memory IDs to delete
            
        Returns:
            Number of successfully deleted memories
        """
        if not memory_ids:
            return 0
        
        self.logger.info(f"Deleting {len(memory_ids)} existing memories")
        
        deleted_count = 0
        
        # Delete in batches (Mem0 supports up to 1000 per batch)
        batch_size = 1000
        for i in range(0, len(memory_ids), batch_size):
            batch = memory_ids[i:i + batch_size]
            
            try:
                self.client.delete_many(memory_ids=batch)
                deleted_count += len(batch)
                self.logger.debug(f"Deleted batch of {len(batch)} memories")
                
            except APIError as e:
                self.logger.error(f"Failed to delete memory batch: {e}")
        
        self.logger.info(f"Successfully deleted {deleted_count} memories")
        return deleted_count
    
    def check_for_duplicates(self, memories: List[ExtractedMemory]) -> List[ExtractedMemory]:
        """Check for duplicate memories and filter them out.
        
        Args:
            memories: List of memories to check
            
        Returns:
            List of unique memories
        """
        # Get existing memories
        existing_memories = self.get_existing_memories()
        existing_content = {mem.get("memory", "") for mem in existing_memories}
        
        # Filter out duplicates
        unique_memories = []
        for memory in memories:
            if memory.content not in existing_content:
                unique_memories.append(memory)
            else:
                self.logger.debug(f"Skipping duplicate memory: {memory.content[:50]}...")
        
        self.logger.info(f"Filtered {len(memories) - len(unique_memories)} duplicate memories")
        return unique_memories
    
    def validate_memory(self, memory: ExtractedMemory) -> bool:
        """Validate a memory before upload.
        
        Args:
            memory: Memory to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        if not memory.content or not memory.content.strip():
            return False
        
        # Check confidence threshold
        if memory.confidence < settings.confidence_threshold:
            return False
        
        # Check category
        if memory.category not in settings.memory_categories:
            return False
        
        # Check content length (reasonable limits)
        if len(memory.content) > 1000:  # Too long
            return False
        
        if len(memory.content) < 10:  # Too short
            return False
        
        return True
    
    def prepare_memories_for_upload(self, memories: List[ExtractedMemory]) -> List[ExtractedMemory]:
        """Prepare memories for upload by filtering and validating.
        
        Args:
            memories: Raw extracted memories
            
        Returns:
            Cleaned and validated memories ready for upload
        """
        self.logger.info(f"Preparing {len(memories)} memories for upload")
        
        # Validate memories
        valid_memories = [mem for mem in memories if self.validate_memory(mem)]
        self.logger.info(f"Validated {len(valid_memories)} memories")
        
        # Remove duplicates within the batch
        unique_memories = self._remove_internal_duplicates(valid_memories)
        self.logger.info(f"Removed {len(valid_memories) - len(unique_memories)} internal duplicates")
        
        # Check against existing memories in Mem0
        final_memories = self.check_for_duplicates(unique_memories)
        
        return final_memories
    
    def _remove_internal_duplicates(self, memories: List[ExtractedMemory]) -> List[ExtractedMemory]:
        """Remove duplicate memories within the current batch.
        
        Args:
            memories: List of memories to deduplicate
            
        Returns:
            List of unique memories
        """
        seen_content = set()
        unique_memories = []
        
        for memory in memories:
            # Normalize content for comparison
            normalized_content = memory.content.lower().strip()
            
            if normalized_content not in seen_content:
                seen_content.add(normalized_content)
                unique_memories.append(memory)
        
        return unique_memories