"""Local Mem0 memory loader for self-hosted OpenMemory server."""

import logging
import time
import requests
from typing import List, Dict, Optional
from dataclasses import asdict

from ..extractors.ollama_extractor import ExtractedMemory
from ..config.settings import settings


class LocalMem0Loader:
    """Loader for uploading memories to local OpenMemory server."""
    
    def __init__(self, base_url: str = "http://localhost:8765", user_id: str = "default_user"):
        self.base_url = base_url.rstrip('/')
        self.user_id = user_id
        self.logger = logging.getLogger(__name__)
        self.uploaded_count = 0
        self.failed_count = 0
        
        # Test connection to local server
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to local OpenMemory server."""
        try:
            response = requests.get(f"{self.base_url}/api/v1/config/", timeout=5)
            response.raise_for_status()
            self.logger.info("Successfully connected to local OpenMemory server")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to connect to local OpenMemory server at {self.base_url}: {e}")
            self.logger.info("Make sure OpenMemory server is running with 'make up' in the mem0/openmemory directory")
            raise
    
    def load_memories(self, memories: List[ExtractedMemory], 
                     batch_size: int = None) -> Dict[str, int]:
        """Load memories to local OpenMemory server.
        
        Args:
            memories: List of ExtractedMemory objects to upload
            batch_size: Size of batches for upload (default: from settings)
            
        Returns:
            Dictionary with upload statistics
        """
        if not memories:
            return {"total_processed": 0, "uploaded": 0, "failed": 0, "success_rate": 0.0}
        
        batch_size = batch_size or settings.batch_size
        self.uploaded_count = 0
        self.failed_count = 0
        
        # Process memories in batches
        for i in range(0, len(memories), batch_size):
            batch = memories[i:i + batch_size]
            self._upload_batch(batch)
            
            # Rate limiting
            time.sleep(1.0 / settings.requests_per_minute * 60)
        
        total_processed = self.uploaded_count + self.failed_count
        success_rate = self.uploaded_count / total_processed if total_processed > 0 else 0.0
        
        return {
            "total_processed": total_processed,
            "uploaded": self.uploaded_count,
            "failed": self.failed_count,
            "success_rate": success_rate
        }
    
    def _upload_batch(self, batch: List[ExtractedMemory]):
        """Upload a batch of memories."""
        for memory in batch:
            try:
                self._upload_single_memory(memory)
                self.uploaded_count += 1
            except Exception as e:
                self.logger.error(f"Failed to upload memory: {e}")
                self.failed_count += 1
    
    def _upload_single_memory(self, memory: ExtractedMemory):
        """Upload a single memory to local OpenMemory server."""
        # Prepare memory data for OpenMemory API
        memory_data = {
            "user_id": self.user_id,
            "text": memory.content,
            "metadata": {
                "category": memory.category,
                "confidence": memory.confidence,
                "context": memory.context,
                **memory.metadata
            }
        }
        
        # Make API call to local OpenMemory server
        response = requests.post(
            f"{self.base_url}/api/v1/memories/",
            json=memory_data,
            timeout=30
        )
        
        if response.status_code not in [200, 201]:
            self.logger.error(f"Failed to upload memory: {response.status_code} - {response.text}")
            raise Exception(f"Upload failed: {response.status_code}")
        
        self.logger.debug(f"Successfully uploaded memory: {memory.content[:50]}...")
    
    def get_existing_memories(self) -> List[Dict]:
        """Get existing memories from local OpenMemory server."""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/memories/",
                params={"user_id": self.user_id},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to get existing memories: {e}")
            return []
    
    def delete_existing_memories(self, memory_ids: List[str]) -> int:
        """Delete existing memories from local OpenMemory server."""
        try:
            response = requests.delete(
                f"{self.base_url}/api/v1/memories/",
                json={"memory_ids": memory_ids, "user_id": self.user_id},
                timeout=30
            )
            if response.status_code == 200:
                return len(memory_ids)
            else:
                self.logger.warning(f"Failed to delete memories: {response.status_code}")
                return 0
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to delete memories: {e}")
            return 0
    
    def prepare_memories_for_upload(self, memories: List[ExtractedMemory]) -> List[ExtractedMemory]:
        """Prepare memories for upload to local server."""
        # For local server, we can upload all memories as-is
        # No need to transform to cloud format
        return memories
    
    def search_memories(self, query: str, limit: int = 10) -> List[Dict]:
        """Search memories in local OpenMemory server."""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/memories/",
                params={
                    "user_id": self.user_id,
                    "search_query": query,
                    "page_size": limit
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to search memories: {e}")
            return []