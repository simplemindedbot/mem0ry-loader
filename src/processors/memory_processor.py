"""Memory processing and deduplication utilities."""

import logging
from typing import List, Dict, Set
from collections import defaultdict
from dataclasses import dataclass

from ..extractors.ollama_extractor import ExtractedMemory


@dataclass
class ProcessingStats:
    """Statistics from memory processing."""
    total_input: int
    total_output: int
    duplicates_removed: int
    low_confidence_filtered: int
    merged_memories: int
    categories: Dict[str, int]


class MemoryProcessor:
    """Processor for cleaning and deduplicating extracted memories."""
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)
    
    def process_memories(self, memories: List[ExtractedMemory]) -> tuple[List[ExtractedMemory], ProcessingStats]:
        """Process a list of extracted memories.
        
        Args:
            memories: List of raw extracted memories
            
        Returns:
            Tuple of (processed_memories, processing_stats)
        """
        self.logger.info(f"Processing {len(memories)} extracted memories")
        
        stats = ProcessingStats(
            total_input=len(memories),
            total_output=0,
            duplicates_removed=0,
            low_confidence_filtered=0,
            merged_memories=0,
            categories=defaultdict(int)
        )
        
        # Step 1: Filter by confidence
        confident_memories = self._filter_by_confidence(memories, stats)
        
        # Step 2: Remove exact duplicates
        unique_memories = self._remove_duplicates(confident_memories, stats)
        
        # Step 3: Merge similar memories
        merged_memories = self._merge_similar_memories(unique_memories, stats)
        
        # Step 4: Normalize and clean
        cleaned_memories = self._clean_memories(merged_memories, stats)
        
        # Update final stats
        stats.total_output = len(cleaned_memories)
        for memory in cleaned_memories:
            stats.categories[memory.category] += 1
        
        self.logger.info(f"Processing complete. Output: {len(cleaned_memories)} memories")
        return cleaned_memories, stats
    
    def _filter_by_confidence(self, memories: List[ExtractedMemory], 
                            stats: ProcessingStats) -> List[ExtractedMemory]:
        """Filter memories by confidence threshold."""
        filtered = []
        
        for memory in memories:
            if memory.confidence >= self.confidence_threshold:
                filtered.append(memory)
            else:
                stats.low_confidence_filtered += 1
                self.logger.debug(f"Filtered low confidence memory: {memory.content[:50]}...")
        
        self.logger.info(f"Filtered {stats.low_confidence_filtered} low confidence memories")
        return filtered
    
    def _remove_duplicates(self, memories: List[ExtractedMemory], 
                         stats: ProcessingStats) -> List[ExtractedMemory]:
        """Remove exact duplicate memories."""
        seen_content = set()
        unique_memories = []
        
        for memory in memories:
            # Normalize content for comparison
            normalized_content = self._normalize_content(memory.content)
            
            if normalized_content not in seen_content:
                seen_content.add(normalized_content)
                unique_memories.append(memory)
            else:
                stats.duplicates_removed += 1
                self.logger.debug(f"Removed duplicate: {memory.content[:50]}...")
        
        self.logger.info(f"Removed {stats.duplicates_removed} duplicate memories")
        return unique_memories
    
    def _merge_similar_memories(self, memories: List[ExtractedMemory], 
                               stats: ProcessingStats) -> List[ExtractedMemory]:
        """Merge similar memories to reduce redundancy."""
        # Group memories by category for more efficient processing
        category_groups = defaultdict(list)
        for memory in memories:
            category_groups[memory.category].append(memory)
        
        merged_memories = []
        
        for category, category_memories in category_groups.items():
            merged_category = self._merge_within_category(category_memories, stats)
            merged_memories.extend(merged_category)
        
        self.logger.info(f"Merged {stats.merged_memories} similar memories")
        return merged_memories
    
    def _merge_within_category(self, memories: List[ExtractedMemory], 
                              stats: ProcessingStats) -> List[ExtractedMemory]:
        """Merge similar memories within a category."""
        if len(memories) <= 1:
            return memories
        
        merged = []
        used_indices = set()
        
        for i, memory1 in enumerate(memories):
            if i in used_indices:
                continue
            
            similar_memories = [memory1]
            used_indices.add(i)
            
            for j, memory2 in enumerate(memories[i+1:], i+1):
                if j in used_indices:
                    continue
                
                if self._are_similar_memories(memory1, memory2):
                    similar_memories.append(memory2)
                    used_indices.add(j)
            
            if len(similar_memories) > 1:
                merged_memory = self._merge_memory_group(similar_memories)
                merged.append(merged_memory)
                stats.merged_memories += len(similar_memories) - 1
            else:
                merged.append(memory1)
        
        return merged
    
    def _are_similar_memories(self, memory1: ExtractedMemory, 
                            memory2: ExtractedMemory) -> bool:
        """Check if two memories are similar enough to merge."""
        # Must be same category
        if memory1.category != memory2.category:
            return False
        
        # Check content similarity
        content1 = self._normalize_content(memory1.content)
        content2 = self._normalize_content(memory2.content)
        
        # Simple similarity check based on word overlap
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return False
        
        # Calculate Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        similarity = intersection / union if union > 0 else 0
        
        # Merge if similarity is high enough
        return similarity >= 0.7
    
    def _merge_memory_group(self, memories: List[ExtractedMemory]) -> ExtractedMemory:
        """Merge a group of similar memories into one."""
        # Use the memory with highest confidence as base
        base_memory = max(memories, key=lambda m: m.confidence)
        
        # Combine content from all memories
        combined_content = self._combine_memory_content(memories)
        
        # Average confidence
        avg_confidence = sum(m.confidence for m in memories) / len(memories)
        
        # Combine contexts
        combined_context = " | ".join(set(m.context for m in memories if m.context))
        
        # Merge metadata
        combined_metadata = {}
        for memory in memories:
            combined_metadata.update(memory.metadata or {})
        
        return ExtractedMemory(
            content=combined_content,
            category=base_memory.category,
            confidence=avg_confidence,
            context=combined_context,
            metadata=combined_metadata
        )
    
    def _combine_memory_content(self, memories: List[ExtractedMemory]) -> str:
        """Combine content from multiple memories intelligently."""
        contents = [m.content for m in memories]
        
        # If contents are very similar, use the longest one
        if len(contents) <= 3:
            return max(contents, key=len)
        
        # For more complex cases, combine unique information
        # This is a simplified approach - could be enhanced with NLP
        unique_parts = []
        seen_words = set()
        
        for content in sorted(contents, key=len, reverse=True):
            words = set(content.lower().split())
            if not words.issubset(seen_words):
                unique_parts.append(content)
                seen_words.update(words)
        
        return "; ".join(unique_parts)
    
    def _clean_memories(self, memories: List[ExtractedMemory], 
                       stats: ProcessingStats) -> List[ExtractedMemory]:
        """Clean and normalize memory content."""
        cleaned = []
        
        for memory in memories:
            cleaned_content = self._clean_content(memory.content)
            
            if cleaned_content:  # Only keep non-empty memories
                cleaned_memory = ExtractedMemory(
                    content=cleaned_content,
                    category=memory.category,
                    confidence=memory.confidence,
                    context=memory.context,
                    metadata=memory.metadata
                )
                cleaned.append(cleaned_memory)
        
        return cleaned
    
    def _normalize_content(self, content: str) -> str:
        """Normalize content for comparison."""
        # Remove extra whitespace
        content = " ".join(content.split())
        
        # Remove common prefixes/suffixes
        content = content.strip('"\'.,;:!?')
        
        # Convert to lowercase for comparison
        return content.lower()
    
    def _clean_content(self, content: str) -> str:
        """Clean memory content for storage."""
        # Remove extra whitespace
        content = " ".join(content.split())
        
        # Remove common prefixes that might be added by extraction
        prefixes_to_remove = [
            "Remember:",
            "User preference:",
            "User likes:",
            "User dislikes:",
            "Important:",
            "Note:",
            "Memory:"
        ]
        
        for prefix in prefixes_to_remove:
            if content.startswith(prefix):
                content = content[len(prefix):].strip()
        
        # Ensure proper capitalization
        if content and content[0].islower():
            content = content[0].upper() + content[1:]
        
        return content
    
    def get_category_distribution(self, memories: List[ExtractedMemory]) -> Dict[str, int]:
        """Get distribution of memories by category."""
        distribution = defaultdict(int)
        for memory in memories:
            distribution[memory.category] += 1
        return dict(distribution)
    
    def get_confidence_statistics(self, memories: List[ExtractedMemory]) -> Dict[str, float]:
        """Get confidence statistics for memories."""
        if not memories:
            return {"min": 0, "max": 0, "avg": 0, "median": 0}
        
        confidences = [m.confidence for m in memories]
        confidences.sort()
        
        n = len(confidences)
        median = confidences[n//2] if n % 2 == 1 else (confidences[n//2-1] + confidences[n//2]) / 2
        
        return {
            "min": min(confidences),
            "max": max(confidences),
            "avg": sum(confidences) / len(confidences),
            "median": median
        }