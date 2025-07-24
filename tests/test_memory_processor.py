"""Tests for memory processor."""

from typing import Any, Dict, List

import pytest

from src.extractors.ollama_extractor import ExtractedMemory
from src.processors.memory_processor import MemoryProcessor, ProcessingStats


class TestMemoryProcessor:
    """Test cases for MemoryProcessor."""

    def test_init(self):
        """Test processor initialization."""
        processor = MemoryProcessor(confidence_threshold=0.8)
        assert processor.confidence_threshold == 0.8
        assert hasattr(processor, "logger")

    def test_init_default_threshold(self):
        """Test processor initialization with default threshold."""
        processor = MemoryProcessor()
        assert processor.confidence_threshold == 0.7

    def test_process_memories_empty_list(self):
        """Test processing empty memory list."""
        processor = MemoryProcessor()
        memories, stats = processor.process_memories([])

        assert memories == []
        assert stats.total_input == 0
        assert stats.total_output == 0
        assert stats.duplicates_removed == 0
        assert stats.low_confidence_filtered == 0
        assert stats.merged_memories == 0

    def test_filter_by_confidence(self):
        """Test confidence filtering."""
        processor = MemoryProcessor(confidence_threshold=0.8)

        memories = [
            ExtractedMemory("High confidence", "fact", 0.9, "context1"),
            ExtractedMemory("Low confidence", "fact", 0.6, "context2"),
            ExtractedMemory("Medium confidence", "preference", 0.8, "context3"),
        ]

        stats = ProcessingStats(0, 0, 0, 0, 0, {})
        filtered = processor._filter_by_confidence(memories, stats)

        assert len(filtered) == 2
        assert stats.low_confidence_filtered == 1
        assert filtered[0].confidence == 0.9
        assert filtered[1].confidence == 0.8

    def test_remove_duplicates(self):
        """Test duplicate removal."""
        processor = MemoryProcessor()

        memories = [
            ExtractedMemory("User likes Python", "preference", 0.9, "context1"),
            ExtractedMemory(
                "User likes python", "preference", 0.8, "context2"
            ),  # Duplicate (case insensitive)
            ExtractedMemory("User uses JavaScript", "skill", 0.7, "context3"),
        ]

        stats = ProcessingStats(0, 0, 0, 0, 0, {})
        unique = processor._remove_duplicates(memories, stats)

        assert len(unique) == 2
        assert stats.duplicates_removed == 1
        assert "Python" in unique[0].content
        assert "JavaScript" in unique[1].content

    def test_normalize_content(self):
        """Test content normalization."""
        processor = MemoryProcessor()

        # Test whitespace normalization
        assert (
            processor._normalize_content("  multiple   spaces  ") == "multiple spaces"
        )

        # Test punctuation stripping
        assert processor._normalize_content('"quoted text"') == "quoted text"
        assert (
            processor._normalize_content("text with punctuation!")
            == "text with punctuation"
        )

        # Test case conversion
        assert processor._normalize_content("Mixed CASE Text") == "mixed case text"

    def test_clean_content(self):
        """Test content cleaning."""
        processor = MemoryProcessor()

        # Test prefix removal
        assert (
            processor._clean_content("Remember: User likes coffee")
            == "User likes coffee"
        )
        assert (
            processor._clean_content("User preference: Working at night")
            == "Working at night"
        )

        # Test capitalization
        assert processor._clean_content("user likes python") == "User likes python"

        # Test whitespace cleaning
        assert processor._clean_content("  extra   spaces  ") == "Extra spaces"

    def test_are_similar_memories_same_category(self):
        """Test similarity detection for same category memories."""
        processor = MemoryProcessor()

        memory1 = ExtractedMemory(
            "User likes Python programming", "preference", 0.9, "context1"
        )
        memory2 = ExtractedMemory(
            "User really likes Python programming", "preference", 0.8, "context2"
        )
        memory3 = ExtractedMemory(
            "User dislikes meetings", "preference", 0.9, "context3"
        )

        # Similar memories
        assert processor._are_similar_memories(memory1, memory2)

        # Different memories in same category
        assert not processor._are_similar_memories(memory1, memory3)

    def test_are_similar_memories_different_category(self):
        """Test similarity detection for different category memories."""
        processor = MemoryProcessor()

        memory1 = ExtractedMemory("User likes Python", "preference", 0.9, "context1")
        memory2 = ExtractedMemory("User knows Python", "skill", 0.9, "context2")

        # Same content but different categories should not be similar
        assert not processor._are_similar_memories(memory1, memory2)

    def test_merge_memory_group(self):
        """Test merging a group of similar memories."""
        processor = MemoryProcessor()

        memories = [
            ExtractedMemory(
                "User likes Python", "preference", 0.9, "context1", {"source": "conv1"}
            ),
            ExtractedMemory(
                "User enjoys Python programming",
                "preference",
                0.8,
                "context2",
                {"source": "conv2"},
            ),
            ExtractedMemory(
                "User prefers Python development",
                "preference",
                0.85,
                "context3",
                {"source": "conv1"},
            ),
        ]

        merged = processor._merge_memory_group(memories)

        assert merged.category == "preference"
        assert merged.confidence == pytest.approx(0.85, abs=0.01)  # Average confidence
        assert "context1" in merged.context
        assert "context2" in merged.context
        assert "context3" in merged.context
        assert merged.metadata["source"] in ["conv1", "conv2"]

    def test_merge_within_category(self):
        """Test merging memories within a category."""
        processor = MemoryProcessor()

        memories = [
            ExtractedMemory(
                "User likes Python programming", "preference", 0.9, "context1"
            ),
            ExtractedMemory(
                "User really likes Python programming", "preference", 0.8, "context2"
            ),
            ExtractedMemory("User dislikes meetings", "preference", 0.9, "context3"),
        ]

        stats = ProcessingStats(0, 0, 0, 0, 0, {})
        merged = processor._merge_within_category(memories, stats)

        # Should merge first two (similar) but keep third separate
        assert len(merged) == 2
        assert stats.merged_memories == 1

    def test_merge_similar_memories(self):
        """Test the complete similar memory merging process."""
        processor = MemoryProcessor()

        memories = [
            ExtractedMemory(
                "User likes Python programming", "preference", 0.9, "context1"
            ),
            ExtractedMemory(
                "User really likes Python programming", "preference", 0.8, "context2"
            ),
            ExtractedMemory("User is software engineer", "fact", 0.9, "context3"),
            ExtractedMemory("User is a software engineer", "fact", 0.85, "context4"),
        ]

        stats = ProcessingStats(0, 0, 0, 0, 0, {})
        merged = processor._merge_similar_memories(memories, stats)

        # Should merge within each category
        assert len(merged) == 2  # One preference, one fact
        assert stats.merged_memories == 2  # Two pairs merged

    def test_process_memories_complete_pipeline(
        self, sample_memories: List[Dict[str, Any]]
    ):
        """Test the complete memory processing pipeline."""
        processor = MemoryProcessor(confidence_threshold=0.8)

        # Convert sample memories to ExtractedMemory objects
        extracted_memories = [
            ExtractedMemory(
                content=mem["content"],
                category=mem["category"],
                confidence=mem["confidence"],
                context=mem["source"],
                metadata={"conversation_id": mem["conversation_id"]},
            )
            for mem in sample_memories
        ]

        processed_memories, stats = processor.process_memories(extracted_memories)

        # Check stats
        assert stats.total_input == 4
        assert stats.total_output > 0
        assert stats.low_confidence_filtered == 0  # All memories above 0.8 threshold
        assert stats.duplicates_removed == 1  # One duplicate

        # Check processed memories
        assert len(processed_memories) == stats.total_output
        assert all(mem.confidence >= 0.8 for mem in processed_memories)

        # Check categories are counted
        assert sum(stats.categories.values()) == stats.total_output

    def test_get_category_distribution(self):
        """Test category distribution calculation."""
        processor = MemoryProcessor()

        memories = [
            ExtractedMemory("Memory 1", "fact", 0.9, "context1"),
            ExtractedMemory("Memory 2", "fact", 0.8, "context2"),
            ExtractedMemory("Memory 3", "preference", 0.9, "context3"),
        ]

        distribution = processor.get_category_distribution(memories)

        assert distribution["fact"] == 2
        assert distribution["preference"] == 1
        assert len(distribution) == 2

    def test_get_confidence_statistics(self):
        """Test confidence statistics calculation."""
        processor = MemoryProcessor()

        memories = [
            ExtractedMemory("Memory 1", "fact", 0.7, "context1"),
            ExtractedMemory("Memory 2", "fact", 0.8, "context2"),
            ExtractedMemory("Memory 3", "preference", 0.9, "context3"),
        ]

        stats = processor.get_confidence_statistics(memories)

        assert stats["min"] == 0.7
        assert stats["max"] == 0.9
        assert stats["avg"] == pytest.approx(0.8, abs=0.01)
        assert stats["median"] == 0.8

    def test_get_confidence_statistics_empty(self):
        """Test confidence statistics with empty memory list."""
        processor = MemoryProcessor()

        stats = processor.get_confidence_statistics([])

        assert stats["min"] == 0
        assert stats["max"] == 0
        assert stats["avg"] == 0
        assert stats["median"] == 0

    def test_combine_memory_content_simple(self):
        """Test combining memory content for simple cases."""
        processor = MemoryProcessor()

        memories = [
            ExtractedMemory("User likes Python", "preference", 0.9, "context1"),
            ExtractedMemory(
                "User enjoys Python programming", "preference", 0.8, "context2"
            ),
        ]

        combined = processor._combine_memory_content(memories)

        # Should return the longer content
        assert "programming" in combined

    def test_combine_memory_content_complex(self):
        """Test combining memory content for complex cases."""
        processor = MemoryProcessor()

        memories = [
            ExtractedMemory("User likes Python", "preference", 0.9, "context1"),
            ExtractedMemory("User enjoys JavaScript", "preference", 0.8, "context2"),
            ExtractedMemory("User prefers TypeScript", "preference", 0.85, "context3"),
            ExtractedMemory("User dislikes Java", "preference", 0.9, "context4"),
        ]

        combined = processor._combine_memory_content(memories)

        # Should combine unique information
        assert ";" in combined  # Multiple parts joined
        assert len(combined) > max(len(m.content) for m in memories)
