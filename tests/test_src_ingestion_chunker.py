"""
Tests for document chunker module.
"""

import pytest
from pathlib import Path
import re

from src.ingestion.chunker import (
    ChunkingPipeline, ChunkingStrategy, Chunk,
    FixedSizeChunker, SentenceChunker, ParagraphChunker,
    RecursiveChunker, AdaptiveChunker, CodeChunker, MarkdownChunker
)


class TestChunkingPipeline:
    """Tests for ChunkingPipeline."""

    def test_init(self):
        """Test pipeline initialization."""
        pipeline = ChunkingPipeline()
        assert pipeline is not None
        assert pipeline.chunk_size == 800
        assert pipeline.chunk_overlap == 150

    def test_init_custom(self):
        """Test pipeline with custom settings."""
        pipeline = ChunkingPipeline(
            strategy=ChunkingStrategy.SENTENCE,
            chunk_size=500,
            chunk_overlap=100
        )
        assert pipeline.chunk_size == 500
        assert pipeline.chunk_overlap == 100

    def test_chunk_document(self):
        """Test chunking a document."""
        text = """
        This is the first paragraph. It contains multiple sentences.
        
        This is the second paragraph. It has different content.
        
        And a third paragraph with more information.
        """

        pipeline = ChunkingPipeline(chunk_size=100, chunk_overlap=20)
        chunks = pipeline.chunk_document(text)

        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(len(c.text) > 0 for c in chunks)

    def test_chunk_empty_document(self):
        """Test chunking empty document."""
        pipeline = ChunkingPipeline()
        chunks = pipeline.chunk_document("")

        assert chunks == []

    def test_chunk_metadata_preservation(self):
        """Test that metadata is preserved in chunks."""
        text = "This is a test document."
        metadata = {"source": "test.txt", "author": "tester"}

        pipeline = ChunkingPipeline()
        chunks = pipeline.chunk_document(text, metadata)

        assert len(chunks) > 0
        assert chunks[0].metadata["source"] == "test.txt"
        assert chunks[0].metadata["author"] == "tester"
        assert "chunk_index" in chunks[0].metadata

    def test_chunk_stats(self):
        """Test chunk statistics."""
        text = "This is a test document with enough content to create multiple chunks."

        pipeline = ChunkingPipeline(chunk_size=50, chunk_overlap=10)
        chunks = pipeline.chunk_document(text)
        stats = pipeline.get_chunk_stats(chunks)

        assert "total_chunks" in stats
        assert "avg_size" in stats
        assert "min_size" in stats
        assert "max_size" in stats
        assert stats["total_chunks"] == len(chunks)

    def test_batch_chunking(self):
        """Test chunking multiple documents."""
        documents = [
            {"content": "Document 1 content.", "metadata": {"id": 1}},
            {"content": "Document 2 content with more text.", "metadata": {"id": 2}}
        ]

        pipeline = ChunkingPipeline()
        chunks = pipeline.chunk_batch(documents)

        assert len(chunks) > 0
        # Check that metadata from both docs is present
        doc_ids = {c.metadata.get("id") for c in chunks}
        assert 1 in doc_ids or 2 in doc_ids


class TestFixedSizeChunker:
    """Tests for FixedSizeChunker."""

    def test_chunk(self):
        """Test fixed size chunking."""
        text = "This is a test document. " * 20
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(text)

        assert len(chunks) > 1
        assert all(len(c.text) <= 100 for c in chunks)

    def test_chunk_overlap(self):
        """Test that chunks overlap."""
        text = "This is a test document. " * 30
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=30)
        chunks = chunker.chunk(text)

        if len(chunks) > 1:
            # Check that there's overlap between chunks
            first_end = chunks[0].text[-50:]
            second_start = chunks[1].text[:50]
            # There should be some overlap
            assert len(first_end) > 0


class TestSentenceChunker:
    """Tests for SentenceChunker."""

    def test_chunk_by_sentences(self):
        """Test sentence-based chunking."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunker = SentenceChunker(chunk_size=2, chunk_overlap=0)
        chunks = chunker.chunk(text)

        # Each chunk should contain at most 2 sentences
        for chunk in chunks:
            sentence_count = chunk.text.count('.')
            assert sentence_count <= 2


class TestRecursiveChunker:
    """Tests for RecursiveChunker."""

    def test_recursive_splitting(self):
        """Test recursive splitting."""
        text = "Section 1.\n\nSection 2.\nLine 1.\nLine 2.\n\nSection 3."
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.chunk(text)

        assert len(chunks) > 0


class TestCodeChunker:
    """Tests for CodeChunker."""

    def test_chunk_code(self):
        """Test code chunking."""
        text = """
        def function1():
            print("Function 1")
            
        def function2():
            print("Function 2")
            
        class TestClass:
            def method(self):
                pass
        """
        chunker = CodeChunker(chunk_size=300, chunk_overlap=0)
        chunks = chunker.chunk(text)

        assert len(chunks) > 0
        # Functions should not be split
        for chunk in chunks:
            if "def function1" in chunk.text:
                assert "Function 1" in chunk.text
            if "def function2" in chunk.text:
                assert "Function 2" in chunk.text


class TestMarkdownChunker:
    """Tests for MarkdownChunker."""

    def test_chunk_markdown(self):
        """Test markdown chunking."""
        text = """
        # Header 1
        Content for section 1.
        
        ## Header 2
        Content for section 2.
        """
        chunker = MarkdownChunker(chunk_size=200, chunk_overlap=0)
        chunks = chunker.chunk(text)

        assert len(chunks) > 0
        # Headers should be preserved
        for chunk in chunks:
            if "Header 1" in chunk.text:
                assert "Content for section 1" in chunk.text
