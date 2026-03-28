"""
Tests for vector store module.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.retrieval.vector_store import FAISSVectorStore, SearchResult, FAISSHybridStore


class TestFAISSVectorStore:
    """Tests for FAISSVectorStore."""

    def test_init(self):
        """Test vector store initialization."""
        store = FAISSVectorStore(dimension=384, index_type="FlatIP")
        assert store.dimension == 384
        assert store.get_size() == 0

    def test_add_embeddings(self, sample_embeddings, sample_texts):
        """Test adding embeddings to store."""
        dimension = len(sample_embeddings[0])
        store = FAISSVectorStore(dimension=dimension)
        store.add_embeddings(sample_embeddings[:5], sample_texts[:5])

        assert store.get_size() == 5

    def test_search(self, sample_embeddings, sample_texts):
        """Test similarity search."""
        dimension = len(sample_embeddings[0])
        store = FAISSVectorStore(dimension=dimension)
        store.add_embeddings(sample_embeddings, sample_texts)

        query = sample_embeddings[0]
        results = store.search(query, top_k=3)

        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.score >= 0 for r in results)

    def test_search_with_threshold(self, sample_embeddings, sample_texts):
        """Test search with score threshold."""
        dimension = len(sample_embeddings[0])
        store = FAISSVectorStore(dimension=dimension)
        store.add_embeddings(sample_embeddings, sample_texts)

        query = sample_embeddings[0]
        results = store.search(query, top_k=5, score_threshold=0.5)

        for result in results:
            assert result.score >= 0.5

    def test_search_empty_store(self):
        """Test search on empty store."""
        store = FAISSVectorStore(dimension=384)
        query = np.random.randn(384).tolist()
        results = store.search(query, top_k=5)

        assert results == []

    def test_save_and_load(self, sample_embeddings, sample_texts, temp_dir):
        """Test saving and loading vector store."""
        dimension = len(sample_embeddings[0])
        store = FAISSVectorStore(dimension=dimension)
        store.add_embeddings(sample_embeddings[:5], sample_texts[:5])

        save_path = temp_dir / "vector_store"
        store.save(str(save_path))

        new_store = FAISSVectorStore(dimension=dimension)
        new_store.load(str(save_path))

        assert new_store.get_size() == store.get_size()

    def test_clear(self, sample_embeddings, sample_texts):
        """Test clearing vector store."""
        dimension = len(sample_embeddings[0])
        store = FAISSVectorStore(dimension=dimension)
        store.add_embeddings(sample_embeddings[:3], sample_texts[:3])

        assert store.get_size() == 3
        store.clear()
        assert store.get_size() == 0

    def test_get_stats(self, sample_embeddings, sample_texts):
        """Test getting store statistics."""
        dimension = len(sample_embeddings[0])
        store = FAISSVectorStore(dimension=dimension)
        store.add_embeddings(sample_embeddings[:3], sample_texts[:3])

        stats = store.get_stats()
        assert "total_vectors" in stats
        assert "dimension" in stats
        assert stats["total_vectors"] == 3


class TestFAISSHybridStore:
    """Tests for FAISSHybridStore."""

    def test_hybrid_search(self, sample_embeddings, sample_texts):
        """Test hybrid search."""
        dimension = len(sample_embeddings[0])
        store = FAISSHybridStore(dimension=dimension)
        store.add_embeddings(sample_embeddings, sample_texts)

        query_embedding = sample_embeddings[0]
        results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="machine learning",
            top_k=3
        )

        assert len(results) <= 3
        assert all(isinstance(r, SearchResult) for r in results)
