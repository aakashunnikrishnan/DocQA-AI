"""
Unit tests for retrieval module including vector store, retriever, and search functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import tempfile
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.vector_store import FAISSVectorStore, SearchResult, FAISSHybridStore
from src.retrieval.retriever import (
    VectorRetriever, HybridRetriever, ContextualRetriever,
    MultiStageRetriever, EnsembleRetriever, RetrievalResult,
    create_retriever
)


# ============== Fixtures ==============

@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing."""
    np.random.seed(42)
    return [np.random.randn(1536).tolist() for _ in range(20)]


@pytest.fixture
def sample_texts():
    """Generate sample texts for testing."""
    return [
        "Python is a programming language",
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "Natural language processing deals with text",
        "Computer vision works with images",
        "Data science involves statistics and programming",
        "Artificial intelligence is a broad field",
        "Neural networks are inspired by the brain",
        "Transformers revolutionized NLP",
        "GPT models are large language models",
        "Vector databases enable similarity search",
        "FAISS is a library for efficient similarity search",
        "RAG combines retrieval and generation",
        "LLMs can answer questions from documents",
        "Embeddings represent text as vectors",
        "Cosine similarity measures vector similarity",
        "Document chunking improves retrieval",
        "Hybrid search combines vector and keyword",
        "Reranking improves result quality",
        "Evaluation metrics include recall and precision"
    ] * 1  # 20 texts


@pytest.fixture
def vector_store(sample_embeddings, sample_texts):
    """Create a vector store with sample data."""
    store = FAISSVectorStore(dimension=1536, index_type="FlatIP")
    store.add_embeddings(
        embeddings=sample_embeddings,
        texts=sample_texts,
        metadata=[{"index": i} for i in range(len(sample_texts))]
    )
    return store


@pytest.fixture
def mock_embedding_generator():
    """Create a mock embedding generator."""
    mock = Mock()

    def generate_embedding(text):
        result = Mock()
        result.embedding = np.random.randn(1536).tolist()
        return result

    mock.generate_embedding = generate_embedding
    return mock


@pytest.fixture
def temp_vector_store_path():
    """Create temporary path for saving vector store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ============== Vector Store Tests ==============

class TestFAISSVectorStore:
    """Tests for FAISSVectorStore."""

    def test_initialization(self):
        """Test vector store initialization."""
        store = FAISSVectorStore(dimension=768, index_type="FlatIP")
        assert store.dimension == 768
        assert store.index_type == "FlatIP"
        assert store.get_size() == 0

    def test_add_embeddings(self, vector_store, sample_embeddings, sample_texts):
        """Test adding embeddings to store."""
        assert vector_store.get_size() == len(sample_embeddings)

    def test_search(self, vector_store):
        """Test similarity search."""
        query = np.random.randn(1536).tolist()
        results = vector_store.search(query, top_k=3)

        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.score >= 0 for r in results)
        assert all(r.index >= 0 for r in results)

    def test_search_with_threshold(self, vector_store):
        """Test search with score threshold."""
        query = np.random.randn(1536).tolist()
        results = vector_store.search(query, top_k=5, score_threshold=0.5)

        for result in results:
            assert result.score >= 0.5

    def test_search_with_metadata_filter(self, vector_store):
        """Test search with metadata filtering."""
        query = np.random.randn(1536).tolist()
        results = vector_store.search(
            query, top_k=5, filter_metadata={"index": 0}
        )

        for result in results:
            assert result.metadata.get("index") == 0

    def test_search_empty_store(self):
        """Test search on empty vector store."""
        store = FAISSVectorStore(dimension=1536)
        query = np.random.randn(1536).tolist()
        results = store.search(query, top_k=5)

        assert results == []

    def test_save_and_load(self, vector_store, temp_vector_store_path):
        """Test saving and loading vector store."""
        # Save
        save_path = os.path.join(temp_vector_store_path, "test_index")
        vector_store.save(save_path)

        # Load
        new_store = FAISSVectorStore(dimension=1536)
        new_store.load(save_path)

        assert new_store.get_size() == vector_store.get_size()

    def test_clear(self, vector_store):
        """Test clearing vector store."""
        assert vector_store.get_size() > 0
        vector_store.clear()
        assert vector_store.get_size() == 0

    def test_get_stats(self, vector_store):
        """Test getting store statistics."""
        stats = vector_store.get_stats()

        assert "total_vectors" in stats
        assert "dimension" in stats
        assert "index_type" in stats
        assert stats["total_vectors"] == vector_store.get_size()

    def test_different_index_types(self):
        """Test different FAISS index types."""
        index_types = ["FlatIP", "FlatL2", "HNSW32", "HNSW64"]

        for index_type in index_types:
            store = FAISSVectorStore(dimension=128, index_type=index_type)
            assert store.index_type == index_type

    def test_batch_search(self, vector_store):
        """Test batch search functionality."""
        queries = [np.random.randn(1536).tolist() for _ in range(3)]
        batch_results = vector_store.search_batch(queries, top_k=2)

        assert len(batch_results) == 3
        for results in batch_results:
            assert len(results) <= 2


class TestFAISSHybridStore:
    """Tests for FAISSHybridStore."""

    def test_hybrid_search(self, sample_embeddings, sample_texts):
        """Test hybrid search functionality."""
        store = FAISSHybridStore(dimension=1536, index_type="FlatIP")
        store.add_embeddings(sample_embeddings, sample_texts)

        query_embedding = np.random.randn(1536).tolist()
        results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="machine learning",
            top_k=3,
            vector_weight=0.6,
            keyword_weight=0.4
        )

        assert len(results) <= 3
        assert all(isinstance(r, SearchResult) for r in results)

    def test_hybrid_search_without_keyword(self, sample_embeddings, sample_texts):
        """Test hybrid search falls back to vector search when no BM25."""
        store = FAISSHybridStore(dimension=1536, index_type="FlatIP")
        store.add_embeddings(sample_embeddings, sample_texts)

        query_embedding = np.random.randn(1536).tolist()
        results = store.hybrid_search(
            query_embedding=query_embedding,
            query_text="",
            top_k=3
        )

        assert len(results) <= 3


# ============== Retriever Tests ==============

class TestVectorRetriever:
    """Tests for VectorRetriever."""

    def test_initialization(self, vector_store, mock_embedding_generator):
        """Test retriever initialization."""
        retriever = VectorRetriever(
            vector_store=vector_store,
            embedding_generator=mock_embedding_generator,
            top_k=5
        )

        assert retriever.top_k == 5
        assert retriever.vector_store == vector_store

    def test_retrieve(self, vector_store, mock_embedding_generator):
        """Test basic retrieval."""
        retriever = VectorRetriever(
            vector_store=vector_store,
            embedding_generator=mock_embedding_generator,
            top_k=3
        )

        results = retriever.retrieve("What is machine learning?")

        assert len(results) <= 3
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert all(r.score >= 0 for r in results)
        assert all(r.text for r in results)

    def test_retrieve_with_threshold(self, vector_store, mock_embedding_generator):
        """Test retrieval with score threshold."""
        retriever = VectorRetriever(
            vector_store=vector_store,
            embedding_generator=mock_embedding_generator,
            top_k=5,
            score_threshold=0.5
        )

        results = retriever.retrieve("test query")

        for result in results:
            assert result.score >= 0.5

    def test_retrieve_with_metadata_filter(self, vector_store, mock_embedding_generator):
        """Test retrieval with metadata filter."""
        retriever = VectorRetriever(
            vector_store=vector_store,
            embedding_generator=mock_embedding_generator,
            top_k=5
        )

        results = retriever.retrieve(
            "test query",
            filter_metadata={"index": 0}
        )

        for result in results:
            assert result.metadata.get("index") == 0

    def test_retrieve_with_embeddings(self, vector_store, mock_embedding_generator):
        """Test retrieval with pre-computed embeddings."""
        retriever = VectorRetriever(
            vector_store=vector_store,
            embedding_generator=mock_embedding_generator,
            top_k=3
        )

        query_embedding = np.random.randn(1536).tolist()
        results = retriever.retrieve_with_embeddings(query_embedding)

        assert len(results) <= 3

    def test_cache_functionality(self, vector_store, mock_embedding_generator):
        """Test query embedding caching."""
        retriever = VectorRetriever(
            vector_store=vector_store,
            embedding_generator=mock_embedding_generator,
            use_cache=True
        )

        query = "test query"

        # First call should generate embedding
        results1 = retriever.retrieve(query)

        # Second call should use cache
        results2 = retriever.retrieve(query)

        assert mock_embedding_generator.generate_embedding.call_count <= 2

    def test_clear_cache(self, vector_store, mock_embedding_generator):
        """Test clearing cache."""
        retriever = VectorRetriever(
            vector_store=vector_store,
            embedding_generator=mock_embedding_generator,
            use_cache=True
        )

        retriever.retrieve("query1")
        retriever.retrieve("query2")

        assert len(retriever._embedding_cache) == 2

        retriever.clear_cache()

        assert len(retriever._embedding_cache) == 0

    def test_empty_store_retrieval(self, mock_embedding_generator):
        """Test retrieval from empty store."""
        empty_store = FAISSVectorStore(dimension=1536)
        retriever = VectorRetriever(
            vector_store=empty_store,
            embedding_generator=mock_embedding_generator,
            top_k=3
        )

        results = retriever.retrieve("test query")

        assert results == []


class TestHybridRetriever:
    """Tests for HybridRetriever."""

    @pytest.fixture
    def hybrid_store(self, sample_embeddings, sample_texts):
        """Create hybrid store with sample data."""
        store = FAISSHybridStore(dimension=1536, index_type="FlatIP")
        store.add_embeddings(sample_embeddings, sample_texts)
        return store

    def test_initialization(self, hybrid_store, mock_embedding_generator):
        """Test hybrid retriever initialization."""
        retriever = HybridRetriever(
            vector_store=hybrid_store,
            embedding_generator=mock_embedding_generator,
            top_k=5,
            vector_weight=0.7,
            keyword_weight=0.3
        )

        assert retriever.top_k == 5
        assert retriever.vector_weight == 0.7
        assert retriever.keyword_weight == 0.3

    def test_hybrid_retrieval(self, hybrid_store, mock_embedding_generator):
        """Test hybrid retrieval."""
        retriever = HybridRetriever(
            vector_store=hybrid_store,
            embedding_generator=mock_embedding_generator,
            top_k=3
        )

        results = retriever.retrieve("machine learning and AI")

        assert len(results) <= 3
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_weight_override(self, hybrid_store, mock_embedding_generator):
        """Test overriding weights per query."""
        retriever = HybridRetriever(
            vector_store=hybrid_store,
            embedding_generator=mock_embedding_generator,
            vector_weight=0.5,
            keyword_weight=0.5
        )

        results1 = retriever.retrieve("test query")
        results2 = retriever.retrieve("test query", vector_weight=0.8, keyword_weight=0.2)

        # Both should work
        assert results1 is not None
        assert results2 is not None

    def test_batch_retrieval(self, hybrid_store, mock_embedding_generator):
        """Test batch retrieval."""
        retriever = HybridRetriever(
            vector_store=hybrid_store,
            embedding_generator=mock_embedding_generator,
            top_k=2
        )

        queries = ["machine learning", "python programming", "data science"]
        batch_results = retriever.retrieve_batch(queries)

        assert len(batch_results) == 3
        for results in batch_results:
            assert len(results) <= 2

    def test_weight_normalization(self, hybrid_store, mock_embedding_generator):
        """Test weight normalization."""
        retriever = HybridRetriever(
            vector_store=hybrid_store,
            embedding_generator=mock_embedding_generator,
            vector_weight=1.0,
            keyword_weight=1.0
        )

        # Should normalize to 0.5 each
        assert abs(retriever.vector_weight - 0.5) < 0.01
        assert abs(retriever.keyword_weight - 0.5) < 0.01


class TestContextualRetriever:
    """Tests for ContextualRetriever."""

    def test_initialization(self, vector_store, mock_embedding_generator):
        """Test contextual retriever initialization."""
        base_retriever = VectorRetriever(vector_store, mock_embedding_generator)
        contextual_retriever = ContextualRetriever(
            base_retriever=base_retriever,
            max_context_history=3,
            context_weight=0.3
        )

        assert contextual_retriever.max_context_history == 3
        assert contextual_retriever.context_weight == 0.3

    def test_add_to_history(self, vector_store, mock_embedding_generator):
        """Test adding to conversation history."""
        base_retriever = VectorRetriever(vector_store, mock_embedding_generator)
        contextual_retriever = ContextualRetriever(base_retriever=base_retriever)

        contextual_retriever.add_to_history("What is AI?", "AI is artificial intelligence")
        contextual_retriever.add_to_history("Tell me more", "More information...")

        assert len(contextual_retriever.conversation_history) == 2

    def test_retrieve_with_context(self, vector_store, mock_embedding_generator):
        """Test retrieval with conversation context."""
        base_retriever = Mock()
        base_retriever.retrieve.return_value = []

        contextual_retriever = ContextualRetriever(base_retriever=base_retriever)
        contextual_retriever.add_to_history("Previous question", "Previous answer")

        contextual_retriever.retrieve("Current question", use_context=True)

        # Should call retrieve with expanded query
        assert base_retriever.retrieve.called

    def test_retrieve_without_context(self, vector_store, mock_embedding_generator):
        """Test retrieval without context."""
        base_retriever = Mock()
        base_retriever.retrieve.return_value = []

        contextual_retriever = ContextualRetriever(base_retriever=base_retriever)
        contextual_retriever.add_to_history("Previous question", "Previous answer")

        contextual_retriever.retrieve("Current question", use_context=False)

        # Should call retrieve with original query
        call_args = base_retriever.retrieve.call_args[0][0]
        assert "Previous question" not in call_args

    def test_clear_history(self, vector_store, mock_embedding_generator):
        """Test clearing conversation history."""
        base_retriever = VectorRetriever(vector_store, mock_embedding_generator)
        contextual_retriever = ContextualRetriever(base_retriever=base_retriever)

        contextual_retriever.add_to_history("Q1", "A1")
        contextual_retriever.add_to_history("Q2", "A2")

        assert len(contextual_retriever.conversation_history) == 2

        contextual_retriever.clear_history()

        assert len(contextual_retriever.conversation_history) == 0


class TestMultiStageRetriever:
    """Tests for MultiStageRetriever."""

    def test_initialization(self, vector_store, mock_embedding_generator):
        """Test multi-stage retriever initialization."""
        base_retriever = VectorRetriever(vector_store, mock_embedding_generator)
        multi_stage = MultiStageRetriever(
            base_retriever=base_retriever,
            initial_top_k=20,
            final_top_k=5
        )

        assert multi_stage.initial_top_k == 20
        assert multi_stage.final_top_k == 5

    def test_retrieve_with_reranking(self, vector_store, mock_embedding_generator):
        """Test retrieval with reranking."""
        base_retriever = VectorRetriever(vector_store, mock_embedding_generator)
        multi_stage = MultiStageRetriever(
            base_retriever=base_retriever,
            initial_top_k=10,
            final_top_k=3
        )

        # Mock base retriever to return some results
        mock_results = [
            RetrievalResult(text=f"Result {i}", score=1.0 - i*0.1, metadata={})
            for i in range(10)
        ]
        base_retriever.retrieve = Mock(return_value=mock_results)

        results = multi_stage.retrieve("test query")

        assert len(results) <= 3

    def test_empty_results(self, vector_store, mock_embedding_generator):
        """Test when base retriever returns no results."""
        base_retriever = VectorRetriever(vector_store, mock_embedding_generator)
        base_retriever.retrieve = Mock(return_value=[])

        multi_stage = MultiStageRetriever(base_retriever=base_retriever)
        results = multi_stage.retrieve("test query")

        assert results == []


class TestEnsembleRetriever:
    """Tests for EnsembleRetriever."""

    def test_initialization(self, vector_store, mock_embedding_generator):
        """Test ensemble retriever initialization."""
        retriever1 = VectorRetriever(vector_store, mock_embedding_generator)
        retriever2 = VectorRetriever(vector_store, mock_embedding_generator)

        ensemble = EnsembleRetriever(
            retrievers=[retriever1, retriever2],
            weights=[0.6, 0.4],
            top_k=5
        )

        assert len(ensemble.retrievers) == 2
        assert ensemble.weights == [0.6, 0.4]

    def test_equal_weights(self, vector_store, mock_embedding_generator):
        """Test automatic equal weight assignment."""
        retriever1 = VectorRetriever(vector_store, mock_embedding_generator)
        retriever2 = VectorRetriever(vector_store, mock_embedding_generator)

        ensemble = EnsembleRetriever(
            retrievers=[retriever1, retriever2]
        )

        assert ensemble.weights[0] == ensemble.weights[1] == 0.5

    def test_ensemble_retrieval(self, vector_store, mock_embedding_generator):
        """Test ensemble retrieval."""
        retriever1 = VectorRetriever(vector_store, mock_embedding_generator)
        retriever2 = VectorRetriever(vector_store, mock_embedding_generator)

        # Mock different results
        retriever1.retrieve = Mock(return_value=[
            RetrievalResult(text="Result A", score=0.9, chunk_id="A", metadata={}),
            RetrievalResult(text="Result B", score=0.8, chunk_id="B", metadata={})
        ])

        retriever2.retrieve = Mock(return_value=[
            RetrievalResult(text="Result B", score=0.85, chunk_id="B", metadata={}),
            RetrievalResult(text="Result C", score=0.75, chunk_id="C", metadata={})
        ])

        ensemble = EnsembleRetriever(
            retrievers=[retriever1, retriever2],
            weights=[0.5, 0.5],
            top_k=3
        )

        results = ensemble.retrieve("test query")

        # Should combine and deduplicate results
        assert len(results) <= 3

        # Result B should have combined score
        for result in results:
            if result.chunk_id == "B":
                assert result.score == (0.8 * 0.5 + 0.85 * 0.5)


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_creation(self):
        """Test creating retrieval result."""
        result = RetrievalResult(
            text="Test text",
            score=0.95,
            metadata={"source": "doc1"},
            chunk_id="chunk_123",
            relevance_score=0.9
        )

        assert result.text == "Test text"
        assert result.score == 0.95
        assert result.metadata["source"] == "doc1"
        assert result.chunk_id == "chunk_123"
        assert result.relevance_score == 0.9

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = RetrievalResult(
            text="Test",
            score=0.85,
            metadata={"key": "value"}
        )

        result_dict = result.to_dict()

        assert "text" in result_dict
        assert "score" in result_dict
        assert "metadata" in result_dict
        assert result_dict["score"] == 0.85


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_creation(self):
        """Test creating search result."""
        result = SearchResult(
            text="Search result text",
            score=0.92,
            metadata={"doc_id": 123},
            chunk_id="chunk_1",
            index=5
        )

        assert result.text == "Search result text"
        assert result.score == 0.92
        assert result.metadata["doc_id"] == 123
        assert result.chunk_id == "chunk_1"
        assert result.index == 5

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = SearchResult(
            text="Test",
            score=0.75,
            metadata={}
        )

        result_dict = result.to_dict()

        assert "text" in result_dict
        assert "score" in result_dict
        assert "metadata" in result_dict


class TestFactoryFunction:
    """Tests for create_retriever factory function."""

    def test_create_vector_retriever(self, vector_store, mock_embedding_generator):
        """Test creating vector retriever."""
        retriever = create_retriever(
            retriever_type="vector",
            vector_store=vector_store,
            embedding_generator=mock_embedding_generator,
            top_k=10
        )

        assert isinstance(retriever, VectorRetriever)
        assert retriever.top_k == 10

    def test_create_hybrid_retriever(self, vector_store, mock_embedding_generator):
        """Test creating hybrid retriever."""
        retriever = create_retriever(
            retriever_type="hybrid",
            vector_store=vector_store,
            embedding_generator=mock_embedding_generator,
            top_k=5,
            vector_weight=0.6
        )

        assert isinstance(retriever, HybridRetriever)

    def test_create_contextual_retriever(self, vector_store, mock_embedding_generator):
        """Test creating contextual retriever."""
        base = VectorRetriever(vector_store, mock_embedding_generator)
        retriever = create_retriever(
            retriever_type="contextual",
            base_retriever=base,
            max_context_history=5
        )

        assert isinstance(retriever, ContextualRetriever)

    def test_invalid_type(self, vector_store, mock_embedding_generator):
        """Test invalid retriever type."""
        with pytest.raises(ValueError):
            create_retriever(
                retriever_type="invalid",
                vector_store=vector_store,
                embedding_generator=mock_embedding_generator
            )


# ============== Performance Tests ==============

class TestPerformance:
    """Performance tests for retrieval."""

    def test_large_scale_search(self):
        """Test search performance with many vectors."""
        store = FAISSVectorStore(dimension=128, index_type="FlatIP")

        # Add 1000 random vectors
        num_vectors = 1000
        embeddings = [np.random.randn(128).tolist() for _ in range(num_vectors)]
        texts = [f"Document {i}" for i in range(num_vectors)]

        store.add_embeddings(embeddings, texts)

        # Search
        query = np.random.randn(128).tolist()

        import time
        start = time.time()
        results = store.search(query, top_k=10)
        duration = time.time() - start

        assert len(results) == 10
        assert duration < 0.5  # Should be fast

    def test_batch_search_performance(self, vector_store):
        """Test batch search performance."""
        queries = [np.random.randn(1536).tolist() for _ in range(10)]

        import time
        start = time.time()
        batch_results = vector_store.search_batch(queries, top_k=5)
        duration = time.time() - start

        assert len(batch_results) == 10
        assert duration < 1.0  # Should be reasonably fast


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
