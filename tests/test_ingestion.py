"""
Integration tests for DocQA AI system.
Tests the full pipeline from document ingestion to query processing.
"""

import os
import sys
import json
import pytest
import asyncio
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import ChunkingPipeline, ChunkingStrategy
from src.ingestion.embedding_generator import BatchEmbeddingGenerator
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.retriever import VectorRetriever, create_retriever
from src.retrieval.hybrid_search import HybridSearcher, create_hybrid_searcher
from src.generation.llm_interface import LLMInterface
from src.generation.prompt_templates import get_rag_prompt
from src.generation.response_postprocess import postprocess_response
from src.evaluation.metrics import calculate_metrics
from src.evaluation.faithfulness import evaluate_faithfulness
from src.utils.config import get_config
from src.utils.logger import setup_logging

# Setup logging for tests
setup_logging(level="WARNING", log_to_file=False)

logger = logging.getLogger(__name__)


# ============================================================
# Test Fixtures
# ============================================================

@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        {
            "name": "sample1.txt",
            "content": """
            Machine learning is a subset of artificial intelligence that enables systems to learn and improve 
            from experience without being explicitly programmed. It uses algorithms to find patterns in data 
            and make predictions or decisions. The three main types of machine learning are supervised learning, 
            unsupervised learning, and reinforcement learning.
            """
        },
        {
            "name": "sample2.txt",
            "content": """
            Deep learning is a subset of machine learning that uses neural networks with multiple layers to 
            learn hierarchical representations of data. Neural networks are computational models inspired by 
            biological neural networks. They consist of interconnected nodes or neurons organized in layers.
            """
        },
        {
            "name": "sample3.txt",
            "content": """
            Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the 
            interaction between computers and human language. It enables computers to understand, interpret, 
            and generate human language. Key NLP tasks include sentiment analysis, named entity recognition, 
            machine translation, text summarization, and question answering.
            """
        },
        {
            "name": "sample4.txt",
            "content": """
            Computer vision is a field of artificial intelligence that enables computers to interpret and 
            understand visual information from the world. It involves acquiring, processing, analyzing, and 
            understanding images and video. Key computer vision tasks include image classification, object 
            detection, image segmentation, facial recognition, and optical character recognition.
            """
        },
        {
            "name": "sample5.txt",
            "content": """
            RAG (Retrieval-Augmented Generation) is a framework that combines retrieval-based and generation-based 
            approaches to improve the quality and accuracy of AI responses. It enhances large language models by 
            incorporating information retrieval from external knowledge sources, helping to reduce hallucinations 
            and improve factual accuracy.
            """
        }
    ]


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_files(temp_dir, sample_documents):
    """Create sample files in temporary directory."""
    files = []
    for doc in sample_documents:
        file_path = Path(temp_dir) / doc["name"]
        with open(file_path, 'w') as f:
            f.write(doc["content"])
        files.append(str(file_path))
    return files


@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return [
        {
            "query": "What is machine learning?",
            "expected_keywords": ["subset", "artificial intelligence", "algorithms", "patterns"]
        },
        {
            "query": "What are neural networks?",
            "expected_keywords": ["computational models", "biological", "neurons", "layers"]
        },
        {
            "query": "What is Natural Language Processing?",
            "expected_keywords": ["field", "artificial intelligence", "interaction", "understand", "generate"]
        },
        {
            "query": "What is computer vision?",
            "expected_keywords": ["interpret", "understand", "visual information", "images", "video"]
        },
        {
            "query": "What is RAG?",
            "expected_keywords": ["Retrieval-Augmented Generation", "retrieval", "generation", "hallucinations"]
        }
    ]


@pytest.fixture
def mock_embedding_generator(monkeypatch):
    """Mock embedding generator for testing."""
    class MockEmbeddingGenerator:
        def __init__(self, *args, **kwargs):
            self.dimension = 384
            self.model = "test-model"

        def generate_embeddings(self, chunks, **kwargs):
            import numpy as np
            results = []
            for chunk in chunks:
                text = chunk.get("text", "") if isinstance(chunk, dict) else chunk
                # Create deterministic embeddings based on text hash
                import hashlib
                hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
                np.random.seed(hash_val % 2**32)
                embedding = np.random.randn(self.dimension).tolist()

                # Create result object
                class Result:
                    def __init__(self):
                        self.embedding = embedding
                        self.text = text
                        self.metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
                        self.model = self.model
                        self.tokens_used = len(text) // 4
                results.append(Result())
            return results

        async def generate_embeddings_async(self, chunks, **kwargs):
            return self.generate_embeddings(chunks, **kwargs)

        def get_embedding_dimension(self):
            return self.dimension

        def clear_cache(self):
            pass

    return MockEmbeddingGenerator()


@pytest.fixture
def mock_llm_interface(monkeypatch):
    """Mock LLM interface for testing."""
    class MockLLMInterface:
        def __init__(self, *args, **kwargs):
            self.model = "test-model"
            self.provider = "test"

        def generate(self, messages, **kwargs):
            # Return a simple response based on the query
            query = messages[-1]["content"] if messages else ""

            class Response:
                def __init__(self, content):
                    self.content = content
                    self.model = "test-model"
                    self.provider = "test"
                    self.prompt_tokens = 10
                    self.completion_tokens = 20
                    self.total_tokens = 30
                    self.cost = 0.0
                    self.finish_reason = "stop"
                    self.latency_ms = 100
                    self.raw_response = None

            # Extract question from prompt
            if "Question:" in query:
                question = query.split("Question:")[-1].split("\n")[0].strip()
            else:
                question = query[:100]

            answer = f"This is a test response about: {question[:50]}..."
            return Response(answer)

        def generate_simple(self, prompt, system_prompt=None):
            # Extract question from prompt
            if "Question:" in prompt:
                question = prompt.split("Question:")[-1].split("\n")[0].strip()
            else:
                question = prompt[:100]

            return f"This is a test response about: {question[:50]}..."

        def get_model_info(self):
            return {"model": "test-model", "provider": "test"}

    return MockLLMInterface()


@pytest.fixture
def mock_retriever(monkeypatch):
    """Mock retriever for testing."""
    class MockRetriever:
        def __init__(self, *args, **kwargs):
            self.top_k = 5

        def retrieve(self, query, top_k=None):
            class Result:
                def __init__(self, text, score=0.8):
                    self.text = text
                    self.score = score
                    self.metadata = {"source": "test_doc.txt"}
                    self.chunk_id = f"chunk_{hash(text) % 1000}"
                    self.index = 0

            # Return mock results based on query
            if "machine learning" in query.lower():
                return [
                    Result("Machine learning is a subset of artificial intelligence that enables systems to learn from data.", 0.95),
                    Result("The three main types of machine learning are supervised, unsupervised, and reinforcement learning.", 0.85)
                ]
            elif "neural" in query.lower():
                return [
                    Result("Neural networks are computational models inspired by biological neural networks.", 0.95),
                    Result("Deep learning uses neural networks with multiple layers.", 0.85)
                ]
            elif "NLP" in query or "Natural Language" in query:
                return [
                    Result("NLP is a field of AI that focuses on the interaction between computers and human language.", 0.95),
                    Result("Key NLP tasks include sentiment analysis, machine translation, and question answering.", 0.85)
                ]
            elif "computer vision" in query.lower():
                return [
                    Result("Computer vision enables computers to interpret and understand visual information.", 0.95),
                    Result("Key computer vision tasks include image classification and object detection.", 0.85)
                ]
            elif "RAG" in query or "Retrieval-Augmented" in query:
                return [
                    Result("RAG combines retrieval-based and generation-based approaches to improve AI responses.", 0.95),
                    Result("RAG helps reduce hallucinations in AI responses.", 0.85)
                ]
            else:
                return [Result(f"Test result for: {query[:30]}...", 0.7)]

        def retrieve_with_embeddings(self, embedding, top_k=None):
            return self.retrieve("test query", top_k)

    return MockRetriever()


@pytest.fixture
def integration_components(mock_embedding_generator, mock_llm_interface, mock_retriever):
    """Set up integration test components."""
    return {
        "embedding_generator": mock_embedding_generator,
        "llm_interface": mock_llm_interface,
        "retriever": mock_retriever
    }


# ============================================================
# Integration Tests
# ============================================================

class TestDocumentIngestion:
    """Integration tests for document ingestion pipeline."""

    def test_load_documents(self, sample_files):
        """Test loading documents from files."""
        loader = DocumentLoader()
        documents = []

        for file_path in sample_files:
            doc = loader.load_document(file_path)
            documents.append(doc)

        assert len(documents) == len(sample_files)
        for doc in documents:
            assert "content" in doc
            assert "metadata" in doc
            assert "file_path" in doc
            assert len(doc["content"]) > 0

    def test_chunk_documents(self, sample_documents):
        """Test chunking documents."""
        chunker = ChunkingPipeline(
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=200,
            chunk_overlap=50
        )

        all_chunks = []
        for doc in sample_documents:
            chunks = chunker.chunk_document(doc["content"])
            all_chunks.extend(chunks)

        assert len(all_chunks) > 0
        for chunk in all_chunks:
            assert chunk.text is not None
            assert len(chunk.text) > 0
            assert hasattr(chunk, 'metadata')

    def test_embedding_generation(self, sample_documents, mock_embedding_generator):
        """Test embedding generation."""
        chunks = []
        for doc in sample_documents[:2]:
            chunker = ChunkingPipeline(chunk_size=300, chunk_overlap=50)
            doc_chunks = chunker.chunk_document(doc["content"])
            chunks.extend(doc_chunks)

        chunk_data = [{"text": c.text, "metadata": c.metadata} for c in chunks]
        embeddings = mock_embedding_generator.generate_embeddings(chunk_data)

        assert len(embeddings) == len(chunks)
        for emb in embeddings:
            assert hasattr(emb, 'embedding')
            assert len(emb.embedding) == mock_embedding_generator.dimension

    def test_vector_store_operations(self, sample_documents, mock_embedding_generator):
        """Test vector store operations."""
        # Chunk documents
        chunker = ChunkingPipeline(chunk_size=300, chunk_overlap=50)
        chunks = []
        for doc in sample_documents[:2]:
            doc_chunks = chunker.chunk_document(doc["content"])
            chunks.extend(doc_chunks)

        # Generate embeddings
        chunk_data = [{"text": c.text, "metadata": c.metadata} for c in chunks]
        embeddings = mock_embedding_generator.generate_embeddings(chunk_data)

        # Create vector store
        dimension = mock_embedding_generator.dimension
        vector_store = FAISSVectorStore(dimension=dimension, index_type="FlatIP")

        # Add embeddings
        vectors = [e.embedding for e in embeddings]
        texts = [e.text for e in embeddings]
        metadata_list = [e.metadata for e in embeddings]

        vector_store.add_embeddings(vectors, texts, metadata_list)

        assert vector_store.get_size() == len(embeddings)

        # Search
        query = "machine learning"
        query_embedding = mock_embedding_generator.generate_embeddings([{"text": query}])[0].embedding
        results = vector_store.search(query_embedding, top_k=3)

        assert len(results) > 0
        for result in results:
            assert result.text is not None
            assert result.score > 0
            assert result.index >= 0


class TestRetrieval:
    """Integration tests for retrieval system."""

    def test_vector_retrieval(self, sample_documents, mock_embedding_generator):
        """Test vector retrieval."""
        # Set up vector store
        chunker = ChunkingPipeline(chunk_size=300, chunk_overlap=50)
        chunks = []
        for doc in sample_documents:
            doc_chunks = chunker.chunk_document(doc["content"])
            chunks.extend(doc_chunks)

        chunk_data = [{"text": c.text, "metadata": c.metadata} for c in chunks]
        embeddings = mock_embedding_generator.generate_embeddings(chunk_data)

        dimension = mock_embedding_generator.dimension
        vector_store = FAISSVectorStore(dimension=dimension, index_type="FlatIP")

        vectors = [e.embedding for e in embeddings]
        texts = [e.text for e in embeddings]
        metadata_list = [e.metadata for e in embeddings]
        vector_store.add_embeddings(vectors, texts, metadata_list)

        # Create retriever
        retriever = create_retriever(
            retriever_type="vector",
            vector_store=vector_store,
            embedding_generator=mock_embedding_generator,
            top_k=5
        )

        # Test retrieval
        queries = [
            "What is machine learning?",
            "What are neural networks?",
            "What is NLP?"
        ]

        for query in queries:
            results = retriever.retrieve(query, top_k=3)
            assert len(results) > 0
            for result in results:
                assert result.text is not None
                assert result.score > 0

    def test_hybrid_retrieval(self, sample_documents, mock_embedding_generator):
        """Test hybrid retrieval."""
        # Set up vector store
        chunker = ChunkingPipeline(chunk_size=300, chunk_overlap=50)
        chunks = []
        for doc in sample_documents:
            doc_chunks = chunker.chunk_document(doc["content"])
            chunks.extend(doc_chunks)

        chunk_data = [{"text": c.text, "metadata": c.metadata} for c in chunks]
        embeddings = mock_embedding_generator.generate_embeddings(chunk_data)

        dimension = mock_embedding_generator.dimension
        vector_store = FAISSVectorStore(dimension=dimension, index_type="FlatIP")

        vectors = [e.embedding for e in embeddings]
        texts = [e.text for e in embeddings]
        vector_store.add_embeddings(vectors, texts, [{} for _ in texts])

        # Create hybrid searcher
        hybrid_searcher = create_hybrid_searcher(
            vector_store=vector_store,
            embedding_generator=mock_embedding_generator,
            vector_weight=0.7,
            keyword_weight=0.3,
            top_k=5
        )

        # Index documents for BM25
        hybrid_searcher.index_documents(texts)

        # Test search
        query = "What is machine learning and neural networks?"
        results = hybrid_searcher.search(query, top_k=3)

        assert len(results) > 0
        for result in results:
            assert result.text is not None
            assert result.combined_score > 0
            assert hasattr(result, 'vector_score')
            assert hasattr(result, 'keyword_score')


class TestGeneration:
    """Integration tests for response generation."""

    def test_llm_generation(self, mock_llm_interface):
        """Test LLM generation."""
        prompt = "What is machine learning?"
        response = mock_llm_interface.generate_simple(prompt)

        assert response is not None
        assert len(response) > 0
        assert "test response" in response.lower()

    def test_rag_pipeline(self, sample_documents, mock_embedding_generator, mock_llm_interface):
        """Test RAG pipeline."""
        # Set up components
        chunker = ChunkingPipeline(chunk_size=300, chunk_overlap=50)
        chunks = []
        for doc in sample_documents[:3]:
            doc_chunks = chunker.chunk_document(doc["content"])
            chunks.extend(doc_chunks)

        chunk_data = [{"text": c.text, "metadata": c.metadata} for c in chunks]
        embeddings = mock_embedding_generator.generate_embeddings(chunk_data)

        dimension = mock_embedding_generator.dimension
        vector_store = FAISSVectorStore(dimension=dimension, index_type="FlatIP")

        vectors = [e.embedding for e in embeddings]
        texts = [e.text for e in embeddings]
        vector_store.add_embeddings(vectors, texts, [{} for _ in texts])

        retriever = create_retriever(
            retriever_type="vector",
            vector_store=vector_store,
            embedding_generator=mock_embedding_generator,
            top_k=3
        )

        # Test RAG pipeline
        queries = [
            "What is machine learning?",
            "What are neural networks?"
        ]

        for query in queries:
            # Retrieve
            results = retriever.retrieve(query, top_k=3)
            assert len(results) > 0

            # Prepare context
            context_chunks = [
                {"text": r.text, "source": "test_doc.txt"}
                for r in results
            ]

            # Generate prompt
            prompt = get_rag_prompt(question=query, chunks=context_chunks)
            assert prompt is not None
            assert len(prompt) > 0

            # Generate response
            response = mock_llm_interface.generate_simple(prompt)
            assert response is not None
            assert len(response) > 0

    def test_response_postprocessing(self):
        """Test response post-processing."""
        test_response = """
        The answer is 42. I think that's correct.
        
        Here are the key points:
        - Point 1
        - Point 2
        - Point 3
        """

        processed = postprocess_response(test_response, aggressive_cleaning=True)

        assert processed is not None
        assert processed.cleaned_text is not None
        assert len(processed.cleaned_text) > 0
        assert hasattr(processed, 'confidence')
        assert hasattr(processed, 'has_hallucination')


class TestFullPipeline:
    """End-to-end integration tests."""

    def test_full_pipeline(self, sample_documents, mock_embedding_generator, mock_llm_interface):
        """Test the complete pipeline from ingestion to response."""
        # Step 1: Chunk documents
        chunker = ChunkingPipeline(
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=300,
            chunk_overlap=50
        )

        chunks = []
        for doc in sample_documents[:3]:
            doc_chunks = chunker.chunk_document(doc["content"])
            chunks.extend(doc_chunks)

        assert len(chunks) > 0

        # Step 2: Generate embeddings
        chunk_data = [{"text": c.text, "metadata": c.metadata} for c in chunks]
        embeddings = mock_embedding_generator.generate_embeddings(chunk_data)

        assert len(embeddings) > 0

        # Step 3: Store in vector store
        dimension = mock_embedding_generator.dimension
        vector_store = FAISSVectorStore(dimension=dimension, index_type="FlatIP")

        vectors = [e.embedding for e in embeddings]
        texts = [e.text for e in embeddings]
        vector_store.add_embeddings(vectors, texts, [{} for _ in texts])

        assert vector_store.get_size() > 0

        # Step 4: Create retriever
        retriever = create_retriever(
            retriever_type="vector",
            vector_store=vector_store,
            embedding_generator=mock_embedding_generator,
            top_k=3
        )

        # Step 5: Query
        query = "What is machine learning and how does it work?"

        # Retrieve
        retrieval_results = retriever.retrieve(query, top_k=3)
        assert len(retrieval_results) > 0

        # Step 6: Generate response
        context_chunks = [
            {"text": r.text, "source": "test_doc.txt"}
            for r in retrieval_results
        ]

        prompt = get_rag_prompt(question=query, chunks=context_chunks)
        response = mock_llm_interface.generate_simple(prompt)

        assert response is not None
        assert len(response) > 0

        # Step 7: Post-process
        processed = postprocess_response(response, aggressive_cleaning=True)

        assert processed.cleaned_text is not None
        assert len(processed.cleaned_text) > 0

    def test_evaluation_metrics(self, sample_queries):
        """Test evaluation metrics on generated responses."""
        # Generate mock responses
        candidates = []
        references = []

        for q in sample_queries[:3]:
            query = q["query"]
            # Mock response
            candidate = f"This is a test response about: {query[:30]}..."
            candidates.append(candidate)
            references.append(f"This is a reference answer about: {query[:30]}...")

        # Calculate metrics
        metrics = calculate_metrics(
            candidates,
            references,
            metrics=['bleu', 'rouge', 'meteor', 'f1']
        )

        assert metrics is not None
        assert 'bleu' in metrics
        assert 'rouge1_fmeasure' in metrics or 'rouge' in metrics

    def test_faithfulness_evaluation(self):
        """Test faithfulness evaluation."""
        source = """
        Machine learning is a subset of artificial intelligence that enables systems to learn 
        and improve from experience without being explicitly programmed.
        """

        faithful_response = """
        Machine learning is a subset of artificial intelligence that enables systems to learn from data.
        """

        hallucinated_response = """
        Machine learning was invented in 1980 by John Smith and uses quantum computers.
        """

        # Test faithful response
        result = evaluate_faithfulness(
            faithful_response,
            source,
            method="token",
            threshold=0.3
        )
        assert result.score > 0.3

        # Test hallucinated response
        result = evaluate_faithfulness(
            hallucinated_response,
            source,
            method="token",
            threshold=0.3
        )
        assert result.score < 0.3


class TestErrorHandling:
    """Integration tests for error handling."""

    def test_empty_document_handling(self):
        """Test handling of empty documents."""
        chunker = ChunkingPipeline()
        chunks = chunker.chunk_document("")

        assert chunks == []

    def test_invalid_file_handling(self):
        """Test handling of invalid files."""
        loader = DocumentLoader()

        with pytest.raises(FileNotFoundError):
            loader.load_document("/nonexistent/file.txt")

    def test_missing_query_handling(self, mock_retriever):
        """Test handling of empty queries."""
        results = mock_retriever.retrieve("", top_k=3)
        assert results is not None

    def test_vector_store_empty_search(self):
        """Test search on empty vector store."""
        vector_store = FAISSVectorStore(dimension=384)
        query_embedding = [0.0] * 384

        results = vector_store.search(query_embedding, top_k=5)
        assert results == []


class TestPerformance:
    """Performance integration tests."""

    def test_batch_processing_performance(self, sample_documents, mock_embedding_generator):
        """Test batch processing performance."""
        chunker = ChunkingPipeline(chunk_size=200, chunk_overlap=50)

        chunks = []
        for doc in sample_documents:
            doc_chunks = chunker.chunk_document(doc["content"])
            chunks.extend(doc_chunks)

        chunk_data = [{"text": c.text, "metadata": c.metadata} for c in chunks]

        start_time = time.time()
        embeddings = mock_embedding_generator.generate_embeddings(chunk_data)
        duration = time.time() - start_time

        assert len(embeddings) == len(chunks)
        assert duration < 10.0  # Should be fast for integration tests

    def test_search_performance(self, sample_documents, mock_embedding_generator):
        """Test search performance."""
        # Set up vector store with documents
        chunker = ChunkingPipeline(chunk_size=300, chunk_overlap=50)
        chunks = []
        for doc in sample_documents:
            doc_chunks = chunker.chunk_document(doc["content"])
            chunks.extend(doc_chunks)

        chunk_data = [{"text": c.text, "metadata": c.metadata} for c in chunks]
        embeddings = mock_embedding_generator.generate_embeddings(chunk_data)

        dimension = mock_embedding_generator.dimension
        vector_store = FAISSVectorStore(dimension=dimension, index_type="FlatIP")

        vectors = [e.embedding for e in embeddings]
        texts = [e.text for e in embeddings]
        vector_store.add_embeddings(vectors, texts, [{} for _ in texts])

        # Test search performance
        query_embedding = mock_embedding_generator.generate_embeddings([{"text": "test query"}])[0].embedding

        start_time = time.time()
        results = vector_store.search(query_embedding, top_k=3)
        duration = time.time() - start_time

        assert len(results) > 0
        assert duration < 1.0  # Should be fast


# ============================================================
# Test Runner
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--capture=no"])
