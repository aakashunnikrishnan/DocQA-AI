"""
Pytest configuration and fixtures for DocQA AI tests.
"""

import os
import sys
import pytest
import tempfile
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from unittest.mock import Mock, MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import get_config, ConfigManager
from src.utils.logger import setup_logging

# Setup test logging
setup_logging(level="WARNING", log_to_file=False)


@pytest.fixture(scope="session")
def test_config():
    """Get test configuration."""
    config = get_config()
    config.environment = "test"
    config.debug = True
    config.processing.chunk_size = 200
    config.processing.chunk_overlap = 50
    config.retrieval.top_k = 3
    return config


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_texts():
    """Return sample texts for testing."""
    return [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing deals with text and language.",
        "Computer vision enables machines to understand images.",
        "Reinforcement learning involves agents learning through interaction.",
        "Python is a popular programming language for data science.",
        "TensorFlow and PyTorch are popular deep learning frameworks.",
        "Data science combines statistics and programming skills."
    ]


@pytest.fixture
def sample_embeddings(sample_texts):
    """Generate sample embeddings for testing."""
    np.random.seed(42)
    dimension = 384
    return [np.random.randn(dimension).tolist() for _ in range(len(sample_texts))]


@pytest.fixture
def mock_embedding_generator():
    """Create a mock embedding generator."""
    mock = Mock()

    def generate_embedding(text):
        result = Mock()
        result.embedding = np.random.randn(384).tolist()
        result.text = text
        result.metadata = {}
        result.model = "test-model"
        result.tokens_used = len(text) // 4
        return result

    def generate_embeddings(chunks, **kwargs):
        results = []
        for chunk in chunks:
            text = chunk.get("text", "") if isinstance(chunk, dict) else chunk
            result = Mock()
            result.embedding = np.random.randn(384).tolist()
            result.text = text
            result.metadata = chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
            result.model = "test-model"
            result.tokens_used = len(text) // 4
            results.append(result)
        return results

    mock.generate_embedding = generate_embedding
    mock.generate_embeddings = generate_embeddings
    mock.get_embedding_dimension = Mock(return_value=384)
    mock.clear_cache = Mock()
    mock.model = "test-model"

    return mock


@pytest.fixture
def mock_llm_interface():
    """Create a mock LLM interface."""
    mock = Mock()

    def generate_simple(prompt, system_prompt=None):
        return f"Mock response for: {prompt[:50]}..."

    def generate(messages, **kwargs):
        class Response:
            def __init__(self):
                self.content = "Mock response"
                self.model = "test-model"
                self.provider = "test"
                self.prompt_tokens = 10
                self.completion_tokens = 20
                self.total_tokens = 30
                self.cost = 0.0
                self.finish_reason = "stop"
                self.latency_ms = 100
                self.raw_response = None
                self.retry_count = 0
        return Response()

    mock.generate_simple = generate_simple
    mock.generate = generate
    mock.model = "test-model"
    mock.provider = "test"

    return mock


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        {
            "name": "doc1.txt",
            "content": """
            Machine learning is a subset of artificial intelligence that enables systems 
            to learn and improve from experience without being explicitly programmed.
            The three main types are supervised learning, unsupervised learning, 
            and reinforcement learning.
            """
        },
        {
            "name": "doc2.txt",
            "content": """
            Deep learning is a subset of machine learning that uses neural networks 
            with multiple layers. Neural networks are computational models inspired by 
            biological neural networks.
            """
        },
        {
            "name": "doc3.txt",
            "content": """
            Natural Language Processing (NLP) is a field of artificial intelligence 
            that focuses on the interaction between computers and human language.
            Key tasks include sentiment analysis, named entity recognition, and 
            machine translation.
            """
        }
    ]


@pytest.fixture
def sample_qa_pairs():
    """Create sample QA pairs for testing."""
    return [
        {
            "question": "What is machine learning?",
            "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "context": "Machine learning is a subset of AI that enables systems to learn and improve from experience."
        },
        {
            "question": "What are neural networks?",
            "answer": "Neural networks are computational models inspired by biological neural networks.",
            "context": "Deep learning uses neural networks with multiple layers."
        },
        {
            "question": "What is NLP?",
            "answer": "NLP is a field of AI that focuses on the interaction between computers and human language.",
            "context": "Natural Language Processing deals with text and language understanding."
        }
    ]


@pytest.fixture
def sample_conversations():
    """Create sample conversations for testing."""
    return [
        {
            "messages": [
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a subset of AI..."}
            ],
            "system_prompt": "You are a helpful assistant."
        }
    ]


@pytest.fixture
def mock_retriever():
    """Create a mock retriever."""
    mock = Mock()

    def retrieve(query, top_k=5):
        class Result:
            def __init__(self, text, score=0.8):
                self.text = text
                self.score = score
                self.metadata = {"source": "test_doc.txt"}
                self.chunk_id = f"chunk_{hash(text) % 1000}"
                self.index = 0

        return [
            Result(f"Result for: {query[:30]}...", 0.95),
            Result(f"Another result for: {query[:30]}...", 0.85)
        ][:top_k]

    mock.retrieve = retrieve
    mock.top_k = 5

    return mock


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    mock = Mock()
    mock.get_size = Mock(return_value=100)
    mock.add_embeddings = Mock(return_value=[0, 1, 2])
    mock.search = Mock(return_value=[])
    mock.clear = Mock()
    mock.save = Mock()
    mock.load = Mock()
    mock.dimension = 384

    return mock


# Test data for various modules
@pytest.fixture
def test_data_dir():
    """Get the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def test_config_path():
    """Get test config path."""
    return Path(__file__).parent / "config" / "test.yaml"
