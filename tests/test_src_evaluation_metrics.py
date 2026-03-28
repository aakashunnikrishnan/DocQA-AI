"""
Tests for evaluation metrics module.
"""

import pytest
from src.evaluation.metrics import (
    BleuScorer, RougeScorer, MeteorScorer, F1Scorer,
    ExactMatchScorer, TextPreprocessor, EvaluationPipeline,
    RetrievalMetrics
)


class TestTextPreprocessor:
    """Tests for TextPreprocessor."""

    def test_tokenize(self):
        """Test tokenization."""
        preprocessor = TextPreprocessor()
        tokens = preprocessor.tokenize("This is a test.")

        assert len(tokens) > 0
        assert "test" in tokens

    def test_tokenize_lowercase(self):
        """Test lowercase tokenization."""
        preprocessor = TextPreprocessor(lowercase=True)
        tokens = preprocessor.tokenize("Hello World")

        assert "hello" in tokens
        assert "world" in tokens

    def test_tokenize_remove_punctuation(self):
        """Test punctuation removal."""
        preprocessor = TextPreprocessor(remove_punctuation=True)
        tokens = preprocessor.tokenize("Hello, world!")

        assert "Hello" in tokens
        assert "world" in tokens
        assert "," not in tokens
        assert "!" not in tokens


class TestBleuScorer:
    """Tests for BleuScorer."""

    def test_score(self):
        """Test BLEU scoring."""
        scorer = BleuScorer()
        candidate = "Machine learning is a subset of AI."
        reference = "Machine learning is a field of artificial intelligence."

        score = scorer.score(candidate, reference)
        assert score >= 0
        assert score <= 100

    def test_score_with_preprocessor(self):
        """Test BLEU scoring with preprocessor."""
        scorer = BleuScorer()
        preprocessor = TextPreprocessor()

        candidate = "Machine learning is a subset of AI."
        reference = "Machine learning is a field of artificial intelligence."

        score = scorer.score(candidate, reference, preprocessor)
        assert score >= 0


class TestRougeScorer:
    """Tests for RougeScorer."""

    def test_score(self):
        """Test ROUGE scoring."""
        try:
            scorer = RougeScorer()
            candidate = "Machine learning is a subset of AI."
            reference = "Machine learning is a field of artificial intelligence."

            scores = scorer.score(candidate, reference)
            assert "rouge1_fmeasure" in scores
            assert scores["rouge1_fmeasure"] >= 0

        except ImportError:
            pytest.skip("rouge_score not installed")


class TestMeteorScorer:
    """Tests for MeteorScorer."""

    def test_score(self):
        """Test METEOR scoring."""
        try:
            scorer = MeteorScorer()
            candidate = "Machine learning is a subset of AI."
            reference = "Machine learning is a field of artificial intelligence."

            score = scorer.score(candidate, reference)
            assert score >= 0
            assert score <= 100

        except ImportError:
            pytest.skip("nltk not installed")


class TestF1Scorer:
    """Tests for F1Scorer."""

    def test_score(self):
        """Test F1 scoring."""
        scorer = F1Scorer()
        candidate = "machine learning subset ai"
        reference = "machine learning field artificial intelligence"

        score = scorer.score(candidate, reference)
        assert score >= 0
        assert score <= 100


class TestExactMatchScorer:
    """Tests for ExactMatchScorer."""

    def test_score_exact_match(self):
        """Test exact match scoring."""
        candidate = "Machine learning is a subset of AI."
        reference = "Machine learning is a subset of AI."

        score = ExactMatchScorer.score(candidate, reference)
        assert score == 1.0

    def test_score_no_match(self):
        """Test no match scoring."""
        candidate = "Machine learning is a subset of AI."
        reference = "Different text entirely."

        score = ExactMatchScorer.score(candidate, reference)
        assert score == 0.0


class TestRetrievalMetrics:
    """Tests for RetrievalMetrics."""

    def test_mrr(self):
        """Test MRR calculation."""
        relevant = [[0, 1], [1, 2]]
        retrieved = [[0, 2, 1], [0, 1, 2]]

        mrr = RetrievalMetrics.mean_reciprocal_rank(relevant, retrieved)
        assert mrr > 0

    def test_recall_at_k(self):
        """Test Recall@K calculation."""
        relevant = [[0, 1, 2], [1, 2]]
        retrieved = [[0, 1, 3], [1, 0, 2]]

        recall = RetrievalMetrics.recall_at_k(relevant, retrieved, k=2)
        assert recall >= 0
        assert recall <= 1

    def test_precision_at_k(self):
        """Test Precision@K calculation."""
        relevant = [[0, 1], [1, 2]]
        retrieved = [[0, 1, 3], [1, 0, 2]]

        precision = RetrievalMetrics.precision_at_k(relevant, retrieved, k=2)
        assert precision >= 0
        assert precision <= 1


class TestEvaluationPipeline:
    """Tests for EvaluationPipeline."""

    def test_evaluate_generation(self):
        """Test generation evaluation."""
        pipeline = EvaluationPipeline(metrics=['bleu', 'f1'])
        candidates = ["Test response 1", "Test response 2"]
        references = ["Expected response 1", "Expected response 2"]

        results = pipeline.evaluate_generation(candidates, references)

        assert "bleu" in results or "f1" in results
        assert len(results) > 0

    def test_evaluate_retrieval(self):
        """Test retrieval evaluation."""
        pipeline = EvaluationPipeline()
        relevant = [[0, 1], [1, 2]]
        retrieved = [[0, 1, 3], [1, 0, 2]]

        results = pipeline.evaluate_retrieval(relevant, retrieved)

        assert "mrr" in results
        assert "recall@5" in results or "recall@1" in results
