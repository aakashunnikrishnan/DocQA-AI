"""
Relevance scoring module for evaluating retrieval quality.
Provides multiple approaches for measuring relevance:
- Semantic similarity (embedding-based)
- Token overlap (lexical matching)
- NLI-based relevance
- Cross-encoder relevance
- Hybrid relevance scoring
- MRR, NDCG, Precision@K, Recall@K
"""

import re
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from collections import defaultdict
import math
from functools import lru_cache

# Try importing NLP libraries
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not installed. Install with: pip install transformers")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not installed. Install with: pip install scikit-learn")

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logging.warning("rank-bm25 not installed. Install with: pip install rank-bm25")

logger = logging.getLogger(__name__)


@dataclass
class RelevanceResult:
    """Result of relevance scoring."""
    score: float  # 0-1 relevance score
    is_relevant: bool
    confidence: float
    method: str
    explanations: List[str] = field(default_factory=list)
    features: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "is_relevant": self.is_relevant,
            "confidence": self.confidence,
            "method": self.method,
            "explanations": self.explanations[:5],
            "features": self.features,
            "metadata": self.metadata
        }


class RelevanceScorer:
    """Base class for relevance scorers."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def score(
        self,
        query: str,
        document: str,
        **kwargs
    ) -> RelevanceResult:
        """Score relevance between query and document."""
        raise NotImplementedError

    def batch_score(
        self,
        queries: List[str],
        documents: List[str],
        **kwargs
    ) -> List[RelevanceResult]:
        """Score relevance for multiple pairs."""
        results = []
        for query, doc in zip(queries, documents):
            results.append(self.score(query, doc, **kwargs))
        return results


class SemanticRelevanceScorer(RelevanceScorer):
    """
    Semantic relevance scoring using sentence embeddings.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.5,
        device: str = "cpu"
    ):
        """
        Initialize semantic relevance scorer.

        Args:
            model_name: Sentence transformer model name
            threshold: Relevance threshold
            device: Device to run model on
        """
        super().__init__(threshold)

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")

        self.model_name = model_name
        self.device = device

        # Initialize model
        self.model = SentenceTransformer(model_name, device=device)

        logger.info(f"Initialized SemanticRelevanceScorer with model={model_name}")

    def score(
        self,
        query: str,
        document: str,
        **kwargs
    ) -> RelevanceResult:
        """
        Score relevance using semantic similarity.

        Args:
            query: Query text
            document: Document text
            **kwargs: Additional arguments

        Returns:
            RelevanceResult object
        """
        if not query or not document:
            return RelevanceResult(
                score=0.0,
                is_relevant=False,
                confidence=0.0,
                method="semantic",
                explanations=["Empty query or document"]
            )

        try:
            # Generate embeddings
            query_emb = self.model.encode(query, convert_to_tensor=True)
            doc_emb = self.model.encode(document, convert_to_tensor=True)

            # Compute similarity
            similarity = util.cos_sim(query_emb, doc_emb).item()

            # Determine relevance
            is_relevant = similarity >= self.threshold

            # Calculate confidence (based on distance from threshold)
            confidence = 1.0 - abs(similarity - self.threshold) / self.threshold
            confidence = max(0.0, min(1.0, confidence))

            return RelevanceResult(
                score=similarity,
                is_relevant=is_relevant,
                confidence=confidence,
                method="semantic",
                explanations=[
                    f"Semantic similarity: {similarity:.3f}",
                    f"Threshold: {self.threshold:.3f}",
                    f"Relevant: {is_relevant}"
                ],
                features={
                    "similarity": similarity,
                    "threshold": self.threshold
                },
                metadata={
                    "model": self.model_name,
                    "query_length": len(query),
                    "document_length": len(document)
                }
            )

        except Exception as e:
            logger.warning(f"Semantic relevance scoring failed: {e}")
            return RelevanceResult(
                score=0.0,
                is_relevant=False,
                confidence=0.0,
                method="semantic",
                explanations=[f"Scoring failed: {str(e)}"]
            )

    def score_batch(
        self,
        queries: List[str],
        documents: List[str],
        batch_size: int = 32,
        **kwargs
    ) -> List[RelevanceResult]:
        """
        Score relevance for multiple pairs efficiently.

        Args:
            queries: List of query texts
            documents: List of document texts
            batch_size: Batch size for encoding

        Returns:
            List of RelevanceResult objects
        """
        if not queries or not documents:
            return []

        try:
            # Encode all texts
            all_texts = queries + documents
            embeddings = self.model.encode(
                all_texts,
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=False
            )

            # Split embeddings
            query_embs = embeddings[:len(queries)]
            doc_embs = embeddings[len(queries):]

            # Compute similarities
            results = []
            for i, (q_emb, d_emb) in enumerate(zip(query_embs, doc_embs)):
                similarity = util.cos_sim(q_emb, d_emb).item()
                is_relevant = similarity >= self.threshold
                confidence = 1.0 - abs(similarity - self.threshold) / self.threshold
                confidence = max(0.0, min(1.0, confidence))

                results.append(RelevanceResult(
                    score=similarity,
                    is_relevant=is_relevant,
                    confidence=confidence,
                    method="semantic",
                    explanations=[
                        f"Semantic similarity: {similarity:.3f}",
                        f"Relevant: {is_relevant}"
                    ],
                    features={"similarity": similarity},
                    metadata={
                        "model": self.model_name,
                        "query": queries[i][:100],
                        "document": documents[i][:100]
                    }
                ))

            return results

        except Exception as e:
            logger.warning(f"Batch semantic scoring failed: {e}")
            return []


class TFIDFRelevanceScorer(RelevanceScorer):
    """
    Relevance scoring using TF-IDF and cosine similarity.
    """

    def __init__(
        self,
        threshold: float = 0.3,
        max_features: int = 10000,
        stop_words: str = "english"
    ):
        """
        Initialize TF-IDF relevance scorer.

        Args:
            threshold: Relevance threshold
            max_features: Maximum number of features
            stop_words: Stop words to remove
        """
        super().__init__(threshold)

        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not installed. Install with: pip install scikit-learn")

        self.max_features = max_features
        self.stop_words = stop_words
        self.vectorizer = None

        logger.info(f"Initialized TFIDFRelevanceScorer")

    def _fit_vectorizer(self, texts: List[str]):
        """Fit TF-IDF vectorizer on texts."""
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words=self.stop_words,
            lowercase=True
        )
        self.vectorizer.fit(texts)

    def score(
        self,
        query: str,
        document: str,
        **kwargs
    ) -> RelevanceResult:
        """
        Score relevance using TF-IDF similarity.

        Args:
            query: Query text
            document: Document text
            **kwargs: Additional arguments

        Returns:
            RelevanceResult object
        """
        if not query or not document:
            return RelevanceResult(
                score=0.0,
                is_relevant=False,
                confidence=0.0,
                method="tfidf",
                explanations=["Empty query or document"]
            )

        try:
            # Fit vectorizer if not already fitted
            if self.vectorizer is None:
                self._fit_vectorizer([query, document])

            # Transform texts
            query_vec = self.vectorizer.transform([query])
            doc_vec = self.vectorizer.transform([document])

            # Compute similarity
            similarity = cosine_similarity(query_vec, doc_vec)[0][0]

            # Determine relevance
            is_relevant = similarity >= self.threshold

            # Calculate confidence
            confidence = 1.0 - abs(similarity - self.threshold) / self.threshold
            confidence = max(0.0, min(1.0, confidence))

            # Get top features
            feature_names = self.vectorizer.get_feature_names_out()
            query_features = query_vec.toarray()[0]
            top_indices = query_features.argsort()[-5:][::-1]
            top_features = [feature_names[i] for i in top_indices if query_features[i] > 0]

            return RelevanceResult(
                score=similarity,
                is_relevant=is_relevant,
                confidence=confidence,
                method="tfidf",
                explanations=[
                    f"TF-IDF similarity: {similarity:.3f}",
                    f"Relevant: {is_relevant}",
                    f"Top features: {', '.join(top_features[:5])}"
                ],
                features={
                    "similarity": similarity,
                    "threshold": self.threshold,
                    "top_features": top_features[:5]
                },
                metadata={
                    "query_length": len(query),
                    "document_length": len(document)
                }
            )

        except Exception as e:
            logger.warning(f"TF-IDF relevance scoring failed: {e}")
            return RelevanceResult(
                score=0.0,
                is_relevant=False,
                confidence=0.0,
                method="tfidf",
                explanations=[f"Scoring failed: {str(e)}"]
            )


class BM25RelevanceScorer(RelevanceScorer):
    """
    Relevance scoring using BM25 (Okapi BM25) algorithm.
    """

    def __init__(
        self,
        threshold: float = 0.3,
        k1: float = 1.5,
        b: float = 0.75
    ):
        """
        Initialize BM25 relevance scorer.

        Args:
            threshold: Relevance threshold
            k1: BM25 k1 parameter
            b: BM25 b parameter
        """
        super().__init__(threshold)

        if not BM25_AVAILABLE:
            raise ImportError("rank-bm25 not installed. Install with: pip install rank-bm25")

        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.corpus = []

        logger.info(f"Initialized BM25RelevanceScorer")

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25."""
        # Lowercase and split
        tokens = text.lower().split()

        # Remove punctuation
        tokens = [t.strip('.,!?;:()[]{}"\'') for t in tokens]

        # Remove empty tokens
        tokens = [t for t in tokens if t]

        return tokens

    def _fit_bm25(self, texts: List[str]):
        """Fit BM25 on texts."""
        tokenized_corpus = [self._tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        self.corpus = texts

    def score(
        self,
        query: str,
        document: str,
        **kwargs
    ) -> RelevanceResult:
        """
        Score relevance using BM25.

        Args:
            query: Query text
            document: Document text
            **kwargs: Additional arguments

        Returns:
            RelevanceResult object
        """
        if not query or not document:
            return RelevanceResult(
                score=0.0,
                is_relevant=False,
                confidence=0.0,
                method="bm25",
                explanations=["Empty query or document"]
            )

        try:
            # Fit BM25 if not already fitted
            if self.bm25 is None:
                self._fit_bm25([document])

            # Tokenize query
            query_tokens = self._tokenize(query)

            # Get BM25 scores
            scores = self.bm25.get_scores(query_tokens)

            # Get score for document (if multiple documents, use first)
            score = scores[0] if len(scores) > 0 else 0.0

            # Normalize score to [0, 1] (approximate)
            max_score = 10.0  # Approximate max BM25 score
            normalized_score = min(1.0, score / max_score)

            # Determine relevance
            is_relevant = normalized_score >= self.threshold

            # Calculate confidence
            confidence = 1.0 - abs(normalized_score - self.threshold) / self.threshold
            confidence = max(0.0, min(1.0, confidence))

            return RelevanceResult(
                score=normalized_score,
                is_relevant=is_relevant,
                confidence=confidence,
                method="bm25",
                explanations=[
                    f"BM25 score: {score:.3f} (normalized: {normalized_score:.3f})",
                    f"Relevant: {is_relevant}",
                    f"Query tokens: {len(query_tokens)}"
                ],
                features={
                    "bm25_score": score,
                    "normalized_score": normalized_score,
                    "threshold": self.threshold
                },
                metadata={
                    "query_length": len(query),
                    "document_length": len(document),
                    "k1": self.k1,
                    "b": self.b
                }
            )

        except Exception as e:
            logger.warning(f"BM25 relevance scoring failed: {e}")
            return RelevanceResult(
                score=0.0,
                is_relevant=False,
                confidence=0.0,
                method="bm25",
                explanations=[f"Scoring failed: {str(e)}"]
            )


class CrossEncoderRelevanceScorer(RelevanceScorer):
    """
    Relevance scoring using cross-encoder models.
    More accurate but slower than bi-encoder approaches.
    """

    SUPPORTED_MODELS = {
        "ms-marco-MiniLM-L-6-v2": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "ms-marco-MiniLM-L-12-v2": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "ms-marco-bert-base-v2": "cross-encoder/ms-marco-bert-base-v2",
        "distilroberta-base-msmarco": "cross-encoder/distilroberta-base-msmarco",
    }

    def __init__(
        self,
        model_name: str = "ms-marco-MiniLM-L-6-v2",
        threshold: float = 0.5,
        device: str = "cpu",
        batch_size: int = 32
    ):
        """
        Initialize cross-encoder relevance scorer.

        Args:
            model_name: Cross-encoder model name
            threshold: Relevance threshold
            device: Device to run model on
            batch_size: Batch size for inference
        """
        super().__init__(threshold)

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")

        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        # Get model path
        model_path = self.SUPPORTED_MODELS.get(model_name, model_name)

        # Initialize model
        self.model = CrossEncoder(
            model_path,
            device=device,
            num_labels=1
        )

        logger.info(f"Initialized CrossEncoderRelevanceScorer with model={model_name}")

    def score(
        self,
        query: str,
        document: str,
        **kwargs
    ) -> RelevanceResult:
        """
        Score relevance using cross-encoder.

        Args:
            query: Query text
            document: Document text
            **kwargs: Additional arguments

        Returns:
            RelevanceResult object
        """
        if not query or not document:
            return RelevanceResult(
                score=0.0,
                is_relevant=False,
                confidence=0.0,
                method="cross_encoder",
                explanations=["Empty query or document"]
            )

        try:
            # Score pair
            score = self.model.predict([(query, document)])[0]

            # Normalize score to [0, 1] using sigmoid
            normalized_score = 1.0 / (1.0 + np.exp(-float(score)))

            # Determine relevance
            is_relevant = normalized_score >= self.threshold

            # Calculate confidence
            confidence = 1.0 - abs(normalized_score - self.threshold) / self.threshold
            confidence = max(0.0, min(1.0, confidence))

            return RelevanceResult(
                score=normalized_score,
                is_relevant=is_relevant,
                confidence=confidence,
                method="cross_encoder",
                explanations=[
                    f"Cross-encoder score: {score:.3f} (normalized: {normalized_score:.3f})",
                    f"Relevant: {is_relevant}"
                ],
                features={
                    "raw_score": score,
                    "normalized_score": normalized_score,
                    "threshold": self.threshold
                },
                metadata={
                    "model": self.model_name,
                    "query_length": len(query),
                    "document_length": len(document)
                }
            )

        except Exception as e:
            logger.warning(f"Cross-encoder scoring failed: {e}")
            return RelevanceResult(
                score=0.0,
                is_relevant=False,
                confidence=0.0,
                method="cross_encoder",
                explanations=[f"Scoring failed: {str(e)}"]
            )

    def batch_score(
        self,
        queries: List[str],
        documents: List[str],
        **kwargs
    ) -> List[RelevanceResult]:
        """
        Score relevance for multiple pairs efficiently.

        Args:
            queries: List of query texts
            documents: List of document texts
            **kwargs: Additional arguments

        Returns:
            List of RelevanceResult objects
        """
        if not queries or not documents:
            return []

        try:
            # Create pairs
            pairs = list(zip(queries, documents))

            # Predict scores
            scores = self.model.predict(pairs, batch_size=self.batch_size)

            # Process results
            results = []
            for i, score in enumerate(scores):
                normalized_score = 1.0 / (1.0 + np.exp(-float(score)))
                is_relevant = normalized_score >= self.threshold
                confidence = 1.0 - abs(normalized_score - self.threshold) / self.threshold
                confidence = max(0.0, min(1.0, confidence))

                results.append(RelevanceResult(
                    score=normalized_score,
                    is_relevant=is_relevant,
                    confidence=confidence,
                    method="cross_encoder",
                    explanations=[
                        f"Score: {score:.3f}",
                        f"Relevant: {is_relevant}"
                    ],
                    features={
                        "raw_score": score,
                        "normalized_score": normalized_score
                    },
                    metadata={
                        "model": self.model_name,
                        "query": queries[i][:100],
                        "document": documents[i][:100]
                    }
                ))

            return results

        except Exception as e:
            logger.warning(f"Batch cross-encoder scoring failed: {e}")
            return []


class HybridRelevanceScorer(RelevanceScorer):
    """
    Hybrid relevance scoring combining multiple methods.
    """

    def __init__(
        self,
        methods: List[str] = ["semantic", "tfidf", "bm25"],
        weights: Dict[str, float] = None,
        threshold: float = 0.5,
        **kwargs
    ):
        """
        Initialize hybrid relevance scorer.

        Args:
            methods: Methods to use ('semantic', 'tfidf', 'bm25', 'cross_encoder')
            weights: Weights for each method
            threshold: Relevance threshold
            **kwargs: Additional arguments for scorers
        """
        super().__init__(threshold)

        self.methods = methods

        # Set weights
        self.weights = weights or {
            "semantic": 0.4,
            "tfidf": 0.3,
            "bm25": 0.3
        }

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}

        # Initialize scorers
        self.scorers = {}

        if "semantic" in methods:
            self.scorers["semantic"] = SemanticRelevanceScorer(
                threshold=threshold,
                **kwargs
            )

        if "tfidf" in methods:
            self.scorers["tfidf"] = TFIDFRelevanceScorer(
                threshold=threshold,
                **kwargs
            )

        if "bm25" in methods:
            self.scorers["bm25"] = BM25RelevanceScorer(
                threshold=threshold,
                **kwargs
            )

        if "cross_encoder" in methods:
            self.scorers["cross_encoder"] = CrossEncoderRelevanceScorer(
                threshold=threshold,
                **kwargs
            )

        logger.info(f"Initialized HybridRelevanceScorer with methods={methods}")

    def score(
        self,
        query: str,
        document: str,
        **kwargs
    ) -> RelevanceResult:
        """
        Score relevance using multiple methods.

        Args:
            query: Query text
            document: Document text
            **kwargs: Additional arguments

        Returns:
            RelevanceResult object
        """
        if not query or not document:
            return RelevanceResult(
                score=0.0,
                is_relevant=False,
                confidence=0.0,
                method="hybrid",
                explanations=["Empty query or document"]
            )

        results = []
        all_explanations = []
        combined_score = 0.0
        total_weight = 0.0
        features = {}

        for method, scorer in self.scorers.items():
            try:
                result = scorer.score(query, document, **kwargs)
                weight = self.weights.get(method, 0.2)
                combined_score += result.score * weight
                total_weight += weight

                results.append(result)
                all_explanations.extend(result.explanations)
                features.update(result.features)

            except Exception as e:
                logger.warning(f"Method {method} failed: {e}")

        # Normalize combined score
        if total_weight > 0:
            combined_score = combined_score / total_weight
        else:
            combined_score = 0.0

        # Determine relevance
        is_relevant = combined_score >= self.threshold

        # Calculate confidence (based on method agreement)
        if results:
            method_scores = [r.score for r in results]
            confidence = 1.0 - np.std(method_scores) if len(method_scores) > 1 else 0.5
            confidence = max(0.0, min(1.0, confidence))
        else:
            confidence = 0.0

        return RelevanceResult(
            score=combined_score,
            is_relevant=is_relevant,
            confidence=confidence,
            method="hybrid",
            explanations=all_explanations[:5],
            features={
                "combined_score": combined_score,
                "method_scores": {r.method: r.score for r in results},
                **features
            },
            metadata={
                "methods": self.methods,
                "weights": self.weights,
                "num_methods": len(self.scorers)
            }
        )


class RelevanceEvaluator:
    """
    Complete relevance evaluation pipeline with retrieval metrics.
    """

    def __init__(
        self,
        scorer: Optional[RelevanceScorer] = None,
        scorer_type: str = "hybrid",
        threshold: float = 0.5,
        **kwargs
    ):
        """
        Initialize relevance evaluator.

        Args:
            scorer: Pre-initialized scorer (optional)
            scorer_type: Type of scorer to use
            threshold: Relevance threshold
            **kwargs: Additional arguments for scorer
        """
        if scorer:
            self.scorer = scorer
        else:
            self.scorer = self._create_scorer(scorer_type, threshold, **kwargs)

        self.threshold = threshold

    def _create_scorer(self, scorer_type: str, threshold: float, **kwargs) -> RelevanceScorer:
        """Create relevance scorer."""
        if scorer_type == "semantic":
            return SemanticRelevanceScorer(threshold=threshold, **kwargs)
        elif scorer_type == "tfidf":
            return TFIDFRelevanceScorer(threshold=threshold, **kwargs)
        elif scorer_type == "bm25":
            return BM25RelevanceScorer(threshold=threshold, **kwargs)
        elif scorer_type == "cross_encoder":
            return CrossEncoderRelevanceScorer(threshold=threshold, **kwargs)
        elif scorer_type == "hybrid":
            return HybridRelevanceScorer(threshold=threshold, **kwargs)
        else:
            raise ValueError(f"Unsupported scorer type: {scorer_type}")

    def evaluate_single(
        self,
        query: str,
        document: str,
        ground_truth: bool = None
    ) -> Dict[str, Any]:
        """
        Evaluate relevance for a single query-document pair.

        Args:
            query: Query text
            document: Document text
            ground_truth: Ground truth relevance label

        Returns:
            Evaluation results
        """
        result = self.scorer.score(query, document)

        eval_result = {
            "score": result.score,
            "is_relevant": result.is_relevant,
            "confidence": result.confidence,
            "method": result.method,
            "explanations": result.explanations,
            "features": result.features
        }

        if ground_truth is not None:
            eval_result["ground_truth"] = ground_truth
            eval_result["correct"] = result.is_relevant == ground_truth

        return eval_result

    def evaluate_retrieval(
        self,
        queries: List[str],
        retrieved_documents: List[List[str]],
        relevant_documents: List[List[str]],
        top_k: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance.

        Args:
            queries: List of query texts
            retrieved_documents: List of retrieved document texts for each query
            relevant_documents: List of relevant document texts for each query
            top_k: K values for metrics

        Returns:
            Dictionary of retrieval metrics
        """
        metrics = {}

        # Convert to sets for faster lookup
        relevant_sets = [set(docs) for docs in relevant_documents]

        for k in top_k:
            # Precision@K
            precisions = []
            recalls = []

            for i, retrieved in enumerate(retrieved_documents):
                retrieved_k = retrieved[:k]
                relevant_set = relevant_sets[i] if i < len(relevant_sets) else set()

                if not retrieved_k:
                    precisions.append(0.0)
                    recalls.append(0.0)
                    continue

                hits = sum(1 for doc in retrieved_k if doc in relevant_set)
                precision = hits / len(retrieved_k)
                recall = hits / len(relevant_set) if relevant_set else 0.0

                precisions.append(precision)
                recalls.append(recall)

            metrics[f"precision@{k}"] = np.mean(precisions)
            metrics[f"recall@{k}"] = np.mean(recalls)

        # MRR
        mrr = 0.0
        for i, retrieved in enumerate(retrieved_documents):
            relevant_set = relevant_sets[i] if i < len(relevant_sets) else set()
            for rank, doc in enumerate(retrieved, 1):
                if doc in relevant_set:
                    mrr += 1.0 / rank
                    break

        metrics["mrr"] = mrr / len(queries) if queries else 0.0

        # Mean Average Precision (MAP)
        map_score = 0.0
        for i, retrieved in enumerate(retrieved_documents):
            relevant_set = relevant_sets[i] if i < len(relevant_sets) else set()
            if not relevant_set:
                continue

            hits = 0
            precision_sum = 0.0
            for rank, doc in enumerate(retrieved, 1):
                if doc in relevant_set:
                    hits += 1
                    precision_sum += hits / rank

            map_score += precision_sum / len(relevant_set)

        metrics["map"] = map_score / len(queries) if queries else 0.0

        return metrics

    def evaluate_ranking(
        self,
        query: str,
        documents: List[str],
        ground_truth: List[bool] = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate ranking quality for a single query.

        Args:
            query: Query text
            documents: List of documents to rank
            ground_truth: Ground truth relevance labels
            top_k: K for metrics

        Returns:
            Evaluation results
        """
        # Score all documents
        scores = []
        for doc in documents:
            result = self.scorer.score(query, doc)
            scores.append(result)

        # Sort by score
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i].score, reverse=True)
        sorted_scores = [scores[i] for i in sorted_indices]
        sorted_docs = [documents[i] for i in sorted_indices]

        eval_result = {
            "scores": [s.score for s in sorted_scores],
            "is_relevant": [s.is_relevant for s in sorted_scores],
            "documents": sorted_docs[:top_k],
            "ranking": list(range(1, min(top_k + 1, len(sorted_docs) + 1)))
        }

        if ground_truth:
            sorted_truth = [ground_truth[i] for i in sorted_indices]

            # Compute NDCG@K
            ndcg = 0.0
            dcg = 0.0
            idcg = 0.0

            # Calculate DCG
            for i, rel in enumerate(sorted_truth[:top_k]):
                if rel:
                    dcg += 1.0 / math.log2(i + 2)

            # Calculate IDCG
            true_rels = sorted(ground_truth, reverse=True)
            for i, rel in enumerate(true_rels[:top_k]):
                if rel:
                    idcg += 1.0 / math.log2(i + 2)

            if idcg > 0:
                ndcg = dcg / idcg

            eval_result["ndcg"] = ndcg
            eval_result["ground_truth"] = sorted_truth[:top_k]

        return eval_result


# ============================================================
# Convenience Functions
# ============================================================

def evaluate_relevance(
    query: str,
    document: str,
    method: str = "hybrid",
    threshold: float = 0.5
) -> RelevanceResult:
    """
    Quick function to evaluate relevance.

    Args:
        query: Query text
        document: Document text
        method: Method to use ('semantic', 'tfidf', 'bm25', 'cross_encoder', 'hybrid')
        threshold: Relevance threshold

    Returns:
        RelevanceResult object
    """
    scorer = None

    if method == "hybrid":
        scorer = HybridRelevanceScorer(threshold=threshold)
    elif method == "semantic":
        scorer = SemanticRelevanceScorer(threshold=threshold)
    elif method == "tfidf":
        scorer = TFIDFRelevanceScorer(threshold=threshold)
    elif method == "bm25":
        scorer = BM25RelevanceScorer(threshold=threshold)
    elif method == "cross_encoder":
        scorer = CrossEncoderRelevanceScorer(threshold=threshold)
    else:
        raise ValueError(f"Unsupported method: {method}")

    return scorer.score(query, document)


def batch_evaluate_relevance(
    queries: List[str],
    documents: List[str],
    method: str = "hybrid",
    threshold: float = 0.5
) -> List[RelevanceResult]:
    """
    Batch evaluate relevance for multiple pairs.

    Args:
        queries: List of query texts
        documents: List of document texts
        method: Method to use
        threshold: Relevance threshold

    Returns:
        List of RelevanceResult objects
    """
    scorer = None

    if method == "hybrid":
        scorer = HybridRelevanceScorer(threshold=threshold)
    elif method == "semantic":
        scorer = SemanticRelevanceScorer(threshold=threshold)
    elif method == "tfidf":
        scorer = TFIDFRelevanceScorer(threshold=threshold)
    elif method == "bm25":
        scorer = BM25RelevanceScorer(threshold=threshold)
    elif method == "cross_encoder":
        scorer = CrossEncoderRelevanceScorer(threshold=threshold)
    else:
        raise ValueError(f"Unsupported method: {method}")

    return scorer.batch_score(queries, documents)


def aggregate_relevance_scores(
    results: List[RelevanceResult]
) -> Dict[str, float]:
    """
    Aggregate relevance scores from multiple evaluations.

    Args:
        results: List of RelevanceResult objects

    Returns:
        Dictionary with aggregated statistics
    """
    if not results:
        return {
            "mean_score": 0.0,
            "std_score": 0.0,
            "min_score": 0.0,
            "max_score": 0.0,
            "relevant_ratio": 0.0,
            "total": 0
        }

    scores = [r.score for r in results]
    is_relevant = [r.is_relevant for r in results]

    return {
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "min_score": np.min(scores),
        "max_score": np.max(scores),
        "relevant_ratio": np.mean(is_relevant),
        "total": len(results)
    }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Sample texts
    query = "What is machine learning?"

    relevant_doc = """
    Machine learning is a subset of artificial intelligence that enables systems to learn 
    and improve from experience without being explicitly programmed. It uses algorithms 
    to find patterns in data and make predictions.
    """

    irrelevant_doc = """
    The weather today is sunny with a high of 75 degrees. There is a 20% chance of rain 
    later in the evening. It's a perfect day for outdoor activities.
    """

    # Test different methods
    print("Testing Relevance Scoring...")
    print("=" * 60)

    # Semantic scoring
    print("\n1. Semantic Scoring:")
    result = evaluate_relevance(query, relevant_doc, method="semantic")
    print(f"  Relevant document score: {result.score:.3f} - {'Relevant' if result.is_relevant else 'Not Relevant'}")

    result = evaluate_relevance(query, irrelevant_doc, method="semantic")
    print(f"  Irrelevant document score: {result.score:.3f} - {'Relevant' if result.is_relevant else 'Not Relevant'}")

    # Hybrid scoring
    print("\n2. Hybrid Scoring:")
    result = evaluate_relevance(query, relevant_doc, method="hybrid")
    print(f"  Relevant document score: {result.score:.3f} - {'Relevant' if result.is_relevant else 'Not Relevant'}")
    print(f"  Features: {result.features}")

    result = evaluate_relevance(query, irrelevant_doc, method="hybrid")
    print(f"  Irrelevant document score: {result.score:.3f} - {'Relevant' if result.is_relevant else 'Not Relevant'}")

    # Batch scoring
    print("\n3. Batch Scoring:")
    queries = [query, query]
    documents = [relevant_doc, irrelevant_doc]
    results = batch_evaluate_relevance(queries, documents, method="hybrid")

    for i, result in enumerate(results):
        print(f"  Pair {i+1}: {result.score:.3f} - {'Relevant' if result.is_relevant else 'Not Relevant'}")

    # Aggregated statistics
    print("\n4. Aggregated Statistics:")
    stats = aggregate_relevance_scores(results)
    for key, value in stats.items():
        print(f"  {key}: {value:.3f}")
