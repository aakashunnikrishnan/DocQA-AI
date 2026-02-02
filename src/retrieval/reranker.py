"""
Reranking module for improving retrieval quality.
Supports multiple reranking strategies:
- Cross-encoder reranking (using sentence-transformers or custom models)
- MMR (Maximum Marginal Relevance) for diversity
- Reciprocal Rank Fusion for ensemble reranking
- Learning-to-rank with feature-based scoring
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import re
from collections import defaultdict

from src.retrieval.retriever import RetrievalResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Try importing sentence-transformers for cross-encoder
try:
    from sentence_transformers import CrossEncoder
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Cross-encoder reranking will not work.")

# Try importing transformers for zero-shot reranking
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed. Some reranking features will not work.")


class RerankStrategy(Enum):
    """Available reranking strategies."""
    CROSS_ENCODER = "cross_encoder"
    MMR = "mmr"
    RECIPROCAL_RANK = "reciprocal_rank"
    FEATURE_BASED = "feature_based"
    ENSEMBLE = "ensemble"
    NONE = "none"


@dataclass
class RerankResult:
    """Result of reranking operation."""
    original_index: int
    text: str
    score: float
    rerank_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "original_index": self.original_index,
            "text": self.text,
            "score": self.score,
            "rerank_score": self.rerank_score,
            "metadata": self.metadata,
            "features": self.features
        }


class CrossEncoderReranker:
    """
    Reranker using cross-encoder models for better relevance scoring.
    Cross-encoders process query-document pairs together for more accurate scoring.
    """

    SUPPORTED_MODELS = {
        "ms-marco-MiniLM-L-6-v2": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "ms-marco-MiniLM-L-12-v2": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "ms-marco-MiniLM-L-4-v2": "cross-encoder/ms-marco-MiniLM-L-4-v2",
        "ms-marco-bert-base-v2": "cross-encoder/ms-marco-bert-base-v2",
        "distilroberta-base-msmarco": "cross-encoder/distilroberta-base-msmarco",
        "miniLM-L6-H384-uncased": "cross-encoder/miniLM-L6-H384-uncased",
    }

    def __init__(
        self,
        model_name: str = "ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        max_length: int = 512,
        use_gpu: bool = False
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: Name of the cross-encoder model
            device: Device to run model on ('cpu' or 'cuda')
            batch_size: Batch size for inference
            max_length: Maximum sequence length
            use_gpu: Whether to use GPU if available
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for cross-encoder reranking. "
                "Install with: pip install sentence-transformers"
            )

        # Get model path
        model_path = self.SUPPORTED_MODELS.get(model_name, model_name)

        # Determine device
        if use_gpu:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        # Initialize model
        self.model = CrossEncoder(
            model_path,
            device=device,
            max_length=max_length,
            num_labels=1
        )

        logger.info(f"Initialized CrossEncoderReranker with model={model_name}, device={device}")

    def rerank(
        self,
        query: str,
        candidates: List[Union[RetrievalResult, Dict[str, Any]]],
        top_k: Optional[int] = None,
        normalize_scores: bool = True
    ) -> List[RerankResult]:
        """
        Rerank candidates using cross-encoder.

        Args:
            query: Query string
            candidates: List of candidate results
            top_k: Number of top candidates to return
            normalize_scores: Whether to normalize scores to [0, 1]

        Returns:
            List of RerankResult objects sorted by relevance
        """
        if not candidates:
            return []

        # Prepare query-document pairs
        pairs = []
        original_indices = []

        for idx, candidate in enumerate(candidates):
            # Extract text
            if isinstance(candidate, RetrievalResult):
                text = candidate.text
                metadata = candidate.metadata
                original_score = candidate.score
            elif isinstance(candidate, dict):
                text = candidate.get("text", "")
                metadata = candidate.get("metadata", {})
                original_score = candidate.get("score", 0.0)
            else:
                logger.warning(f"Unknown candidate type: {type(candidate)}")
                continue

            pairs.append((query, text))
            original_indices.append(idx)

        if not pairs:
            return []

        # Get cross-encoder scores
        try:
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False
            )

            # Handle single score or list
            if isinstance(scores, (int, float)):
                scores = [scores]
            elif isinstance(scores, np.ndarray):
                scores = scores.tolist()

        except Exception as e:
            logger.error(f"Cross-encoder prediction failed: {e}")
            # Fallback to original scores
            scores = [0.0] * len(pairs)

        # Create rerank results
        results = []
        for idx, (score, pair_idx) in enumerate(zip(scores, original_indices)):
            candidate = candidates[pair_idx]

            if isinstance(candidate, RetrievalResult):
                text = candidate.text
                metadata = candidate.metadata
                original_score = candidate.score
            else:
                text = candidate.get("text", "")
                metadata = candidate.get("metadata", {})
                original_score = candidate.get("score", 0.0)

            # Normalize score if needed
            if normalize_scores:
                # Cross-encoder scores are typically logits, normalize with sigmoid
                score = 1.0 / (1.0 + np.exp(-float(score))) if score is not None else 0.0
            else:
                score = float(score) if score is not None else 0.0

            results.append(RerankResult(
                original_index=pair_idx,
                text=text,
                score=original_score,
                rerank_score=score,
                metadata=metadata
            ))

        # Sort by rerank score
        results.sort(key=lambda x: x.rerank_score, reverse=True)

        # Return top_k if specified
        if top_k and top_k < len(results):
            results = results[:top_k]

        return results


class MMRReranker:
    """
    Maximum Marginal Relevance reranker for improving diversity.
    Balances relevance and diversity in retrieval results.
    """

    def __init__(
        self,
        lambda_param: float = 0.5,
        diversity_weight: float = 0.3,
        use_embeddings: bool = True,
        embedding_generator: Optional[Any] = None
    ):
        """
        Initialize MMR reranker.

        Args:
            lambda_param: Trade-off between relevance and diversity (0 = max diversity, 1 = max relevance)
            diversity_weight: Weight for diversity in scoring
            use_embeddings: Whether to use embeddings for diversity calculation
            embedding_generator: Embedding generator for computing diversity
        """
        self.lambda_param = lambda_param
        self.diversity_weight = diversity_weight
        self.use_embeddings = use_embeddings
        self.embedding_generator = embedding_generator

        logger.info(f"Initialized MMRReranker with lambda={lambda_param}")

    def _compute_similarity_matrix(
        self,
        texts: List[str],
        embeddings: Optional[List[List[float]]] = None
    ) -> np.ndarray:
        """
        Compute similarity matrix between texts.

        Args:
            texts: List of texts
            embeddings: Optional pre-computed embeddings

        Returns:
            Similarity matrix
        """
        if self.use_embeddings and embeddings:
            # Use provided embeddings
            emb_array = np.array(embeddings)
            # Compute cosine similarity
            norm_emb = emb_array / np.linalg.norm(emb_array, axis=1, keepdims=True)
            similarity = np.dot(norm_emb, norm_emb.T)
            return similarity

        # Fallback: use TF-IDF
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf = vectorizer.fit_transform(texts)
            similarity = (tfidf * tfidf.T).toarray()
            return similarity
        except ImportError:
            # Fallback: use character overlap
            similarity = np.zeros((len(texts), len(texts)))
            for i, text1 in enumerate(texts):
                for j, text2 in enumerate(texts):
                    if i == j:
                        similarity[i, j] = 1.0
                    else:
                        # Jaccard similarity of character trigrams
                        words1 = set(text1.lower().split())
                        words2 = set(text2.lower().split())
                        if words1 or words2:
                            intersection = len(words1 & words2)
                            union = len(words1 | words2)
                            similarity[i, j] = intersection / union if union > 0 else 0.0
            return similarity

    def rerank(
        self,
        query: str,
        candidates: List[Union[RetrievalResult, Dict[str, Any]]],
        top_k: Optional[int] = None,
        embeddings: Optional[List[List[float]]] = None
    ) -> List[RerankResult]:
        """
        Rerank candidates using MMR.

        Args:
            query: Query string
            candidates: List of candidate results
            top_k: Number of top candidates to return
            embeddings: Pre-computed embeddings for candidates

        Returns:
            List of RerankResult objects sorted by MMR score
        """
        if not candidates:
            return []

        # Extract data
        texts = []
        scores = []
        metadata_list = []
        original_indices = []
        candidate_embeddings = []

        for idx, candidate in enumerate(candidates):
            if isinstance(candidate, RetrievalResult):
                texts.append(candidate.text)
                scores.append(candidate.score)
                metadata_list.append(candidate.metadata)
            elif isinstance(candidate, dict):
                texts.append(candidate.get("text", ""))
                scores.append(candidate.get("score", 0.0))
                metadata_list.append(candidate.get("metadata", {}))
            else:
                continue

            original_indices.append(idx)

            # Get embedding if available
            if embeddings and idx < len(embeddings):
                candidate_embeddings.append(embeddings[idx])

        if not texts:
            return []

        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(texts, candidate_embeddings)

        # MMR algorithm
        n = len(texts)
        selected = []
        selected_indices = set()

        # Normalize scores to [0, 1]
        if scores:
            min_score = min(scores)
            max_score = max(scores)
            if max_score > min_score:
                normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]
            else:
                normalized_scores = [1.0] * n
        else:
            normalized_scores = [1.0] * n

        # MMR selection
        k = top_k or len(candidates)
        k = min(k, n)

        for _ in range(k):
            best_idx = -1
            best_score = -float('inf')

            for i in range(n):
                if i in selected_indices:
                    continue

                # Relevance score
                relevance = normalized_scores[i]

                # Diversity penalty (max similarity to selected)
                if selected_indices:
                    max_similarity = max(similarity_matrix[i, j] for j in selected_indices)
                else:
                    max_similarity = 0.0

                # MMR score
                mmr_score = self.lambda_param * relevance - (1 - self.lambda_param) * max_similarity

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            if best_idx != -1:
                selected.append(best_idx)
                selected_indices.add(best_idx)

        # Create results
        results = []
        for idx in selected:
            result = RerankResult(
                original_index=original_indices[idx],
                text=texts[idx],
                score=scores[idx],
                rerank_score=normalized_scores[idx],
                metadata=metadata_list[idx]
            )
            results.append(result)

        return results


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion for combining multiple ranking lists.
    """

    def __init__(self, k: int = 60):
        """
        Initialize RRF.

        Args:
            k: Parameter for reciprocal rank (higher = more weight to top positions)
        """
        self.k = k
        logger.info(f"Initialized ReciprocalRankFusion with k={k}")

    def rerank(
        self,
        ranking_lists: List[List[Union[RetrievalResult, Dict[str, Any]]]],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Combine multiple ranking lists using RRF.

        Args:
            ranking_lists: List of ranking lists to combine
            top_k: Number of top results to return

        Returns:
            Combined ranked results
        """
        if not ranking_lists:
            return []

        # Collect all unique items
        item_scores = defaultdict(float)
        item_data = {}

        for rank_list in ranking_lists:
            for rank, item in enumerate(rank_list):
                # Extract item key
                if isinstance(item, RetrievalResult):
                    key = item.chunk_id or item.text[:100]
                    text = item.text
                    score = item.score
                    metadata = item.metadata
                    chunk_id = item.chunk_id
                elif isinstance(item, dict):
                    key = item.get("chunk_id", item.get("text", "")[:100])
                    text = item.get("text", "")
                    score = item.get("score", 0.0)
                    metadata = item.get("metadata", {})
                    chunk_id = item.get("chunk_id", "")
                else:
                    continue

                # Compute reciprocal rank score
                rrf_score = 1.0 / (self.k + rank + 1)
                item_scores[key] += rrf_score

                # Store item data if not already stored
                if key not in item_data:
                    item_data[key] = {
                        "text": text,
                        "score": score,
                        "metadata": metadata,
                        "chunk_id": chunk_id
                    }

        # Sort by combined score
        sorted_items = sorted(
            item_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Create results
        results = []
        for key, rrf_score in sorted_items[:top_k] if top_k else sorted_items:
            data = item_data[key]
            results.append(RerankResult(
                original_index=-1,
                text=data["text"],
                score=data["score"],
                rerank_score=rrf_score,
                metadata=data["metadata"]
            ))

        return results


class FeatureBasedReranker:
    """
    Feature-based reranker that combines multiple features for scoring.
    """

    def __init__(
        self,
        feature_weights: Optional[Dict[str, float]] = None,
        normalize_features: bool = True
    ):
        """
        Initialize feature-based reranker.

        Args:
            feature_weights: Weights for different features
            normalize_features: Whether to normalize features
        """
        self.feature_weights = feature_weights or {
            "relevance_score": 0.4,
            "text_length": 0.1,
            "query_overlap": 0.2,
            "position_bonus": 0.1,
            "keyword_coverage": 0.2
        }
        self.normalize_features = normalize_features

        # Normalize weights
        total = sum(self.feature_weights.values())
        if total > 0:
            self.feature_weights = {k: v/total for k, v in self.feature_weights.items()}

        logger.info(f"Initialized FeatureBasedReranker with weights={self.feature_weights}")

    def _compute_features(
        self,
        query: str,
        candidate: Union[RetrievalResult, Dict[str, Any]],
        position: int
    ) -> Dict[str, float]:
        """
        Compute features for a candidate.

        Args:
            query: Query string
            candidate: Candidate result
            position: Position in original ranking

        Returns:
            Feature dictionary
        """
        if isinstance(candidate, RetrievalResult):
            text = candidate.text
            score = candidate.score
            metadata = candidate.metadata
        else:
            text = candidate.get("text", "")
            score = candidate.get("score", 0.0)
            metadata = candidate.get("metadata", {})

        features = {}

        # 1. Relevance score (from retriever)
        features["relevance_score"] = score

        # 2. Text length (normalized)
        features["text_length"] = min(1.0, len(text) / 1000.0)

        # 3. Query overlap (word overlap percentage)
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        overlap = len(query_words & text_words)
        features["query_overlap"] = overlap / max(len(query_words), 1)

        # 4. Position bonus (higher for earlier positions)
        features["position_bonus"] = 1.0 / (1.0 + position * 0.1)

        # 5. Keyword coverage (does it contain important words?)
        important_words = ["important", "key", "critical", "main", "significant"]
        if query_words:
            important_matches = sum(1 for w in important_words if w in text.lower())
            features["keyword_coverage"] = min(1.0, important_matches / 3.0)
        else:
            features["keyword_coverage"] = 0.0

        return features

    def rerank(
        self,
        query: str,
        candidates: List[Union[RetrievalResult, Dict[str, Any]]],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank using feature-based scoring.

        Args:
            query: Query string
            candidates: List of candidate results
            top_k: Number of top results to return

        Returns:
            List of RerankResult objects
        """
        if not candidates:
            return []

        # Compute features for each candidate
        results = []
        all_features = []

        for idx, candidate in enumerate(candidates):
            features = self._compute_features(query, candidate, idx)
            all_features.append(features)

            if isinstance(candidate, RetrievalResult):
                text = candidate.text
                score = candidate.score
                metadata = candidate.metadata
            else:
                text = candidate.get("text", "")
                score = candidate.get("score", 0.0)
                metadata = candidate.get("metadata", {})

            # Compute weighted score
            weighted_score = 0.0
            for feat_name, weight in self.feature_weights.items():
                if feat_name in features:
                    weighted_score += weight * features[feat_name]

            results.append(RerankResult(
                original_index=idx,
                text=text,
                score=score,
                rerank_score=weighted_score,
                metadata=metadata,
                features=features
            ))

        # Sort by rerank score
        results.sort(key=lambda x: x.rerank_score, reverse=True)

        # Return top_k if specified
        if top_k and top_k < len(results):
            results = results[:top_k]

        return results


class EnsembleReranker:
    """
    Ensemble reranker that combines multiple reranking strategies.
    """

    def __init__(
        self,
        rerankers: List[Any],
        weights: Optional[List[float]] = None,
        method: str = "weighted"
    ):
        """
        Initialize ensemble reranker.

        Args:
            rerankers: List of reranker instances
            weights: Weights for each reranker
            method: Combination method ('weighted', 'reciprocal_rank', 'max')
        """
        self.rerankers = rerankers
        self.method = method

        # Set weights
        if weights:
            self.weights = weights
        else:
            self.weights = [1.0 / len(rerankers)] * len(rerankers)

        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

        logger.info(f"Initialized EnsembleReranker with {len(rerankers)} rerankers, method={method}")

    def rerank(
        self,
        query: str,
        candidates: List[Union[RetrievalResult, Dict[str, Any]]],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Rerank using ensemble of rerankers.

        Args:
            query: Query string
            candidates: List of candidate results
            top_k: Number of top results to return

        Returns:
            List of RerankResult objects
        """
        if not candidates:
            return []

        if self.method == "reciprocal_rank":
            # Use RRF to combine rankings
            ranking_lists = []
            for reranker in self.rerankers:
                results = reranker.rerank(query, candidates, top_k=None)
                ranking_lists.append(results)

            fusion = ReciprocalRankFusion()
            combined = fusion.rerank(ranking_lists, top_k)
            return combined

        elif self.method == "max":
            # Take max score from all rerankers
            all_results = []
            for reranker in self.rerankers:
                results = reranker.rerank(query, candidates, top_k=None)
                all_results.extend(results)

            # Group by original index and take max score
            best_scores = {}
            best_results = {}
            for result in all_results:
                idx = result.original_index
                if idx not in best_scores or result.rerank_score > best_scores[idx]:
                    best_scores[idx] = result.rerank_score
                    best_results[idx] = result

            results = list(best_results.values())
            results.sort(key=lambda x: x.rerank_score, reverse=True)

            if top_k and top_k < len(results):
                results = results[:top_k]

            return results

        else:
            # Default: weighted combination
            # Run each reranker
            all_results = []
            for reranker, weight in zip(self.rerankers, self.weights):
                results = reranker.rerank(query, candidates, top_k=None)
                for result in results:
                    result.rerank_score *= weight
                all_results.extend(results)

            # Group by original index and sum scores
            combined_scores = defaultdict(float)
            combined_results = {}
            for result in all_results:
                idx = result.original_index
                combined_scores[idx] += result.rerank_score
                if idx not in combined_results:
                    combined_results[idx] = result

            # Update scores
            for idx, score in combined_scores.items():
                if idx in combined_results:
                    combined_results[idx].rerank_score = score

            results = list(combined_results.values())
            results.sort(key=lambda x: x.rerank_score, reverse=True)

            if top_k and top_k < len(results):
                results = results[:top_k]

            return results


class RerankerPipeline:
    """
    Complete reranking pipeline with multiple stages.
    """

    def __init__(
        self,
        primary_strategy: RerankStrategy = RerankStrategy.CROSS_ENCODER,
        enable_mmr: bool = True,
        enable_ensemble: bool = False,
        lambda_param: float = 0.5,
        diversity_weight: float = 0.3,
        model_name: str = "ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        batch_size: int = 32
    ):
        """
        Initialize reranker pipeline.

        Args:
            primary_strategy: Primary reranking strategy
            enable_mmr: Whether to apply MMR for diversity
            enable_ensemble: Whether to use ensemble reranking
            lambda_param: MMR lambda parameter
            diversity_weight: MMR diversity weight
            model_name: Cross-encoder model name
            device: Device for cross-encoder
            batch_size: Batch size for cross-encoder
        """
        self.primary_strategy = primary_strategy
        self.enable_mmr = enable_mmr
        self.enable_ensemble = enable_ensemble
        self.lambda_param = lambda_param
        self.diversity_weight = diversity_weight

        # Initialize rerankers
        self.rerankers = []

        if primary_strategy == RerankStrategy.CROSS_ENCODER:
            try:
                cross_encoder = CrossEncoderReranker(
                    model_name=model_name,
                    device=device,
                    batch_size=batch_size
                )
                self.rerankers.append(("cross_encoder", cross_encoder))
            except Exception as e:
                logger.warning(f"Failed to initialize cross-encoder: {e}")

        if primary_strategy == RerankStrategy.MMR or enable_mmr:
            mmr_reranker = MMRReranker(
                lambda_param=lambda_param,
                diversity_weight=diversity_weight
            )
            self.rerankers.append(("mmr", mmr_reranker))

        if primary_strategy == RerankStrategy.FEATURE_BASED:
            feature_reranker = FeatureBasedReranker()
            self.rerankers.append(("feature", feature_reranker))

        if enable_ensemble and len(self.rerankers) > 1:
            ensemble = EnsembleReranker(
                [r for _, r in self.rerankers],
                method="weighted"
            )
            self.rerankers = [("ensemble", ensemble)]

        logger.info(f"Initialized RerankerPipeline with {len(self.rerankers)} rerankers")

    def rerank(
        self,
        query: str,
        candidates: List[Union[RetrievalResult, Dict[str, Any]]],
        top_k: Optional[int] = None
    ) -> List[RerankResult]:
        """
        Run reranking pipeline.

        Args:
            query: Query string
            candidates: List of candidate results
            top_k: Number of top results to return

        Returns:
            List of RerankResult objects
        """
        if not candidates or not self.rerankers:
            # Return original candidates if no rerankers
            results = []
            for idx, candidate in enumerate(candidates):
                if isinstance(candidate, RetrievalResult):
                    results.append(RerankResult(
                        original_index=idx,
                        text=candidate.text,
                        score=candidate.score,
                        rerank_score=candidate.score,
                        metadata=candidate.metadata
                    ))
                elif isinstance(candidate, dict):
                    results.append(RerankResult(
                        original_index=idx,
                        text=candidate.get("text", ""),
                        score=candidate.get("score", 0.0),
                        rerank_score=candidate.get("score", 0.0),
                        metadata=candidate.get("metadata", {})
                    ))
            return results

        # Apply first reranker
        name, reranker = self.rerankers[0]
        results = reranker.rerank(query, candidates, top_k=None)

        # Apply additional rerankers
        for name, reranker in self.rerankers[1:]:
            # Convert results back to dict format for next reranker
            candidate_dicts = [
                {
                    "text": r.text,
                    "score": r.rerank_score,
                    "metadata": r.metadata
                }
                for r in results
            ]
            results = reranker.rerank(query, candidate_dicts, top_k=None)

        # Apply MMR if enabled and not already applied
        if self.enable_mmr and self.primary_strategy != RerankStrategy.MMR:
            mmr = MMRReranker(
                lambda_param=self.lambda_param,
                diversity_weight=self.diversity_weight
            )
            candidate_dicts = [
                {
                    "text": r.text,
                    "score": r.rerank_score,
                    "metadata": r.metadata
                }
                for r in results
            ]
            results = mmr.rerank(query, candidate_dicts, top_k=None)

        # Return top_k
        if top_k and top_k < len(results):
            results = results[:top_k]

        return results


# Convenience function
def create_reranker(
    strategy: Union[str, RerankStrategy] = "cross_encoder",
    **kwargs
) -> RerankerPipeline:
    """
    Create a reranker pipeline.

    Args:
        strategy: Reranking strategy
        **kwargs: Additional arguments for reranker

    Returns:
        RerankerPipeline instance
    """
    if isinstance(strategy, str):
        strategy = RerankStrategy(strategy)

    return RerankerPipeline(
        primary_strategy=strategy,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create sample candidates
    candidates = [
        RetrievalResult(
            text="Machine learning is a subset of artificial intelligence.",
            score=0.85,
            metadata={"source": "doc1"}
        ),
        RetrievalResult(
            text="Deep learning uses neural networks for complex pattern recognition.",
            score=0.78,
            metadata={"source": "doc2"}
        ),
        RetrievalResult(
            text="Artificial intelligence encompasses machine learning, deep learning, and more.",
            score=0.75,
            metadata={"source": "doc3"}
        ),
        RetrievalResult(
            text="Neural networks are inspired by biological brains.",
            score=0.65,
            metadata={"source": "doc4"}
        ),
        RetrievalResult(
            text="Natural language processing deals with text and language understanding.",
            score=0.60,
            metadata={"source": "doc5"}
        )
    ]

    query = "What is machine learning and AI?"

    # Test cross-encoder reranking
    print("Testing Cross-Encoder Reranker...")
    reranker = CrossEncoderReranker(model_name="ms-marco-MiniLM-L-6-v2")
    results = reranker.rerank(query, candidates, top_k=3)

    print("\nReranked Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result.rerank_score:.4f} - {result.text[:50]}...")

    # Test MMR reranking
    print("\nTesting MMR Reranker...")
    mmr = MMRReranker(lambda_param=0.7)
    results = mmr.rerank(query, candidates, top_k=3)

    print("\nMMR Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result.rerank_score:.4f} - {result.text[:50]}...")

    # Test complete pipeline
    print("\nTesting Reranker Pipeline...")
    pipeline = RerankerPipeline(
        primary_strategy=RerankStrategy.CROSS_ENCODER,
        enable_mmr=True,
        lambda_param=0.6
    )
    results = pipeline.rerank(query, candidates, top_k=3)

    print("\nPipeline Results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result.rerank_score:.4f} - {result.text[:50]}...")
