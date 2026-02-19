"""
Hybrid search module combining keyword-based (BM25) and vector similarity search.
Provides configurable weighting, normalization, and fusion strategies for optimal retrieval.
"""

import re
import math
import logging
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np

from src.retrieval.vector_store import FAISSVectorStore, SearchResult
from src.retrieval.reranker import RerankerPipeline, RerankResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Try importing BM25
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank-bm25 not installed. Install with: pip install rank-bm25")


class FusionStrategy(Enum):
    """Fusion strategies for combining search results."""
    RECIPROCAL_RANK = "reciprocal_rank"
    SCORE_WEIGHTED = "score_weighted"
    MAX = "max"
    MIN = "min"
    AVERAGE = "average"
    CONCAVE = "concave"
    LINEAR = "linear"


class NormalizationMethod(Enum):
    """Normalization methods for scores."""
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    SIGMOID = "sigmoid"
    RANK = "rank"
    SOFTMAX = "softmax"


@dataclass
class HybridSearchResult:
    """Result from hybrid search."""
    text: str
    vector_score: float
    keyword_score: float
    combined_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""
    index: int = -1
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "vector_score": self.vector_score,
            "keyword_score": self.keyword_score,
            "combined_score": self.combined_score,
            "metadata": self.metadata,
            "chunk_id": self.chunk_id,
            "index": self.index,
            "rank": self.rank
        }


class BM25Index:
    """
    BM25 index for keyword-based search.
    Provides efficient keyword retrieval with configurable parameters.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        language: str = "en",
        stopwords: Optional[List[str]] = None,
        stem: bool = False
    ):
        """
        Initialize BM25 index.

        Args:
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (document length normalization)
            language: Language for tokenization
            stopwords: Custom stopwords list
            stem: Whether to apply stemming
        """
        if not BM25_AVAILABLE:
            raise ImportError("rank-bm25 not installed. Install with: pip install rank-bm25")

        self.k1 = k1
        self.b = b
        self.language = language
        self.stem = stem

        # Default stopwords
        self.stopwords = set(stopwords or [
            'a', 'an', 'the', 'of', 'to', 'for', 'with', 'on', 'at', 'from',
            'by', 'in', 'as', 'is', 'was', 'were', 'are', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must'
        ])

        self.corpus: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._is_built = False

        # Stats
        self.stats = {
            "num_documents": 0,
            "avg_doc_length": 0,
            "total_terms": 0
        }

        logger.info(f"BM25Index initialized: k1={k1}, b={b}, stem={stem}")

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation and special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

        # Split into tokens
        tokens = text.split()

        # Remove stopwords
        tokens = [t for t in tokens if t not in self.stopwords]

        # Apply stemming if enabled
        if self.stem:
            try:
                from nltk.stem import PorterStemmer
                stemmer = PorterStemmer()
                tokens = [stemmer.stem(t) for t in tokens]
            except ImportError:
                logger.warning("nltk not installed for stemming")

        return tokens

    def build_index(self, corpus: List[str]):
        """
        Build BM25 index from corpus.

        Args:
            corpus: List of document texts
        """
        if not corpus:
            logger.warning("Empty corpus provided")
            return

        self.corpus = corpus
        self.tokenized_corpus = [self._tokenize(doc) for doc in corpus]

        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b
        )

        self._is_built = True

        # Update stats
        self.stats["num_documents"] = len(corpus)
        self.stats["avg_doc_length"] = sum(len(tokens) for tokens in self.tokenized_corpus) / len(corpus)
        self.stats["total_terms"] = sum(len(tokens) for tokens in self.tokenized_corpus)

        logger.info(f"BM25 index built: {len(corpus)} documents, "
                   f"avg_doc_length={self.stats['avg_doc_length']:.2f}")

    def search(
        self,
        query: str,
        top_k: int = 10,
        return_scores: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Search using BM25.

        Args:
            query: Query string
            top_k: Number of results to return
            return_scores: Whether to return scores

        Returns:
            List of (document_index, score) tuples
        """
        if not self._is_built:
            logger.warning("BM25 index not built")
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Get top-k results
        if top_k >= len(scores):
            indices = list(range(len(scores)))
            scores_list = scores.tolist()
        else:
            # Get top-k indices
            indices = np.argsort(scores)[-top_k:][::-1]
            scores_list = scores[indices].tolist()

        if return_scores:
            return [(idx, scores_list[i]) for i, idx in enumerate(indices)]
        else:
            return [(idx, 0.0) for idx in indices]

    def get_document(self, index: int) -> Optional[str]:
        """Get document by index."""
        if 0 <= index < len(self.corpus):
            return self.corpus[index]
        return None

    def add_documents(self, documents: List[str]):
        """
        Add documents to the index (rebuilds index).

        Args:
            documents: List of new documents
        """
        if not documents:
            return

        # Extend corpus
        self.corpus.extend(documents)

        # Rebuild index
        self.build_index(self.corpus)

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            **self.stats,
            "is_built": self._is_built,
            "k1": self.k1,
            "b": self.b,
            "stem": self.stem
        }


class ScoreNormalizer:
    """
    Normalize scores from different sources for fair combination.
    """

    @staticmethod
    def normalize(
        scores: List[float],
        method: NormalizationMethod = NormalizationMethod.MIN_MAX,
        **kwargs
    ) -> List[float]:
        """
        Normalize scores using specified method.

        Args:
            scores: List of scores to normalize
            method: Normalization method
            **kwargs: Additional parameters

        Returns:
            Normalized scores (0-1 range)
        """
        if not scores:
            return []

        if method == NormalizationMethod.MIN_MAX:
            return ScoreNormalizer._min_max(scores)
        elif method == NormalizationMethod.Z_SCORE:
            return ScoreNormalizer._z_score(scores)
        elif method == NormalizationMethod.SIGMOID:
            return ScoreNormalizer._sigmoid(scores)
        elif method == NormalizationMethod.RANK:
            return ScoreNormalizer._rank(scores)
        elif method == NormalizationMethod.SOFTMAX:
            return ScoreNormalizer._softmax(scores)
        else:
            return ScoreNormalizer._min_max(scores)

    @staticmethod
    def _min_max(scores: List[float]) -> List[float]:
        """Min-Max normalization."""
        if len(scores) == 0:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]

    @staticmethod
    def _z_score(scores: List[float]) -> List[float]:
        """Z-score normalization."""
        if len(scores) < 2:
            return [0.5] * len(scores)
        mean = sum(scores) / len(scores)
        std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
        if std == 0:
            return [0.5] * len(scores)

        # Normalize to [0, 1] range
        z_scores = [(s - mean) / std for s in scores]
        min_z = min(z_scores)
        max_z = max(z_scores)
        if max_z == min_z:
            return [0.5] * len(scores)
        return [(z - min_z) / (max_z - min_z) for z in z_scores]

    @staticmethod
    def _sigmoid(scores: List[float], k: float = 1.0) -> List[float]:
        """Sigmoid normalization."""
        return [1.0 / (1.0 + math.exp(-s * k)) for s in scores]

    @staticmethod
    def _rank(scores: List[float]) -> List[float]:
        """Rank-based normalization."""
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        ranks = [0] * len(scores)
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = 1.0 - (rank / len(scores))
        return ranks

    @staticmethod
    def _softmax(scores: List[float], temperature: float = 1.0) -> List[float]:
        """Softmax normalization."""
        if not scores:
            return []

        # Prevent overflow
        max_score = max(scores)
        exp_scores = [math.exp((s - max_score) / temperature) for s in scores]
        sum_exp = sum(exp_scores)
        if sum_exp == 0:
            return [1.0 / len(scores)] * len(scores)
        return [e / sum_exp for e in exp_scores]


class ScoreFuser:
    """
    Fuse scores from multiple sources using different strategies.
    """

    @staticmethod
    def fuse(
        scores_list: List[List[float]],
        strategy: FusionStrategy = FusionStrategy.RECIPROCAL_RANK,
        weights: Optional[List[float]] = None,
        **kwargs
    ) -> List[float]:
        """
        Fuse scores from multiple sources.

        Args:
            scores_list: List of score lists (each list corresponds to one source)
            strategy: Fusion strategy
            weights: Weights for each source
            **kwargs: Additional parameters

        Returns:
            Fused scores
        """
        if not scores_list:
            return []

        if weights is None:
            weights = [1.0 / len(scores_list)] * len(scores_list)

        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        if strategy == FusionStrategy.RECIPROCAL_RANK:
            return ScoreFuser._reciprocal_rank(scores_list, weights, **kwargs)
        elif strategy == FusionStrategy.SCORE_WEIGHTED:
            return ScoreFuser._score_weighted(scores_list, weights, **kwargs)
        elif strategy == FusionStrategy.MAX:
            return ScoreFuser._max(scores_list)
        elif strategy == FusionStrategy.MIN:
            return ScoreFuser._min(scores_list)
        elif strategy == FusionStrategy.AVERAGE:
            return ScoreFuser._average(scores_list, weights)
        elif strategy == FusionStrategy.CONCAVE:
            return ScoreFuser._concave(scores_list, weights)
        elif strategy == FusionStrategy.LINEAR:
            return ScoreFuser._linear(scores_list, weights)
        else:
            return ScoreFuser._reciprocal_rank(scores_list, weights, **kwargs)

    @staticmethod
    def _reciprocal_rank(
        scores_list: List[List[float]],
        weights: List[float],
        k: int = 60
    ) -> List[float]:
        """Reciprocal Rank Fusion."""
        # Convert scores to ranks
        all_items = set()
        for scores in scores_list:
            all_items.update(range(len(scores)))

        fused = defaultdict(float)

        for source_idx, scores in enumerate(scores_list):
            # Sort by score
            sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

            for rank, idx in enumerate(sorted_indices):
                rrf_score = weights[source_idx] / (k + rank + 1)
                fused[idx] += rrf_score

        # Convert to list
        max_idx = max(all_items) if all_items else 0
        result = [0.0] * (max_idx + 1)
        for idx, score in fused.items():
            result[idx] = score

        return result

    @staticmethod
    def _score_weighted(
        scores_list: List[List[float]],
        weights: List[float]
    ) -> List[float]:
        """Weighted sum of scores."""
        if len(scores_list) != len(weights):
            raise ValueError("Number of score lists must match number of weights")

        # Align all score lists to the same length
        max_len = max(len(scores) for scores in scores_list)

        aligned_scores = []
        for scores in scores_list:
            if len(scores) < max_len:
                # Pad with zeros
                aligned = scores + [0.0] * (max_len - len(scores))
            else:
                aligned = scores[:max_len]
            aligned_scores.append(aligned)

        # Weighted sum
        fused = [0.0] * max_len
        for i in range(max_len):
            for j, scores in enumerate(aligned_scores):
                if i < len(scores):
                    fused[i] += weights[j] * scores[i]

        return fused

    @staticmethod
    def _max(scores_list: List[List[float]]) -> List[float]:
        """Take maximum score for each position."""
        max_len = max(len(scores) for scores in scores_list)

        fused = [0.0] * max_len
        for i in range(max_len):
            max_score = 0.0
            for scores in scores_list:
                if i < len(scores):
                    max_score = max(max_score, scores[i])
            fused[i] = max_score

        return fused

    @staticmethod
    def _min(scores_list: List[List[float]]) -> List[float]:
        """Take minimum score for each position."""
        max_len = max(len(scores) for scores in scores_list)

        fused = [1.0] * max_len
        for i in range(max_len):
            min_score = 1.0
            for scores in scores_list:
                if i < len(scores):
                    min_score = min(min_score, scores[i])
            fused[i] = min_score

        return fused

    @staticmethod
    def _average(scores_list: List[List[float]], weights: List[float]) -> List[float]:
        """Weighted average of scores."""
        return ScoreFuser._score_weighted(scores_list, weights)

    @staticmethod
    def _concave(scores_list: List[List[float]], weights: List[float]) -> List[float]:
        """Concave fusion (emphasizes higher scores)."""
        weighted_scores = ScoreFuser._score_weighted(scores_list, weights)
        return [s ** 0.5 for s in weighted_scores]

    @staticmethod
    def _linear(scores_list: List[List[float]], weights: List[float]) -> List[float]:
        """Linear fusion (simple weighted sum)."""
        return ScoreFuser._score_weighted(scores_list, weights)


class HybridSearcher:
    """
    Hybrid search combining BM25 keyword search and vector similarity.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedding_generator: Any,
        fusion_strategy: FusionStrategy = FusionStrategy.RECIPROCAL_RANK,
        normalization_method: NormalizationMethod = NormalizationMethod.MIN_MAX,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        top_k: int = 10,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        use_stemming: bool = False,
        rerank_results: bool = True,
        reranker: Optional[RerankerPipeline] = None
    ):
        """
        Initialize hybrid searcher.

        Args:
            vector_store: FAISS vector store instance
            embedding_generator: Embedding generator for queries
            fusion_strategy: Strategy for fusing scores
            normalization_method: Method for normalizing scores
            vector_weight: Weight for vector scores
            keyword_weight: Weight for keyword scores
            top_k: Number of results to return
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
            use_stemming: Whether to use stemming for BM25
            rerank_results: Whether to rerank results
            reranker: Reranker pipeline instance
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.fusion_strategy = fusion_strategy
        self.normalization_method = normalization_method
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.top_k = top_k
        self.rerank_results = rerank_results
        self.reranker = reranker

        # Initialize BM25 index
        self.bm25_index = BM25Index(
            k1=bm25_k1,
            b=bm25_b,
            stem=use_stemming
        )

        # Whether BM25 is built
        self._bm25_built = False

        logger.info(f"HybridSearcher initialized: fusion={fusion_strategy.value}, "
                   f"vector_weight={vector_weight}, keyword_weight={keyword_weight}")

    def index_documents(self, texts: List[str], chunk_ids: Optional[List[str]] = None):
        """
        Index documents for keyword search.

        Args:
            texts: List of document texts
            chunk_ids: Optional chunk IDs
        """
        if not texts:
            return

        # Build BM25 index
        self.bm25_index.build_index(texts)
        self._bm25_built = True

        logger.info(f"Indexed {len(texts)} documents for keyword search")

    def sync_with_vector_store(self):
        """
        Sync BM25 index with vector store.
        """
        if not self.vector_store or not self.vector_store.texts:
            return

        texts = self.vector_store.texts
        self.index_documents(texts)

    def _get_vector_scores(
        self,
        query_embedding: List[float],
        top_k: int
    ) -> List[Tuple[int, float]]:
        """
        Get vector similarity scores.

        Args:
            query_embedding: Query embedding
            top_k: Number of results

        Returns:
            List of (index, score) tuples
        """
        if not self.vector_store or self.vector_store.get_size() == 0:
            return []

        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2  # Get more for fusion
        )

        return [(r.index, r.score) for r in results]

    def _get_keyword_scores(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[int, float]]:
        """
        Get BM25 keyword scores.

        Args:
            query: Query string
            top_k: Number of results

        Returns:
            List of (index, score) tuples
        """
        if not self._bm25_built:
            return []

        return self.bm25_index.search(query, top_k=top_k * 2)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        vector_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None,
        fusion_strategy: Optional[FusionStrategy] = None,
        normalize_scores: bool = True,
        filter_metadata: Optional[Dict[str, Any]] = None,
        rerank: Optional[bool] = None
    ) -> List[HybridSearchResult]:
        """
        Perform hybrid search combining vector and keyword search.

        Args:
            query: Query string
            top_k: Number of results to return
            vector_weight: Weight for vector scores
            keyword_weight: Weight for keyword scores
            fusion_strategy: Fusion strategy
            normalize_scores: Whether to normalize scores
            filter_metadata: Filter results by metadata
            rerank: Whether to rerank results

        Returns:
            List of HybridSearchResult objects
        """
        if top_k is None:
            top_k = self.top_k

        if vector_weight is None:
            vector_weight = self.vector_weight

        if keyword_weight is None:
            keyword_weight = self.keyword_weight

        if fusion_strategy is None:
            fusion_strategy = self.fusion_strategy

        if rerank is None:
            rerank = self.rerank_results

        # Get query embedding
        query_embedding = None
        try:
            embedding_result = self.embedding_generator.generate_embeddings([query])
            if embedding_result:
                query_embedding = embedding_result[0].embedding
        except Exception as e:
            logger.warning(f"Failed to generate query embedding: {e}")

        # Get scores from both sources
        vector_results = []
        keyword_results = []

        if query_embedding and self.vector_store:
            vector_results = self._get_vector_scores(query_embedding, top_k * 2)

        if self._bm25_built:
            keyword_results = self._get_keyword_scores(query, top_k * 2)

        # Combine results
        combined = self._combine_results(
            vector_results,
            keyword_results,
            vector_weight,
            keyword_weight,
            fusion_strategy,
            normalize_scores,
            filter_metadata
        )

        # Apply reranking if enabled
        if rerank and self.reranker:
            combined = self._apply_reranking(query, combined)

        # Convert to HybridSearchResult objects
        results = []
        for rank, (idx, vector_score, keyword_score, combined_score) in enumerate(combined):
            # Get text and metadata from vector store
            text = self.vector_store.texts[idx] if idx < len(self.vector_store.texts) else ""
            metadata = self.vector_store.metadata[idx] if idx < len(self.vector_store.metadata) else {}
            chunk_id = self.vector_store.chunk_ids[idx] if idx < len(self.vector_store.chunk_ids) else ""

            results.append(HybridSearchResult(
                text=text,
                vector_score=vector_score,
                keyword_score=keyword_score,
                combined_score=combined_score,
                metadata=metadata,
                chunk_id=chunk_id,
                index=idx,
                rank=rank + 1
            ))

        return results[:top_k]

    def _combine_results(
        self,
        vector_results: List[Tuple[int, float]],
        keyword_results: List[Tuple[int, float]],
        vector_weight: float,
        keyword_weight: float,
        fusion_strategy: FusionStrategy,
        normalize_scores: bool,
        filter_metadata: Optional[Dict[str, Any]]
    ) -> List[Tuple[int, float, float, float]]:
        """
        Combine vector and keyword results.

        Returns:
            List of (index, vector_score, keyword_score, combined_score)
        """
        # Create dictionaries for quick lookup
        vector_dict = {idx: score for idx, score in vector_results}
        keyword_dict = {idx: score for idx, score in keyword_results}

        # Get all unique indices
        all_indices = set(vector_dict.keys()) | set(keyword_dict.keys())

        if not all_indices:
            return []

        # Prepare score lists for normalization
        vector_scores = [vector_dict.get(idx, 0.0) for idx in all_indices]
        keyword_scores = [keyword_dict.get(idx, 0.0) for idx in all_indices]

        # Normalize scores if requested
        if normalize_scores:
            vector_scores = ScoreNormalizer.normalize(
                vector_scores,
                self.normalization_method
            )
            keyword_scores = ScoreNormalizer.normalize(
                keyword_scores,
                self.normalization_method
            )

        # Apply weights
        weighted_vector = [s * vector_weight for s in vector_scores]
        weighted_keyword = [s * keyword_weight for s in keyword_scores]

        # Fuse scores
        fused_scores = ScoreFuser.fuse(
            [weighted_vector, weighted_keyword],
            fusion_strategy,
            [vector_weight, keyword_weight]
        )

        # Combine results
        combined = []
        indices_list = list(all_indices)

        for i, idx in enumerate(indices_list):
            combined.append((
                idx,
                vector_scores[i] if i < len(vector_scores) else 0.0,
                keyword_scores[i] if i < len(keyword_scores) else 0.0,
                fused_scores[i] if i < len(fused_scores) else 0.0
            ))

        # Sort by combined score
        combined.sort(key=lambda x: x[3], reverse=True)

        # Apply metadata filter if provided
        if filter_metadata and self.vector_store:
            filtered = []
            for idx, v_score, k_score, c_score in combined:
                if idx < len(self.vector_store.metadata):
                    metadata = self.vector_store.metadata[idx]
                    if self._matches_filter(metadata, filter_metadata):
                        filtered.append((idx, v_score, k_score, c_score))
            combined = filtered

        return combined

    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    def _apply_reranking(
        self,
        query: str,
        combined: List[Tuple[int, float, float, float]]
    ) -> List[Tuple[int, float, float, float]]:
        """
        Apply reranking to combined results.

        Args:
            query: Query string
            combined: List of (index, vector_score, keyword_score, combined_score)

        Returns:
            Reranked results
        """
        if not combined or not self.reranker:
            return combined

        # Convert to candidate format
        candidates = []
        for idx, v_score, k_score, c_score in combined:
            if idx < len(self.vector_store.texts):
                candidates.append({
                    "text": self.vector_store.texts[idx],
                    "score": c_score,
                    "metadata": self.vector_store.metadata[idx] if idx < len(self.vector_store.metadata) else {},
                    "chunk_id": self.vector_store.chunk_ids[idx] if idx < len(self.vector_store.chunk_ids) else ""
                })

        # Rerank
        reranked = self.reranker.rerank(query, candidates, top_k=len(candidates))

        # Convert back to combined format
        reranked_combined = []
        for result in reranked:
            # Find the original index
            idx = -1
            for i, (orig_idx, _, _, _) in enumerate(combined):
                if orig_idx < len(self.vector_store.texts) and self.vector_store.texts[orig_idx] == result.text:
                    idx = orig_idx
                    break

            if idx != -1:
                # Get original scores
                orig_v_score = 0.0
                orig_k_score = 0.0
                for orig_idx, v_score, k_score, _ in combined:
                    if orig_idx == idx:
                        orig_v_score = v_score
                        orig_k_score = k_score
                        break

                reranked_combined.append((
                    idx,
                    orig_v_score,
                    orig_k_score,
                    result.rerank_score
                ))

        return reranked_combined

    def batch_search(
        self,
        queries: List[str],
        top_k: Optional[int] = None
    ) -> List[List[HybridSearchResult]]:
        """
        Perform hybrid search for multiple queries.

        Args:
            queries: List of query strings
            top_k: Number of results per query

        Returns:
            List of result lists
        """
        if top_k is None:
            top_k = self.top_k

        results = []
        for query in queries:
            results.append(self.search(query, top_k))

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get search statistics."""
        stats = {
            "fusion_strategy": self.fusion_strategy.value,
            "normalization_method": self.normalization_method.value,
            "vector_weight": self.vector_weight,
            "keyword_weight": self.keyword_weight,
            "top_k": self.top_k,
            "bm25_built": self._bm25_built,
        }

        if self._bm25_built:
            stats["bm25"] = self.bm25_index.get_stats()

        if self.vector_store:
            stats["vector_store"] = {
                "size": self.vector_store.get_size(),
                "dimension": self.vector_store.dimension
            }

        return stats


# Convenience function
def create_hybrid_searcher(
    vector_store: FAISSVectorStore,
    embedding_generator: Any,
    fusion_strategy: Union[str, FusionStrategy] = "reciprocal_rank",
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
    top_k: int = 10,
    rerank: bool = True,
    **kwargs
) -> HybridSearcher:
    """
    Create a hybrid searcher with default settings.

    Args:
        vector_store: FAISS vector store
        embedding_generator: Embedding generator
        fusion_strategy: Fusion strategy
        vector_weight: Weight for vector scores
        keyword_weight: Weight for keyword scores
        top_k: Number of results
        rerank: Whether to rerank results
        **kwargs: Additional arguments

    Returns:
        HybridSearcher instance
    """
    if isinstance(fusion_strategy, str):
        fusion_strategy = FusionStrategy(fusion_strategy)

    # Create reranker if needed
    reranker = None
    if rerank:
        from src.retrieval.reranker import RerankerPipeline, RerankStrategy
        reranker = RerankerPipeline(
            primary_strategy=RerankStrategy.CROSS_ENCODER,
            enable_mmr=True
        )

    return HybridSearcher(
        vector_store=vector_store,
        embedding_generator=embedding_generator,
        fusion_strategy=fusion_strategy,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
        top_k=top_k,
        rerank_results=rerank,
        reranker=reranker,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Sample documents
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing deals with text and language.",
        "Computer vision enables machines to understand images.",
        "Reinforcement learning involves agents learning through interaction.",
        "Python is a popular programming language for data science.",
        "TensorFlow and PyTorch are popular deep learning frameworks.",
        "Data science combines statistics and programming skills.",
        "AI applications include recommendation systems and autonomous vehicles.",
        "Neural networks are inspired by biological brains."
    ]

    # Create vector store (mock)
    class MockVectorStore:
        def __init__(self, texts):
            self.texts = texts
            self.metadata = [{} for _ in texts]
            self.chunk_ids = [f"chunk_{i}" for i in range(len(texts))]

        def get_size(self):
            return len(self.texts)

        def search(self, query_embedding, top_k):
            # Mock search - return random results
            import random
            indices = list(range(len(self.texts)))
            random.shuffle(indices)
            return [SearchResult(text=self.texts[i], score=random.random(), index=i) for i in indices[:top_k]]

    # Mock embedding generator
    class MockEmbeddingGenerator:
        def generate_embeddings(self, query):
            import random
            return [type('obj', (object,), {'embedding': [random.random() for _ in range(1536)]})()]

    # Create hybrid searcher
    vector_store = MockVectorStore(documents)
    embedding_generator = MockEmbeddingGenerator()

    searcher = create_hybrid_searcher(
        vector_store=vector_store,
        embedding_generator=embedding_generator,
        vector_weight=0.6,
        keyword_weight=0.4,
        top_k=5
    )

    # Index documents for BM25
    searcher.index_documents(documents)

    # Search
    query = "What is machine learning and AI?"
    results = searcher.search(query)

    print(f"\nHybrid Search Results for: '{query}'")
    print("=" * 60)
    for result in results:
        print(f"Rank {result.rank}: Combined={result.combined_score:.4f} "
              f"(Vector={result.vector_score:.4f}, Keyword={result.keyword_score:.4f})")
        print(f"  {result.text[:80]}...")
