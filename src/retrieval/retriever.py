"""
Basic retriever implementation for document retrieval using vector similarity and hybrid search.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import numpy as np

from src.retrieval.vector_store import FAISSVectorStore, FAISSHybridStore, SearchResult
from src.utils.logger import get_logger, log_function_call

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Represents a retrieval result with additional metadata."""
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""
    relevance_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "score": self.score,
            "relevance_score": self.relevance_score,
            "metadata": self.metadata,
            "chunk_id": self.chunk_id
        }


class BaseRetriever:
    """Base class for all retrievers."""

    def __init__(self, top_k: int = 5, score_threshold: Optional[float] = None):
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(self, query: str, **kwargs) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query."""
        raise NotImplementedError

    def batch_retrieve(self, queries: List[str], **kwargs) -> List[List[RetrievalResult]]:
        """Retrieve for multiple queries."""
        return [self.retrieve(query, **kwargs) for query in queries]


class VectorRetriever(BaseRetriever):
    """
    Vector-based retriever using embeddings and similarity search.
    """

    def __init__(
        self,
        vector_store: FAISSVectorStore,
        embedding_generator: Any,  # OpenAIEmbeddingGenerator
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        use_cache: bool = True
    ):
        """
        Initialize vector retriever.

        Args:
            vector_store: FAISS vector store instance
            embedding_generator: Embedding generator for queries
            top_k: Number of results to return
            score_threshold: Minimum similarity score threshold
            use_cache: Whether to cache query embeddings
        """
        super().__init__(top_k, score_threshold)
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.use_cache = use_cache
        self._embedding_cache: Dict[str, List[float]] = {}

        logger.info(f"Initialized VectorRetriever with top_k={top_k}")

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query with caching."""
        if self.use_cache and query in self._embedding_cache:
            logger.debug(f"Using cached embedding for query: {query[:50]}...")
            return self._embedding_cache[query]

        # Generate embedding
        result = self.embedding_generator.generate_embedding(query)
        embedding = result.embedding

        # Cache if enabled
        if self.use_cache:
            self._embedding_cache[query] = embedding

        return embedding

    @log_function_call(level="INFO")
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        return_scores: bool = True
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents using vector similarity.

        Args:
            query: Query text
            top_k: Override default top_k
            score_threshold: Override default threshold
            filter_metadata: Filter results by metadata
            return_scores: Include similarity scores

        Returns:
            List of RetrievalResult objects
        """
        # Get query embedding
        query_embedding = self._get_query_embedding(query)

        # Search vector store
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k or self.top_k,
            score_threshold=score_threshold or self.score_threshold,
            filter_metadata=filter_metadata
        )

        # Convert to RetrievalResult
        retrieval_results = []
        for result in results:
            retrieval_results.append(RetrievalResult(
                text=result.text,
                score=result.score,
                metadata=result.metadata,
                chunk_id=result.chunk_id,
                relevance_score=result.score
            ))

        logger.info(f"Retrieved {len(retrieval_results)} documents for query: {query[:50]}...")
        return retrieval_results

    def retrieve_with_embeddings(
        self,
        query_embedding: List[float],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve using pre-computed query embedding.

        Args:
            query_embedding: Pre-computed query embedding
            top_k: Number of results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of RetrievalResult objects
        """
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k or self.top_k,
            score_threshold=score_threshold or self.score_threshold
        )

        return [
            RetrievalResult(
                text=result.text,
                score=result.score,
                metadata=result.metadata,
                chunk_id=result.chunk_id,
                relevance_score=result.score
            )
            for result in results
        ]

    def clear_cache(self):
        """Clear query embedding cache."""
        self._embedding_cache.clear()
        logger.info("Cleared query embedding cache")


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining vector similarity and keyword search (BM25).
    """

    def __init__(
        self,
        vector_store: FAISSHybridStore,
        embedding_generator: Any,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        use_cache: bool = True
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_store: FAISS hybrid vector store
            embedding_generator: Embedding generator for queries
            top_k: Number of results to return
            score_threshold: Minimum combined score threshold
            vector_weight: Weight for vector similarity (0-1)
            keyword_weight: Weight for keyword relevance (0-1)
            use_cache: Whether to cache query embeddings
        """
        super().__init__(top_k, score_threshold)
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.use_cache = use_cache
        self._embedding_cache: Dict[str, List[float]] = {}

        # Validate weights
        total = vector_weight + keyword_weight
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total}, normalizing to 1.0")
            self.vector_weight = vector_weight / total
            self.keyword_weight = keyword_weight / total

        logger.info(f"Initialized HybridRetriever with top_k={top_k}, "
                   f"vector_weight={self.vector_weight}, keyword_weight={self.keyword_weight}")

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query with caching."""
        if self.use_cache and query in self._embedding_cache:
            return self._embedding_cache[query]

        result = self.embedding_generator.generate_embedding(query)
        embedding = result.embedding

        if self.use_cache:
            self._embedding_cache[query] = embedding

        return embedding

    @log_function_call(level="INFO")
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        vector_weight: Optional[float] = None,
        keyword_weight: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve documents using hybrid search.

        Args:
            query: Query text
            top_k: Override default top_k
            score_threshold: Override default threshold
            vector_weight: Override vector weight for this query
            keyword_weight: Override keyword weight for this query
            filter_metadata: Filter results by metadata

        Returns:
            List of RetrievalResult objects
        """
        # Get query embedding
        query_embedding = self._get_query_embedding(query)

        # Use provided weights or defaults
        v_weight = vector_weight if vector_weight is not None else self.vector_weight
        k_weight = keyword_weight if keyword_weight is not None else self.keyword_weight

        # Perform hybrid search
        results = self.vector_store.hybrid_search(
            query_embedding=query_embedding,
            query_text=query,
            top_k=top_k or self.top_k,
            vector_weight=v_weight,
            keyword_weight=k_weight,
            score_threshold=score_threshold or self.score_threshold
        )

        # Apply metadata filter if needed (post-filtering)
        if filter_metadata:
            results = [
                r for r in results
                if self._matches_filter(r.metadata, filter_metadata)
            ]

        # Convert to RetrievalResult
        retrieval_results = []
        for result in results:
            retrieval_results.append(RetrievalResult(
                text=result.text,
                score=result.score,
                metadata=result.metadata,
                chunk_id=result.chunk_id,
                relevance_score=result.score
            ))

        logger.info(f"Hybrid retrieval found {len(retrieval_results)} documents for query: {query[:50]}...")
        return retrieval_results

    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    def retrieve_batch(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[List[RetrievalResult]]:
        """
        Retrieve for multiple queries efficiently.

        Args:
            queries: List of query texts
            top_k: Number of results per query
            score_threshold: Minimum score threshold

        Returns:
            List of result lists for each query
        """
        # Get embeddings for all queries
        query_embeddings = [self._get_query_embedding(q) for q in queries]

        # Batch search
        batch_results = self.vector_store.search_batch(
            query_embeddings=query_embeddings,
            top_k=top_k or self.top_k,
            score_threshold=score_threshold or self.score_threshold
        )

        # Convert to RetrievalResult
        all_results = []
        for results in batch_results:
            retrieval_results = [
                RetrievalResult(
                    text=r.text,
                    score=r.score,
                    metadata=r.metadata,
                    chunk_id=r.chunk_id,
                    relevance_score=r.score
                )
                for r in results
            ]
            all_results.append(retrieval_results)

        logger.info(f"Batch retrieval completed for {len(queries)} queries")
        return all_results

    def clear_cache(self):
        """Clear query embedding cache."""
        self._embedding_cache.clear()
        logger.info("Cleared query embedding cache")


class ContextualRetriever(BaseRetriever):
    """
    Retriever that uses conversation context for improved retrieval.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        max_context_history: int = 5,
        context_weight: float = 0.3
    ):
        """
        Initialize contextual retriever.

        Args:
            base_retriever: Underlying retriever (vector or hybrid)
            max_context_history: Maximum number of previous turns to consider
            context_weight: Weight for context in query expansion
        """
        super().__init__(base_retriever.top_k, base_retriever.score_threshold)
        self.base_retriever = base_retriever
        self.max_context_history = max_context_history
        self.context_weight = context_weight
        self.conversation_history: List[Dict[str, str]] = []

        logger.info(f"Initialized ContextualRetriever with context_weight={context_weight}")

    def add_to_history(self, query: str, response: str):
        """Add query-response pair to conversation history."""
        self.conversation_history.append({
            "query": query,
            "response": response
        })

        # Keep only recent history
        if len(self.conversation_history) > self.max_context_history:
            self.conversation_history.pop(0)

    def _expand_query_with_context(self, query: str) -> str:
        """Expand query using conversation context."""
        if not self.conversation_history:
            return query

        # Build context string
        context_parts = []
        for turn in self.conversation_history[-self.max_context_history:]:
            context_parts.append(f"Previous Q: {turn['query']}")
            context_parts.append(f"Previous A: {turn['response'][:200]}")

        context = "\n".join(context_parts)

        # Expanded query
        expanded_query = f"{query}\n\nContext:\n{context}"

        logger.debug(f"Expanded query with {len(self.conversation_history)} context turns")
        return expanded_query

    def retrieve(
        self,
        query: str,
        use_context: bool = True,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve documents with optional context.

        Args:
            query: Query text
            use_context: Whether to use conversation context
            top_k: Number of results to return
            score_threshold: Minimum score threshold

        Returns:
            List of RetrievalResult objects
        """
        if use_context and self.conversation_history:
            expanded_query = self._expand_query_with_context(query)
            results = self.base_retriever.retrieve(
                expanded_query,
                top_k=top_k,
                score_threshold=score_threshold
            )
        else:
            results = self.base_retriever.retrieve(
                query,
                top_k=top_k,
                score_threshold=score_threshold
            )

        return results

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history.clear()
        logger.info("Cleared conversation history")


class MultiStageRetriever(BaseRetriever):
    """
    Multi-stage retriever that first retrieves candidates then reranks them.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        reranker: Optional[Any] = None,  # Will implement reranker later
        initial_top_k: int = 20,
        final_top_k: int = 5
    ):
        """
        Initialize multi-stage retriever.

        Args:
            base_retriever: Primary retriever for candidate selection
            reranker: Reranker model for fine-grained scoring
            initial_top_k: Number of candidates to retrieve initially
            final_top_k: Number of results after reranking
        """
        super().__init__(final_top_k, None)
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.initial_top_k = initial_top_k
        self.final_top_k = final_top_k

        logger.info(f"Initialized MultiStageRetriever with initial_k={initial_top_k}, final_k={final_top_k}")

    @log_function_call(level="INFO")
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve with multi-stage candidate selection and reranking.

        Args:
            query: Query text
            top_k: Number of final results
            score_threshold: Minimum score threshold

        Returns:
            List of RetrievalResult objects
        """
        # Stage 1: Retrieve initial candidates
        candidates = self.base_retriever.retrieve(
            query,
            top_k=self.initial_top_k,
            score_threshold=score_threshold
        )

        if not candidates:
            return []

        # Stage 2: Rerank if reranker is available
        if self.reranker:
            candidates = self._rerank(query, candidates)

        # Stage 3: Return top-k results
        final_k = top_k or self.final_top_k
        results = candidates[:final_k]

        logger.info(f"Multi-stage retrieval: {len(candidates)} candidates -> {len(results)} final")
        return results

    def _rerank(self, query: str, candidates: List[RetrievalResult]) -> List[RetrievalResult]:
        """Rerank candidates using cross-encoder or other model."""
        # Placeholder for reranking logic
        # Will be implemented when reranker module is ready
        logger.debug(f"Reranking {len(candidates)} candidates")

        # For now, just return candidates sorted by score
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates


class EnsembleRetriever(BaseRetriever):
    """
    Ensemble retriever that combines results from multiple retrievers.
    """

    def __init__(
        self,
        retrievers: List[BaseRetriever],
        weights: Optional[List[float]] = None,
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ):
        """
        Initialize ensemble retriever.

        Args:
            retrievers: List of retriever instances
            weights: Weight for each retriever (default: equal weights)
            top_k: Number of final results
            score_threshold: Minimum score threshold
        """
        super().__init__(top_k, score_threshold)
        self.retrievers = retrievers

        # Set weights
        if weights:
            self.weights = weights
        else:
            self.weights = [1.0 / len(retrievers)] * len(retrievers)

        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

        logger.info(f"Initialized EnsembleRetriever with {len(retrievers)} retrievers")

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[RetrievalResult]:
        """
        Combine results from all retrievers using weighted scoring.

        Args:
            query: Query text
            top_k: Number of final results
            score_threshold: Minimum score threshold

        Returns:
            List of RetrievalResult objects
        """
        # Collect results from all retrievers
        all_results: Dict[str, RetrievalResult] = {}
        result_scores: Dict[str, float] = {}

        for retriever, weight in zip(self.retrievers, self.weights):
            results = retriever.retrieve(
                query,
                top_k=self.top_k * 2,  # Get more from each retriever
                score_threshold=score_threshold
            )

            for result in results:
                chunk_id = result.chunk_id

                if chunk_id not in all_results:
                    all_results[chunk_id] = result
                    result_scores[chunk_id] = 0.0

                # Add weighted score
                result_scores[chunk_id] += result.score * weight

        # Convert to list and sort by combined score
        combined_results = []
        for chunk_id, score in result_scores.items():
            result = all_results[chunk_id]
            result.score = score
            result.relevance_score = score
            combined_results.append(result)

        # Sort by score and return top-k
        combined_results.sort(key=lambda x: x.score, reverse=True)
        final_k = top_k or self.top_k
        final_results = combined_results[:final_k]

        logger.info(f"Ensemble retrieval: combined {len(combined_results)} unique results")
        return final_results


# Convenience factory function
def create_retriever(
    retriever_type: str = "vector",
    vector_store: Optional[FAISSVectorStore] = None,
    embedding_generator: Optional[Any] = None,
    top_k: int = 5,
    **kwargs
) -> BaseRetriever:
    """
    Factory function to create a retriever instance.

    Args:
        retriever_type: Type of retriever ('vector', 'hybrid', 'contextual', 'multi_stage', 'ensemble')
        vector_store: FAISS vector store instance
        embedding_generator: Embedding generator instance
        top_k: Number of results to return
        **kwargs: Additional retriever-specific arguments

    Returns:
        Retriever instance
    """
    if retriever_type == "vector":
        if not vector_store or not embedding_generator:
            raise ValueError("vector_store and embedding_generator required for vector retriever")
        return VectorRetriever(vector_store, embedding_generator, top_k=top_k, **kwargs)

    elif retriever_type == "hybrid":
        if not vector_store or not embedding_generator:
            raise ValueError("vector_store and embedding_generator required for hybrid retriever")
        return HybridRetriever(vector_store, embedding_generator, top_k=top_k, **kwargs)

    elif retriever_type == "contextual":
        base_retriever = kwargs.get("base_retriever")
        if not base_retriever:
            raise ValueError("base_retriever required for contextual retriever")
        return ContextualRetriever(base_retriever, **kwargs)

    elif retriever_type == "multi_stage":
        base_retriever = kwargs.get("base_retriever")
        if not base_retriever:
            raise ValueError("base_retriever required for multi-stage retriever")
        return MultiStageRetriever(base_retriever, top_k=top_k, **kwargs)

    elif retriever_type == "ensemble":
        retrievers = kwargs.get("retrievers")
        if not retrievers:
            raise ValueError("retrievers list required for ensemble retriever")
        return EnsembleRetriever(retrievers, top_k=top_k, **kwargs)

    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")


if __name__ == "__main__":
    # Example usage (requires vector_store and embedding_generator)
    import numpy as np
    from src.retrieval.vector_store import create_vector_store
    from src.ingestion.embedding_generator import OpenAIEmbeddingGenerator

    # Initialize components
    store = create_vector_store(dimension=1536, index_type="HNSW64")

    # Add some dummy data
    dummy_embeddings = [np.random.randn(1536).tolist() for _ in range(10)]
    dummy_texts = [
        "Python is a programming language",
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "Natural language processing deals with text",
        "Computer vision works with images"
    ] * 2
    store.add_embeddings(dummy_embeddings[:10], dummy_texts[:10])

    # Initialize retriever (requires OpenAI API key)
    # embedding_gen = OpenAIEmbeddingGenerator()
    # retriever = VectorRetriever(store, embedding_gen, top_k=3)

    # results = retriever.retrieve("What is machine learning?")
    # for result in results:
    #     print(f"Score: {result.score:.4f} - {result.text}")

    print("Retriever module ready. Configure with your API keys to test.")
