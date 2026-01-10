"""
Vector store implementation using FAISS for efficient similarity search.
Supports multiple index types, serialization, and hybrid search capabilities.
"""

import os
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np

import faiss
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result from vector store."""
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = ""
    index: int = -1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "score": float(self.score),
            "metadata": self.metadata,
            "chunk_id": self.chunk_id,
            "index": self.index
        }


class FAISSVectorStore:
    """
    FAISS-based vector store for efficient similarity search.

    Supports index types:
    - FlatIP: Exact search with inner product (cosine similarity)
    - FlatL2: Exact search with L2 distance
    - HNSW32: Hierarchical Navigable Small World (faster, approximate)
    - HNSW64: HNSW with 64 connections
    - IVF: Inverted File Index (scalable to millions of vectors)
    """

    INDEX_TYPES = {
        "FlatIP": lambda d: faiss.IndexFlatIP(d),  # Inner product (cosine)
        "FlatL2": lambda d: faiss.IndexFlatL2(d),  # L2 distance
        "HNSW32": lambda d: faiss.IndexHNSWFlat(d, 32),
        "HNSW64": lambda d: faiss.IndexHNSWFlat(d, 64),
        "IVF": lambda d: faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, 100),
    }

    def __init__(
        self,
        dimension: int = 1536,
        index_type: str = "HNSW64",
        metric: str = "cosine",
        index_path: Optional[str] = None,
        use_gpu: bool = False,
        ef_search: int = 100,
        ef_construction: int = 200,
        m: int = 16,
        nlist: int = 100
    ):
        """
        Initialize FAISS vector store.

        Args:
            dimension: Embedding dimension
            index_type: Type of FAISS index to use
            metric: Similarity metric ('cosine' or 'l2')
            index_path: Path to load existing index
            use_gpu: Whether to use GPU (requires faiss-gpu)
            ef_search: HNSW search parameter (higher = more accurate but slower)
            ef_construction: HNSW construction parameter
            m: HNSW number of connections per layer
            nlist: IVF number of clusters
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.ef_search = ef_search
        self.ef_construction = ef_construction
        self.m = m
        self.nlist = nlist

        # Data storage
        self.texts: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.chunk_ids: List[str] = []

        # Create or load index
        self.index = self._create_index()

        # Configure HNSW parameters if applicable
        if "HNSW" in index_type and hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = ef_search
            self.index.hnsw.efConstruction = ef_construction

        # Load existing index if provided
        if index_path and Path(index_path).exists():
            self.load(index_path)

        # GPU support
        self.use_gpu = use_gpu
        if use_gpu and faiss.get_num_gpus() > 0:
            self._move_to_gpu()

        logger.info(f"Initialized FAISS vector store with {index_type} index, dimension={dimension}")

    def _create_index(self):
        """Create FAISS index based on configuration."""
        # Normalize vectors for cosine similarity if using IP
        if self.metric == "cosine" and "IP" in self.index_type:
            # We'll normalize vectors before adding
            pass

        index_factory = self.INDEX_TYPES.get(self.index_type)
        if not index_factory:
            logger.warning(f"Unknown index type {self.index_type}, using FlatIP")
            index_factory = self.INDEX_TYPES["FlatIP"]

        index = index_factory(self.dimension)

        # Configure IVF index
        if self.index_type == "IVF" and hasattr(index, "make_direct_map"):
            index.make_direct_map()
            index.nprobe = 10  # Number of clusters to search

        return index

    def _move_to_gpu(self):
        """Move index to GPU for faster search."""
        try:
            if faiss.get_num_gpus() > 0:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("Moved FAISS index to GPU")
            else:
                logger.warning("No GPU available, using CPU")
        except Exception as e:
            logger.warning(f"Failed to move index to GPU: {e}")

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity."""
        faiss.normalize_L2(vectors)
        return vectors

    def add_embeddings(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        chunk_ids: Optional[List[str]] = None
    ) -> List[int]:
        """
        Add embeddings to the vector store.

        Args:
            embeddings: List of embedding vectors
            texts: List of corresponding text chunks
            metadata: Optional metadata for each chunk
            chunk_ids: Optional IDs for each chunk

        Returns:
            List of indices where embeddings were added
        """
        if not embeddings:
            logger.warning("No embeddings to add")
            return []

        # Convert to numpy array
        vectors = np.array(embeddings).astype(np.float32)

        # Normalize if using cosine similarity
        if self.metric == "cosine":
            vectors = self._normalize_vectors(vectors)

        # Add to index
        indices = []
        start_idx = len(self.texts)

        for i in range(vectors.shape[0]):
            # Add single vector (FAISS requires 2D array)
            self.index.add(vectors[i:i+1])
            indices.append(start_idx + i)

        # Store text and metadata
        self.texts.extend(texts)

        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{} for _ in texts])

        if chunk_ids:
            self.chunk_ids.extend(chunk_ids)
        else:
            self.chunk_ids.extend([f"chunk_{start_idx + i}" for i in range(len(texts))])

        logger.info(f"Added {len(embeddings)} embeddings. Total: {len(self.texts)}")

        return indices

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score threshold
            filter_metadata: Filter results by metadata (exact match)

        Returns:
            List of SearchResult objects
        """
        if len(self.texts) == 0:
            logger.warning("Vector store is empty")
            return []

        # Prepare query vector
        query = np.array([query_embedding]).astype(np.float32)

        # Normalize if using cosine similarity
        if self.metric == "cosine":
            query = self._normalize_vectors(query)

        # Adjust top_k if we have fewer documents
        actual_k = min(top_k, len(self.texts))

        # Search
        try:
            scores, indices = self.index.search(query, actual_k)

            # Flatten results (first row)
            scores = scores[0]
            indices = indices[0]

        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return []

        # Build results
        results = []
        for score, idx in zip(scores, indices):
            if idx == -1 or idx >= len(self.texts):
                continue

            # Convert score based on metric
            if self.metric == "cosine":
                # Cosine similarity is already in [0,1] range
                similarity = float(score)
            elif self.metric == "l2":
                # Convert L2 distance to similarity (smaller distance = higher similarity)
                similarity = 1.0 / (1.0 + float(score))
            else:
                similarity = float(score)

            # Apply threshold
            if score_threshold and similarity < score_threshold:
                continue

            # Apply metadata filter
            if filter_metadata:
                if not self._matches_filter(self.metadata[idx], filter_metadata):
                    continue

            results.append(SearchResult(
                text=self.texts[idx],
                score=similarity,
                metadata=self.metadata[idx].copy(),
                chunk_id=self.chunk_ids[idx],
                index=int(idx)
            ))

        return results

    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    def search_batch(
        self,
        query_embeddings: List[List[float]],
        top_k: int = 5,
        score_threshold: Optional[float] = None
    ) -> List[List[SearchResult]]:
        """
        Search for multiple queries in batch.

        Args:
            query_embeddings: List of query embedding vectors
            top_k: Number of results per query
            score_threshold: Minimum similarity score threshold

        Returns:
            List of result lists for each query
        """
        if len(self.texts) == 0:
            return [[] for _ in query_embeddings]

        # Prepare query matrix
        queries = np.array(query_embeddings).astype(np.float32)

        # Normalize if using cosine similarity
        if self.metric == "cosine":
            queries = self._normalize_vectors(queries)

        # Adjust top_k
        actual_k = min(top_k, len(self.texts))

        # Search
        try:
            all_scores, all_indices = self.index.search(queries, actual_k)
        except Exception as e:
            logger.error(f"Batch FAISS search failed: {e}")
            return [[] for _ in query_embeddings]

        # Build results
        batch_results = []
        for query_idx, (scores, indices) in enumerate(zip(all_scores, all_indices)):
            results = []
            for score, idx in zip(scores, indices):
                if idx == -1 or idx >= len(self.texts):
                    continue

                if self.metric == "cosine":
                    similarity = float(score)
                else:
                    similarity = 1.0 / (1.0 + float(score))

                if score_threshold and similarity < score_threshold:
                    continue

                results.append(SearchResult(
                    text=self.texts[idx],
                    score=similarity,
                    metadata=self.metadata[idx].copy(),
                    chunk_id=self.chunk_ids[idx],
                    index=int(idx)
                ))

            batch_results.append(results)

        return batch_results

    def delete(self, indices: List[int]):
        """
        Delete embeddings by indices.
        Note: FAISS doesn't support direct deletion, so we rebuild index.

        Args:
            indices: List of indices to delete
        """
        if not indices:
            return

        # Create mask of indices to keep
        keep_mask = [i for i in range(len(self.texts)) if i not in indices]

        if not keep_mask:
            # Delete everything
            self.clear()
            return

        # Keep only selected items
        self.texts = [self.texts[i] for i in keep_mask]
        self.metadata = [self.metadata[i] for i in keep_mask]
        self.chunk_ids = [self.chunk_ids[i] for i in keep_mask]

        # Rebuild index
        self._rebuild_index()

        logger.info(f"Deleted {len(indices)} items. Remaining: {len(self.texts)}")

    def _rebuild_index(self):
        """Rebuild index from current embeddings."""
        if not self.texts:
            self.clear()
            return

        # We need to get embeddings again - this requires storing them
        # For now, warn that this is not implemented
        logger.warning("Rebuilding index requires original embeddings. Use update_embeddings method instead.")

    def update_embedding(self, index: int, embedding: List[float], text: str, metadata: Dict[str, Any]):
        """
        Update an embedding at specific index.

        Args:
            index: Index to update
            embedding: New embedding vector
            text: New text
            metadata: New metadata
        """
        if index < 0 or index >= len(self.texts):
            raise IndexError(f"Index {index} out of range")

        # Update stored data
        self.texts[index] = text
        self.metadata[index] = metadata

        # Update index (rebuild needed for FAISS)
        # This is inefficient; for production, consider using a different vector store
        logger.warning("Updating individual embedding requires index rebuild. Use batch updates when possible.")

    def clear(self):
        """Clear all data from vector store."""
        self.texts = []
        self.metadata = []
        self.chunk_ids = []
        self.index = self._create_index()

        logger.info("Cleared vector store")

    def save(self, path: str):
        """
        Save vector store to disk.

        Args:
            path: Directory path to save to
        """
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = save_path / "index.faiss"
        faiss.write_index(self.index, str(index_path))

        # Save metadata
        data_path = save_path / "data.pkl"
        with open(data_path, 'wb') as f:
            pickle.dump({
                'texts': self.texts,
                'metadata': self.metadata,
                'chunk_ids': self.chunk_ids,
                'dimension': self.dimension,
                'index_type': self.index_type,
                'metric': self.metric
            }, f)

        logger.info(f"Saved vector store to {path}")

    def load(self, path: str):
        """
        Load vector store from disk.

        Args:
            path: Directory path to load from
        """
        load_path = Path(path)

        if not load_path.exists():
            raise FileNotFoundError(f"Vector store path not found: {path}")

        # Load FAISS index
        index_path = load_path / "index.faiss"
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))

        # Load metadata
        data_path = load_path / "data.pkl"
        if data_path.exists():
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                self.texts = data.get('texts', [])
                self.metadata = data.get('metadata', [])
                self.chunk_ids = data.get('chunk_ids', [])

        logger.info(f"Loaded vector store from {path}: {len(self.texts)} items")

    def get_size(self) -> int:
        """Get number of vectors in store."""
        return len(self.texts)

    def get_vector(self, index: int) -> Optional[np.ndarray]:
        """
        Get vector at specific index (requires reconstruct method).
        Note: Not all FAISS indices support reconstruction.
        """
        try:
            if hasattr(self.index, 'reconstruct'):
                vector = np.zeros(self.dimension, dtype=np.float32)
                self.index.reconstruct(index, vector)
                return vector
        except Exception as e:
            logger.warning(f"Failed to reconstruct vector at index {index}: {e}")

        return None

    def train_index(self, embeddings: List[List[float]]):
        """
        Train index (required for IVF indexes).

        Args:
            embeddings: Training embeddings
        """
        if not hasattr(self.index, 'train'):
            logger.info("Index doesn't require training")
            return

        vectors = np.array(embeddings).astype(np.float32)

        if self.metric == "cosine":
            vectors = self._normalize_vectors(vectors)

        self.index.train(vectors)
        logger.info("Index trained")

    def merge_from(self, other: 'FAISSVectorStore'):
        """
        Merge another vector store into this one.

        Args:
            other: Another FAISSVectorStore instance
        """
        if not other.texts:
            return

        # Merge data
        self.texts.extend(other.texts)
        self.metadata.extend(other.metadata)
        self.chunk_ids.extend(other.chunk_ids)

        # Merge indices (requires rebuilding)
        # For now, we need to re-add all vectors
        logger.warning("Merging requires rebuilding index. Consider adding embeddings directly.")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        return {
            "total_vectors": len(self.texts),
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "index_size_bytes": self.index.ntotal * self.dimension * 4 if hasattr(self.index, 'ntotal') else 0,
            "average_text_length": np.mean([len(t) for t in self.texts]) if self.texts else 0,
            "has_metadata": sum(1 for m in self.metadata if m) > 0
        }


class FAISSHybridStore(FAISSVectorStore):
    """
    FAISS vector store with hybrid search capabilities (vector + keyword).
    Uses BM25 for keyword search and combines with vector similarity.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bm25 = None
        self._bm25_corpus = []

    def _init_bm25(self):
        """Initialize BM25 for keyword search."""
        try:
            from rank_bm25 import BM25Okapi
            if self._bm25_corpus:
                # Tokenize texts
                tokenized_corpus = [self._tokenize(text) for text in self._bm25_corpus]
                self._bm25 = BM25Okapi(tokenized_corpus)
        except ImportError:
            logger.warning("rank_bm25 not installed. Install with: pip install rank-bm25")
            self._bm25 = None

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for BM25."""
        import re
        # Convert to lowercase and split on non-alphanumeric
        tokens = re.findall(r'\w+', text.lower())
        return tokens

    def add_embeddings(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        chunk_ids: Optional[List[str]] = None
    ) -> List[int]:
        """Add embeddings and prepare for hybrid search."""
        # Add to vector store
        indices = super().add_embeddings(embeddings, texts, metadata, chunk_ids)

        # Add to BM25 corpus
        self._bm25_corpus.extend(texts)
        self._init_bm25()

        return indices

    def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        top_k: int = 5,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining vector similarity and keyword relevance.

        Args:
            query_embedding: Query embedding vector
            query_text: Original query text for keyword search
            top_k: Number of results to return
            vector_weight: Weight for vector similarity (0-1)
            keyword_weight: Weight for keyword relevance (0-1)
            score_threshold: Minimum combined score threshold

        Returns:
            List of SearchResult objects
        """
        # Get vector search results (get more for reranking)
        vector_k = top_k * 3
        vector_results = self.search(query_embedding, vector_k, score_threshold=None)

        if not vector_results:
            return []

        # Get keyword scores if BM25 is available
        keyword_scores = {}
        if self._bm25 and query_text:
            tokenized_query = self._tokenize(query_text)
            bm25_scores = self._bm25.get_scores(tokenized_query)

            # Normalize BM25 scores to [0, 1]
            max_score = max(bm25_scores) if bm25_scores else 1
            for i, score in enumerate(bm25_scores):
                if i < len(self.texts):
                    normalized_score = score / max_score if max_score > 0 else 0
                    keyword_scores[i] = normalized_score

        # Combine scores
        for result in vector_results:
            vector_score = result.score
            keyword_score = keyword_scores.get(result.index, 0)

            # Weighted combination
            combined_score = (vector_weight * vector_score) + (keyword_weight * keyword_score)
            result.score = combined_score

        # Sort by combined score
        vector_results.sort(key=lambda x: x.score, reverse=True)

        # Apply threshold and return top_k
        results = []
        for result in vector_results[:top_k]:
            if score_threshold is None or result.score >= score_threshold:
                results.append(result)

        return results

    def clear(self):
        """Clear all data including BM25 corpus."""
        super().clear()
        self._bm25_corpus = []
        self._bm25 = None
        logger.info("Cleared hybrid vector store")


# Convenience function
def create_vector_store(
    dimension: int = 1536,
    index_type: str = "HNSW64",
    metric: str = "cosine",
    index_path: Optional[str] = None,
    use_gpu: bool = False
) -> FAISSVectorStore:
    """
    Create a FAISS vector store with specified configuration.

    Args:
        dimension: Embedding dimension
        index_type: Type of FAISS index
        metric: Similarity metric
        index_path: Path to load existing index
        use_gpu: Whether to use GPU

    Returns:
        FAISSVectorStore instance
    """
    return FAISSVectorStore(
        dimension=dimension,
        index_type=index_type,
        metric=metric,
        index_path=index_path,
        use_gpu=use_gpu
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create vector store
    store = create_vector_store(dimension=1536, index_type="HNSW64")

    # Generate dummy embeddings
    dummy_embeddings = [np.random.randn(1536).tolist() for _ in range(10)]
    dummy_texts = [f"This is document {i}" for i in range(10)]

    # Add embeddings
    store.add_embeddings(dummy_embeddings, dummy_texts)

    # Search
    query = np.random.randn(1536).tolist()
    results = store.search(query, top_k=3)

    print(f"Search results: {len(results)}")
    for i, result in enumerate(results):
        print(f"  {i+1}. Score: {result.score:.4f} - {result.text[:50]}")

    # Save
    store.save("./test_vector_store")

    # Load
    new_store = create_vector_store(index_path="./test_vector_store")
    print(f"Loaded store with {new_store.get_size()} vectors")
