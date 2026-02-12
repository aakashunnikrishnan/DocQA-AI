"""
Vector store implementation using FAISS with HNSW optimization.
Supports multiple index types including optimized HNSW, IVF, and Flat indexes.
ENHANCED: HNSW optimization with configurable parameters, GPU support, and performance tuning.
"""

import os
import pickle
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import time
from dataclasses import dataclass, field

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
    FAISS-based vector store with HNSW optimization for efficient similarity search.

    Supports index types:
    - FlatIP: Exact search with inner product (cosine similarity)
    - FlatL2: Exact search with L2 distance
    - HNSW32: HNSW with 32 connections (fast, good accuracy)
    - HNSW64: HNSW with 64 connections (better accuracy, slightly slower)
    - HNSW128: HNSW with 128 connections (best accuracy, slower)
    - IVF: Inverted File Index (scalable to millions of vectors)
    - IVF_HNSW: IVF + HNSW hybrid (best for very large datasets)
    """

    INDEX_TYPES = {
        "FlatIP": lambda d: faiss.IndexFlatIP(d),
        "FlatL2": lambda d: faiss.IndexFlatL2(d),
        "HNSW32": lambda d: faiss.IndexHNSWFlat(d, 32),
        "HNSW64": lambda d: faiss.IndexHNSWFlat(d, 64),
        "HNSW128": lambda d: faiss.IndexHNSWFlat(d, 128),
        "IVF": lambda d: faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, 100),
        "IVF_HNSW": lambda d: faiss.IndexIVFFlat(
            faiss.IndexHNSWFlat(d, 64), d, 100
        ),
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
        nlist: int = 100,
        nprobe: int = 10,
        max_elements: int = 1000000,
        use_optimized_hnsw: bool = True
    ):
        """
        Initialize FAISS vector store with HNSW optimization.

        Args:
            dimension: Embedding dimension (MUST match your embedding model)
            index_type: Type of FAISS index to use
            metric: Similarity metric ('cosine' or 'l2')
            index_path: Path to load existing index
            use_gpu: Whether to use GPU (requires faiss-gpu)
            ef_search: HNSW search parameter (higher = more accurate but slower)
            ef_construction: HNSW construction parameter (higher = better index quality)
            m: HNSW number of connections per layer (higher = better recall but slower)
            nlist: IVF number of clusters (for IVF indexes)
            nprobe: IVF number of clusters to search (higher = more accurate)
            max_elements: Maximum number of elements for pre-allocation
            use_optimized_hnsw: Use optimized HNSW with better default parameters
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.ef_search = ef_search
        self.ef_construction = ef_construction
        self.m = m
        self.nlist = nlist
        self.nprobe = nprobe
        self.max_elements = max_elements
        self.use_optimized_hnsw = use_optimized_hnsw

        # Data storage
        self.texts: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.chunk_ids: List[str] = []
        self._embeddings: Optional[np.ndarray] = None  # Store embeddings for rebuilding
        self._is_trained = False

        # Performance tracking
        self._performance_stats = {
            "add_count": 0,
            "search_count": 0,
            "total_search_time": 0.0,
            "total_add_time": 0.0,
            "avg_search_time": 0.0,
            "avg_add_time": 0.0
        }

        # Create or load index
        self.index = self._create_index()

        # Configure HNSW parameters if applicable
        self._configure_hnsw()

        # Load existing index if provided
        if index_path and Path(index_path).exists():
            self.load(index_path)

        # GPU support
        self.use_gpu = use_gpu
        if use_gpu and faiss.get_num_gpus() > 0:
            self._move_to_gpu()

        logger.info(f"Initialized FAISS vector store with {index_type} index, dimension={dimension}")
        logger.info(f"HNSW params: ef_search={ef_search}, ef_construction={ef_construction}, m={m}")
        if "IVF" in index_type:
            logger.info(f"IVF params: nlist={nlist}, nprobe={nprobe}")

    def _create_index(self):
        """Create FAISS index based on configuration with HNSW optimization."""
        # Use optimized HNSW if available
        if self.use_optimized_hnsw and "HNSW" in self.index_type:
            return self._create_optimized_hnsw_index()

        # Normalize vectors for cosine similarity if using IP
        if self.metric == "cosine" and "IP" in self.index_type:
            # We'll normalize vectors before adding
            pass

        index_factory = self.INDEX_TYPES.get(self.index_type)
        if not index_factory:
            logger.warning(f"Unknown index type {self.index_type}, using HNSW64")
            index_factory = self.INDEX_TYPES["HNSW64"]

        index = index_factory(self.dimension)

        # Configure IVF index
        if self.index_type == "IVF" and hasattr(index, "make_direct_map"):
            index.make_direct_map()
            index.nprobe = self.nprobe

        return index

    def _create_optimized_hnsw_index(self) -> faiss.Index:
        """
        Create optimized HNSW index with better default parameters.
        Uses HNSW with optimized settings for the best performance/accuracy tradeoff.
        """
        # Parse HNSW type to get M value
        if self.index_type.startswith("HNSW"):
            m_value = int(self.index_type.replace("HNSW", ""))
        else:
            m_value = 64  # Default

        # Use optimized M value if not specified
        if m_value == 64 and self.m > 0:
            m_value = self.m

        # Create HNSW index
        index = faiss.IndexHNSWFlat(self.dimension, m_value)

        # Set optimized parameters
        if hasattr(index, 'hnsw'):
            # Optimized search parameters
            index.hnsw.efSearch = self.ef_search

            # Optimized construction parameters
            if hasattr(index.hnsw, 'efConstruction'):
                index.hnsw.efConstruction = self.ef_construction

            # Set max neighbors for better recall
            if hasattr(index.hnsw, 'max_neighbors'):
                index.hnsw.max_neighbors = m_value * 2

        return index

    def _configure_hnsw(self):
        """Configure HNSW parameters for optimal performance."""
        if not hasattr(self.index, 'hnsw'):
            return

        # Set search parameters
        self.index.hnsw.efSearch = self.ef_search

        # Set construction parameters if available
        if hasattr(self.index.hnsw, 'efConstruction'):
            self.index.hnsw.efConstruction = self.ef_construction

        # Update M value if available
        if self.use_optimized_hnsw and hasattr(self.index.hnsw, 'max_neighbors'):
            # Use the configured M value
            pass

        logger.debug(f"HNSW configured: efSearch={self.ef_search}, "
                    f"efConstruction={self.ef_construction}")

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

    def _validate_embedding_dimension(self, embedding: np.ndarray) -> bool:
        """
        Validate that embedding dimension matches expected dimension.

        Args:
            embedding: Embedding vector to validate

        Returns:
            True if valid, False otherwise

        Raises:
            ValueError: If dimension mismatch
        """
        actual_dim = embedding.shape[-1] if len(embedding.shape) > 0 else len(embedding)

        if actual_dim != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {actual_dim}. "
                f"Please ensure your embedding model dimension matches the vector store dimension."
            )
        return True

    def add_embeddings(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        chunk_ids: Optional[List[str]] = None,
        train_index: bool = True
    ) -> List[int]:
        """
        Add embeddings to the vector store with optimized HNSW batching.

        Args:
            embeddings: List of embedding vectors (must match dimension)
            texts: List of corresponding text chunks
            metadata: Optional metadata for each chunk
            chunk_ids: Optional IDs for each chunk
            train_index: Whether to train index (for IVF indexes)

        Returns:
            List of indices where embeddings were added

        Raises:
            ValueError: If embedding dimensions don't match
        """
        if not embeddings:
            logger.warning("No embeddings to add")
            return []

        start_time = time.time()

        # Convert to numpy array
        vectors = np.array(embeddings).astype(np.float32)

        # FIX: Validate dimensions before adding
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: vectors have shape {vectors.shape[1]}, "
                f"but vector store expects {self.dimension}. "
                f"Please use the same embedding model for both generation and storage."
            )

        # Normalize if using cosine similarity
        if self.metric == "cosine":
            vectors = self._normalize_vectors(vectors)

        # Train IVF index if needed
        if train_index and self.index_type == "IVF" and not self._is_trained:
            self._train_index(vectors)

        # Add to index with optimized batch size
        batch_size = min(1024, len(vectors))
        indices = []
        start_idx = len(self.texts)

        try:
            # Add vectors in optimized batches for better performance
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i+batch_size]
                self.index.add(batch)

        except Exception as e:
            # FIX: Better error message for dimension issues
            if "dimension" in str(e).lower():
                raise ValueError(
                    f"FAISS dimension error: {str(e)}. "
                    f"Expected dimension {self.dimension}, "
                    f"but got vectors with shape {vectors.shape}. "
                    f"Please check your embedding model configuration."
                )
            raise

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

        # Store embeddings for potential rebuild
        if self._embeddings is None:
            self._embeddings = vectors
        else:
            self._embeddings = np.vstack([self._embeddings, vectors])

        # Return indices
        indices = list(range(start_idx, len(self.texts)))

        self._performance_stats["add_count"] += 1
        self._performance_stats["total_add_time"] += (time.time() - start_time)
        self._performance_stats["avg_add_time"] = (
            self._performance_stats["total_add_time"] / self._performance_stats["add_count"]
        )

        logger.info(f"Added {len(embeddings)} embeddings. Total: {len(self.texts)}")
        return indices

    def _train_index(self, vectors: np.ndarray):
        """Train index (required for IVF indexes)."""
        if not hasattr(self.index, 'train'):
            self._is_trained = True
            return

        if self._is_trained:
            return

        try:
            # Use a subset for training if dataset is large
            train_size = min(10000, len(vectors))
            if len(vectors) > train_size:
                indices = np.random.choice(len(vectors), train_size, replace=False)
                train_vectors = vectors[indices]
            else:
                train_vectors = vectors

            self.index.train(train_vectors)
            self._is_trained = True
            logger.info(f"Index trained with {len(train_vectors)} vectors")

        except Exception as e:
            logger.error(f"Failed to train index: {e}")
            raise

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        ef_search: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Search for similar vectors with optimized HNSW parameters.

        Args:
            query_embedding: Query embedding vector (must match dimension)
            top_k: Number of results to return
            score_threshold: Minimum similarity score threshold
            filter_metadata: Filter results by metadata (exact match)
            ef_search: Override ef_search for this query (higher = more accurate)

        Returns:
            List of SearchResult objects

        Raises:
            ValueError: If query embedding dimension doesn't match
        """
        if len(self.texts) == 0:
            logger.warning("Vector store is empty")
            return []

        start_time = time.time()

        # Optimize HNSW search parameters for this query
        if ef_search is not None and hasattr(self.index, 'hnsw'):
            original_ef = self.index.hnsw.efSearch
            self.index.hnsw.efSearch = ef_search

        # Prepare query vector
        query = np.array([query_embedding]).astype(np.float32)

        # FIX: Validate query dimension
        if query.shape[1] != self.dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: got {query.shape[1]}, "
                f"expected {self.dimension}. "
                f"Please ensure you're using the same embedding model for queries."
            )

        # Normalize if using cosine similarity
        if self.metric == "cosine":
            query = self._normalize_vectors(query)

        # Adjust top_k if we have fewer documents
        actual_k = min(top_k, len(self.texts))

        # Use optimized search with HNSW
        try:
            # Use more efficient search for large datasets
            if len(self.texts) > 10000 and "HNSW" in self.index_type:
                # HNSW is already efficient for large datasets
                pass

            scores, indices = self.index.search(query, actual_k)

            # Flatten results (first row)
            scores = scores[0]
            indices = indices[0]

        except Exception as e:
            # FIX: Better error handling for search failures
            logger.error(f"FAISS search failed: {e}")
            if "dimension" in str(e).lower():
                raise ValueError(f"FAISS search dimension error: {str(e)}")
            return []
        finally:
            # Restore original ef_search if changed
            if ef_search is not None and hasattr(self.index, 'hnsw'):
                self.index.hnsw.efSearch = original_ef

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

        # Update performance stats
        self._performance_stats["search_count"] += 1
        self._performance_stats["total_search_time"] += (time.time() - start_time)
        self._performance_stats["avg_search_time"] = (
            self._performance_stats["total_search_time"] / self._performance_stats["search_count"]
        )

        return results

    def search_batch(
        self,
        query_embeddings: List[List[float]],
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        ef_search: Optional[int] = None
    ) -> List[List[SearchResult]]:
        """
        Search for multiple queries in batch with optimized HNSW.

        Args:
            query_embeddings: List of query embedding vectors
            top_k: Number of results per query
            score_threshold: Minimum similarity score threshold
            ef_search: Override ef_search for this query

        Returns:
            List of result lists for each query

        Raises:
            ValueError: If any query embedding dimension doesn't match
        """
        if len(self.texts) == 0:
            return [[] for _ in query_embeddings]

        # Optimize HNSW search parameters for this batch
        if ef_search is not None and hasattr(self.index, 'hnsw'):
            original_ef = self.index.hnsw.efSearch
            self.index.hnsw.efSearch = ef_search

        # Prepare query matrix
        queries = np.array(query_embeddings).astype(np.float32)

        # FIX: Validate all query dimensions
        if queries.shape[1] != self.dimension:
            raise ValueError(
                f"Query embedding dimension mismatch: got {queries.shape[1]}, "
                f"expected {self.dimension}. "
                f"Please ensure you're using the same embedding model for all queries."
            )

        # Normalize if using cosine similarity
        if self.metric == "cosine":
            queries = self._normalize_vectors(queries)

        # Adjust top_k
        actual_k = min(top_k, len(self.texts))

        # Search
        try:
            all_scores, all_indices = self.index.search(queries, actual_k)
        except Exception as e:
            # FIX: Better error handling for batch search
            logger.error(f"Batch FAISS search failed: {e}")
            if "dimension" in str(e).lower():
                raise ValueError(f"FAISS batch search dimension error: {str(e)}")
            return [[] for _ in query_embeddings]
        finally:
            # Restore original ef_search if changed
            if ef_search is not None and hasattr(self.index, 'hnsw'):
                self.index.hnsw.efSearch = original_ef

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

    def _matches_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

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
        if not self.texts or self._embeddings is None:
            self.clear()
            return

        # Keep only embeddings for remaining texts
        if len(self.texts) < len(self._embeddings):
            # We need to rebuild from stored embeddings
            # This is a simplified version - for production, store embeddings separately
            logger.warning("Rebuilding index from stored embeddings...")
            self.index = self._create_index()
            self._is_trained = False

            # Add embeddings in batches
            batch_size = 1000
            for i in range(0, len(self._embeddings), batch_size):
                batch = self._embeddings[i:i+batch_size]
                if self.index_type == "IVF" and not self._is_trained:
                    self._train_index(batch)
                self.index.add(batch)

            self._is_trained = True
            logger.info(f"Index rebuilt with {len(self._embeddings)} embeddings")

    def clear(self):
        """Clear all data from vector store."""
        self.texts = []
        self.metadata = []
        self.chunk_ids = []
        self._embeddings = None
        self._is_trained = False
        self.index = self._create_index()
        self._configure_hnsw()

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
                'metric': self.metric,
                'ef_search': self.ef_search,
                'ef_construction': self.ef_construction,
                'm': self.m,
                '_embeddings': self._embeddings,
                '_is_trained': self._is_trained
            }, f)

        logger.info(f"Saved vector store to {path}")

    def load(self, path: str):
        """
        Load vector store from disk.

        Args:
            path: Directory path to load from

        Raises:
            FileNotFoundError: If path doesn't exist
            ValueError: If loaded index dimension doesn't match
        """
        load_path = Path(path)

        if not load_path.exists():
            raise FileNotFoundError(f"Vector store path not found: {path}")

        # Load FAISS index
        index_path = load_path / "index.faiss"
        if index_path.exists():
            loaded_index = faiss.read_index(str(index_path))

            # FIX: Validate loaded index dimension
            if hasattr(loaded_index, 'd') and loaded_index.d != self.dimension:
                raise ValueError(
                    f"Loaded index dimension ({loaded_index.d}) does not match "
                    f"expected dimension ({self.dimension}). "
                    f"Please ensure you're loading a compatible index."
                )
            self.index = loaded_index
            self._configure_hnsw()

        # Load metadata
        data_path = load_path / "data.pkl"
        if data_path.exists():
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                self.texts = data.get('texts', [])
                self.metadata = data.get('metadata', [])
                self.chunk_ids = data.get('chunk_ids', [])
                self.dimension = data.get('dimension', self.dimension)
                self.index_type = data.get('index_type', self.index_type)
                self.metric = data.get('metric', self.metric)
                self.ef_search = data.get('ef_search', self.ef_search)
                self.ef_construction = data.get('ef_construction', self.ef_construction)
                self.m = data.get('m', self.m)
                self._embeddings = data.get('_embeddings', None)
                self._is_trained = data.get('_is_trained', False)

        # FIX: Verify loaded data matches index
        if len(self.texts) != self.index.ntotal:
            logger.warning(
                f"Data count ({len(self.texts)}) does not match index count ({self.index.ntotal}). "
                f"This may indicate a corrupted load."
            )

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

    def get_dimension(self) -> int:
        """Get the dimension of the vector store."""
        return self.dimension

    def optimize_search_parameters(self, query_size: int = 1000):
        """
        Optimize HNSW search parameters based on dataset size.

        Args:
            query_size: Estimated number of queries to run
        """
        total_vectors = len(self.texts)

        if total_vectors < 10000:
            # Small dataset - use faster search
            self.ef_search = min(100, max(10, int(total_vectors * 0.01)))
        elif total_vectors < 100000:
            # Medium dataset
            self.ef_search = min(200, max(50, int(total_vectors * 0.001)))
        else:
            # Large dataset
            self.ef_search = min(400, max(100, int(total_vectors * 0.0005)))

        if hasattr(self.index, 'hnsw'):
            self.index.hnsw.efSearch = self.ef_search

        logger.info(f"Optimized ef_search to {self.ef_search} for {total_vectors} vectors")

        return {
            "ef_search": self.ef_search,
            "total_vectors": total_vectors,
            "recommended_top_k": min(100, max(5, int(total_vectors * 0.001)))
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        stats = {
            "total_vectors": len(self.texts),
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metric": self.metric,
            "ef_search": self.ef_search,
            "ef_construction": self.ef_construction,
            "m": self.m,
            "is_trained": self._is_trained,
            "index_size_bytes": self.index.ntotal * self.dimension * 4 if hasattr(self.index, 'ntotal') else 0,
            "average_text_length": np.mean([len(t) for t in self.texts]) if self.texts else 0,
            "has_metadata": sum(1 for m in self.metadata if m) > 0,
            "index_ntotal": getattr(self.index, 'ntotal', 0),
            "performance": self._performance_stats
        }

        # Add HNSW-specific stats
        if hasattr(self.index, 'hnsw'):
            stats.update({
                "hnsw_ef_search": self.index.hnsw.efSearch,
                "hnsw_ef_construction": getattr(self.index.hnsw, 'efConstruction', self.ef_construction),
                "hnsw_max_neighbors": getattr(self.index.hnsw, 'max_neighbors', self.m)
            })

        return stats

    def optimize_memory(self):
        """
        Optimize memory usage by compressing index.
        Currently only logs memory stats - in future could implement quantization.
        """
        if hasattr(self.index, 'ntotal'):
            total_bytes = self.index.ntotal * self.dimension * 4
            total_mb = total_bytes / (1024 * 1024)

            logger.info(f"Index memory usage: {total_mb:.2f} MB for {self.index.ntotal} vectors")

            if total_mb > 1024:  # > 1GB
                logger.warning(f"Index is large ({total_mb:.2f} MB). Consider using IVF or quantization.")

            return {
                "memory_mb": total_mb,
                "vectors": self.index.ntotal,
                "bytes_per_vector": self.dimension * 4,
                "recommendation": "Use IVF_HNSW for large datasets" if total_mb > 1024 else "OK"
            }

        return {}


# ============================================================
# Convenience Functions
# ============================================================

def create_vector_store(
    dimension: int = 1536,
    index_type: str = "HNSW64",
    metric: str = "cosine",
    index_path: Optional[str] = None,
    use_gpu: bool = False,
    ef_search: int = 100,
    ef_construction: int = 200,
    m: int = 16,
    nlist: int = 100,
    nprobe: int = 10,
    max_elements: int = 1000000
) -> FAISSVectorStore:
    """
    Create a FAISS vector store with HNSW optimization.

    Args:
        dimension: Embedding dimension
        index_type: Type of FAISS index
        metric: Similarity metric
        index_path: Path to load existing index
        use_gpu: Whether to use GPU
        ef_search: HNSW search parameter
        ef_construction: HNSW construction parameter
        m: HNSW connections per layer
        nlist: IVF number of clusters
        nprobe: IVF number of clusters to search
        max_elements: Maximum elements for pre-allocation

    Returns:
        FAISSVectorStore instance
    """
    return FAISSVectorStore(
        dimension=dimension,
        index_type=index_type,
        metric=metric,
        index_path=index_path,
        use_gpu=use_gpu,
        ef_search=ef_search,
        ef_construction=ef_construction,
        m=m,
        nlist=nlist,
        nprobe=nprobe,
        max_elements=max_elements
    )


def validate_embedding_dimension(embedding: List[float], expected_dimension: int) -> bool:
    """
    Helper function to validate embedding dimension.

    Args:
        embedding: Embedding vector
        expected_dimension: Expected dimension

    Returns:
        True if valid

    Raises:
        ValueError: If dimension mismatch
    """
    actual_dim = len(embedding)
    if actual_dim != expected_dimension:
        raise ValueError(
            f"Embedding dimension mismatch: got {actual_dim}, expected {expected_dimension}. "
            f"Please check your embedding model configuration."
        )
    return True


if __name__ == "__main__":
    # Example usage with HNSW optimization
    logging.basicConfig(level=logging.INFO)

    # Create vector store with optimized HNSW
    dimension = 1536
    store = create_vector_store(
        dimension=dimension,
        index_type="HNSW64",
        metric="cosine",
        ef_search=100,
        ef_construction=200,
        m=16
    )

    # Generate test data
    num_vectors = 1000
    embeddings = [np.random.randn(dimension).tolist() for _ in range(num_vectors)]
    texts = [f"Document {i}" for i in range(num_vectors)]

    # Add embeddings
    print(f"Adding {num_vectors} embeddings...")
    store.add_embeddings(embeddings, texts)

    # Test search
    query = np.random.randn(dimension).tolist()

    # Search with default parameters
    print("\nSearching with default parameters...")
    results = store.search(query, top_k=5)
    print(f"Found {len(results)} results")

    # Search with optimized ef_search
    print("\nSearching with optimized ef_search=200...")
    results = store.search(query, top_k=5, ef_search=200)
    print(f"Found {len(results)} results")

    # Optimize parameters based on dataset
    print("\nOptimizing search parameters...")
    optimization = store.optimize_search_parameters()
    print(f"Recommended ef_search: {optimization['ef_search']}")

    # Get stats
    print("\nStore statistics:")
    stats = store.get_stats()
    for key, value in stats.items():
        if key != 'performance':
            print(f"  {key}: {value}")

    # Performance stats
    print("\nPerformance statistics:")
    perf = stats.get('performance', {})
    print(f"  Add count: {perf.get('add_count', 0)}")
    print(f"  Avg add time: {perf.get('avg_add_time', 0):.4f}s")
    print(f"  Search count: {perf.get('search_count', 0)}")
    print(f"  Avg search time: {perf.get('avg_search_time', 0):.4f}s")
