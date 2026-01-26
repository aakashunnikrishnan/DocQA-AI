"""
OpenAI embeddings integration for generating vector representations of text chunks.
ENHANCED: Batch processing, rate limiting, async support, and caching for large-scale ingestion.
"""

import os
import logging
import time
import asyncio
import hashlib
from typing import List, Dict, Any, Optional, Union, Tuple, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from collections import deque
import threading

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryError
)

# OpenAI imports
try:
    from openai import OpenAI, AsyncOpenAI
    from openai.types import CreateEmbeddingResponse
    from openai import RateLimitError, APIError, APIConnectionError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI package not installed. Install with: pip install openai")

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Represents an embedding result for a text chunk."""
    text: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0
    model: str = ""
    tokens_used: int = 0
    batch_id: str = ""
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "chunk_index": self.chunk_index,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "batch_id": self.batch_id,
            "embedding_dimension": len(self.embedding)
        }

    def get_numpy_array(self) -> np.ndarray:
        """Get embedding as numpy array."""
        return np.array(self.embedding, dtype=np.float32)


class RateLimiter:
    """
    Rate limiter for API calls with token bucket algorithm.
    """

    def __init__(self, requests_per_minute: int = 50, tokens_per_minute: int = 100000):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
            tokens_per_minute: Maximum tokens per minute
        """
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute

        self.request_interval = 60.0 / requests_per_minute
        self.token_interval = 60.0 / tokens_per_minute if tokens_per_minute > 0 else 0

        self.request_timestamps = deque(maxlen=requests_per_minute)
        self.token_timestamps = deque(maxlen=tokens_per_minute)

        self._lock = threading.Lock()

        logger.info(f"RateLimiter initialized: {requests_per_minute} req/min, {tokens_per_minute} tokens/min")

    def wait(self, tokens: int = 0):
        """
        Wait until rate limit allows request.

        Args:
            tokens: Number of tokens for this request
        """
        with self._lock:
            now = time.time()

            # Check request rate
            if len(self.request_timestamps) >= self.requests_per_minute:
                oldest = self.request_timestamps[0]
                wait_time = self.request_interval - (now - oldest)
                if wait_time > 0:
                    time.sleep(wait_time)
                    now = time.time()

            # Check token rate
            if tokens > 0 and self.tokens_per_minute > 0:
                if len(self.token_timestamps) >= self.tokens_per_minute:
                    oldest = self.token_timestamps[0]
                    wait_time = self.token_interval - (now - oldest)
                    if wait_time > 0:
                        time.sleep(wait_time)
                        now = time.time()

            # Record request
            self.request_timestamps.append(now)
            if tokens > 0:
                for _ in range(tokens):
                    self.token_timestamps.append(now)

    async def wait_async(self, tokens: int = 0):
        """
        Async version of wait.

        Args:
            tokens: Number of tokens for this request
        """
        with self._lock:
            now = time.time()

            if len(self.request_timestamps) >= self.requests_per_minute:
                oldest = self.request_timestamps[0]
                wait_time = self.request_interval - (now - oldest)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    now = time.time()

            if tokens > 0 and self.tokens_per_minute > 0:
                if len(self.token_timestamps) >= self.tokens_per_minute:
                    oldest = self.token_timestamps[0]
                    wait_time = self.token_interval - (now - oldest)
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                        now = time.time()

            self.request_timestamps.append(now)
            if tokens > 0:
                for _ in range(tokens):
                    self.token_timestamps.append(now)

    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        with self._lock:
            return {
                "requests_per_minute": self.requests_per_minute,
                "tokens_per_minute": self.tokens_per_minute,
                "current_requests": len(self.request_timestamps),
                "current_tokens": len(self.token_timestamps),
                "request_interval": self.request_interval,
                "token_interval": self.token_interval
            }


class EmbeddingCache:
    """
    Persistent cache for embeddings to avoid redundant API calls.
    Supports disk persistence with LRU eviction.
    """

    def __init__(self, cache_dir: str = "./data/embeddings/cache", max_size: int = 10000):
        """
        Initialize embedding cache.

        Args:
            cache_dir: Directory for cache storage
            max_size: Maximum number of entries in cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self._cache: Dict[str, EmbeddingResult] = {}
        self._access_order: List[str] = []
        self._lock = threading.Lock()
        self._load_cache()

        logger.info(f"EmbeddingCache initialized: {cache_dir}, max_size={max_size}")

    def _get_hash_key(self, text: str, model: str) -> str:
        """Generate hash key for text and model combination."""
        content = f"{text}:{model}".encode('utf-8')
        return hashlib.md5(content).hexdigest()

    def _get_cache_file(self) -> Path:
        """Get cache file path."""
        return self.cache_dir / "embeddings_cache.npz"

    def _load_cache(self):
        """Load cached embeddings from disk."""
        cache_file = self._get_cache_file()
        if cache_file.exists():
            try:
                data = np.load(cache_file, allow_pickle=True)
                keys = data['keys']
                # Load embeddings as list of lists
                embeddings_data = data['embeddings']

                for key, emb_data in zip(keys, embeddings_data):
                    # Reconstruct EmbeddingResult
                    if isinstance(emb_data, dict):
                        result = EmbeddingResult(
                            text=emb_data.get('text', ''),
                            embedding=emb_data.get('embedding', []),
                            metadata=emb_data.get('metadata', {}),
                            model=emb_data.get('model', ''),
                            tokens_used=emb_data.get('tokens_used', 0)
                        )
                        self._cache[key] = result
                        self._access_order.append(key)

                logger.info(f"Loaded {len(self._cache)} cached embeddings from {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self._cache = {}
                self._access_order = []

    def _save_cache(self):
        """Save cached embeddings to disk."""
        if not self._cache:
            return

        cache_file = self._get_cache_file()
        try:
            keys = list(self._cache.keys())
            # Convert embeddings to serializable format
            embeddings_data = []
            for key in keys:
                result = self._cache[key]
                embeddings_data.append({
                    'text': result.text,
                    'embedding': result.embedding,
                    'metadata': result.metadata,
                    'model': result.model,
                    'tokens_used': result.tokens_used
                })

            np.savez_compressed(
                cache_file,
                keys=np.array(keys),
                embeddings=np.array(embeddings_data, dtype=object)
            )
            logger.debug(f"Saved {len(self._cache)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def get(self, text: str, model: str) -> Optional[EmbeddingResult]:
        """Get cached embedding if available."""
        with self._lock:
            key = self._get_hash_key(text, model)
            if key in self._cache:
                # Update access order (move to end)
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]
            return None

    def set(self, text: str, model: str, result: EmbeddingResult):
        """Cache an embedding result."""
        with self._lock:
            key = self._get_hash_key(text, model)

            # Evict if at max size
            if len(self._cache) >= self.max_size and key not in self._cache:
                # Remove least recently used
                if self._access_order:
                    oldest_key = self._access_order.pop(0)
                    if oldest_key in self._cache:
                        del self._cache[oldest_key]

            self._cache[key] = result
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            # Periodically save cache (every 100 items)
            if len(self._cache) % 100 == 0:
                self._save_cache()

    def set_batch(self, texts: List[str], model: str, results: List[EmbeddingResult]):
        """Cache multiple embeddings at once."""
        with self._lock:
            for text, result in zip(texts, results):
                key = self._get_hash_key(text, model)
                self._cache[key] = result
                if key in self._access_order:
                    self._access_order.remove(key)
                self._access_order.append(key)

            # Save if cache size changed significantly
            if len(self._cache) % 100 < len(results):
                self._save_cache()

    def clear(self):
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
            self._access_order = []
            cache_file = self._get_cache_file()
            if cache_file.exists():
                cache_file.unlink()
            logger.info("Cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "cache_dir": str(self.cache_dir),
                "usage_percent": (len(self._cache) / self.max_size * 100) if self.max_size > 0 else 0
            }


class BatchEmbeddingGenerator:
    """
    Optimized batch embedding generator with rate limiting, caching, and retry logic.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        batch_size: int = 20,
        max_retries: int = 3,
        timeout: int = 60,
        dimensions: Optional[int] = None,
        organization: Optional[str] = None,
        rate_limit_requests: int = 50,
        rate_limit_tokens: int = 100000,
        use_cache: bool = True,
        cache_dir: str = "./data/embeddings/cache",
        max_cache_size: int = 10000,
        show_progress: bool = True
    ):
        """
        Initialize batch embedding generator.

        Args:
            model: OpenAI embedding model name
            api_key: OpenAI API key
            base_url: Custom API base URL
            batch_size: Number of texts to embed in one batch
            max_retries: Maximum number of retries for failed requests
            timeout: Request timeout in seconds
            dimensions: Output dimensions (for 3-small/3-large models)
            organization: OpenAI organization ID
            rate_limit_requests: Maximum requests per minute
            rate_limit_tokens: Maximum tokens per minute
            use_cache: Whether to use embedding cache
            cache_dir: Directory for cache storage
            max_cache_size: Maximum cache entries
            show_progress: Whether to show progress bar
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package is required. Install with: pip install openai"
            )

        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
        self.show_progress = show_progress

        # Model configuration
        self.model_config = self._get_model_config(model, dimensions)

        # Initialize clients
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            timeout=timeout
        )

        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            timeout=timeout
        )

        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            requests_per_minute=rate_limit_requests,
            tokens_per_minute=rate_limit_tokens
        )

        # Initialize cache
        self.use_cache = use_cache
        self.cache = EmbeddingCache(cache_dir, max_cache_size) if use_cache else None

        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "batches_processed": 0,
            "total_items": 0,
            "start_time": None,
            "end_time": None
        }

        logger.info(f"BatchEmbeddingGenerator initialized: model={model}, batch_size={batch_size}")
        logger.info(f"Rate limit: {rate_limit_requests} req/min, {rate_limit_tokens} tokens/min")
        logger.info(f"Cache: {'enabled' if use_cache else 'disabled'}")

    def _get_model_config(self, model: str, dimensions: Optional[int]) -> Dict[str, Any]:
        """Get model configuration."""
        model_configs = {
            "text-embedding-3-small": {
                "dimension": dimensions or 1536,
                "max_tokens": 8191,
                "cost_per_1k_tokens": 0.00002,
                "description": "Most cost-effective embedding model"
            },
            "text-embedding-3-large": {
                "dimension": dimensions or 3072,
                "max_tokens": 8191,
                "cost_per_1k_tokens": 0.00013,
                "description": "Most powerful embedding model"
            },
            "text-embedding-ada-002": {
                "dimension": dimensions or 1536,
                "max_tokens": 8191,
                "cost_per_1k_tokens": 0.00010,
                "description": "Legacy Ada v2 model"
            }
        }

        if model in model_configs:
            config = model_configs[model].copy()
            if dimensions and dimensions <= config["dimension"]:
                config["dimension"] = dimensions
            return config
        else:
            return {
                "dimension": dimensions or 1536,
                "max_tokens": 8191,
                "cost_per_1k_tokens": 0.00002,
                "description": "Custom model"
            }

    def _truncate_text(self, text: str) -> str:
        """Truncate text to maximum token limit."""
        max_tokens = self.model_config["max_tokens"]
        if len(text) <= max_tokens * 4:  # Rough estimate: 4 chars per token
            return text

        # More accurate truncation using tiktoken
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            if len(tokens) > max_tokens:
                tokens = tokens[:max_tokens]
                return encoding.decode(tokens)
        except ImportError:
            # Fallback to character-based truncation
            logger.warning("tiktoken not installed. Using approximate truncation.")
            max_chars = max_tokens * 4
            if len(text) > max_chars:
                text = text[:max_chars]

        return text

    def _estimate_tokens(self, texts: List[str]) -> int:
        """Estimate total tokens for a batch."""
        total_chars = sum(len(self._truncate_text(t)) for t in texts)
        return int(total_chars / 4)  # Rough estimate

    def _calculate_cost(self, total_tokens: int) -> float:
        """Calculate cost for embedding tokens."""
        cost_per_1k = self.model_config["cost_per_1k_tokens"]
        return (total_tokens / 1000) * cost_per_1k

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APIError, APIConnectionError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _generate_embeddings_batch(self, texts: List[str]) -> Tuple[List[List[float]], int]:
        """
        Generate embeddings for a batch of texts with retry logic.

        Returns:
            Tuple of (embeddings list, total_tokens_used)
        """
        # Truncate texts
        truncated_texts = [self._truncate_text(t) for t in texts]

        # Estimate tokens for rate limiting
        estimated_tokens = self._estimate_tokens(truncated_texts)

        # Apply rate limiting
        self.rate_limiter.wait(tokens=estimated_tokens)

        try:
            response: CreateEmbeddingResponse = self.client.embeddings.create(
                model=self.model,
                input=truncated_texts,
                dimensions=self.model_config.get("dimension")
            )

            embeddings = [data.embedding for data in response.data]
            total_tokens = response.usage.total_tokens

            self.stats["successful_requests"] += 1
            self.stats["total_tokens"] += total_tokens
            self.stats["total_cost"] += self._calculate_cost(total_tokens)

            return embeddings, total_tokens

        except Exception as e:
            self.stats["failed_requests"] += 1
            logger.error(f"Batch embedding generation failed: {e}")
            raise

    def generate_embeddings(
        self,
        chunks: List[Union[str, Dict[str, Any]]],
        show_progress: Optional[bool] = None,
        return_cached: bool = True
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple chunks with batch processing and caching.

        Args:
            chunks: List of either text strings or dicts with 'text' and 'metadata' keys
            show_progress: Whether to show progress bar (overrides default)
            return_cached: Whether to return cached results immediately

        Returns:
            List of EmbeddingResult objects
        """
        self.stats["start_time"] = time.time()

        # Parse chunks
        texts = []
        metadata_list = []
        chunk_indices = []

        for i, chunk in enumerate(chunks):
            if isinstance(chunk, str):
                texts.append(chunk)
                metadata_list.append({})
            elif isinstance(chunk, dict):
                texts.append(chunk.get("text", ""))
                metadata_list.append(chunk.get("metadata", {}))
            else:
                raise TypeError(f"Unsupported chunk type: {type(chunk)}")
            chunk_indices.append(i)

        # Check cache first
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []
        uncached_metadata = []

        if self.use_cache:
            for idx, text in enumerate(texts):
                cached = self.cache.get(text, self.model)
                if cached and return_cached:
                    # Copy metadata to cached result
                    cached.metadata.update(metadata_list[idx])
                    cached.chunk_index = chunk_indices[idx]
                    results[idx] = cached
                    self.stats["cache_hits"] += 1
                else:
                    uncached_indices.append(idx)
                    uncached_texts.append(text)
                    uncached_metadata.append(metadata_list[idx])
                    self.stats["cache_misses"] += 1
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts
            uncached_metadata = metadata_list

        # Generate embeddings for uncached chunks in batches
        if uncached_texts:
            logger.info(f"Generating embeddings for {len(uncached_texts)} uncached chunks")

            batch_embeddings = []
            batch_tokens = []

            # Process in batches
            total_batches = (len(uncached_texts) + self.batch_size - 1) // self.batch_size

            iterator = range(0, len(uncached_texts), self.batch_size)
            if self.show_progress if show_progress is None else show_progress:
                try:
                    from tqdm import tqdm
                    iterator = tqdm(iterator, total=total_batches, desc="Generating embeddings")
                except ImportError:
                    pass

            for i in iterator:
                batch_texts = uncached_texts[i:i + self.batch_size]

                try:
                    embeddings, tokens = self._generate_embeddings_batch(batch_texts)
                    batch_embeddings.extend(embeddings)
                    batch_tokens.append(tokens)
                    self.stats["batches_processed"] += 1

                except Exception as e:
                    logger.error(f"Failed to process batch starting at index {i}: {e}")
                    # Add zero embeddings for failed batch
                    for _ in batch_texts:
                        batch_embeddings.append([0.0] * self.model_config["dimension"])

            # Create results for uncached chunks
            for idx, (embedding, text, metadata, original_idx) in enumerate(
                zip(batch_embeddings, uncached_texts, uncached_metadata, uncached_indices)
            ):
                result = EmbeddingResult(
                    text=text,
                    embedding=embedding,
                    metadata=metadata,
                    chunk_index=chunk_indices[original_idx],
                    model=self.model,
                    tokens_used=self._estimate_tokens([text])
                )

                # Cache the result
                if self.use_cache:
                    self.cache.set(text, self.model, result)

                results[original_idx] = result
                self.stats["total_items"] += 1

        # Calculate processing time
        self.stats["end_time"] = time.time()
        processing_time = self.stats["end_time"] - self.stats["start_time"]

        # Log statistics
        self._log_stats(processing_time)

        return results

    async def generate_embeddings_async(
        self,
        chunks: List[Union[str, Dict[str, Any]]],
        max_concurrent: int = 10,
        show_progress: Optional[bool] = None
    ) -> List[EmbeddingResult]:
        """
        Asynchronously generate embeddings with concurrency control.

        Args:
            chunks: List of chunks to embed
            max_concurrent: Maximum number of concurrent requests
            show_progress: Whether to show progress bar

        Returns:
            List of EmbeddingResult objects
        """
        self.stats["start_time"] = time.time()

        # Parse chunks
        texts = []
        metadata_list = []
        chunk_indices = []

        for i, chunk in enumerate(chunks):
            if isinstance(chunk, str):
                texts.append(chunk)
                metadata_list.append({})
            else:
                texts.append(chunk.get("text", ""))
                metadata_list.append(chunk.get("metadata", {}))
            chunk_indices.append(i)

        # Check cache first
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        if self.use_cache:
            for idx, text in enumerate(texts):
                cached = self.cache.get(text, self.model)
                if cached:
                    cached.metadata.update(metadata_list[idx])
                    cached.chunk_index = chunk_indices[idx]
                    results[idx] = cached
                    self.stats["cache_hits"] += 1
                else:
                    uncached_indices.append(idx)
                    uncached_texts.append(text)
                    self.stats["cache_misses"] += 1
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts

        if uncached_texts:
            # Create semaphore for concurrency control
            semaphore = asyncio.Semaphore(max_concurrent)

            async def process_one(text, original_idx):
                async with semaphore:
                    # Estimate tokens
                    estimated_tokens = self._estimate_tokens([text])

                    # Apply rate limiting
                    await self.rate_limiter.wait_async(tokens=estimated_tokens)

                    # Generate embedding
                    truncated = self._truncate_text(text)

                    try:
                        response = await self.async_client.embeddings.create(
                            model=self.model,
                            input=[truncated],
                            dimensions=self.model_config.get("dimension")
                        )

                        embedding = response.data[0].embedding
                        total_tokens = response.usage.total_tokens

                        self.stats["successful_requests"] += 1
                        self.stats["total_tokens"] += total_tokens
                        self.stats["total_cost"] += self._calculate_cost(total_tokens)

                        return embedding, total_tokens

                    except Exception as e:
                        self.stats["failed_requests"] += 1
                        logger.error(f"Async embedding failed for text: {e}")
                        return [0.0] * self.model_config["dimension"], 0

            # Process all chunks with concurrency control
            tasks = [
                process_one(text, idx)
                for idx, text in zip(uncached_indices, uncached_texts)
            ]

            # Show progress if requested
            if self.show_progress if show_progress is None else show_progress:
                try:
                    from tqdm.asyncio import tqdm
                    results_async = await tqdm.gather(*tasks, desc="Generating embeddings")
                except ImportError:
                    results_async = await asyncio.gather(*tasks)
            else:
                results_async = await asyncio.gather(*tasks)

            # Process results
            for (embedding, tokens), original_idx in zip(results_async, uncached_indices):
                result = EmbeddingResult(
                    text=texts[original_idx],
                    embedding=embedding,
                    metadata=metadata_list[original_idx],
                    chunk_index=chunk_indices[original_idx],
                    model=self.model,
                    tokens_used=tokens
                )

                if self.use_cache:
                    self.cache.set(texts[original_idx], self.model, result)

                results[original_idx] = result
                self.stats["total_items"] += 1

        self.stats["end_time"] = time.time()
        processing_time = self.stats["end_time"] - self.stats["start_time"]
        self._log_stats(processing_time)

        return results

    def generate_embeddings_stream(
        self,
        chunks: List[Union[str, Dict[str, Any]]]
    ) -> Iterator[EmbeddingResult]:
        """
        Generate embeddings as a stream, yielding results as they complete.

        Args:
            chunks: List of chunks to embed

        Yields:
            EmbeddingResult objects as they are generated
        """
        # Parse chunks
        texts = []
        metadata_list = []

        for chunk in chunks:
            if isinstance(chunk, str):
                texts.append(chunk)
                metadata_list.append({})
            else:
                texts.append(chunk.get("text", ""))
                metadata_list.append(chunk.get("metadata", {}))

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_metadata = metadata_list[i:i + self.batch_size]

            # Check cache for each text
            batch_results = []
            texts_to_process = []
            indices_to_process = []

            for idx, text in enumerate(batch_texts):
                if self.use_cache:
                    cached = self.cache.get(text, self.model)
                    if cached:
                        cached.metadata.update(batch_metadata[idx])
                        batch_results.append(cached)
                        self.stats["cache_hits"] += 1
                        continue

                texts_to_process.append(text)
                indices_to_process.append(idx)
                self.stats["cache_misses"] += 1

            # Process uncached texts
            if texts_to_process:
                try:
                    embeddings, tokens = self._generate_embeddings_batch(texts_to_process)

                    for idx, (embedding, text, metadata) in enumerate(
                        zip(embeddings, texts_to_process,
                            [batch_metadata[i] for i in indices_to_process])
                    ):
                        result = EmbeddingResult(
                            text=text,
                            embedding=embedding,
                            metadata=metadata,
                            model=self.model,
                            tokens_used=self._estimate_tokens([text])
                        )

                        if self.use_cache:
                            self.cache.set(text, self.model, result)

                        batch_results.append(result)
                        self.stats["total_items"] += 1

                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
                    # Yield zero embeddings for failed items
                    for text in texts_to_process:
                        result = EmbeddingResult(
                            text=text,
                            embedding=[0.0] * self.model_config["dimension"],
                            metadata={},
                            model=self.model
                        )
                        batch_results.append(result)

            # Yield results in original order
            for result in batch_results:
                yield result

    def _log_stats(self, processing_time: float):
        """Log processing statistics."""
        total_items = self.stats["total_items"]
        total_tokens = self.stats["total_tokens"]
        total_cost = self.stats["total_cost"]
        cache_hits = self.stats["cache_hits"]
        cache_misses = self.stats["cache_misses"]

        logger.info("=" * 50)
        logger.info("EMBEDDING GENERATION STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Total items processed:  {total_items}")
        logger.info(f"Total tokens used:      {total_tokens}")
        logger.info(f"Total cost:             ${total_cost:.6f}")
        logger.info(f"Cache hits:             {cache_hits}")
        logger.info(f"Cache misses:           {cache_misses}")

        if cache_hits + cache_misses > 0:
            hit_rate = cache_hits / (cache_hits + cache_misses)
            logger.info(f"Cache hit rate:         {hit_rate:.2%}")

        logger.info(f"Batches processed:      {self.stats['batches_processed']}")
        logger.info(f"Successful requests:    {self.stats['successful_requests']}")
        logger.info(f"Failed requests:        {self.stats['failed_requests']}")
        logger.info(f"Processing time:        {processing_time:.2f}s")

        if total_items > 0:
            items_per_second = total_items / processing_time
            logger.info(f"Throughput:             {items_per_second:.1f} items/s")

        logger.info("=" * 50)

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        cache_stats = self.cache.get_stats() if self.cache else {}
        rate_limiter_stats = self.rate_limiter.get_stats()

        return {
            **self.stats,
            "cache": cache_stats,
            "rate_limiter": rate_limiter_stats,
            "model": self.model,
            "dimension": self.model_config["dimension"],
            "batch_size": self.batch_size
        }

    def clear_cache(self):
        """Clear the embedding cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Cache cleared")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model": self.model,
            "dimension": self.model_config["dimension"],
            "max_tokens": self.model_config["max_tokens"],
            "cost_per_1k_tokens": self.model_config["cost_per_1k_tokens"],
            "description": self.model_config["description"]
        }


# Convenience function for batch embedding
def generate_embeddings_batch(
    texts: List[str],
    model: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    batch_size: int = 20,
    use_cache: bool = True,
    show_progress: bool = True
) -> List[List[float]]:
    """
    Quick helper function to generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed
        model: OpenAI embedding model
        api_key: OpenAI API key
        batch_size: Batch size for processing
        use_cache: Whether to use cache
        show_progress: Whether to show progress

    Returns:
        List of embedding vectors
    """
    generator = BatchEmbeddingGenerator(
        model=model,
        api_key=api_key,
        batch_size=batch_size,
        use_cache=use_cache,
        show_progress=show_progress
    )

    chunks = [{"text": text, "metadata": {}} for text in texts]
    results = generator.generate_embeddings(chunks, show_progress=show_progress)

    return [result.embedding for result in results]


if __name__ == "__main__":
    # Example usage with batch processing
    logging.basicConfig(level=logging.INFO)

    # Create generator with batch support
    generator = BatchEmbeddingGenerator(
        model="text-embedding-3-small",
        batch_size=10,
        rate_limit_requests=50,
        use_cache=True
    )

    # Sample texts
    sample_texts = [
        "This is the first document.",
        "This is the second document with more content.",
        "And a third one for good measure.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers."
    ]

    # Generate embeddings in batch
    chunks = [{"text": t, "metadata": {"index": i}} for i, t in enumerate(sample_texts)]
    results = generator.generate_embeddings(chunks, show_progress=True)

    print(f"\nGenerated {len(results)} embeddings:")
    for i, result in enumerate(results):
        print(f"  Document {i}: {len(result.embedding)} dimensions, tokens: {result.tokens_used}")

    # Get stats
    stats = generator.get_stats()
    print(f"\nStatistics:")
    print(f"  Total items: {stats['total_items']}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Total cost: ${stats['total_cost']:.6f}")
