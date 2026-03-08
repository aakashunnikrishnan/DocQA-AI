"""
OpenAI embeddings integration for generating vector representations of text chunks.
OPTIMIZED: Advanced batching with dynamic batch sizing, throughput optimization, and caching.
"""

import os
import logging
import time
import asyncio
import hashlib
from typing import List, Dict, Any, Optional, Union, Tuple, Iterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from collections import deque
import threading
import math
from functools import wraps

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


class BatchOptimizer:
    """
    Dynamic batch optimizer for embedding generation.
    Adjusts batch size based on token usage, latency, and success rates.
    """

    def __init__(
        self,
        initial_batch_size: int = 20,
        min_batch_size: int = 5,
        max_batch_size: int = 100,
        target_tokens_per_batch: int = 8000,
        adjustment_factor: float = 0.1,
        cooldown_seconds: int = 60
    ):
        """
        Initialize batch optimizer.

        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            target_tokens_per_batch: Target tokens per batch
            adjustment_factor: Factor for batch size adjustments
            cooldown_seconds: Cooldown between adjustments
        """
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_tokens_per_batch = target_tokens_per_batch
        self.adjustment_factor = adjustment_factor
        self.cooldown_seconds = cooldown_seconds

        # Statistics
        self.batch_stats = {
            "total_batches": 0,
            "successful_batches": 0,
            "failed_batches": 0,
            "avg_tokens_per_batch": 0,
            "avg_latency_ms": 0,
            "batch_sizes": [],
            "token_counts": [],
            "latencies": []
        }

        self._last_adjustment_time = 0
        self._lock = threading.Lock()

        logger.info(f"BatchOptimizer initialized: initial={initial_batch_size}, "
                   f"min={min_batch_size}, max={max_batch_size}")

    def record_batch_result(
        self,
        batch_size: int,
        tokens_used: int,
        latency_ms: float,
        success: bool
    ):
        """
        Record batch performance statistics.

        Args:
            batch_size: Number of items in batch
            tokens_used: Total tokens in batch
            latency_ms: Batch processing time
            success: Whether batch was successful
        """
        with self._lock:
            self.batch_stats["total_batches"] += 1
            self.batch_stats["batch_sizes"].append(batch_size)
            self.batch_stats["token_counts"].append(tokens_used)
            self.batch_stats["latencies"].append(latency_ms)

            if success:
                self.batch_stats["successful_batches"] += 1
            else:
                self.batch_stats["failed_batches"] += 1

            # Update averages
            self.batch_stats["avg_tokens_per_batch"] = np.mean(self.batch_stats["token_counts"])
            self.batch_stats["avg_latency_ms"] = np.mean(self.batch_stats["latencies"])

    def get_optimal_batch_size(self, estimated_tokens: int = 0) -> int:
        """
        Get optimal batch size based on current performance.

        Args:
            estimated_tokens: Estimated tokens for current batch

        Returns:
            Optimal batch size
        """
        with self._lock:
            # Check if enough data for optimization
            if self.batch_stats["total_batches"] < 5:
                return self.current_batch_size

            current_time = time.time()
            if current_time - self._last_adjustment_time < self.cooldown_seconds:
                return self.current_batch_size

            # Calculate optimal size based on token targets
            if estimated_tokens > 0:
                # Adjust based on token estimate
                target_size = int(self.target_tokens_per_batch / estimated_tokens) * 2
                target_size = max(self.min_batch_size, min(self.max_batch_size, target_size))
            else:
                # Use historical data
                token_ratio = self.target_tokens_per_batch / max(1, self.batch_stats["avg_tokens_per_batch"])
                target_size = int(self.current_batch_size * token_ratio)

            # Apply adjustment factor
            adjustment = (target_size - self.current_batch_size) * self.adjustment_factor
            new_size = int(self.current_batch_size + adjustment)

            # Clamp to limits
            new_size = max(self.min_batch_size, min(self.max_batch_size, new_size))

            # Check latency-based adjustment
            if self.batch_stats["avg_latency_ms"] > 5000:  # >5 seconds
                # Reduce batch size if latency is high
                new_size = max(self.min_batch_size, int(new_size * 0.8))

            # Update if changed significantly
            if abs(new_size - self.current_batch_size) > 2:
                self.current_batch_size = new_size
                self._last_adjustment_time = current_time
                logger.info(f"Adjusted batch size to {new_size} (was {self.current_batch_size})")

            return self.current_batch_size

    def get_stats(self) -> Dict[str, Any]:
        """Get batch optimizer statistics."""
        with self._lock:
            total = self.batch_stats["total_batches"]
            if total == 0:
                success_rate = 0.0
            else:
                success_rate = self.batch_stats["successful_batches"] / total

            return {
                "current_batch_size": self.current_batch_size,
                "min_batch_size": self.min_batch_size,
                "max_batch_size": self.max_batch_size,
                "total_batches": total,
                "successful_batches": self.batch_stats["successful_batches"],
                "failed_batches": self.batch_stats["failed_batches"],
                "success_rate": success_rate,
                "avg_tokens_per_batch": self.batch_stats["avg_tokens_per_batch"],
                "avg_latency_ms": self.batch_stats["avg_latency_ms"],
                "target_tokens_per_batch": self.target_tokens_per_batch
            }


class AdaptiveBatchProcessor:
    """
    Adaptive batch processor for efficient embedding generation.
    Handles dynamic batching, parallel processing, and throughput optimization.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        dimension: int,
        batch_optimizer: Optional[BatchOptimizer] = None,
        max_workers: int = 4,
        use_parallel: bool = True,
        timeout: int = 60
    ):
        """
        Initialize adaptive batch processor.

        Args:
            client: OpenAI client
            model: Model name
            dimension: Embedding dimension
            batch_optimizer: Batch optimizer instance
            max_workers: Maximum parallel workers
            use_parallel: Whether to use parallel processing
            timeout: Request timeout
        """
        self.client = client
        self.model = model
        self.dimension = dimension
        self.batch_optimizer = batch_optimizer or BatchOptimizer()
        self.max_workers = max_workers
        self.use_parallel = use_parallel
        self.timeout = timeout

        self._executor = ThreadPoolExecutor(max_workers=max_workers) if use_parallel else None

        # Rate limiting
        self._rate_limiter = None

        logger.info(f"AdaptiveBatchProcessor initialized: max_workers={max_workers}, "
                   f"use_parallel={use_parallel}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((RateLimitError, APIError, APIConnectionError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _process_batch_sync(
        self,
        texts: List[str],
        batch_id: str
    ) -> Tuple[List[List[float]], int]:
        """
        Process a single batch synchronously.

        Args:
            texts: List of texts to embed
            batch_id: Batch identifier

        Returns:
            Tuple of (embeddings, total_tokens)
        """
        start_time = time.time()

        try:
            response: CreateEmbeddingResponse = self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimension
            )

            embeddings = [data.embedding for data in response.data]
            total_tokens = response.usage.total_tokens

            latency_ms = (time.time() - start_time) * 1000

            # Record batch performance
            self.batch_optimizer.record_batch_result(
                batch_size=len(texts),
                tokens_used=total_tokens,
                latency_ms=latency_ms,
                success=True
            )

            logger.debug(f"Batch {batch_id}: {len(texts)} texts, {total_tokens} tokens, "
                        f"{latency_ms:.0f}ms")

            return embeddings, total_tokens

        except Exception as e:
            # Record failure
            latency_ms = (time.time() - start_time) * 1000
            self.batch_optimizer.record_batch_result(
                batch_size=len(texts),
                tokens_used=0,
                latency_ms=latency_ms,
                success=False
            )
            raise

    def _process_batch_parallel(
        self,
        batches: List[Tuple[List[str], str]]
    ) -> List[Tuple[List[List[float]], int]]:
        """
        Process multiple batches in parallel.

        Args:
            batches: List of (texts, batch_id) tuples

        Returns:
            List of (embeddings, total_tokens) tuples
        """
        if not self._executor:
            return [self._process_batch_sync(texts, batch_id) for texts, batch_id in batches]

        futures = []
        for texts, batch_id in batches:
            future = self._executor.submit(self._process_batch_sync, texts, batch_id)
            futures.append(future)

        results = []
        for future in as_completed(futures):
            try:
                result = future.result(timeout=self.timeout)
                results.append(result)
            except Exception as e:
                logger.error(f"Parallel batch processing failed: {e}")
                # Return zero embeddings for failed batch
                results.append(([[0.0] * self.dimension] * len(batches[0][0]), 0))

        return results

    def process(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> Tuple[List[List[float]], int, List[str]]:
        """
        Process texts with adaptive batching.

        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress

        Returns:
            Tuple of (embeddings, total_tokens, batch_ids)
        """
        if not texts:
            return [], 0, []

        # Estimate tokens for batch optimization
        estimated_tokens = sum(len(t) // 4 for t in texts)  # Rough estimate

        # Get optimal batch size
        batch_size = self.batch_optimizer.get_optimal_batch_size(
            estimated_tokens // max(1, len(texts))
        )

        # Create batches
        batches = []
        batch_ids = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_id = f"batch_{len(batches)}"
            batches.append((batch_texts, batch_id))
            batch_ids.append(batch_id)

        logger.info(f"Processing {len(texts)} texts in {len(batches)} batches "
                   f"(batch_size={batch_size})")

        # Process batches
        all_embeddings = []
        total_tokens = 0

        if self.use_parallel and len(batches) > 1:
            # Parallel processing
            results = self._process_batch_parallel(batches)
            for embeddings, tokens in results:
                all_embeddings.extend(embeddings)
                total_tokens += tokens
        else:
            # Sequential processing
            for batch_texts, batch_id in batches:
                embeddings, tokens = self._process_batch_sync(batch_texts, batch_id)
                all_embeddings.extend(embeddings)
                total_tokens += tokens

        return all_embeddings, total_tokens, batch_ids


class SmartBatchGenerator:
    """
    Smart batch generator that groups texts by estimated token count.
    Optimizes batch composition for better API utilization.
    """

    def __init__(
        self,
        max_tokens_per_batch: int = 8000,
        max_texts_per_batch: int = 100,
        min_texts_per_batch: int = 1
    ):
        """
        Initialize smart batch generator.

        Args:
            max_tokens_per_batch: Maximum tokens per batch
            max_texts_per_batch: Maximum texts per batch
            min_texts_per_batch: Minimum texts per batch
        """
        self.max_tokens_per_batch = max_tokens_per_batch
        self.max_texts_per_batch = max_texts_per_batch
        self.min_texts_per_batch = min_texts_per_batch

        logger.info(f"SmartBatchGenerator initialized: max_tokens={max_tokens_per_batch}, "
                   f"max_texts={max_texts_per_batch}")

    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens in a text."""
        return len(text) // 4  # Rough estimate (1 token ≈ 4 chars)

    def generate_batches(
        self,
        texts: List[str],
        token_estimates: Optional[List[int]] = None
    ) -> List[List[str]]:
        """
        Generate optimized batches.

        Args:
            texts: List of texts
            token_estimates: Optional pre-computed token estimates

        Returns:
            List of batches
        """
        if not texts:
            return []

        # Get token estimates
        if token_estimates is None:
            token_estimates = [self.estimate_tokens(t) for t in texts]

        # Pair texts with their estimates
        items = list(zip(texts, token_estimates))

        # Sort by token count (larger texts first)
        items.sort(key=lambda x: x[1], reverse=True)

        batches = []
        current_batch = []
        current_tokens = 0

        for text, tokens in items:
            # Check if adding this text would exceed limits
            if (current_tokens + tokens > self.max_tokens_per_batch or
                len(current_batch) >= self.max_texts_per_batch):

                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0

            current_batch.append(text)
            current_tokens += tokens

        # Add last batch
        if current_batch:
            batches.append(current_batch)

        # Ensure minimum batch size (merge small batches)
        merged_batches = []
        for batch in batches:
            if len(batch) < self.min_texts_per_batch and merged_batches:
                # Merge with previous batch
                merged_batches[-1].extend(batch)
            else:
                merged_batches.append(batch)

        return merged_batches


class OptimizedEmbeddingGenerator:
    """
    Optimized embedding generator with advanced batching, caching, and performance tuning.
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
        show_progress: bool = True,
        use_adaptive_batching: bool = True,
        max_workers: int = 4,
        use_parallel: bool = True,
        smart_batching: bool = True
    ):
        """
        Initialize optimized embedding generator.

        Args:
            model: OpenAI embedding model name
            api_key: OpenAI API key
            base_url: Custom API base URL
            batch_size: Initial batch size
            max_retries: Maximum number of retries
            timeout: Request timeout in seconds
            dimensions: Output dimensions
            organization: OpenAI organization ID
            rate_limit_requests: Maximum requests per minute
            rate_limit_tokens: Maximum tokens per minute
            use_cache: Whether to use embedding cache
            cache_dir: Directory for cache storage
            max_cache_size: Maximum cache entries
            show_progress: Whether to show progress bar
            use_adaptive_batching: Whether to use adaptive batching
            max_workers: Maximum parallel workers
            use_parallel: Whether to use parallel processing
            smart_batching: Whether to use smart batching
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package is required. Install with: pip install openai"
            )

        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.show_progress = show_progress
        self.use_adaptive_batching = use_adaptive_batching
        self.use_parallel = use_parallel
        self.smart_batching = smart_batching

        # Model configuration
        self.model_config = self._get_model_config(model, dimensions)
        self.dimension = self.model_config["dimension"]

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

        # Initialize cache
        self.use_cache = use_cache
        self.cache = None
        if use_cache:
            from src.utils.cache import DiskCache
            self.cache = DiskCache(
                name="embeddings",
                cache_dir=cache_dir,
                max_size_mb=1024,
                default_ttl=86400 * 7  # 7 days
            )

        # Initialize batch optimizer
        self.batch_optimizer = BatchOptimizer(
            initial_batch_size=batch_size,
            min_batch_size=max(1, batch_size // 4),
            max_batch_size=min(100, batch_size * 4),
            target_tokens_per_batch=8000
        ) if use_adaptive_batching else None

        # Initialize adaptive batch processor
        self.batch_processor = AdaptiveBatchProcessor(
            client=self.client,
            model=self.model,
            dimension=self.dimension,
            batch_optimizer=self.batch_optimizer,
            max_workers=max_workers,
            use_parallel=use_parallel,
            timeout=timeout
        ) if use_adaptive_batching else None

        # Initialize smart batch generator
        self.batch_generator = SmartBatchGenerator(
            max_tokens_per_batch=8000,
            max_texts_per_batch=100
        ) if smart_batching else None

        # Statistics
        self.stats = {
            "total_items": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "processing_time": 0.0,
            "batches_processed": 0
        }

        logger.info(f"OptimizedEmbeddingGenerator initialized: model={model}, "
                   f"dimension={self.dimension}, batch_size={batch_size}, "
                   f"adaptive_batching={use_adaptive_batching}, "
                   f"parallel={use_parallel}, smart_batching={smart_batching}")

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
        if len(text) <= max_tokens * 4:
            return text

        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            if len(tokens) > max_tokens:
                tokens = tokens[:max_tokens]
                return encoding.decode(tokens)
        except ImportError:
            max_chars = max_tokens * 4
            if len(text) > max_chars:
                text = text[:max_chars]

        return text

    def _estimate_tokens(self, texts: List[str]) -> int:
        """Estimate total tokens for a batch."""
        return sum(len(t) // 4 for t in texts)

    def _calculate_cost(self, total_tokens: int) -> float:
        """Calculate cost for embedding tokens."""
        cost_per_1k = self.model_config["cost_per_1k_tokens"]
        return (total_tokens / 1000) * cost_per_1k

    def generate_embeddings(
        self,
        chunks: List[Union[str, Dict[str, Any]]],
        show_progress: Optional[bool] = None,
        return_cached: bool = True,
        use_batching: bool = True
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings with optimized batching.

        Args:
            chunks: List of either text strings or dicts with 'text' and 'metadata' keys
            show_progress: Whether to show progress bar
            return_cached: Whether to return cached results immediately
            use_batching: Whether to use batching (always True for large datasets)

        Returns:
            List of EmbeddingResult objects
        """
        start_time = time.time()
        show_prog = show_progress if show_progress is not None else self.show_progress

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
                cached = self.cache.get(text, self.model) if self.cache else None
                if cached and return_cached:
                    # Convert cached embedding to list if needed
                    if isinstance(cached, np.ndarray):
                        cached = cached.tolist()

                    result = EmbeddingResult(
                        text=text,
                        embedding=cached,
                        metadata=metadata_list[idx],
                        chunk_index=chunk_indices[idx],
                        model=self.model
                    )
                    results[idx] = result
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

        # Generate embeddings for uncached chunks
        if uncached_texts:
            logger.info(f"Generating embeddings for {len(uncached_texts)} uncached chunks")

            # Truncate texts
            truncated_texts = [self._truncate_text(t) for t in uncached_texts]

            # Use smart batching if enabled
            if self.smart_batching and self.batch_generator:
                batches = self.batch_generator.generate_batches(truncated_texts)
            else:
                # Use fixed batch size
                batch_size = self.batch_optimizer.current_batch_size if self.batch_optimizer else 20
                batches = [truncated_texts[i:i+batch_size] for i in range(0, len(truncated_texts), batch_size)]

            # Process batches
            all_embeddings = []
            total_tokens = 0

            if self.use_adaptive_batching and self.batch_processor:
                # Use adaptive batch processor
                embeddings, tokens, batch_ids = self.batch_processor.process(
                    truncated_texts,
                    show_progress=show_prog
                )
                all_embeddings = embeddings
                total_tokens = tokens
                self.stats["batches_processed"] = len(batch_ids)
            else:
                # Simple batch processing
                if show_prog:
                    try:
                        from tqdm import tqdm
                        iterator = tqdm(batches, desc="Generating embeddings")
                    except ImportError:
                        iterator = batches
                else:
                    iterator = batches

                for batch in iterator:
                    try:
                        response = self.client.embeddings.create(
                            model=self.model,
                            input=batch,
                            dimensions=self.dimension
                        )

                        embeddings = [data.embedding for data in response.data]
                        all_embeddings.extend(embeddings)
                        total_tokens += response.usage.total_tokens
                        self.stats["batches_processed"] += 1

                    except Exception as e:
                        logger.error(f"Batch processing failed: {e}")
                        # Add zero embeddings for failed batch
                        for _ in batch:
                            all_embeddings.append([0.0] * self.dimension)

            # Create results for uncached chunks
            for idx, (embedding, text, metadata, original_idx) in enumerate(
                zip(all_embeddings, uncached_texts, uncached_metadata, uncached_indices)
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
                if self.use_cache and self.cache:
                    self.cache.set(text, self.model, embedding)

                results[original_idx] = result
                self.stats["total_items"] += 1

            # Update cost
            self.stats["total_tokens"] += total_tokens
            self.stats["total_cost"] += self._calculate_cost(total_tokens)

        # Calculate processing time
        self.stats["processing_time"] = time.time() - start_time

        # Log statistics
        self._log_stats()

        return results

    def _log_stats(self):
        """Log generation statistics."""
        total_items = self.stats["total_items"]
        total_tokens = self.stats["total_tokens"]
        total_cost = self.stats["total_cost"]
        cache_hits = self.stats["cache_hits"]
        cache_misses = self.stats["cache_misses"]
        processing_time = self.stats["processing_time"]

        logger.info("=" * 60)
        logger.info("EMBEDDING GENERATION STATISTICS (Optimized)")
        logger.info("=" * 60)
        logger.info(f"Total items processed:  {total_items}")
        logger.info(f"Total tokens used:      {total_tokens}")
        logger.info(f"Total cost:             ${total_cost:.6f}")
        logger.info(f"Cache hits:             {cache_hits}")
        logger.info(f"Cache misses:           {cache_misses}")

        if cache_hits + cache_misses > 0:
            hit_rate = cache_hits / (cache_hits + cache_misses)
            logger.info(f"Cache hit rate:         {hit_rate:.2%}")

        logger.info(f"Batches processed:      {self.stats['batches_processed']}")
        logger.info(f"Processing time:        {processing_time:.2f}s")

        if total_items > 0 and processing_time > 0:
            items_per_second = total_items / processing_time
            tokens_per_second = total_tokens / processing_time
            logger.info(f"Throughput:             {items_per_second:.1f} items/s")
            logger.info(f"Token throughput:       {tokens_per_second:.0f} tokens/s")

        if self.batch_optimizer:
            logger.info(f"Current batch size:     {self.batch_optimizer.current_batch_size}")

        logger.info("=" * 60)

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        stats = self.stats.copy()

        if self.cache:
            stats["cache_stats"] = self.cache.get_stats() if hasattr(self.cache, 'get_stats') else {}

        if self.batch_optimizer:
            stats["batch_optimizer"] = self.batch_optimizer.get_stats()

        stats["model_info"] = {
            "model": self.model,
            "dimension": self.dimension,
            "max_tokens": self.model_config["max_tokens"],
            "cost_per_1k_tokens": self.model_config["cost_per_1k_tokens"]
        }

        return stats

    def clear_cache(self):
        """Clear the embedding cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Embedding cache cleared")


# ============================================================
# Convenience Function
# ============================================================

def generate_embeddings_optimized(
    texts: List[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 20,
    use_cache: bool = True,
    show_progress: bool = True,
    use_adaptive_batching: bool = True,
    use_parallel: bool = True,
    smart_batching: bool = True
) -> List[List[float]]:
    """
    Quick helper function to generate embeddings with optimized batching.

    Args:
        texts: List of text strings to embed
        model: OpenAI embedding model
        batch_size: Initial batch size
        use_cache: Whether to use cache
        show_progress: Whether to show progress
        use_adaptive_batching: Whether to use adaptive batching
        use_parallel: Whether to use parallel processing
        smart_batching: Whether to use smart batching

    Returns:
        List of embedding vectors
    """
    generator = OptimizedEmbeddingGenerator(
        model=model,
        batch_size=batch_size,
        use_cache=use_cache,
        show_progress=show_progress,
        use_adaptive_batching=use_adaptive_batching,
        use_parallel=use_parallel,
        smart_batching=smart_batching
    )

    chunks = [{"text": t, "metadata": {}} for t in texts]
    results = generator.generate_embeddings(chunks)

    return [result.embedding for result in results]


if __name__ == "__main__":
    # Example usage with optimized batching
    logging.basicConfig(level=logging.INFO)

    # Create optimized generator
    generator = OptimizedEmbeddingGenerator(
        model="text-embedding-3-small",
        batch_size=20,
        use_adaptive_batching=True,
        use_parallel=True,
        smart_batching=True,
        use_cache=True
    )

    # Sample texts
    sample_texts = [
        "This is the first document." * 10,
        "This is the second document with more content." * 20,
        "And a third one for good measure." * 30,
        "Machine learning is a subset of artificial intelligence." * 40,
        "Deep learning uses neural networks with multiple layers." * 50
    ]

    # Generate embeddings
    chunks = [{"text": t, "metadata": {"index": i}} for i, t in enumerate(sample_texts)]
    results = generator.generate_embeddings(chunks, show_progress=True)

    print(f"\nGenerated {len(results)} embeddings:")
    for i, result in enumerate(results):
        print(f"  Document {i}: {len(result.embedding)} dimensions")

    # Get stats
    stats = generator.get_stats()
    print(f"\nPerformance Statistics:")
    print(f"  Throughput: {stats.get('throughput_items_per_second', 0):.1f} items/s")
    print(f"  Total cost: ${stats.get('total_cost', 0):.6f}")
