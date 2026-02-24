"""
Batch query processing module for efficient handling of multiple queries.
Supports parallel processing, rate limiting, progress tracking, and result aggregation.
"""

import asyncio
import time
import logging
from typing import List, Dict, Any, Optional, Union, Callable, Iterator, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue
from collections import defaultdict
import json

from src.utils.logger import get_logger
from src.utils.monitoring import get_performance_monitor, measure

logger = get_logger(__name__)


class BatchStatus(Enum):
    """Batch processing status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


class ItemStatus(Enum):
    """Individual item status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_concurrent: int = 10
    batch_size: int = 100
    rate_limit_per_second: Optional[float] = None
    timeout_seconds: int = 60
    retry_count: int = 3
    retry_delay: float = 1.0
    progress_update_interval: int = 1
    stop_on_error: bool = False
    continue_on_error: bool = True
    max_retries_per_item: int = 3
    item_timeout_seconds: int = 30


@dataclass
class BatchItem:
    """Individual batch item."""
    id: str
    data: Any
    status: ItemStatus = ItemStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Result of batch processing."""
    id: str
    status: BatchStatus
    total_items: int
    completed_items: int
    failed_items: int
    skipped_items: int
    processing_time_ms: float
    items: List[BatchItem]
    errors: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "status": self.status.value,
            "total_items": self.total_items,
            "completed_items": self.completed_items,
            "failed_items": self.failed_items,
            "skipped_items": self.skipped_items,
            "processing_time_ms": self.processing_time_ms,
            "success_rate": self.completed_items / self.total_items if self.total_items > 0 else 0,
            "items": [
                {
                    "id": item.id,
                    "status": item.status.value,
                    "processing_time_ms": item.processing_time_ms,
                    "error": item.error
                }
                for item in self.items
            ],
            "errors": self.errors[:10],
            "metadata": self.metadata,
            "started_at": self.started_at,
            "completed_at": self.completed_at
        }


class BatchProcessor:
    """
    Process multiple queries or tasks in batch with parallel execution.
    """

    def __init__(
        self,
        config: Optional[BatchConfig] = None,
        executor: Optional[ThreadPoolExecutor] = None
    ):
        """
        Initialize batch processor.

        Args:
            config: Batch configuration
            executor: Thread pool executor for parallel processing
        """
        self.config = config or BatchConfig()
        self.executor = executor or ThreadPoolExecutor(max_workers=self.config.max_concurrent)

        # Processing state
        self._items: Dict[str, BatchItem] = {}
        self._results: Dict[str, Any] = {}
        self._errors: Dict[str, str] = {}
        self._status = BatchStatus.PENDING
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

        # Rate limiting
        self._rate_limiter = None
        if self.config.rate_limit_per_second:
            self._rate_limiter = RateLimiter(self.config.rate_limit_per_second)

        # Progress tracking
        self._progress_callbacks: List[Callable] = []
        self._completed_count = 0
        self._failed_count = 0
        self._total_count = 0
        self._lock = threading.Lock()

        logger.info(f"BatchProcessor initialized: max_concurrent={config.max_concurrent}, "
                   f"rate_limit={config.rate_limit_per_second}")

    def add_item(self, item_id: str, data: Any, metadata: Optional[Dict[str, Any]] = None):
        """
        Add an item to the batch.

        Args:
            item_id: Unique item identifier
            data: Item data to process
            metadata: Optional metadata
        """
        with self._lock:
            if item_id in self._items:
                logger.warning(f"Item {item_id} already exists, overwriting")

            self._items[item_id] = BatchItem(
                id=item_id,
                data=data,
                metadata=metadata or {},
                status=ItemStatus.PENDING
            )
            self._total_count += 1

    def add_items(self, items: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None):
        """
        Add multiple items to the batch.

        Args:
            items: Dictionary of item_id -> data
            metadata: Optional metadata for all items
        """
        for item_id, data in items.items():
            self.add_item(item_id, data, metadata)

    def add_items_from_list(
        self,
        items: List[Any],
        id_generator: Optional[Callable[[int], str]] = None
    ):
        """
        Add items from a list.

        Args:
            items: List of items to process
            id_generator: Function to generate IDs (default: item_{index})
        """
        for i, data in enumerate(items):
            item_id = id_generator(i) if id_generator else f"item_{i}"
            self.add_item(item_id, data)

    def process(
        self,
        processor_fn: Callable[[Any], Any],
        progress_callback: Optional[Callable[[int, int, BatchStatus], None]] = None
    ) -> BatchResult:
        """
        Process all items in the batch.

        Args:
            processor_fn: Function to process each item (takes data, returns result)
            progress_callback: Optional progress callback (completed, total, status)

        Returns:
            BatchResult object
        """
        self._status = BatchStatus.RUNNING
        self._start_time = time.time()

        if progress_callback:
            self._progress_callbacks.append(progress_callback)

        try:
            # Create processing tasks
            items_to_process = [
                (item_id, item) for item_id, item in self._items.items()
                if item.status == ItemStatus.PENDING
            ]

            if not items_to_process:
                logger.warning("No items to process")
                return self._create_result()

            # Process items with parallel execution
            self._process_items_parallel(processor_fn, items_to_process)

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            self._status = BatchStatus.FAILED
            raise

        finally:
            self._end_time = time.time()
            if self._status == BatchStatus.RUNNING:
                self._status = BatchStatus.COMPLETED

            # Final progress update
            self._update_progress()

        return self._create_result()

    async def process_async(
        self,
        processor_fn: Callable[[Any], Any],
        progress_callback: Optional[Callable[[int, int, BatchStatus], None]] = None
    ) -> BatchResult:
        """
        Process all items asynchronously.

        Args:
            processor_fn: Function to process each item
            progress_callback: Optional progress callback

        Returns:
            BatchResult object
        """
        self._status = BatchStatus.RUNNING
        self._start_time = time.time()

        if progress_callback:
            self._progress_callbacks.append(progress_callback)

        try:
            items_to_process = [
                (item_id, item) for item_id, item in self._items.items()
                if item.status == ItemStatus.PENDING
            ]

            if not items_to_process:
                logger.warning("No items to process")
                return self._create_result()

            # Process items asynchronously
            await self._process_items_async(processor_fn, items_to_process)

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            self._status = BatchStatus.FAILED
            raise

        finally:
            self._end_time = time.time()
            if self._status == BatchStatus.RUNNING:
                self._status = BatchStatus.COMPLETED

            self._update_progress()

        return self._create_result()

    def _process_items_parallel(
        self,
        processor_fn: Callable[[Any], Any],
        items: List[Tuple[str, BatchItem]]
    ):
        """
        Process items in parallel using thread pool.
        """
        # Create tasks
        tasks = []
        for item_id, item in items:
            task = self.executor.submit(
                self._process_single_item,
                item_id,
                item,
                processor_fn
            )
            tasks.append(task)

        # Wait for all tasks to complete
        for task in tasks:
            try:
                task.result()
            except Exception as e:
                logger.error(f"Task failed: {e}")
                if self.config.stop_on_error:
                    raise

    async def _process_items_async(
        self,
        processor_fn: Callable[[Any], Any],
        items: List[Tuple[str, BatchItem]]
    ):
        """
        Process items asynchronously with concurrency control.
        """
        semaphore = asyncio.Semaphore(self.config.max_concurrent)

        async def process_with_semaphore(item_id: str, item: BatchItem):
            async with semaphore:
                return await self._process_single_item_async(
                    item_id,
                    item,
                    processor_fn
                )

        # Create tasks
        tasks = []
        for item_id, item in items:
            task = asyncio.create_task(process_with_semaphore(item_id, item))
            tasks.append(task)

        # Wait for all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed: {result}")
                if self.config.stop_on_error:
                    raise result

    def _process_single_item(
        self,
        item_id: str,
        item: BatchItem,
        processor_fn: Callable[[Any], Any]
    ):
        """
        Process a single item with retry logic.
        """
        item.status = ItemStatus.PROCESSING
        item.start_time = time.time()

        # Rate limiting
        if self._rate_limiter:
            self._rate_limiter.wait()

        try:
            # Process with retries
            for attempt in range(self.config.max_retries_per_item):
                try:
                    result = processor_fn(item.data)

                    # Success
                    item.status = ItemStatus.COMPLETED
                    item.result = result
                    item.end_time = time.time()
                    item.processing_time_ms = (item.end_time - item.start_time) * 1000

                    with self._lock:
                        self._completed_count += 1
                        self._results[item_id] = result

                    self._update_progress()
                    return

                except Exception as e:
                    item.retry_count = attempt + 1

                    if attempt < self.config.max_retries_per_item - 1:
                        time.sleep(self.config.retry_delay * (attempt + 1))
                        continue
                    else:
                        # All retries exhausted
                        raise

        except Exception as e:
            # Failed
            item.status = ItemStatus.FAILED
            item.error = str(e)
            item.end_time = time.time()
            item.processing_time_ms = (item.end_time - item.start_time) * 1000

            with self._lock:
                self._failed_count += 1
                self._errors[item_id] = str(e)

            logger.error(f"Item {item_id} failed: {e}")
            self._update_progress()

            if self.config.stop_on_error:
                raise

    async def _process_single_item_async(
        self,
        item_id: str,
        item: BatchItem,
        processor_fn: Callable[[Any], Any]
    ):
        """
        Process a single item asynchronously with retry logic.
        """
        item.status = ItemStatus.PROCESSING
        item.start_time = time.time()

        # Rate limiting
        if self._rate_limiter:
            await self._rate_limiter.wait_async()

        try:
            # Process with retries
            for attempt in range(self.config.max_retries_per_item):
                try:
                    # Check if processor is async
                    if asyncio.iscoroutinefunction(processor_fn):
                        result = await processor_fn(item.data)
                    else:
                        # Run sync function in thread pool
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            self.executor,
                            processor_fn,
                            item.data
                        )

                    # Success
                    item.status = ItemStatus.COMPLETED
                    item.result = result
                    item.end_time = time.time()
                    item.processing_time_ms = (item.end_time - item.start_time) * 1000

                    with self._lock:
                        self._completed_count += 1
                        self._results[item_id] = result

                    self._update_progress()
                    return

                except Exception as e:
                    item.retry_count = attempt + 1

                    if attempt < self.config.max_retries_per_item - 1:
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                        continue
                    else:
                        raise

        except Exception as e:
            # Failed
            item.status = ItemStatus.FAILED
            item.error = str(e)
            item.end_time = time.time()
            item.processing_time_ms = (item.end_time - item.start_time) * 1000

            with self._lock:
                self._failed_count += 1
                self._errors[item_id] = str(e)

            logger.error(f"Item {item_id} failed: {e}")
            self._update_progress()

            if self.config.stop_on_error:
                raise

    def _update_progress(self):
        """Update progress callbacks."""
        with self._lock:
            completed = self._completed_count + self._failed_count
            total = self._total_count

        for callback in self._progress_callbacks:
            try:
                callback(completed, total, self._status)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    def _create_result(self) -> BatchResult:
        """Create batch result."""
        processing_time_ms = (time.time() - self._start_time) * 1000 if self._start_time else 0

        return BatchResult(
            id=f"batch_{int(time.time())}",
            status=self._status,
            total_items=self._total_count,
            completed_items=self._completed_count,
            failed_items=self._failed_count,
            skipped_items=0,
            processing_time_ms=processing_time_ms,
            items=list(self._items.values()),
            errors=[
                {"id": item_id, "error": error}
                for item_id, error in self._errors.items()
            ],
            completed_at=time.time()
        )

    def cancel(self):
        """Cancel batch processing."""
        self._status = BatchStatus.CANCELLED
        logger.info("Batch processing cancelled")

    def reset(self):
        """Reset processor state."""
        with self._lock:
            self._items.clear()
            self._results.clear()
            self._errors.clear()
            self._completed_count = 0
            self._failed_count = 0
            self._total_count = 0
            self._status = BatchStatus.PENDING
            self._start_time = None
            self._end_time = None

        logger.info("Batch processor reset")

    def get_item_status(self, item_id: str) -> Optional[ItemStatus]:
        """Get status of a specific item."""
        item = self._items.get(item_id)
        return item.status if item else None

    def get_item_result(self, item_id: str) -> Optional[Any]:
        """Get result of a specific item."""
        return self._results.get(item_id)

    def get_item_error(self, item_id: str) -> Optional[str]:
        """Get error of a specific item."""
        return self._errors.get(item_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "status": self._status.value,
            "total_items": self._total_count,
            "completed_items": self._completed_count,
            "failed_items": self._failed_count,
            "pending_items": self._total_count - self._completed_count - self._failed_count,
            "success_rate": self._completed_count / self._total_count if self._total_count > 0 else 0,
            "processing_time_ms": (time.time() - self._start_time) * 1000 if self._start_time else 0
        }


class QueryBatchProcessor:
    """
    Specialized batch processor for query processing with LLM.
    """

    def __init__(
        self,
        llm_interface: Any,
        retriever: Any,
        config: Optional[BatchConfig] = None
    ):
        """
        Initialize query batch processor.

        Args:
            llm_interface: LLM interface for generation
            retriever: Retriever for document retrieval
            config: Batch configuration
        """
        self.llm_interface = llm_interface
        self.retriever = retriever
        self.config = config or BatchConfig()

        self.batch_processor = BatchProcessor(self.config)
        self._results_cache: Dict[str, Any] = {}

        logger.info("QueryBatchProcessor initialized")

    def process_queries(
        self,
        queries: List[str],
        top_k: int = 5,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        include_sources: bool = True,
        progress_callback: Optional[Callable[[int, int, BatchStatus], None]] = None
    ) -> BatchResult:
        """
        Process multiple queries in batch.

        Args:
            queries: List of query strings
            top_k: Number of documents to retrieve per query
            temperature: LLM temperature
            max_tokens: Max tokens for response
            include_sources: Include source citations
            progress_callback: Progress callback

        Returns:
            BatchResult object
        """
        # Clear previous state
        self.batch_processor.reset()

        # Add queries as items
        for i, query in enumerate(queries):
            item_data = {
                "query": query,
                "top_k": top_k,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "include_sources": include_sources
            }
            self.batch_processor.add_item(f"query_{i}", item_data)

        # Process queries
        return self.batch_processor.process(
            self._process_single_query,
            progress_callback
        )

    async def process_queries_async(
        self,
        queries: List[str],
        top_k: int = 5,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        include_sources: bool = True,
        progress_callback: Optional[Callable[[int, int, BatchStatus], None]] = None
    ) -> BatchResult:
        """
        Process multiple queries asynchronously.

        Args:
            queries: List of query strings
            top_k: Number of documents to retrieve per query
            temperature: LLM temperature
            max_tokens: Max tokens for response
            include_sources: Include source citations
            progress_callback: Progress callback

        Returns:
            BatchResult object
        """
        self.batch_processor.reset()

        for i, query in enumerate(queries):
            item_data = {
                "query": query,
                "top_k": top_k,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "include_sources": include_sources
            }
            self.batch_processor.add_item(f"query_{i}", item_data)

        return await self.batch_processor.process_async(
            self._process_single_query_async,
            progress_callback
        )

    def _process_single_query(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single query (synchronous).

        Args:
            data: Query data

        Returns:
            Query result
        """
        query = data["query"]
        top_k = data["top_k"]
        temperature = data["temperature"]
        max_tokens = data["max_tokens"]
        include_sources = data["include_sources"]

        # Check cache
        cache_key = f"{query}_{top_k}_{temperature}_{max_tokens}"
        if cache_key in self._results_cache:
            return self._results_cache[cache_key]

        with measure("query_processing", {"batch": "true"}):
            # Retrieve documents
            retrieval_results = self.retriever.retrieve(query, top_k=top_k)

            if not retrieval_results:
                result = {
                    "query": query,
                    "answer": "No relevant information found.",
                    "confidence": 0.0,
                    "sources": []
                }
                self._results_cache[cache_key] = result
                return result

            # Generate prompt
            from src.generation.prompt_templates import get_rag_prompt
            context_chunks = [
                {"text": r.text, "source": r.metadata.get("file_path", "Unknown")}
                for r in retrieval_results[:3]
            ]
            prompt = get_rag_prompt(question=query, chunks=context_chunks)

            # Generate response
            response = self.llm_interface.generate_simple(
                prompt,
                system_prompt="You are a helpful assistant that answers questions based on provided documents."
            )

            # Post-process
            from src.generation.response_postprocess import postprocess_response
            processed = postprocess_response(response, aggressive_cleaning=True)

            # Prepare sources
            sources = []
            if include_sources:
                for r in retrieval_results[:5]:
                    sources.append({
                        "text": r.text[:500] + "..." if len(r.text) > 500 else r.text,
                        "score": r.score,
                        "metadata": r.metadata
                    })

            result = {
                "query": query,
                "answer": processed.cleaned_text,
                "confidence": processed.confidence,
                "sources": sources,
                "has_hallucination": processed.has_hallucination
            }

            self._results_cache[cache_key] = result
            return result

    async def _process_single_query_async(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single query (asynchronous).
        """
        # Use sync version in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._process_single_query,
            data
        )

    def get_results(self, batch_result: BatchResult) -> List[Dict[str, Any]]:
        """
        Extract results from batch result.

        Args:
            batch_result: BatchResult object

        Returns:
            List of query results
        """
        results = []
        for item in batch_result.items:
            if item.status == ItemStatus.COMPLETED:
                results.append(item.result)
            elif item.status == ItemStatus.FAILED:
                results.append({
                    "query": item.data.get("query", "Unknown"),
                    "error": item.error,
                    "status": "failed"
                })
        return results


class RateLimiter:
    """
    Rate limiter for controlling request rate.
    """

    def __init__(self, rate_per_second: float):
        """
        Initialize rate limiter.

        Args:
            rate_per_second: Maximum requests per second
        """
        self.rate_per_second = rate_per_second
        self.min_interval = 1.0 / rate_per_second if rate_per_second > 0 else 0
        self._last_request_time: float = 0
        self._lock = threading.Lock()

    def wait(self):
        """Wait until rate limit allows next request."""
        if self.min_interval <= 0:
            return

        with self._lock:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time

            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)

            self._last_request_time = time.time()

    async def wait_async(self):
        """Wait asynchronously until rate limit allows next request."""
        if self.min_interval <= 0:
            return

        with self._lock:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time

            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                await asyncio.sleep(sleep_time)

            self._last_request_time = time.time()


# ============================================================
# Convenience Functions
# ============================================================

def batch_process_queries(
    queries: List[str],
    llm_interface: Any,
    retriever: Any,
    top_k: int = 5,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    include_sources: bool = True,
    max_concurrent: int = 10,
    progress_callback: Optional[Callable[[int, int, BatchStatus], None]] = None
) -> BatchResult:
    """
    Quick function to batch process queries.

    Args:
        queries: List of queries
        llm_interface: LLM interface
        retriever: Retriever
        top_k: Number of documents to retrieve
        temperature: LLM temperature
        max_tokens: Max tokens
        include_sources: Include sources
        max_concurrent: Maximum concurrent requests
        progress_callback: Progress callback

    Returns:
        BatchResult object
    """
    config = BatchConfig(max_concurrent=max_concurrent)
    processor = QueryBatchProcessor(llm_interface, retriever, config)
    return processor.process_queries(
        queries,
        top_k,
        temperature,
        max_tokens,
        include_sources,
        progress_callback
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Mock components for testing
    class MockLLM:
        def generate_simple(self, prompt, system_prompt=None):
            return f"Mock response for: {prompt[:50]}..."

    class MockRetriever:
        def retrieve(self, query, top_k=5):
            return [
                type('obj', (object,), {
                    'text': f"Mock document for {query}",
                    'score': 0.9,
                    'metadata': {'source': 'mock'}
                }) for _ in range(top_k)
            ]

    # Create processor
    llm = MockLLM()
    retriever = MockRetriever()
    processor = QueryBatchProcessor(llm, retriever)

    # Process queries
    queries = [
        "What is machine learning?",
        "How does deep learning work?",
        "What are neural networks?",
        "Explain transformers in AI",
        "What is natural language processing?"
    ]

    print("Processing queries in batch...")

    def progress_callback(completed, total, status):
        print(f"  Progress: {completed}/{total} ({status.value})")

    result = processor.process_queries(
        queries,
        top_k=3,
        progress_callback=progress_callback
    )

    # Get results
    results = processor.get_results(result)

    print(f"\nBatch Results:")
    print(f"  Status: {result.status.value}")
    print(f"  Completed: {result.completed_items}/{result.total_items}")
    print(f"  Processing time: {result.processing_time_ms:.0f}ms")
    print(f"  Success rate: {result.completed_items/result.total_items*100:.1f}%")

    for i, res in enumerate(results[:3]):
        print(f"\n  Query {i+1}: {res.get('query', 'Unknown')}")
        print(f"  Answer: {res.get('answer', '')[:100]}...")
