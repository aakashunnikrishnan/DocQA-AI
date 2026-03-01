#!/usr/bin/env python3
"""
Load testing script for the DocQA AI system.
Tests API performance under various load conditions with configurable parameters.
"""

import os
import sys
import json
import time
import asyncio
import argparse
import random
import statistics
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import threading
import concurrent.futures

import aiohttp
import requests
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


# ============================================================
# Data Models
# ============================================================

@dataclass
class TestResult:
    """Result of a single test request."""
    success: bool
    status_code: int
    response_time_ms: float
    error: Optional[str] = None
    response_size: int = 0
    timestamp: float = field(default_factory=time.time)
    tokens_used: int = 0
    endpoint: str = ""
    method: str = "POST"


@dataclass
class LoadTestSummary:
    """Summary of load test results."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    min_response_time_ms: float
    max_response_time_ms: float
    avg_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    requests_per_second: float
    total_duration_seconds: float
    errors_by_status: Dict[str, int]
    endpoints: Dict[str, Dict[str, Any]]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": self.success_rate,
            "min_response_time_ms": self.min_response_time_ms,
            "max_response_time_ms": self.max_response_time_ms,
            "avg_response_time_ms": self.avg_response_time_ms,
            "p50_response_time_ms": self.p50_response_time_ms,
            "p95_response_time_ms": self.p95_response_time_ms,
            "p99_response_time_ms": self.p99_response_time_ms,
            "requests_per_second": self.requests_per_second,
            "total_duration_seconds": self.total_duration_seconds,
            "errors_by_status": self.errors_by_status,
            "endpoints": self.endpoints,
            "timestamp": self.timestamp
        }


# ============================================================
# Test Scenarios
# ============================================================

class TestScenario:
    """Test scenario configurations."""

    # Sample queries for load testing
    QUERIES = [
        "What is machine learning?",
        "How does deep learning work?",
        "What are neural networks?",
        "Explain natural language processing.",
        "What is computer vision?",
        "How does reinforcement learning work?",
        "What are transformers in AI?",
        "Explain the concept of embeddings.",
        "What is transfer learning?",
        "How does attention mechanism work?",
        "What is the difference between supervised and unsupervised learning?",
        "What is RAG in AI?",
        "How do large language models work?",
        "What is fine-tuning in machine learning?",
        "What is a vector database?"
    ]

    @classmethod
    def get_query(cls) -> str:
        """Get a random query."""
        return random.choice(cls.QUERIES)

    @classmethod
    def get_query_batch(cls, size: int) -> List[str]:
        """Get a batch of random queries."""
        return random.sample(cls.QUERIES, min(size, len(cls.QUERIES)))


# ============================================================
# Load Test Runner
# ============================================================

class LoadTestRunner:
    """
    Run load tests against the DocQA API.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        concurrency: int = 10,
        total_requests: int = 100,
        timeout: int = 60,
        warmup_requests: int = 5
    ):
        """
        Initialize load test runner.

        Args:
            base_url: Base URL of the API
            api_key: API key for authentication
            concurrency: Number of concurrent requests
            total_requests: Total number of requests to make
            timeout: Request timeout in seconds
            warmup_requests: Number of warmup requests
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.concurrency = concurrency
        self.total_requests = total_requests
        self.timeout = timeout
        self.warmup_requests = warmup_requests

        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

        self.results: List[TestResult] = []
        self._lock = threading.Lock()

        logger.info(f"LoadTestRunner initialized: concurrency={concurrency}, "
                   f"total_requests={total_requests}, timeout={timeout}")

    def _check_health(self) -> bool:
        """Check if the API is healthy."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ API is healthy")
                return True
            else:
                logger.error(f"❌ API returned status {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ API health check failed: {e}")
            return False

    def _warmup(self):
        """Perform warmup requests."""
        logger.info(f"Running {self.warmup_requests} warmup requests...")

        for i in range(self.warmup_requests):
            query = TestScenario.get_query()
            try:
                response = requests.post(
                    f"{self.base_url}/api/v1/query",
                    json={"question": query, "top_k": 3},
                    headers=self.headers,
                    timeout=self.timeout
                )
                if response.status_code == 200:
                    logger.debug(f"Warmup {i+1}: OK")
            except Exception as e:
                logger.debug(f"Warmup {i+1}: Failed - {e}")

        logger.info("✅ Warmup complete")

    async def _make_request(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        data: Dict[str, Any]
    ) -> TestResult:
        """
        Make a single request to the API.

        Args:
            session: aiohttp session
            endpoint: API endpoint
            data: Request data

        Returns:
            TestResult object
        """
        start_time = time.time()

        try:
            async with session.post(
                f"{self.base_url}{endpoint}",
                json=data,
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                response_time = (time.time() - start_time) * 1000

                if response.status == 200:
                    try:
                        response_data = await response.json()
                        response_size = len(str(response_data))
                        tokens_used = response_data.get("tokens_used", 0)
                    except Exception:
                        response_size = 0
                        tokens_used = 0

                    return TestResult(
                        success=True,
                        status_code=response.status,
                        response_time_ms=response_time,
                        response_size=response_size,
                        tokens_used=tokens_used,
                        endpoint=endpoint
                    )
                else:
                    error_text = await response.text()
                    return TestResult(
                        success=False,
                        status_code=response.status,
                        response_time_ms=response_time,
                        error=error_text[:200],
                        endpoint=endpoint
                    )

        except asyncio.TimeoutError:
            return TestResult(
                success=False,
                status_code=0,
                response_time_ms=(time.time() - start_time) * 1000,
                error="Timeout",
                endpoint=endpoint
            )
        except Exception as e:
            return TestResult(
                success=False,
                status_code=0,
                response_time_ms=(time.time() - start_time) * 1000,
                error=str(e)[:200],
                endpoint=endpoint
            )

    async def _run_queries(self, query_type: str = "single", **kwargs):
        """
        Run query load test.

        Args:
            query_type: Type of query test ('single', 'batch')
            **kwargs: Additional arguments
        """
        logger.info(f"Running {query_type} query load test...")

        semaphore = asyncio.Semaphore(self.concurrency)

        async def process_request(session, query, idx):
            async with semaphore:
                if query_type == "single":
                    data = {
                        "question": query,
                        "top_k": kwargs.get("top_k", 5),
                        "temperature": kwargs.get("temperature", 0.7)
                    }
                    return await self._make_request(session, "/api/v1/query", data)
                else:
                    # Batch query
                    data = {
                        "queries": query,  # query is a list for batch
                        "top_k": kwargs.get("top_k", 5),
                        "temperature": kwargs.get("temperature", 0.7)
                    }
                    return await self._make_request(session, "/api/v1/query/batch", data)

        async with aiohttp.ClientSession() as session:
            tasks = []
            queries = []

            for i in range(self.total_requests):
                if query_type == "single":
                    query = TestScenario.get_query()
                    queries.append(query)
                else:
                    batch_size = kwargs.get("batch_size", 5)
                    query = TestScenario.get_query_batch(batch_size)
                    queries.append(query)

                tasks.append(process_request(session, query, i))

            # Execute with progress bar
            results = []
            with tqdm(total=len(tasks), desc="Making requests") as pbar:
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    results.append(result)
                    pbar.update(1)

            with self._lock:
                self.results.extend(results)

    async def _run_ingestion(self, file_paths: List[str], **kwargs):
        """
        Run document ingestion load test.

        Args:
            file_paths: List of file paths to ingest
            **kwargs: Additional arguments
        """
        logger.info(f"Running ingestion load test with {len(file_paths)} files...")

        semaphore = asyncio.Semaphore(min(self.concurrency, 5))  # Limit ingestion concurrency

        async def process_ingestion(session, file_path, idx):
            async with semaphore:
                start_time = time.time()

                try:
                    with open(file_path, 'rb') as f:
                        files = {'files': (Path(file_path).name, f, 'text/plain')}
                        data = {
                            'chunk_size': kwargs.get('chunk_size', 800),
                            'chunk_overlap': kwargs.get('chunk_overlap', 150),
                            'chunking_strategy': kwargs.get('chunking_strategy', 'adaptive')
                        }

                        # Using multipart form data with aiohttp
                        form = aiohttp.FormData()
                        for key, value in data.items():
                            form.add_field(key, str(value))

                        # Add file
                        form.add_field('files', f, filename=Path(file_path).name)

                        async with session.post(
                            f"{self.base_url}/api/v1/documents/ingest",
                            data=form,
                            headers={"Authorization": self.headers.get("Authorization", "")},
                            timeout=aiohttp.ClientTimeout(total=300)
                        ) as response:
                            response_time = (time.time() - start_time) * 1000

                            if response.status == 200:
                                return TestResult(
                                    success=True,
                                    status_code=response.status,
                                    response_time_ms=response_time,
                                    endpoint="/api/v1/documents/ingest"
                                )
                            else:
                                error_text = await response.text()
                                return TestResult(
                                    success=False,
                                    status_code=response.status,
                                    response_time_ms=response_time,
                                    error=error_text[:200],
                                    endpoint="/api/v1/documents/ingest"
                                )

                except Exception as e:
                    return TestResult(
                        success=False,
                        status_code=0,
                        response_time_ms=(time.time() - start_time) * 1000,
                        error=str(e)[:200],
                        endpoint="/api/v1/documents/ingest"
                    )

        async with aiohttp.ClientSession() as session:
            tasks = []
            for i, file_path in enumerate(file_paths[:self.total_requests]):
                tasks.append(process_ingestion(session, file_path, i))

            results = []
            with tqdm(total=len(tasks), desc="Ingesting documents") as pbar:
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    results.append(result)
                    pbar.update(1)

            with self._lock:
                self.results.extend(results)

    def run_query_test(
        self,
        query_type: str = "single",
        top_k: int = 5,
        temperature: float = 0.7,
        batch_size: int = 5
    ) -> LoadTestSummary:
        """
        Run query load test.

        Args:
            query_type: Type of query test ('single', 'batch')
            top_k: Number of documents to retrieve
            temperature: LLM temperature
            batch_size: Batch size for batch queries

        Returns:
            LoadTestSummary
        """
        # Check API health
        if not self._check_health():
            raise RuntimeError("API is not healthy")

        # Warmup
        self._warmup()

        # Reset results
        self.results = []

        # Run test
        asyncio.run(self._run_queries(
            query_type=query_type,
            top_k=top_k,
            temperature=temperature,
            batch_size=batch_size
        ))

        return self._generate_summary()

    def run_ingestion_test(self, file_paths: List[str], **kwargs) -> LoadTestSummary:
        """
        Run ingestion load test.

        Args:
            file_paths: List of file paths to ingest
            **kwargs: Additional arguments

        Returns:
            LoadTestSummary
        """
        if not file_paths:
            raise ValueError("No file paths provided")

        # Check API health
        if not self._check_health():
            raise RuntimeError("API is not healthy")

        # Warmup
        self._warmup()

        # Reset results
        self.results = []

        # Run test
        asyncio.run(self._run_ingestion(file_paths, **kwargs))

        return self._generate_summary()

    def _generate_summary(self) -> LoadTestSummary:
        """Generate summary statistics from results."""
        if not self.results:
            return LoadTestSummary(
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                success_rate=0.0,
                min_response_time_ms=0.0,
                max_response_time_ms=0.0,
                avg_response_time_ms=0.0,
                p50_response_time_ms=0.0,
                p95_response_time_ms=0.0,
                p99_response_time_ms=0.0,
                requests_per_second=0.0,
                total_duration_seconds=0.0,
                errors_by_status={},
                endpoints={}
            )

        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        response_times = [r.response_time_ms for r in self.results if r.success]
        sorted_times = sorted(response_times) if response_times else [0]

        # Calculate duration
        timestamps = [r.timestamp for r in self.results]
        duration = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0

        # Group by endpoint
        endpoints = defaultdict(lambda: {"count": 0, "success": 0, "failed": 0, "response_times": []})
        for r in self.results:
            endpoints[r.endpoint]["count"] += 1
            if r.success:
                endpoints[r.endpoint]["success"] += 1
                endpoints[r.endpoint]["response_times"].append(r.response_time_ms)
            else:
                endpoints[r.endpoint]["failed"] += 1

        endpoints_summary = {}
        for endpoint, data in endpoints.items():
            times = data["response_times"]
            endpoints_summary[endpoint] = {
                "total": data["count"],
                "success": data["success"],
                "failed": data["failed"],
                "success_rate": data["success"] / data["count"] if data["count"] > 0 else 0,
                "avg_response_time_ms": statistics.mean(times) if times else 0,
                "min_response_time_ms": min(times) if times else 0,
                "max_response_time_ms": max(times) if times else 0,
                "p95_response_time_ms": sorted_times[int(len(sorted_times) * 0.95)] if sorted_times else 0
            }

        # Errors by status
        errors_by_status = defaultdict(int)
        for r in failed:
            errors_by_status[str(r.status_code)] += 1

        return LoadTestSummary(
            total_requests=len(self.results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            success_rate=len(successful) / len(self.results) if self.results else 0,
            min_response_time_ms=min(sorted_times) if sorted_times else 0,
            max_response_time_ms=max(sorted_times) if sorted_times else 0,
            avg_response_time_ms=statistics.mean(response_times) if response_times else 0,
            p50_response_time_ms=sorted_times[int(len(sorted_times) * 0.5)] if sorted_times else 0,
            p95_response_time_ms=sorted_times[int(len(sorted_times) * 0.95)] if sorted_times else 0,
            p99_response_time_ms=sorted_times[int(len(sorted_times) * 0.99)] if sorted_times else 0,
            requests_per_second=len(self.results) / duration if duration > 0 else 0,
            total_duration_seconds=duration,
            errors_by_status=dict(errors_by_status),
            endpoints=endpoints_summary
        )


# ============================================================
# CLI Interface
# ============================================================

def create_sample_files(output_dir: str, num_files: int = 10):
    """Create sample text files for ingestion testing."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sample_content = """
    Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to find patterns in data and make predictions or decisions.

    Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn hierarchical representations of data. Neural networks are computational models inspired by biological neural networks.

    Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and human language. It enables computers to understand, interpret, and generate human language.

    Computer vision is a field of artificial intelligence that enables computers to interpret and understand visual information from the world. It involves acquiring, processing, analyzing, and understanding images and video.

    RAG (Retrieval-Augmented Generation) is a framework that combines retrieval-based and generation-based approaches to improve the quality and accuracy of AI responses.
    """

    for i in range(num_files):
        file_path = output_path / f"sample_{i+1}.txt"
        with open(file_path, 'w') as f:
            f.write(f"Document {i+1}\n{sample_content}\n")
        logger.info(f"Created sample file: {file_path}")

    return [str(output_path / f"sample_{i+1}.txt") for i in range(num_files)]


def print_summary(summary: LoadTestSummary):
    """Print load test summary."""
    print("\n" + "=" * 60)
    print("LOAD TEST SUMMARY")
    print("=" * 60)

    print(f"\n📊 Overall Results:")
    print(f"  Total Requests:     {summary.total_requests}")
    print(f"  Successful:         {summary.successful_requests}")
    print(f"  Failed:             {summary.failed_requests}")
    print(f"  Success Rate:       {summary.success_rate:.2%}")

    print(f"\n⏱️  Response Times (ms):")
    print(f"  Min:                {summary.min_response_time_ms:.2f}")
    print(f"  Max:                {summary.max_response_time_ms:.2f}")
    print(f"  Average:            {summary.avg_response_time_ms:.2f}")
    print(f"  P50:                {summary.p50_response_time_ms:.2f}")
    print(f"  P95:                {summary.p95_response_time_ms:.2f}")
    print(f"  P99:                {summary.p99_response_time_ms:.2f}")

    print(f"\n🚀 Throughput:")
    print(f"  Requests/sec:       {summary.requests_per_second:.2f}")
    print(f"  Total Duration:     {summary.total_duration_seconds:.2f}s")

    if summary.errors_by_status:
        print(f"\n❌ Errors by Status:")
        for status, count in summary.errors_by_status.items():
            print(f"  {status}:            {count}")

    print(f"\n📈 Endpoint Breakdown:")
    for endpoint, stats in summary.endpoints.items():
        print(f"  {endpoint}:")
        print(f"    Total: {stats['total']}, Success: {stats['success']}, Failed: {stats['failed']}")
        print(f"    Avg: {stats['avg_response_time_ms']:.2f}ms, P95: {stats['p95_response_time_ms']:.2f}ms")

    print("\n" + "=" * 60)


def main():
    """Main entry point for load test script."""
    parser = argparse.ArgumentParser(
        description="Load test DocQA AI API",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key for authentication"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent requests (default: 10)"
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Total number of requests (default: 100)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds (default: 60)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup requests (default: 5)"
    )

    # Test type
    parser.add_argument(
        "--test-type",
        type=str,
        choices=["query", "ingestion"],
        default="query",
        help="Type of test to run (default: query)"
    )

    # Query test options
    parser.add_argument(
        "--query-type",
        type=str,
        choices=["single", "batch"],
        default="single",
        help="Type of query test (default: single)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Batch size for batch queries (default: 5)"
    )

    # Ingestion test options
    parser.add_argument(
        "--file-dir",
        type=str,
        help="Directory containing files for ingestion test"
    )
    parser.add_argument(
        "--num-files",
        type=int,
        default=10,
        help="Number of sample files to create (default: 10)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Chunk size for ingestion (default: 800)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=150,
        help="Chunk overlap for ingestion (default: 150)"
    )
    parser.add_argument(
        "--chunking-strategy",
        type=str,
        default="adaptive",
        help="Chunking strategy (default: adaptive)"
    )

    # Output options
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output file to save results (JSON)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet output"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level="WARNING" if args.quiet else "INFO", log_to_file=False)

    # Create runner
    runner = LoadTestRunner(
        base_url=args.url,
        api_key=args.api_key,
        concurrency=args.concurrency,
        total_requests=args.requests,
        timeout=args.timeout,
        warmup_requests=args.warmup
    )

    # Run test
    try:
        if args.test_type == "query":
            summary = runner.run_query_test(
                query_type=args.query_type,
                top_k=args.top_k,
                temperature=args.temperature,
                batch_size=args.batch_size
            )
        else:
            # Ingestion test
            if args.file_dir:
                file_paths = [str(p) for p in Path(args.file_dir).glob("*.txt")]
                if not file_paths:
                    raise ValueError(f"No .txt files found in {args.file_dir}")
            else:
                # Create sample files
                sample_dir = Path("./data/sample_files")
                file_paths = create_sample_files(str(sample_dir), args.num_files)

            summary = runner.run_ingestion_test(
                file_paths=file_paths,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                chunking_strategy=args.chunking_strategy
            )

        # Print results
        if not args.quiet:
            print_summary(summary)

        # Save results
        if args.output_file:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(summary.to_dict(), f, indent=2)

            if not args.quiet:
                print(f"\n📁 Results saved to: {output_path}")

    except Exception as e:
        logger.error(f"Load test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
