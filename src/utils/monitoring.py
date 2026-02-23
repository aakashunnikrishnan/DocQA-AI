"""
Performance monitoring module for DocQA AI system.
Provides metrics collection, Prometheus integration, and real-time monitoring.
"""

import os
import time
import json
import logging
import threading
import asyncio
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextlib import contextmanager
import functools
import psutil
import gc

# Try importing Prometheus client
try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, Info,
        generate_latest, REGISTRY, CollectorRegistry,
        CounterMetricFamily, GaugeMetricFamily, HistogramMetricFamily
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("prometheus_client not installed. Install with: pip install prometheus-client")

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """Metric definition."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    type: str = "gauge"  # gauge, counter, histogram


@dataclass
class MetricSummary:
    """Summary of a metric."""
    name: str
    count: int
    sum: float
    min: float
    max: float
    avg: float
    p50: float
    p95: float
    p99: float
    last_value: float
    timestamp: float


class MetricCollector:
    """
    Collect and store metrics with aggregation and statistics.
    """

    def __init__(self, max_history: int = 1000, retention_seconds: int = 3600):
        """
        Initialize metric collector.

        Args:
            max_history: Maximum number of data points per metric
            retention_seconds: Maximum age of data points in seconds
        """
        self.max_history = max_history
        self.retention_seconds = retention_seconds

        self._metrics: Dict[str, deque] = {}
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

        logger.info("MetricCollector initialized")

    def record(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels
        """
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = deque(maxlen=self.max_history)

            metric = Metric(
                name=name,
                value=value,
                timestamp=time.time(),
                labels=labels or {}
            )
            self._metrics[name].append(metric)

    def increment_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """
        Increment a counter metric.

        Args:
            name: Counter name
            value: Increment value
            labels: Optional labels
        """
        with self._lock:
            key = self._get_key(name, labels)
            self._counters[key] += value

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Set a gauge metric.

        Args:
            name: Gauge name
            value: Gauge value
            labels: Optional labels
        """
        with self._lock:
            key = self._get_key(name, labels)
            self._gauges[key] = value

    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """
        Record a histogram value.

        Args:
            name: Histogram name
            value: Value to record
            labels: Optional labels
        """
        with self._lock:
            key = self._get_key(name, labels)
            self._histograms[key].append(value)

            # Limit histogram size
            if len(self._histograms[key]) > 1000:
                self._histograms[key] = self._histograms[key][-500:]

    def _get_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Generate key from name and labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}:{label_str}"

    def _parse_key(self, key: str) -> Tuple[str, Dict[str, str]]:
        """Parse key into name and labels."""
        if ":" not in key:
            return key, {}

        name, label_str = key.split(":", 1)
        labels = {}
        for item in label_str.split(","):
            if "=" in item:
                k, v = item.split("=", 1)
                labels[k] = v
        return name, labels

    def get_metrics(
        self,
        name: Optional[str] = None,
        since: Optional[float] = None
    ) -> List[Metric]:
        """
        Get collected metrics.

        Args:
            name: Filter by metric name
            since: Only return metrics after this timestamp

        Returns:
            List of Metric objects
        """
        with self._lock:
            metrics = []

            for metric_name, values in self._metrics.items():
                if name and metric_name != name:
                    continue

                for metric in values:
                    if since and metric.timestamp < since:
                        continue
                    metrics.append(metric)

            return metrics

    def get_metric_summary(self, name: str) -> Optional[MetricSummary]:
        """
        Get summary statistics for a metric.

        Args:
            name: Metric name

        Returns:
            MetricSummary object or None
        """
        with self._lock:
            if name not in self._metrics:
                return None

            values = [m.value for m in self._metrics[name]]
            if not values:
                return None

            sorted_values = sorted(values)

            return MetricSummary(
                name=name,
                count=len(values),
                sum=sum(values),
                min=min(values),
                max=max(values),
                avg=sum(values) / len(values),
                p50=sorted_values[int(len(values) * 0.5)],
                p95=sorted_values[int(len(values) * 0.95)],
                p99=sorted_values[int(len(values) * 0.99)],
                last_value=values[-1],
                timestamp=time.time()
            )

    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> int:
        """Get counter value."""
        with self._lock:
            key = self._get_key(name, labels)
            return self._counters.get(key, 0)

    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get gauge value."""
        with self._lock:
            key = self._get_key(name, labels)
            return self._gauges.get(key, 0.0)

    def get_histogram_stats(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """Get histogram statistics."""
        with self._lock:
            key = self._get_key(name, labels)
            values = self._histograms.get(key, [])

            if not values:
                return {
                    "count": 0,
                    "sum": 0.0,
                    "avg": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "p50": 0.0,
                    "p95": 0.0,
                    "p99": 0.0
                }

            sorted_values = sorted(values)

            return {
                "count": len(values),
                "sum": sum(values),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "p50": sorted_values[int(len(values) * 0.5)],
                "p95": sorted_values[int(len(values) * 0.95)],
                "p99": sorted_values[int(len(values) * 0.99)]
            }

    def clear(self):
        """Clear all collected metrics."""
        with self._lock:
            self._metrics.clear()
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()

    def _cleanup_loop(self):
        """Background cleanup of old metrics."""
        while True:
            time.sleep(60)  # Run every minute

            with self._lock:
                current_time = time.time()
                cutoff_time = current_time - self.retention_seconds

                # Clean up metrics
                for name in list(self._metrics.keys()):
                    values = self._metrics[name]
                    # Remove old values
                    while values and values[0].timestamp < cutoff_time:
                        values.popleft()

                    # Remove empty deques
                    if not values:
                        del self._metrics[name]


class PerformanceMonitor:
    """
    Performance monitoring with decorators and context managers.
    """

    def __init__(self, collector: Optional[MetricCollector] = None):
        """
        Initialize performance monitor.

        Args:
            collector: Metric collector instance
        """
        self.collector = collector or MetricCollector()
        self.system_metrics = SystemMetrics(self.collector)

        # Start system metrics collection
        self.system_metrics.start()

        logger.info("PerformanceMonitor initialized")

    @contextmanager
    def measure(self, name: str, labels: Optional[Dict[str, str]] = None):
        """
        Context manager for measuring operation duration.

        Args:
            name: Operation name
            labels: Optional labels

        Yields:
            None
        """
        start_time = time.time()

        try:
            yield
        finally:
            duration = (time.time() - start_time) * 1000  # Convert to ms
            self.collector.record_histogram(f"{name}_duration_ms", duration, labels)

    def timed(self, name: str, labels: Optional[Dict[str, str]] = None):
        """
        Decorator for timing functions.

        Args:
            name: Operation name
            labels: Optional labels

        Returns:
            Decorated function
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.measure(name, labels):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    async def timed_async(self, name: str, labels: Optional[Dict[str, str]] = None):
        """
        Async decorator for timing async functions.

        Args:
            name: Operation name
            labels: Optional labels

        Returns:
            Decorated async function
        """
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    return await func(*args, **kwargs)
                finally:
                    duration = (time.time() - start_time) * 1000
                    self.collector.record_histogram(f"{name}_duration_ms", duration, labels)
            return wrapper
        return decorator

    def record_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration_ms: float
    ):
        """
        Record API request metrics.

        Args:
            endpoint: Endpoint path
            method: HTTP method
            status_code: Response status code
            duration_ms: Request duration in milliseconds
        """
        labels = {
            "endpoint": endpoint,
            "method": method,
            "status": str(status_code)
        }

        self.collector.record_histogram("api_request_duration_ms", duration_ms, labels)
        self.collector.increment_counter("api_requests_total", 1, labels)

        if status_code >= 500:
            self.collector.increment_counter("api_errors_total", 1, labels)

    def record_llm_request(
        self,
        provider: str,
        model: str,
        tokens: int,
        duration_ms: float,
        success: bool = True
    ):
        """
        Record LLM request metrics.

        Args:
            provider: LLM provider
            model: Model name
            tokens: Tokens used
            duration_ms: Request duration
            success: Whether request succeeded
        """
        labels = {
            "provider": provider,
            "model": model,
            "success": str(success)
        }

        self.collector.record_histogram("llm_request_duration_ms", duration_ms, labels)
        self.collector.record_histogram("llm_tokens_used", tokens, labels)
        self.collector.increment_counter("llm_requests_total", 1, labels)

        if not success:
            self.collector.increment_counter("llm_errors_total", 1, labels)

    def record_embedding_request(
        self,
        model: str,
        tokens: int,
        duration_ms: float,
        success: bool = True
    ):
        """
        Record embedding request metrics.

        Args:
            model: Embedding model
            tokens: Tokens used
            duration_ms: Request duration
            success: Whether request succeeded
        """
        labels = {
            "model": model,
            "success": str(success)
        }

        self.collector.record_histogram("embedding_request_duration_ms", duration_ms, labels)
        self.collector.record_histogram("embedding_tokens_used", tokens, labels)
        self.collector.increment_counter("embedding_requests_total", 1, labels)

        if not success:
            self.collector.increment_counter("embedding_errors_total", 1, labels)

    def record_retrieval(
        self,
        method: str,
        top_k: int,
        duration_ms: float
    ):
        """
        Record retrieval metrics.

        Args:
            method: Retrieval method
            top_k: Number of results
            duration_ms: Retrieval duration
        """
        labels = {"method": method}

        self.collector.record_histogram("retrieval_duration_ms", duration_ms, labels)
        self.collector.record_histogram("retrieval_top_k", top_k, labels)
        self.collector.increment_counter("retrieval_requests_total", 1, labels)

    def record_cache_hit(self, cache_type: str):
        """
        Record cache hit.

        Args:
            cache_type: Cache type
        """
        self.collector.increment_counter("cache_hits_total", 1, {"type": cache_type})

    def record_cache_miss(self, cache_type: str):
        """
        Record cache miss.

        Args:
            cache_type: Cache type
        """
        self.collector.increment_counter("cache_misses_total", 1, {"type": cache_type})

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current performance statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            "system": self.system_metrics.get_stats(),
            "api_requests": {
                "total": self.collector.get_counter("api_requests_total"),
                "errors": self.collector.get_counter("api_errors_total")
            },
            "llm_requests": {
                "total": self.collector.get_counter("llm_requests_total"),
                "errors": self.collector.get_counter("llm_errors_total")
            },
            "embedding_requests": {
                "total": self.collector.get_counter("embedding_requests_total"),
                "errors": self.collector.get_counter("embedding_errors_total")
            },
            "cache": {
                "hits": self.collector.get_counter("cache_hits_total"),
                "misses": self.collector.get_counter("cache_misses_total")
            }
        }


class SystemMetrics:
    """
    Collect system-level metrics (CPU, memory, disk, network).
    """

    def __init__(self, collector: MetricCollector, interval: int = 10):
        """
        Initialize system metrics collector.

        Args:
            collector: Metric collector
            interval: Collection interval in seconds
        """
        self.collector = collector
        self.interval = interval
        self._running = False
        self._thread = None

        logger.info(f"SystemMetrics initialized with interval={interval}s")

    def start(self):
        """Start collecting system metrics."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()

        logger.info("System metrics collection started")

    def stop(self):
        """Stop collecting system metrics."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

        logger.info("System metrics collection stopped")

    def _collect_loop(self):
        """Background collection loop."""
        while self._running:
            try:
                self._collect()
            except Exception as e:
                logger.warning(f"System metrics collection failed: {e}")

            time.sleep(self.interval)

    def _collect(self):
        """Collect current system metrics."""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0)
        cpu_count = psutil.cpu_count()
        self.collector.set_gauge("system_cpu_usage_percent", cpu_percent)
        self.collector.set_gauge("system_cpu_count", cpu_count)

        # Memory
        memory = psutil.virtual_memory()
        self.collector.set_gauge("system_memory_total_bytes", memory.total)
        self.collector.set_gauge("system_memory_used_bytes", memory.used)
        self.collector.set_gauge("system_memory_available_bytes", memory.available)
        self.collector.set_gauge("system_memory_usage_percent", memory.percent)

        # Disk
        disk = psutil.disk_usage('/')
        self.collector.set_gauge("system_disk_total_bytes", disk.total)
        self.collector.set_gauge("system_disk_used_bytes", disk.used)
        self.collector.set_gauge("system_disk_free_bytes", disk.free)
        self.collector.set_gauge("system_disk_usage_percent", disk.percent)

        # Network
        net = psutil.net_io_counters()
        self.collector.set_gauge("system_network_bytes_sent", net.bytes_sent)
        self.collector.set_gauge("system_network_bytes_recv", net.bytes_recv)
        self.collector.set_gauge("system_network_packets_sent", net.packets_sent)
        self.collector.set_gauge("system_network_packets_recv", net.packets_recv)

        # Process
        process = psutil.Process()
        self.collector.set_gauge("process_cpu_usage_percent", process.cpu_percent())

        process_memory = process.memory_info()
        self.collector.set_gauge("process_memory_rss_bytes", process_memory.rss)
        self.collector.set_gauge("process_memory_vms_bytes", process_memory.vms)

        # Threads and connections
        self.collector.set_gauge("process_thread_count", process.num_threads())
        self.collector.set_gauge("process_open_files", len(process.open_files()))
        self.collector.set_gauge("process_connections", len(process.connections()))

        # Garbage collection
        gc_count = gc.get_count()
        self.collector.set_gauge("gc_generation_0", gc_count[0])
        self.collector.set_gauge("gc_generation_1", gc_count[1])
        self.collector.set_gauge("gc_generation_2", gc_count[2])

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current system statistics.

        Returns:
            Dictionary of system statistics
        """
        return {
            "cpu": {
                "usage_percent": psutil.cpu_percent(interval=0),
                "count": psutil.cpu_count()
            },
            "memory": {
                "total_mb": psutil.virtual_memory().total / 1024 / 1024,
                "used_mb": psutil.virtual_memory().used / 1024 / 1024,
                "available_mb": psutil.virtual_memory().available / 1024 / 1024,
                "percent": psutil.virtual_memory().percent
            },
            "disk": {
                "total_gb": psutil.disk_usage('/').total / 1024 / 1024 / 1024,
                "used_gb": psutil.disk_usage('/').used / 1024 / 1024 / 1024,
                "free_gb": psutil.disk_usage('/').free / 1024 / 1024 / 1024,
                "percent": psutil.disk_usage('/').percent
            },
            "process": {
                "cpu_percent": psutil.Process().cpu_percent(),
                "memory_mb": psutil.Process().memory_info().rss / 1024 / 1024,
                "threads": psutil.Process().num_threads()
            }
        }


class PrometheusExporter:
    """
    Export metrics to Prometheus format.
    """

    def __init__(self, collector: MetricCollector, prefix: str = "docqa"):
        """
        Initialize Prometheus exporter.

        Args:
            collector: Metric collector
            prefix: Metric prefix
        """
        self.collector = collector
        self.prefix = prefix

        if not PROMETHEUS_AVAILABLE:
            logger.warning("prometheus_client not available")

    def generate_metrics(self) -> str:
        """
        Generate Prometheus metrics format.

        Returns:
            Prometheus metrics string
        """
        if not PROMETHEUS_AVAILABLE:
            return "# Prometheus client not available\n"

        try:
            registry = CollectorRegistry()

            # Create metric families
            # API metrics
            api_counter = CounterMetricFamily(
                f"{self.prefix}_api_requests_total",
                "Total API requests",
                labels=["endpoint", "method", "status"]
            )

            # LLM metrics
            llm_counter = CounterMetricFamily(
                f"{self.prefix}_llm_requests_total",
                "Total LLM requests",
                labels=["provider", "model", "success"]
            )

            # Cache metrics
            cache_counter = CounterMetricFamily(
                f"{self.prefix}_cache_hits_total",
                "Total cache hits",
                labels=["type"]
            )
            cache_miss_counter = CounterMetricFamily(
                f"{self.prefix}_cache_misses_total",
                "Total cache misses",
                labels=["type"]
            )

            # System metrics (gauges)
            system_gauges = {}

            # Process counters
            for key, value in self.collector._counters.items():
                name, labels = self.collector._parse_key(key)

                if name == "api_requests_total":
                    api_counter.add_metric([labels.get("endpoint", ""), labels.get("method", ""), labels.get("status", "")], value)
                elif name == "llm_requests_total":
                    llm_counter.add_metric([labels.get("provider", ""), labels.get("model", ""), labels.get("success", "")], value)
                elif name == "cache_hits_total":
                    cache_counter.add_metric([labels.get("type", "")], value)
                elif name == "cache_misses_total":
                    cache_miss_counter.add_metric([labels.get("type", "")], value)

            # System gauges
            for key, value in self.collector._gauges.items():
                name, labels = self.collector._parse_key(key)
                if name.startswith("system_") or name.startswith("process_"):
                    if name not in system_gauges:
                        system_gauges[name] = GaugeMetricFamily(
                            f"{self.prefix}_{name}",
                            f"System metric {name}",
                            labels=list(labels.keys()) if labels else []
                        )
                    system_gauges[name].add_metric(
                        list(labels.values()) if labels else [],
                        value
                    )

            # Register metrics
            registry.register(api_counter)
            registry.register(llm_counter)
            registry.register(cache_counter)
            registry.register(cache_miss_counter)

            for gauge in system_gauges.values():
                registry.register(gauge)

            return generate_latest(registry).decode('utf-8')

        except Exception as e:
            logger.error(f"Failed to generate Prometheus metrics: {e}")
            return "# Error generating metrics\n"


# ============================================================
# Global Instance
# ============================================================

_performance_monitor: Optional[PerformanceMonitor] = None
_metric_collector: Optional[MetricCollector] = None


def get_metric_collector() -> MetricCollector:
    """Get global metric collector instance."""
    global _metric_collector
    if _metric_collector is None:
        _metric_collector = MetricCollector()
    return _metric_collector


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor(get_metric_collector())
    return _performance_monitor


# ============================================================
# Convenience Functions
# ============================================================

@contextmanager
def measure(name: str, labels: Optional[Dict[str, str]] = None):
    """
    Context manager for measuring operation duration.

    Args:
        name: Operation name
        labels: Optional labels

    Yields:
        None
    """
    monitor = get_performance_monitor()
    with monitor.measure(name, labels):
        yield


def timed(name: str, labels: Optional[Dict[str, str]] = None):
    """
    Decorator for timing functions.

    Args:
        name: Operation name
        labels: Optional labels

    Returns:
        Decorated function
    """
    monitor = get_performance_monitor()
    return monitor.timed(name, labels)


def timed_async(name: str, labels: Optional[Dict[str, str]] = None):
    """
    Async decorator for timing async functions.

    Args:
        name: Operation name
        labels: Optional labels

    Returns:
        Decorated async function
    """
    monitor = get_performance_monitor()
    return monitor.timed_async(name, labels)


def record_request(endpoint: str, method: str, status_code: int, duration_ms: float):
    """Record API request metrics."""
    monitor = get_performance_monitor()
    monitor.record_request(endpoint, method, status_code, duration_ms)


def record_llm_request(provider: str, model: str, tokens: int, duration_ms: float, success: bool = True):
    """Record LLM request metrics."""
    monitor = get_performance_monitor()
    monitor.record_llm_request(provider, model, tokens, duration_ms, success)


def record_embedding_request(model: str, tokens: int, duration_ms: float, success: bool = True):
    """Record embedding request metrics."""
    monitor = get_performance_monitor()
    monitor.record_embedding_request(model, tokens, duration_ms, success)


def record_retrieval(method: str, top_k: int, duration_ms: float):
    """Record retrieval metrics."""
    monitor = get_performance_monitor()
    monitor.record_retrieval(method, top_k, duration_ms)


def get_metrics() -> Dict[str, Any]:
    """Get current metrics."""
    monitor = get_performance_monitor()
    return monitor.get_stats()


def get_prometheus_metrics() -> str:
    """Get Prometheus-formatted metrics."""
    collector = get_metric_collector()
    exporter = PrometheusExporter(collector)
    return exporter.generate_metrics()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Testing Performance Monitoring...")

    # Get monitor
    monitor = get_performance_monitor()
    collector = get_metric_collector()

    # Test metrics
    print("\n1. Recording metrics...")

    # Record some metrics
    collector.record("test_metric", 42.5)
    collector.increment_counter("test_counter", 1)
    collector.set_gauge("test_gauge", 3.14)
    collector.record_histogram("test_histogram", 100)
    collector.record_histogram("test_histogram", 200)
    collector.record_histogram("test_histogram", 300)

    # Test context manager
    with measure("test_operation"):
        time.sleep(0.1)

    # Test decorator
    @timed("test_decorator")
    def test_function():
        time.sleep(0.05)
        return "done"

    test_function()

    # Test system metrics
    print("\n2. System metrics collected...")
    time.sleep(2)

    # Get stats
    stats = monitor.get_stats()
    print(f"System CPU: {stats['system']['cpu']['usage_percent']}%")
    print(f"System Memory: {stats['system']['memory']['percent']}%")

    # Get metric summaries
    print("\n3. Metric summaries...")
    for name in ["test_metric", "test_histogram"]:
        summary = collector.get_metric_summary(name)
        if summary:
            print(f"  {name}: avg={summary.avg:.2f}, min={summary.min:.2f}, max={summary.max:.2f}")

    # Get Prometheus metrics
    print("\n4. Prometheus metrics...")
    prometheus_metrics = get_prometheus_metrics()
    print(prometheus_metrics[:500] + "...")

    print("\n✅ Performance monitoring ready!")
