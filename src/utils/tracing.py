"""
OpenTelemetry tracing module for DocQA AI system.
Provides distributed tracing, performance monitoring, and observability.
"""

import os
import time
import json
import logging
from typing import Dict, Any, Optional, List, Callable, Union
from contextlib import contextmanager
from functools import wraps
import asyncio
from datetime import datetime

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Try importing OpenTelemetry
try:
    from opentelemetry import trace
    from opentelemetry.trace import Span, Tracer, SpanKind, Status, StatusCode
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    from opentelemetry.propagate import extract, inject
    from opentelemetry.context import Context, get_current, attach, detach
    from opentelemetry.trace import set_span_in_context
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor
    )
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
    from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
    OPENTELEMETRY_AVAILABLE = True
except ImportError as e:
    OPENTELEMETRY_AVAILABLE = False
    logger.warning(f"OpenTelemetry not available: {e}")


class TracingConfig:
    """Configuration for OpenTelemetry tracing."""

    def __init__(
        self,
        service_name: str = "docqa-ai",
        service_version: str = "1.0.0",
        enabled: bool = True,
        exporter: str = "console",  # console, otlp, none
        otlp_endpoint: Optional[str] = None,
        otlp_insecure: bool = True,
        sample_rate: float = 1.0,
        environment: str = "development",
        enable_fastapi: bool = True,
        enable_httpx: bool = True,
        enable_aiohttp: bool = True,
        enable_asyncio: bool = False,
        batch_export: bool = True,
        max_export_batch_size: int = 512,
        schedule_delay_millis: int = 5000
    ):
        """
        Initialize tracing configuration.

        Args:
            service_name: Name of the service
            service_version: Version of the service
            enabled: Whether tracing is enabled
            exporter: Exporter type ('console', 'otlp', 'none')
            otlp_endpoint: OTLP endpoint URL
            otlp_insecure: Whether to use insecure connection
            sample_rate: Sampling rate (0.0 to 1.0)
            environment: Environment name
            enable_fastapi: Enable FastAPI instrumentation
            enable_httpx: Enable HTTPX instrumentation
            enable_aiohttp: Enable aiohttp instrumentation
            enable_asyncio: Enable asyncio instrumentation
            batch_export: Whether to use batch export
            max_export_batch_size: Maximum batch size for export
            schedule_delay_millis: Schedule delay for batch export
        """
        self.service_name = service_name
        self.service_version = service_version
        self.enabled = enabled
        self.exporter = exporter
        self.otlp_endpoint = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        self.otlp_insecure = otlp_insecure
        self.sample_rate = sample_rate
        self.environment = environment
        self.enable_fastapi = enable_fastapi
        self.enable_httpx = enable_httpx
        self.enable_aiohttp = enable_aiohttp
        self.enable_asyncio = enable_asyncio
        self.batch_export = batch_export
        self.max_export_batch_size = max_export_batch_size
        self.schedule_delay_millis = schedule_delay_millis


class TracingManager:
    """
    Manages OpenTelemetry tracing setup and instrumentation.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if TracingManager._initialized:
            return

        self.config: Optional[TracingConfig] = None
        self.tracer: Optional[Tracer] = None
        self.provider: Optional[TracerProvider] = None
        self.is_enabled = False
        self._span_processors = []

        TracingManager._initialized = True

    def configure(self, config: Optional[TracingConfig] = None) -> 'TracingManager':
        """
        Configure tracing.

        Args:
            config: Tracing configuration

        Returns:
            Self for chaining
        """
        self.config = config or TracingConfig()

        if not self.config.enabled or not OPENTELEMETRY_AVAILABLE:
            self.is_enabled = False
            logger.info("Tracing is disabled")
            return self

        try:
            self._setup_tracing()
            self.is_enabled = True
            logger.info(f"Tracing configured: service={self.config.service_name}, "
                       f"exporter={self.config.exporter}")
        except Exception as e:
            logger.error(f"Failed to configure tracing: {e}")
            self.is_enabled = False

        return self

    def _setup_tracing(self):
        """Setup OpenTelemetry tracing."""
        if not OPENTELEMETRY_AVAILABLE:
            raise ImportError("OpenTelemetry not installed")

        # Create resource
        resource = Resource.create({
            SERVICE_NAME: self.config.service_name,
            SERVICE_VERSION: self.config.service_version,
            "environment": self.config.environment,
            "service.instance.id": os.getenv("HOSTNAME", os.uname().nodename)
        })

        # Create tracer provider
        self.provider = TracerProvider(resource=resource)

        # Setup exporter
        exporter = self._create_exporter()
        if exporter:
            if self.config.batch_export:
                span_processor = BatchSpanProcessor(
                    exporter,
                    max_export_batch_size=self.config.max_export_batch_size,
                    schedule_delay_millis=self.config.schedule_delay_millis
                )
            else:
                span_processor = SimpleSpanProcessor(exporter)

            self.provider.add_span_processor(span_processor)
            self._span_processors.append(span_processor)

        # Set tracer provider
        trace.set_tracer_provider(self.provider)

        # Get tracer
        self.tracer = trace.get_tracer(
            self.config.service_name,
            self.config.service_version
        )

        # Setup instrumentations
        self._setup_instrumentations()

    def _create_exporter(self):
        """Create span exporter based on configuration."""
        if not self.config:
            return None

        if self.config.exporter == "none":
            return None

        if self.config.exporter == "console":
            return ConsoleSpanExporter()

        if self.config.exporter == "otlp":
            if not self.config.otlp_endpoint:
                logger.warning("OTLP endpoint not configured")
                return None

            try:
                return OTLPSpanExporter(
                    endpoint=self.config.otlp_endpoint,
                    insecure=self.config.otlp_insecure
                )
            except Exception as e:
                logger.error(f"Failed to create OTLP exporter: {e}")
                return None

        return None

    def _setup_instrumentations(self):
        """Setup automatic instrumentations."""
        if not self.config:
            return

        if self.config.enable_fastapi:
            try:
                from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
                # FastAPI instrumentation is done separately
                logger.info("FastAPI instrumentation ready")
            except ImportError:
                pass

        if self.config.enable_httpx:
            try:
                from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
                HTTPXClientInstrumentor().instrument()
                logger.info("HTTPX instrumentation enabled")
            except ImportError:
                pass

        if self.config.enable_aiohttp:
            try:
                from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
                AioHttpClientInstrumentor().instrument()
                logger.info("aiohttp instrumentation enabled")
            except ImportError:
                pass

        if self.config.enable_asyncio:
            try:
                from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
                AsyncioInstrumentor().instrument()
                logger.info("asyncio instrumentation enabled")
            except ImportError:
                pass

    def get_tracer(self) -> Optional[Tracer]:
        """Get the tracer instance."""
        if not self.is_enabled:
            return None
        return self.tracer

    def get_provider(self) -> Optional[TracerProvider]:
        """Get the tracer provider."""
        return self.provider

    def is_tracing_enabled(self) -> bool:
        """Check if tracing is enabled."""
        return self.is_enabled

    def shutdown(self):
        """Shutdown tracing and flush spans."""
        if self.provider:
            try:
                self.provider.force_flush()
                self.provider.shutdown()
                logger.info("Tracing shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down tracing: {e}")


# ============================================================
# Tracing Decorators and Context Managers
# ============================================================

class TraceSpan:
    """
    Context manager for creating and managing spans.
    """

    def __init__(
        self,
        name: str,
        tracer: Optional[Tracer] = None,
        attributes: Optional[Dict[str, Any]] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        record_exception: bool = True
    ):
        """
        Initialize trace span.

        Args:
            name: Span name
            tracer: Tracer instance
            attributes: Span attributes
            kind: Span kind
            record_exception: Whether to record exceptions
        """
        self.name = name
        self.tracer = tracer or get_tracer()
        self.attributes = attributes or {}
        self.kind = kind
        self.record_exception = record_exception
        self.span: Optional[Span] = None
        self._context = None

    def __enter__(self):
        if not self.tracer:
            return self

        # Create and start span
        self.span = self.tracer.start_span(
            self.name,
            kind=self.kind,
            attributes=self.attributes
        )

        # Set as current span
        self._context = set_span_in_context(self.span)
        attach(self._context)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.span:
            return

        # Record exception if any
        if exc_val and self.record_exception:
            self.span.record_exception(exc_val)
            self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
        else:
            self.span.set_status(Status(StatusCode.OK))

        # End span
        self.span.end()
        detach(self._context)

    def add_attribute(self, key: str, value: Any):
        """Add an attribute to the current span."""
        if self.span and value is not None:
            self.span.set_attribute(key, value)

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add an event to the current span."""
        if self.span:
            self.span.add_event(name, attributes or {})

    def record_exception(self, exception: Exception):
        """Record an exception in the current span."""
        if self.span:
            self.span.record_exception(exception)
            self.span.set_status(Status(StatusCode.ERROR, str(exception)))


def trace(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    record_exception: bool = True,
    include_args: bool = True,
    include_result: bool = True,
    max_arg_length: int = 500
):
    """
    Decorator for tracing functions.

    Args:
        name: Span name (defaults to function name)
        attributes: Additional attributes
        kind: Span kind
        record_exception: Whether to record exceptions
        include_args: Whether to include function arguments
        include_result: Whether to include return value
        max_arg_length: Maximum length for argument values
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            span_name = name or func.__name__
            span_attrs = attributes or {}

            # Add function info
            span_attrs.update({
                "function.module": func.__module__,
                "function.name": func.__name__,
                "function.qualified_name": f"{func.__module__}.{func.__name__}"
            })

            # Add arguments if requested
            if include_args:
                try:
                    import inspect
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()

                    arg_str = {}
                    for key, value in bound_args.arguments.items():
                        # Skip self, cls, etc.
                        if key in ['self', 'cls']:
                            continue

                        # Convert to string and truncate
                        val_str = str(value)
                        if len(val_str) > max_arg_length:
                            val_str = val_str[:max_arg_length] + "..."
                        arg_str[key] = val_str

                    span_attrs["function.args"] = json.dumps(arg_str)
                except Exception:
                    pass

            # Create span
            with TraceSpan(span_name, None, span_attrs, kind, record_exception) as span:
                try:
                    result = func(*args, **kwargs)

                    # Add result if requested
                    if include_result:
                        try:
                            result_str = str(result)
                            if len(result_str) > max_arg_length:
                                result_str = result_str[:max_arg_length] + "..."
                            span.add_attribute("function.result", result_str)
                        except Exception:
                            pass

                    return result
                except Exception as e:
                    if record_exception:
                        span.record_exception(e)
                    raise

        return wrapper
    return decorator


def trace_async(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    kind: SpanKind = SpanKind.INTERNAL,
    record_exception: bool = True,
    include_args: bool = True,
    include_result: bool = True,
    max_arg_length: int = 500
):
    """
    Decorator for tracing async functions.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            span_name = name or func.__name__
            span_attrs = attributes or {}

            # Add function info
            span_attrs.update({
                "function.module": func.__module__,
                "function.name": func.__name__,
                "function.qualified_name": f"{func.__module__}.{func.__name__}"
            })

            # Add arguments if requested
            if include_args:
                try:
                    import inspect
                    sig = inspect.signature(func)
                    bound_args = sig.bind(*args, **kwargs)
                    bound_args.apply_defaults()

                    arg_str = {}
                    for key, value in bound_args.arguments.items():
                        if key in ['self', 'cls']:
                            continue

                        val_str = str(value)
                        if len(val_str) > max_arg_length:
                            val_str = val_str[:max_arg_length] + "..."
                        arg_str[key] = val_str

                    span_attrs["function.args"] = json.dumps(arg_str)
                except Exception:
                    pass

            # Create span
            with TraceSpan(span_name, None, span_attrs, kind, record_exception) as span:
                try:
                    result = await func(*args, **kwargs)

                    if include_result:
                        try:
                            result_str = str(result)
                            if len(result_str) > max_arg_length:
                                result_str = result_str[:max_arg_length] + "..."
                            span.add_attribute("function.result", result_str)
                        except Exception:
                            pass

                    return result
                except Exception as e:
                    if record_exception:
                        span.record_exception(e)
                    raise

        return wrapper
    return decorator


def traced(
    name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    kind: SpanKind = SpanKind.INTERNAL
):
    """
    Alternative decorator for tracing with simpler API.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            span_name = name or func.__name__
            span_attrs = attributes or {}

            with TraceSpan(span_name, None, span_attrs, kind):
                return func(*args, **kwargs)

        return wrapper
    return decorator


@contextmanager
def trace_context(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    kind: SpanKind = SpanKind.INTERNAL
):
    """
    Context manager for tracing a block of code.

    Args:
        name: Span name
        attributes: Span attributes
        kind: Span kind
    """
    with TraceSpan(name, None, attributes, kind):
        yield


# ============================================================
# Convenience Functions
# ============================================================

_tracing_manager = None


def get_tracing_manager() -> TracingManager:
    """Get the global tracing manager."""
    global _tracing_manager
    if _tracing_manager is None:
        _tracing_manager = TracingManager()
    return _tracing_manager


def get_tracer() -> Optional[Tracer]:
    """Get the global tracer instance."""
    manager = get_tracing_manager()
    return manager.get_tracer()


def setup_tracing(config: Optional[TracingConfig] = None) -> TracingManager:
    """
    Setup global tracing.

    Args:
        config: Tracing configuration

    Returns:
        TracingManager instance
    """
    manager = get_tracing_manager()
    manager.configure(config)
    return manager


def shutdown_tracing():
    """Shutdown global tracing."""
    manager = get_tracing_manager()
    manager.shutdown()


# ============================================================
# Attribute Helpers
# ============================================================

class TraceAttributes:
    """Common trace attributes."""

    # HTTP
    HTTP_METHOD = "http.method"
    HTTP_URL = "http.url"
    HTTP_STATUS_CODE = "http.status_code"
    HTTP_ROUTE = "http.route"
    HTTP_TARGET = "http.target"
    HTTP_USER_AGENT = "http.user_agent"

    # Database
    DB_SYSTEM = "db.system"
    DB_NAME = "db.name"
    DB_USER = "db.user"
    DB_STATEMENT = "db.statement"
    DB_OPERATION = "db.operation"

    # Messaging
    MESSAGING_SYSTEM = "messaging.system"
    MESSAGING_DESTINATION = "messaging.destination"
    MESSAGING_PROTOCOL = "messaging.protocol"

    # LLM specific
    LLM_MODEL = "llm.model"
    LLM_PROVIDER = "llm.provider"
    LLM_TEMPERATURE = "llm.temperature"
    LLM_MAX_TOKENS = "llm.max_tokens"
    LLM_TOKENS_USED = "llm.tokens_used"
    LLM_PROMPT_TOKENS = "llm.prompt_tokens"
    LLM_COMPLETION_TOKENS = "llm.completion_tokens"
    LLM_RESPONSE = "llm.response"

    # Retrieval
    RETRIEVAL_TOP_K = "retrieval.top_k"
    RETRIEVAL_SCORE_THRESHOLD = "retrieval.score_threshold"
    RETRIEVAL_NUM_RESULTS = "retrieval.num_results"

    # Document processing
    DOCUMENT_ID = "document.id"
    DOCUMENT_NAME = "document.name"
    DOCUMENT_SIZE = "document.size"
    DOCUMENT_TYPE = "document.type"
    CHUNK_SIZE = "chunk.size"
    CHUNK_OVERLAP = "chunk.overlap"
    NUM_CHUNKS = "num.chunks"

    # Errors
    ERROR_TYPE = "error.type"
    ERROR_MESSAGE = "error.message"
    ERROR_STACK = "error.stack"

    # Performance
    LATENCY_MS = "latency.ms"
    PROCESSING_TIME_MS = "processing.time.ms"
    CACHE_HIT = "cache.hit"
    CACHE_AGE = "cache.age"

    # User
    USER_ID = "user.id"
    USER_ROLE = "user.role"
    SESSION_ID = "session.id"

    @staticmethod
    def add_attributes(span: Span, attrs: Dict[str, Any]):
        """Add attributes to a span."""
        for key, value in attrs.items():
            if value is not None:
                span.set_attribute(key, str(value))


# ============================================================
# FastAPI Integration
# ============================================================

def instrument_fastapi(app):
    """
    Instrument FastAPI application with OpenTelemetry.

    Args:
        app: FastAPI application
    """
    if not OPENTELEMETRY_AVAILABLE:
        logger.warning("OpenTelemetry not available, skipping FastAPI instrumentation")
        return

    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        manager = get_tracing_manager()
        if not manager.is_tracing_enabled():
            return

        FastAPIInstrumentor.instrument_app(
            app,
            tracer_provider=manager.get_provider(),
            server_request_hook=_server_request_hook,
            client_request_hook=_client_request_hook
        )
        logger.info("FastAPI instrumentation enabled")
    except Exception as e:
        logger.error(f"Failed to instrument FastAPI: {e}")


def _server_request_hook(span: Span, scope: Dict[str, Any]):
    """Hook for server requests."""
    if span and scope:
        span.set_attribute("http.method", scope.get("method", "unknown"))
        span.set_attribute("http.route", scope.get("path", "unknown"))
        span.set_attribute("http.target", scope.get("raw_path", "unknown"))


def _client_request_hook(span: Span, scope: Dict[str, Any]):
    """Hook for client requests."""
    if span and scope:
        span.set_attribute("http.method", scope.get("method", "unknown"))
        span.set_attribute("http.url", scope.get("url", "unknown"))


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    import asyncio

    async def test_tracing():
        """Test tracing functionality."""
        logging.basicConfig(level=logging.INFO)

        print("Testing OpenTelemetry Tracing...")
        print("=" * 60)

        # Configure tracing
        config = TracingConfig(
            service_name="docqa-test",
            service_version="1.0.0",
            enabled=True,
            exporter="console",
            environment="test"
        )

        setup_tracing(config)

        # Test basic tracing
        print("\n📊 Testing basic tracing:")

        @trace(name="test_function", attributes={"test.attr": "value"})
        def test_function():
            print("  Executing test function")
            return "Hello, World!"

        result = test_function()
        print(f"  Result: {result}")

        # Test async tracing
        print("\n📊 Testing async tracing:")

        @trace_async(name="test_async_function")
        async def test_async_function():
            await asyncio.sleep(0.1)
            return "Async result"

        result = await test_async_function()
        print(f"  Result: {result}")

        # Test context manager
        print("\n📊 Testing context manager:")

        with trace_context("test_context", {"context.attr": "value"}):
            time.sleep(0.05)
            print("  Context block executed")

        # Test with attributes
        print("\n📊 Testing with attributes:")

        with trace_context("attribute_test") as span:
            TraceAttributes.add_attributes(span, {
                "test.string": "value",
                "test.int": 42,
                "test.float": 3.14,
                "test.bool": True
            })
            print("  Attributes added")

        # Shutdown
        shutdown_tracing()
        print("\n✅ Tracing test complete")

    asyncio.run(test_tracing())
