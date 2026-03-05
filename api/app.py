"""
FastAPI application for DocQA AI system.
Provides REST API endpoints for document ingestion, querying, and management.
ENHANCED: Graceful shutdown with proper resource cleanup, connection draining, and signal handling.
"""

import os
import sys
import asyncio
import signal
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
from datetime import datetime
import uvicorn
import gc

from fastapi import FastAPI, HTTPException, Request, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.concurrency import run_in_threadpool
import asyncio
import concurrent.futures

from api.routes import router as api_router
from api.websocket import router as websocket_router
from api.background import (
    BackgroundTaskManager,
    TaskStatus,
    task_manager,
    process_ingestion_task,
    cleanup_expired_tasks
)
from src.utils.config import get_config, get_config_manager
from src.utils.logger import get_logger, setup_logging
from src.utils.cache import CacheManager
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.retriever import VectorRetriever, create_retriever
from src.generation.llm_interface import LLMInterface
from src.ingestion.embedding_generator import BatchEmbeddingGenerator

logger = get_logger(__name__)

# Global application state
app_state = {
    "vector_store": None,
    "retriever": None,
    "llm_interface": None,
    "embedding_generator": None,
    "config": None,
    "startup_time": None,
    "shutdown_time": None,
    "request_count": 0,
    "active_connections": 0,
    "background_tasks": {},
    "cache_manager": None,
    "executor": None,  # Thread pool for CPU-intensive tasks
    "shutdown_event": asyncio.Event(),
    "shutdown_started": False,
    "active_requests": set(),  # Track active request IDs
    "ws_connections": set(),   # Track active WebSocket connections
}

# Async lock for state updates
_state_lock = asyncio.Lock()

# Graceful shutdown configuration
SHUTDOWN_TIMEOUT = int(os.getenv("SHUTDOWN_TIMEOUT", "30"))  # Seconds to wait for in-flight requests
FORCE_SHUTDOWN_TIMEOUT = int(os.getenv("FORCE_SHUTDOWN_TIMEOUT", "10"))  # Force shutdown after this


class ShutdownManager:
    """
    Manages graceful shutdown with proper resource cleanup and connection draining.
    """

    def __init__(self, app_state: Dict[str, Any], timeout: int = SHUTDOWN_TIMEOUT):
        self.app_state = app_state
        self.timeout = timeout
        self._shutdown_tasks = []
        self._cleanup_handlers = []

        # Register signal handlers
        self._register_signal_handlers()

    def _register_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        for sig in [signal.SIGTERM, signal.SIGINT]:
            signal.signal(sig, self._signal_handler)

        logger.info("Signal handlers registered for graceful shutdown")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        asyncio.create_task(self._handle_shutdown())

    async def _handle_shutdown(self):
        """Handle shutdown process."""
        if self.app_state["shutdown_started"]:
            return

        self.app_state["shutdown_started"] = True
        self.app_state["shutdown_time"] = datetime.now()

        logger.info("🚦 Graceful shutdown initiated")
        logger.info(f"⏱️  Shutdown timeout: {self.timeout} seconds")

        try:
            # Start shutdown event
            self.app_state["shutdown_event"].set()

            # Step 1: Stop accepting new requests
            await self._stop_accepting_requests()

            # Step 2: Wait for in-flight requests to complete
            await self._wait_for_active_requests()

            # Step 3: Close WebSocket connections
            await self._close_websocket_connections()

            # Step 4: Clean up background tasks
            await self._cleanup_background_tasks()

            # Step 5: Save state and clean up resources
            await self._save_state_and_cleanup()

            # Step 6: Force cleanup if needed
            await self._force_cleanup()

            logger.info("✅ Graceful shutdown completed successfully")

        except asyncio.TimeoutError:
            logger.error(f"❌ Shutdown timed out after {self.timeout} seconds")
            await self._force_cleanup()

        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")
            await self._force_cleanup()

        finally:
            # Exit the process
            sys.exit(0)

    async def _stop_accepting_requests(self):
        """Stop accepting new requests."""
        logger.info("🛑 Stopping acceptance of new requests")

        # Mark application as shutting down
        self.app_state["shutdown_started"] = True

        # Wait a moment for in-flight requests to start completing
        await asyncio.sleep(0.5)

    async def _wait_for_active_requests(self):
        """Wait for active requests to complete."""
        active_requests = len(self.app_state["active_requests"])
        ws_connections = len(self.app_state["ws_connections"])

        if active_requests > 0 or ws_connections > 0:
            logger.info(f"⏳ Waiting for {active_requests} active requests and {ws_connections} WebSocket connections to complete...")

            start_time = time.time()
            while time.time() - start_time < self.timeout:
                active_requests = len(self.app_state["active_requests"])
                ws_connections = len(self.app_state["ws_connections"])

                if active_requests == 0 and ws_connections == 0:
                    logger.info("✅ All active connections completed")
                    break

                logger.debug(f"Still waiting: {active_requests} requests, {ws_connections} WebSocket connections")
                await asyncio.sleep(0.5)

            # Check if timeout was reached
            if active_requests > 0 or ws_connections > 0:
                logger.warning(f"⚠️ Timeout waiting for connections: {active_requests} requests, {ws_connections} WebSocket connections")

    async def _close_websocket_connections(self):
        """Close WebSocket connections gracefully."""
        ws_connections = self.app_state["ws_connections"]

        if ws_connections:
            logger.info(f"🔌 Closing {len(ws_connections)} WebSocket connections...")

            for ws in list(ws_connections):
                try:
                    # Send close message
                    await ws.close(code=1001, reason="Server shutting down")
                except Exception as e:
                    logger.debug(f"Error closing WebSocket: {e}")

            # Wait for connections to close
            await asyncio.sleep(0.5)

    async def _cleanup_background_tasks(self):
        """Clean up background tasks."""
        logger.info("🧹 Cleaning up background tasks...")

        # Cancel all running tasks
        await task_manager.shutdown()

        # Wait for tasks to complete
        await asyncio.sleep(1)

    async def _save_state_and_cleanup(self):
        """Save state and clean up resources."""
        logger.info("💾 Saving state and cleaning up resources...")

        # Save vector store
        if self.app_state["vector_store"] and self.app_state["vector_store"].get_size() > 0:
            try:
                vector_store_path = self.app_state["config"].vector_store.index_path
                if vector_store_path:
                    await run_in_threadpool(
                        self.app_state["vector_store"].save,
                        vector_store_path
                    )
                    logger.info(f"✅ Vector store saved to {vector_store_path}")
            except Exception as e:
                logger.error(f"Failed to save vector store: {e}")

        # Clear cache
        if self.app_state["cache_manager"]:
            try:
                await run_in_threadpool(self.app_state["cache_manager"].clear)
                logger.info("✅ Cache cleared")
            except Exception as e:
                logger.error(f"Failed to clear cache: {e}")

        # Clear embedding cache
        if self.app_state["embedding_generator"]:
            try:
                await run_in_threadpool(self.app_state["embedding_generator"].clear_cache)
                logger.info("✅ Embedding cache cleared")
            except Exception as e:
                logger.error(f"Failed to clear embedding cache: {e}")

        # Shutdown executor
        if self.app_state["executor"]:
            try:
                self.app_state["executor"].shutdown(wait=True)
                logger.info("✅ Thread pool executor shutdown")
            except Exception as e:
                logger.error(f"Failed to shutdown executor: {e}")

        # Run garbage collection
        gc.collect()
        logger.info("✅ Garbage collection completed")

    async def _force_cleanup(self):
        """Force cleanup when shutdown times out."""
        logger.warning("⚠️ Performing force cleanup...")

        # Force close all WebSocket connections
        for ws in list(self.app_state["ws_connections"]):
            try:
                await ws.close(code=1001, reason="Force shutdown")
            except Exception:
                pass

        # Force shutdown executor
        if self.app_state["executor"]:
            try:
                self.app_state["executor"].shutdown(wait=False)
            except Exception:
                pass

        logger.warning("⚠️ Force cleanup completed")

    def add_shutdown_task(self, task: callable):
        """Add a task to be executed during shutdown."""
        self._shutdown_tasks.append(task)

    def add_cleanup_handler(self, handler: callable):
        """Add a cleanup handler."""
        self._cleanup_handlers.append(handler)


# Create shutdown manager
shutdown_manager = ShutdownManager(app_state)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Handles async initialization and graceful shutdown.
    """
    # Startup
    logger.info("🚀 Starting DocQA AI API...")
    start_time = datetime.now()
    app_state["startup_time"] = start_time

    # Initialize thread pool for CPU-intensive tasks
    app_state["executor"] = concurrent.futures.ThreadPoolExecutor(
        max_workers=int(os.getenv("API_WORKERS", 4)),
        thread_name_prefix="docqa_worker"
    )

    # Load configuration
    config = get_config()
    app_state["config"] = config

    # Initialize cache manager
    cache_manager = CacheManager()
    app_state["cache_manager"] = cache_manager

    # Initialize background task manager
    await task_manager.start()

    # Initialize components asynchronously
    try:
        await _initialize_components()
        logger.info("✅ All components initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize components: {e}")
        raise

    # Start background cleanup task
    asyncio.create_task(_background_cleanup())

    logger.info(f"✅ API started at {start_time}")
    logger.info(f"📡 Listening on port {config.api.port}")

    yield

    # Shutdown
    logger.info("🛑 Initiating graceful shutdown...")

    # Trigger shutdown manager
    asyncio.create_task(shutdown_manager._handle_shutdown())

    # Wait for shutdown to complete
    while app_state["shutdown_started"]:
        await asyncio.sleep(1)

    logger.info("✅ API shutdown complete")


async def _initialize_components():
    """Initialize all components asynchronously."""
    config = app_state["config"]

    # Initialize embedding generator
    app_state["embedding_generator"] = BatchEmbeddingGenerator(
        model=config.embedding.model,
        batch_size=config.embedding.batch_size,
        use_cache=config.embedding.cache_enabled,
        rate_limit_requests=50,
        rate_limit_tokens=100000
    )

    # Initialize vector store
    vector_store_path = config.vector_store.index_path
    if vector_store_path and os.path.exists(vector_store_path):
        try:
            vector_store = FAISSVectorStore(
                dimension=config.vector_store.dimension,
                index_type=config.vector_store.index_type,
                index_path=vector_store_path
            )
            app_state["vector_store"] = vector_store
            logger.info(f"✅ Loaded vector store from {vector_store_path}")
        except Exception as e:
            logger.warning(f"Failed to load vector store: {e}")
            # Create new vector store
            vector_store = FAISSVectorStore(
                dimension=config.vector_store.dimension,
                index_type=config.vector_store.index_type
            )
            app_state["vector_store"] = vector_store
    else:
        # Create new vector store
        vector_store = FAISSVectorStore(
            dimension=config.vector_store.dimension,
            index_type=config.vector_store.index_type
        )
        app_state["vector_store"] = vector_store

    # Initialize retriever
    if app_state["vector_store"] and app_state["embedding_generator"]:
        retriever = create_retriever(
            retriever_type="vector",
            vector_store=app_state["vector_store"],
            embedding_generator=app_state["embedding_generator"],
            top_k=config.retrieval.top_k
        )
        app_state["retriever"] = retriever

    # Initialize LLM interface
    app_state["llm_interface"] = LLMInterface(
        provider=config.llm.provider,
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens
    )


async def _background_cleanup():
    """Background task for periodic cleanup."""
    while not app_state["shutdown_event"].is_set():
        try:
            # Clean up expired cache entries
            if app_state["cache_manager"]:
                # Redis cache doesn't need manual cleanup (TTL handles it)
                pass

            # Clean up expired background tasks
            await cleanup_expired_tasks()

            # Update vector store stats
            if app_state["vector_store"]:
                stats = app_state["vector_store"].get_stats()
                logger.debug(f"Vector store stats: {stats['total_vectors']} vectors")

            # Sleep for 1 hour
            await asyncio.sleep(3600)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Background cleanup error: {e}")
            await asyncio.sleep(60)  # Wait 1 minute before retrying


# Middleware for tracking active requests
@app.middleware("http")
async def track_active_requests(request: Request, call_next):
    """Track active requests for graceful shutdown."""
    # Generate request ID
    request_id = f"{datetime.now().timestamp()}-{id(request)}"

    # Track request start
    app_state["active_requests"].add(request_id)
    app_state["request_count"] += 1

    # Check if shutdown is in progress
    if app_state["shutdown_started"]:
        # Reject new requests during shutdown
        app_state["active_requests"].discard(request_id)
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "SHUTTING_DOWN",
                    "message": "Server is shutting down. Please try again later.",
                    "timestamp": datetime.now().isoformat()
                }
            }
        )

    try:
        # Process request
        response = await call_next(request)
        return response

    except Exception as e:
        logger.error(f"Request {request_id} failed: {e}")
        raise

    finally:
        # Track request end
        app_state["active_requests"].discard(request_id)


# Middleware for WebSocket connection tracking
@app.middleware("websocket")
async def track_websocket_connections(request: Request, call_next):
    """Track WebSocket connections for graceful shutdown."""
    ws = await call_next(request)

    # Check if shutdown is in progress
    if app_state["shutdown_started"]:
        await ws.close(code=1001, reason="Server shutting down")
        return ws

    # Track connection
    app_state["ws_connections"].add(ws)

    try:
        return ws
    finally:
        app_state["ws_connections"].discard(ws)


# Create FastAPI application
app = FastAPI(
    title="DocQA AI API",
    description="""
    # DocQA AI - Document Question Answering System
    
    This API provides access to the DocQA AI system for:
    - Ingesting documents (PDF, DOCX, TXT, HTML, MD, CSV, JSON)
    - Asking questions about your documents
    - Managing documents and vector store
    - Monitoring system health and metrics
    
    ## Graceful Shutdown
    The API supports graceful shutdown with:
    - Connection draining
    - State saving
    - Resource cleanup
    - Signal handling (SIGTERM, SIGINT)
    """,
    version="1.0.0",
    contact={
        "name": "DocQA AI Support",
        "email": "support@docqa-ai.com",
        "url": "https://docqa-ai.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately in production
)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests asynchronously."""
    start_time = datetime.now()

    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")

    try:
        response = await call_next(request)

        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds() * 1000

        # Log response
        logger.info(f"Response: {response.status_code} - {duration:.2f}ms")

        # Add duration header
        response.headers["X-Response-Time"] = f"{duration:.2f}ms"

        return response

    except Exception as e:
        logger.error(f"Request failed: {e}", exc_info=True)
        raise


# Add error handling middleware
@app.middleware("http")
async def error_handling(request: Request, call_next):
    """Handle errors and return consistent error responses."""
    try:
        return await call_next(request)
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={
                "error": {
                    "code": e.status_code,
                    "message": e.detail,
                    "timestamp": datetime.now().isoformat()
                }
            }
        )
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": 500,
                    "message": "Internal server error",
                    "timestamp": datetime.now().isoformat()
                }
            }
        )


# Include routers
app.include_router(api_router, prefix="/api/v1")
app.include_router(websocket_router, prefix="/ws")


# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>DocQA AI API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            h1 { color: #2c3e50; }
            .endpoint { background: #f4f6f8; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { display: inline-block; padding: 3px 8px; border-radius: 3px; font-weight: bold; font-size: 12px; }
            .get { background: #61affe; color: white; }
            .post { background: #49cc90; color: white; }
            .delete { background: #f93e3e; color: white; }
            code { background: #e8e8e8; padding: 2px 5px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>📄 DocQA AI API</h1>
        <p>Document Question Answering System</p>
        
        <h2>📚 Documentation</h2>
        <ul>
            <li><a href="/api/docs">Swagger UI</a></li>
            <li><a href="/api/redoc">ReDoc</a></li>
            <li><a href="/api/openapi.json">OpenAPI Schema</a></li>
        </ul>
        
        <h2>🔧 Available Endpoints</h2>
        
        <div class="endpoint">
            <span class="method post">POST</span>
            <code>/api/v1/query</code>
            <p>Ask a question about your documents</p>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span>
            <code>/api/v1/documents/ingest</code>
            <p>Upload documents for ingestion</p>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span>
            <code>/api/v1/documents</code>
            <p>List all ingested documents</p>
        </div>
        
        <div class="endpoint">
            <span class="method delete">DELETE</span>
            <code>/api/v1/documents/{document_id}</code>
            <p>Delete a document</p>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span>
            <code>/api/v1/health</code>
            <p>Check system health</p>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span>
            <code>/api/v1/metrics</code>
            <p>Get system metrics</p>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span>
            <code>/api/v1/tasks/{task_id}</code>
            <p>Check background task status</p>
        </div>
        
        <h2>🏥 Health Status</h2>
        <p><a href="/api/v1/health">Check API health</a></p>
    </body>
    </html>
    """


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    # Check if shutdown is in progress
    if app_state["shutdown_started"]:
        return JSONResponse(
            status_code=503,
            content={
                "status": "shutting_down",
                "message": "Server is shutting down",
                "timestamp": datetime.now().isoformat()
            }
        )

    uptime_seconds = (
        (datetime.now() - app_state["startup_time"]).total_seconds()
        if app_state["startup_time"] else 0
    )

    # Check component status
    components = {
        "vector_store": "ready" if app_state["vector_store"] is not None else "unavailable",
        "retriever": "ready" if app_state["retriever"] is not None else "unavailable",
        "llm_interface": "ready" if app_state["llm_interface"] is not None else "unavailable",
        "cache_manager": "ready" if app_state["cache_manager"] is not None else "unavailable",
        "task_manager": "ready" if task_manager.is_running else "unavailable"
    }

    all_ready = all(v == "ready" for v in components.values())

    return {
        "status": "healthy" if all_ready else "degraded",
        "timestamp": datetime.now().isoformat(),
        "version": app.version,
        "uptime_seconds": uptime_seconds,
        "components": components,
        "vector_store_size": app_state["vector_store"].get_size() if app_state["vector_store"] else 0,
        "active_tasks": len(task_manager.tasks),
        "active_requests": len(app_state["active_requests"]),
        "shutting_down": app_state["shutdown_started"]
    }


@app.get("/metrics")
async def get_metrics():
    """Get system metrics."""
    if app_state["shutdown_started"]:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "SHUTTING_DOWN",
                    "message": "Server is shutting down",
                    "timestamp": datetime.now().isoformat()
                }
            }
        )

    vector_size = 0
    if app_state["vector_store"]:
        vector_size = app_state["vector_store"].get_size()

    # Get cache stats
    cache_stats = {}
    if app_state["cache_manager"]:
        cache_stats = app_state["cache_manager"].get_stats()

    # Get task stats
    task_stats = task_manager.get_stats()

    return {
        "requests": {
            "total": app_state["request_count"],
            "active_connections": len(app_state["active_requests"]),
            "active_websockets": len(app_state["ws_connections"])
        },
        "vector_store": {
            "size": vector_size,
            "dimension": app_state["config"].vector_store.dimension if app_state["config"] else 0
        },
        "cache": cache_stats,
        "tasks": task_stats,
        "system": {
            "uptime_seconds": (
                (datetime.now() - app_state["startup_time"]).total_seconds()
                if app_state["startup_time"] else 0
            ),
            "shutting_down": app_state["shutdown_started"],
            "memory_usage": await _get_memory_usage_async()
        }
    }


async def _get_memory_usage_async() -> Dict[str, float]:
    """Get memory usage statistics asynchronously."""
    try:
        import psutil
        process = psutil.Process()
        memory = process.memory_info()
        return {
            "rss_mb": memory.rss / 1024 / 1024,
            "vms_mb": memory.vms / 1024 / 1024,
            "percent": process.memory_percent()
        }
    except ImportError:
        return {}


# Custom OpenAPI schema
def custom_openapi():
    """Generate custom OpenAPI schema with additional metadata."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Add security scheme
    openapi_schema["components"] = openapi_schema.get("components", {})
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "Authorization"
        }
    }

    # Add global security
    openapi_schema["security"] = [
        {"ApiKeyAuth": []}
    ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


def get_app_state():
    """Get application state (for use in routes)."""
    return app_state


# Graceful shutdown endpoint (for testing)
@app.post("/shutdown")
async def trigger_shutdown():
    """Trigger graceful shutdown (for testing)."""
    if app_state["shutdown_started"]:
        return {
            "status": "shutdown_already_started",
            "message": "Shutdown is already in progress"
        }

    logger.info("Shutdown triggered via API endpoint")
    asyncio.create_task(shutdown_manager._handle_shutdown())

    return {
        "status": "shutdown_initiated",
        "message": "Graceful shutdown has been initiated",
        "shutdown_timeout": SHUTDOWN_TIMEOUT
    }


if __name__ == "__main__":
    # Run directly for development
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
