"""
FastAPI application for DocQA AI system.
Provides REST API endpoints for document ingestion, querying, and management.
ENHANCED: Full async support with background tasks, async database, and async endpoints.
"""

import os
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
from datetime import datetime
import uvicorn

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
from src.utils.cache import CacheManager, async_cached
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
    "request_count": 0,
    "active_connections": 0,
    "background_tasks": {},
    "cache_manager": None,
    "executor": None  # Thread pool for CPU-intensive tasks
}

# Async lock for state updates
_state_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Handles async initialization and cleanup.
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

    yield

    # Shutdown
    logger.info("🛑 Shutting down DocQA AI API...")

    # Save vector store if needed
    await _save_vector_store()

    # Shutdown background task manager
    await task_manager.shutdown()

    # Shutdown thread pool
    if app_state["executor"]:
        app_state["executor"].shutdown(wait=True)

    # Clean up resources
    await _cleanup_resources()

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


async def _save_vector_store():
    """Save vector store asynchronously."""
    if app_state["vector_store"] and app_state["vector_store"].get_size() > 0:
        try:
            vector_store_path = app_state["config"].vector_store.index_path
            if vector_store_path:
                # Run save in thread pool to avoid blocking
                await run_in_threadpool(
                    app_state["vector_store"].save,
                    vector_store_path
                )
                logger.info(f"✅ Vector store saved to {vector_store_path}")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")


async def _cleanup_resources():
    """Clean up resources."""
    # Clear embedding cache
    if app_state["embedding_generator"]:
        try:
            app_state["embedding_generator"].clear_cache()
            logger.info("✅ Embedding cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear embedding cache: {e}")

    # Clear application cache
    if app_state["cache_manager"]:
        try:
            app_state["cache_manager"].clear()
            logger.info("✅ Application cache cleared")
        except Exception as e:
            logger.warning(f"Failed to clear application cache: {e}")


async def _background_cleanup():
    """Background task for periodic cleanup."""
    while True:
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
    
    ## Async Support
    The API supports asynchronous operations for better performance:
    - Concurrent request handling
    - Background tasks for long-running operations (document ingestion)
    - Caching for frequently accessed data
    - WebSocket connections for real-time streaming
    
    ## Authentication
    Include your API key in the `Authorization` header:
