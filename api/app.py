"""
FastAPI application for DocQA AI system.
Provides REST API endpoints for document ingestion, querying, and management.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
from datetime import datetime
import uvicorn

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from api.routes import router as api_router
from api.websocket import router as websocket_router
from src.utils.config import get_config, get_config_manager
from src.utils.logger import get_logger, setup_logging
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.retriever import VectorRetriever, create_retriever
from src.generation.llm_interface import LLMInterface
from src.ingestion.embedding_generator import OpenAIEmbeddingGenerator

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
    "active_connections": 0
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting DocQA AI API...")
    app_state["startup_time"] = datetime.now()

    # Load configuration
    config = get_config()
    app_state["config"] = config

    # Initialize components
    try:
        # Initialize embedding generator
        embedding_generator = OpenAIEmbeddingGenerator(
            model=config.embedding.model,
            batch_size=config.embedding.batch_size,
            use_cache=config.embedding.cache_enabled
        )
        app_state["embedding_generator"] = embedding_generator

        # Initialize vector store
        vector_store = FAISSVectorStore(
            dimension=config.vector_store.dimension,
            index_type=config.vector_store.index_type,
            index_path=config.vector_store.index_path if os.path.exists(config.vector_store.index_path) else None
        )
        app_state["vector_store"] = vector_store

        # Initialize retriever
        retriever = create_retriever(
            retriever_type="vector",
            vector_store=vector_store,
            embedding_generator=embedding_generator,
            top_k=config.retrieval.top_k
        )
        app_state["retriever"] = retriever

        # Initialize LLM interface
        llm_interface = LLMInterface(
            provider=config.llm.provider,
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens
        )
        app_state["llm_interface"] = llm_interface

        logger.info("All components initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        # Continue with partial initialization

    yield

    # Shutdown
    logger.info("Shutting down DocQA AI API...")

    # Save vector store if needed
    if app_state["vector_store"] and app_state["vector_store"].get_size() > 0:
        try:
            vector_store_path = config.vector_store.index_path
            if vector_store_path:
                app_state["vector_store"].save(vector_store_path)
                logger.info(f"Vector store saved to {vector_store_path}")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")


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
    
    ## Authentication
    Include your API key in the `Authorization` header:
