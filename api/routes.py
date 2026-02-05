"""
API routes for DocQA AI system with async support.
Handles document ingestion, querying, and management endpoints.
"""

import os
import json
import asyncio
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, Request, status, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, validator

from api.app import get_app_state, task_manager
from api.schemas import (
    QueryRequest, QueryResponse, DocumentIngestRequest,
    DocumentIngestResponse, DocumentListResponse, DocumentInfo,
    HealthResponse, MetricsResponse, ErrorResponse, TaskStatusResponse
)
from api.background import process_ingestion_task
from src.utils.logger import get_logger
from src.utils.cache import async_cached
from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import ChunkingPipeline
from src.ingestion.embedding_generator import BatchEmbeddingGenerator
from src.retrieval.retriever import RetrievalResult
from src.generation.prompt_templates import get_rag_prompt
from src.generation.response_postprocess import postprocess_response

logger = get_logger(__name__)

# Create router
router = APIRouter(tags=["docqa"])


# ============== Async Helper Functions ==============

async def process_ingestion_async(
    files: List[UploadFile],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    chunking_strategy: str = "adaptive"
) -> DocumentIngestResponse:
    """
    Process document ingestion asynchronously.
    """
    state = get_app_state()
    loader = DocumentLoader()

    # Create temporary directory for uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        uploaded_files = []
        failed_files = []

        # Save uploaded files in parallel
        save_tasks = []
        for file in files:
            save_tasks.append(_save_uploaded_file(file, temp_dir))

        results = await asyncio.gather(*save_tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"File save error: {result}")
                failed_files.append(str(result))
            elif result:
                uploaded_files.append(result)

        if not uploaded_files:
            raise HTTPException(
                status_code=400,
                detail="No valid files uploaded"
            )

        # Load documents in parallel
        load_tasks = []
        for file_path in uploaded_files:
            load_tasks.append(
                run_in_threadpool(loader.load_document, file_path)
            )

        load_results = await asyncio.gather(*load_tasks, return_exceptions=True)
        documents = []
        for result in load_results:
            if isinstance(result, Exception):
                logger.error(f"Document load error: {result}")
                failed_files.append(str(result))
            else:
                documents.append(result)

        if not documents:
            raise HTTPException(
                status_code=400,
                detail="No documents could be loaded"
            )

        # Chunk documents
        chunking_pipeline = ChunkingPipeline(
            strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        chunks = []
        for doc in documents:
            doc_chunks = await run_in_threadpool(
                chunking_pipeline.chunk_document,
                doc["content"],
                doc["metadata"]
            )
            chunks.extend(doc_chunks)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No chunks could be created"
            )

        # Generate embeddings
        embedding_generator = state["embedding_generator"]

        # Prepare chunks for embedding
        chunk_data = [
            {"text": chunk.text, "metadata": chunk.metadata}
            for chunk in chunks
        ]

        # Generate embeddings asynchronously
        embeddings = await embedding_generator.generate_embeddings_async(chunk_data)

        # Store in vector store
        vector_store = state["vector_store"]
        if not vector_store:
            raise HTTPException(
                status_code=503,
                detail="Vector store not initialized"
            )

        # Add embeddings (thread-safe)
        embedding_vectors = [e.embedding for e in embeddings]
        texts = [e.text for e in embeddings]
        metadata_list = [e.metadata for e in embeddings]
        chunk_ids = [f"chunk_{i}" for i in range(len(embeddings))]

        indices = await run_in_threadpool(
            vector_store.add_embeddings,
            embedding_vectors,
            texts,
            metadata_list,
            chunk_ids
        )

        # Generate document IDs
        document_ids = []
        for doc in documents:
            doc_id = f"doc_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(document_ids)}"
            document_ids.append(doc_id)

        return DocumentIngestResponse(
            success=True,
            document_ids=document_ids,
            total_chunks=len(chunks),
            total_documents=len(documents),
            processing_time_seconds=0,
            failed_files=failed_files
        )


async def _save_uploaded_file(file: UploadFile, temp_dir: str) -> Optional[str]:
    """Save uploaded file asynchronously."""
    try:
        file_path = Path(temp_dir) / file.filename

        # Validate file extension
        extension = file_path.suffix.lower()
        state = get_app_state()
        supported = state["config"].processing.supported_extensions
        if extension not in supported:
            raise ValueError(f"Unsupported format: {extension}")

        # Check file size
        max_size = state["config"].processing.max_file_size_mb * 1024 * 1024
        content = await file.read()
        if len(content) > max_size:
            raise ValueError(f"File too large: {len(content)} bytes (max {max_size})")

        # Save file
        with open(file_path, 'wb') as f:
            f.write(content)

        return str(file_path)

    except Exception as e:
        logger.error(f"Failed to save {file.filename}: {e}")
        return None


@async_cached(ttl=300)  # Cache for 5 minutes
async def _get_cached_query_result(question: str, top_k: int) -> Dict[str, Any]:
    """Get cached query result."""
    state = get_app_state()

    # Retrieve documents
    retrieval_results = state["retriever"].retrieve(question, top_k=top_k)

    if not retrieval_results:
        return {
            "answer": "I couldn't find any relevant information in the documents.",
            "confidence": 0.0,
            "sources": [],
            "tokens_used": 0
        }

    # Prepare context
    context_chunks = [
        {"text": r.text, "source": r.metadata.get("file_path", "Unknown")}
        for r in retrieval_results
    ]

    # Generate prompt
    prompt = get_rag_prompt(
        question=question,
        chunks=context_chunks
    )

    # Generate response with LLM
    llm_response = await run_in_threadpool(
        state["llm_interface"].generate_simple,
        prompt,
        system_prompt="You are a helpful assistant that answers questions based on provided documents."
    )

    # Post-process response
    processed_response = await run_in_threadpool(
        postprocess_response,
        llm_response,
        str(context_chunks[:3]),
        True
    )

    # Prepare sources
    sources = []
    for r in retrieval_results[:5]:
        source = {
            "text": r.text[:500] + "..." if len(r.text) > 500 else r.text,
            "score": r.score,
            "metadata": r.metadata
        }
        sources.append(source)

    return {
        "answer": processed_response.cleaned_text,
        "confidence": processed_response.confidence,
        "sources": sources,
        "tokens_used": processed_response.tokens_used,
        "has_hallucination": processed_response.has_hallucination
    }


# ============== API Endpoints ==============

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question",
    description="Ask a question about your documents and get an AI-generated answer"
)
async def query_endpoint(request: QueryRequest):
    """
    Query endpoint for asking questions about documents.
    Supports caching and async processing.
    """
    start_time = time.time()

    state = get_app_state()

    # Validate state
    if not state["retriever"] or not state["llm_interface"]:
        raise HTTPException(
            status_code=503,
            detail="System not fully initialized"
        )

    if state["vector_store"].get_size() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested. Please upload documents first."
        )

    try:
        # Try to get from cache if not streaming
        if not request.stream:
            try:
                cached_result = await _get_cached_query_result(
                    request.question,
                    request.top_k
                )
                return QueryResponse(
                    answer=cached_result["answer"],
                    confidence=cached_result["confidence"],
                    sources=cached_result["sources"],
                    processing_time_ms=(time.time() - start_time) * 1000,
                    tokens_used=cached_result.get("tokens_used", 0),
                    has_hallucination=cached_result.get("has_hallucination", False)
                )
            except Exception as e:
                logger.warning(f"Cache lookup failed: {e}")
                # Fall through to normal processing

        # Normal processing
        result = await _process_query(
            request.question,
            request.top_k,
            request.temperature,
            request.max_tokens,
            request.include_sources
        )

        return QueryResponse(
            answer=result["answer"],
            confidence=result["confidence"],
            sources=result["sources"],
            processing_time_ms=(time.time() - start_time) * 1000,
            tokens_used=result.get("tokens_used", 0),
            has_hallucination=result.get("has_hallucination", False)
        )

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )


async def _process_query(
    question: str,
    top_k: int,
    temperature: Optional[float],
    max_tokens: Optional[int],
    include_sources: bool
) -> Dict[str, Any]:
    """Process query asynchronously."""
    state = get_app_state()

    # Retrieve documents
    retrieval_results = await run_in_threadpool(
        state["retriever"].retrieve,
        question,
        top_k=top_k
    )

    if not retrieval_results:
        return {
            "answer": "I couldn't find any relevant information in the documents to answer your question.",
            "confidence": 0.0,
            "sources": [],
            "tokens_used": 0,
            "has_hallucination": False
        }

    # Prepare context
    context_chunks = [
        {"text": r.text, "source": r.metadata.get("file_path", "Unknown")}
        for r in retrieval_results
    ]

    # Generate prompt
    prompt = get_rag_prompt(
        question=question,
        chunks=context_chunks
    )

    # Generate response with LLM
    llm_response = await run_in_threadpool(
        state["llm_interface"].generate_simple,
        prompt,
        system_prompt="You are a helpful assistant that answers questions based on provided documents."
    )

    # Post-process response
    processed_response = await run_in_threadpool(
        postprocess_response,
        llm_response,
        str(context_chunks[:3]),
        True
    )

    # Prepare sources
    sources = []
    if include_sources:
        for r in retrieval_results[:5]:
            source = {
                "text": r.text[:500] + "..." if len(r.text) > 500 else r.text,
                "score": r.score,
                "metadata": r.metadata
            }
            sources.append(source)

    return {
        "answer": processed_response.cleaned_text,
        "confidence": processed_response.confidence,
        "sources": sources,
        "tokens_used": processed_response.tokens_used,
        "has_hallucination": processed_response.has_hallucination
    }


@router.post(
    "/documents/ingest",
    response_model=dict,
    summary="Ingest documents",
    description="Upload documents for ingestion as a background task"
)
async def ingest_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Documents to upload"),
    chunk_size: int = Form(800, description="Chunk size in characters", ge=100, le=10000),
    chunk_overlap: int = Form(150, description="Chunk overlap in characters", ge=0, le=5000),
    chunking_strategy: str = Form("adaptive", description="Chunking strategy")
):
    """
    Ingest documents asynchronously using background tasks.
    Returns a task ID for tracking progress.
    """
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files provided"
        )

    # Validate file sizes
    config = get_app_state()["config"]
    max_size = config.processing.max_file_size_mb * 1024 * 1024

    for file in files:
        await file.seek(0, 2)
        size = await file.tell()
        await file.seek(0)

        if size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} exceeds maximum size of {config.processing.max_file_size_mb}MB"
            )

    try:
        # Create background task
        task_id = await task_manager.create_task(
            process_ingestion_task,
            files=files,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunking_strategy=chunking_strategy
        )

        return {
            "success": True,
            "task_id": task_id,
            "message": "Document ingestion started as background task",
            "status_url": f"/api/v1/tasks/{task_id}"
        }

    except Exception as e:
        logger.error(f"Ingestion task creation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start ingestion: {str(e)}"
        )


@router.get(
    "/tasks/{task_id}",
    response_model=TaskStatusResponse,
    summary="Get task status",
    description="Get status of a background task"
)
async def get_task_status(task_id: str):
    """
    Get status of a background task.
    """
    status = task_manager.get_task_status(task_id)

    if not status:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )

    return status


@router.get(
    "/tasks",
    summary="List tasks",
    description="List all background tasks"
)
async def list_tasks(
    limit: int = Query(50, description="Maximum number of tasks", ge=1, le=100),
    status: Optional[str] = Query(None, description="Filter by status")
):
    """
    List background tasks.
    """
    tasks = task_manager.list_tasks(limit, status)

    return {
        "tasks": tasks,
        "total": len(tasks)
    }


@router.post(
    "/tasks/{task_id}/cancel",
    summary="Cancel task",
    description="Cancel a running background task"
)
async def cancel_task(task_id: str):
    """
    Cancel a background task.
    """
    success = await task_manager.cancel_task(task_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found or cannot be cancelled"
        )

    return {
        "success": True,
        "message": f"Task {task_id} cancelled"
    }


# ============== Document Management Endpoints ==============

@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List documents",
    description="List all ingested documents"
)
async def list_documents(
    page: int = Query(1, description="Page number", ge=1),
    page_size: int = Query(20, description="Items per page", ge=1, le=100)
):
    """
    List ingested documents.
    """
    state = get_app_state()
    vector_store = state["vector_store"]

    if not vector_store:
        return DocumentListResponse(
            documents=[],
            total=0
        )

    # Get all documents from vector store
    documents = []
    seen_docs = set()

    for i, metadata in enumerate(vector_store.metadata):
        doc_name = metadata.get("file_path", metadata.get("file_name", f"Document_{i}"))

        if doc_name not in seen_docs:
            seen_docs.add(doc_name)
            documents.append({
                "id": f"doc_{i}",
                "name": Path(doc_name).name if doc_name else f"document_{i}",
                "size_bytes": metadata.get("file_size", 0),
                "file_type": metadata.get("file_type", "unknown"),
                "ingested_at": metadata.get("ingested_at", datetime.now().isoformat()),
                "chunk_count": sum(1 for m in vector_store.metadata if m.get("file_path") == doc_name),
                "metadata": metadata
            })

    # Pagination
    total = len(documents)
    start = (page - 1) * page_size
    end = start + page_size
    paginated = documents[start:end]

    return DocumentListResponse(
        documents=paginated,
        total=total
    )


@router.delete(
    "/documents/{document_id}",
    summary="Delete document",
    description="Delete a document from the system"
)
async def delete_document(document_id: str):
    """
    Delete a document by ID.
    """
    state = get_app_state()
    vector_store = state["vector_store"]

    if not vector_store:
        raise HTTPException(
            status_code=404,
            detail="Vector store not initialized"
        )

    # Find indices to delete
    indices_to_delete = []
    doc_name = None

    for i, metadata in enumerate(vector_store.metadata):
        if metadata.get("document_id") == document_id:
            indices_to_delete.append(i)
        elif not doc_name and metadata.get("file_path"):
            if Path(metadata.get("file_path", "")).stem == document_id:
                indices_to_delete.append(i)
                doc_name = metadata.get("file_path")

    if not indices_to_delete:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{document_id}' not found"
        )

    # Delete from vector store (thread-safe)
    await run_in_threadpool(vector_store.delete, indices_to_delete)

    return {
        "success": True,
        "message": f"Deleted document '{document_id}'",
        "chunks_deleted": len(indices_to_delete)
    }


@router.delete(
    "/documents",
    summary="Delete all documents",
    description="Delete all documents from the system"
)
async def delete_all_documents():
    """
    Delete all documents.
    """
    state = get_app_state()
    vector_store = state["vector_store"]

    if not vector_store:
        raise HTTPException(
            status_code=404,
            detail="Vector store not initialized"
        )

    count = vector_store.get_size()
    await run_in_threadpool(vector_store.clear)

    return {
        "success": True,
        "message": f"Deleted all {count} documents",
        "chunks_deleted": count
    }


# ============== Streaming Endpoints ==============

@router.post(
    "/query/stream",
    summary="Ask a question with streaming",
    description="Ask a question and get streaming response"
)
async def query_stream(request: QueryRequest):
    """
    Query with streaming response.
    """
    state = get_app_state()

    if not state["retriever"] or not state["llm_interface"]:
        raise HTTPException(
            status_code=503,
            detail="System not fully initialized"
        )

    if state["vector_store"].get_size() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested"
        )

    async def generate_stream():
        try:
            # Retrieve documents
            retrieval_results = await run_in_threadpool(
                state["retriever"].retrieve,
                request.question,
                top_k=request.top_k
            )

            if not retrieval_results:
                yield json.dumps({"type": "error", "message": "No relevant documents found"}) + "\n"
                return

            # Prepare context
            context_chunks = [
                {"text": r.text, "source": r.metadata.get("file_path", "Unknown")}
                for r in retrieval_results
            ]

            # Generate prompt
            prompt = get_rag_prompt(
                question=request.question,
                chunks=context_chunks
            )

            # Stream response
            llm_response = await run_in_threadpool(
                state["llm_interface"].generate,
                [{"role": "user", "content": prompt}],
                stream=True
            )

            full_response = ""
            for chunk in llm_response:
                if chunk.content:
                    full_response += chunk.content
                    yield json.dumps({
                        "type": "chunk",
                        "content": chunk.content
                    }) + "\n"

            # Post-process and send final answer
            processed = await run_in_threadpool(
                postprocess_response,
                full_response,
                str(context_chunks[:3]),
                True
            )

            # Send sources
            sources = []
            for r in retrieval_results[:5]:
                sources.append({
                    "text": r.text[:500] + "..." if len(r.text) > 500 else r.text,
                    "score": r.score,
                    "metadata": r.metadata
                })

            yield json.dumps({
                "type": "final",
                "answer": processed.cleaned_text,
                "confidence": processed.confidence,
                "sources": sources,
                "has_hallucination": processed.has_hallucination
            }) + "\n"

        except Exception as e:
            yield json.dumps({"type": "error", "message": str(e)}) + "\n"

    return StreamingResponse(
        generate_stream(),
        media_type="application/x-ndjson"
    )


# ============== Configuration Endpoint ==============

@router.get(
    "/config",
    summary="Get configuration",
    description="Get current system configuration (sensitive info redacted)"
)
async def get_config_endpoint():
    """
    Get system configuration.
    """
    state = get_app_state()
    config = state["config"]

    if not config:
        raise HTTPException(
            status_code=503,
            detail="Configuration not available"
        )

    # Redact sensitive information
    config_dict = config.to_dict()

    # Remove API keys
    if "llm" in config_dict and "api_key" in config_dict["llm"]:
        config_dict["llm"]["api_key"] = "***REDACTED***"

    if "embedding" in config_dict and "api_key" in config_dict["embedding"]:
        config_dict["embedding"]["api_key"] = "***REDACTED***"

    if "auth" in config_dict and "api_keys" in config_dict["auth"]:
        config_dict["auth"]["api_keys"] = ["***REDACTED***"] if config_dict["auth"]["api_keys"] else []

    return config_dict
