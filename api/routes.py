"""
API routes for DocQA AI system.
Handles document ingestion, querying, and management endpoints.
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator

from api.app import get_app_state
from api.schemas import (
    QueryRequest, QueryResponse, DocumentIngestRequest,
    DocumentIngestResponse, DocumentListResponse, DocumentInfo,
    HealthResponse, MetricsResponse, ErrorResponse
)
from src.utils.logger import get_logger
from src.ingestion.loader import DocumentLoader
from src.ingestion.chunker import ChunkingPipeline
from src.ingestion.embedding_generator import EmbeddingGeneratorPipeline
from src.retrieval.retriever import RetrievalResult
from src.generation.prompt_templates import get_rag_prompt
from src.generation.response_postprocess import postprocess_response

logger = get_logger(__name__)

# Create router
router = APIRouter(tags=["docqa"])


# ============== Request/Response Models ==============

class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str = Field(..., description="Question to ask", min_length=1, max_length=1000)
    top_k: int = Field(5, description="Number of documents to retrieve", ge=1, le=50)
    stream: bool = Field(False, description="Stream the response")
    include_sources: bool = Field(True, description="Include source citations")
    temperature: Optional[float] = Field(None, description="LLM temperature", ge=0, le=2)
    max_tokens: Optional[int] = Field(None, description="Max tokens for response", ge=1, le=4096)

    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str = Field(..., description="Generated answer")
    confidence: float = Field(..., description="Confidence score", ge=0, le=1)
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source documents")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    tokens_used: int = Field(0, description="Total tokens used")
    has_hallucination: bool = Field(False, description="Whether hallucination was detected")


class DocumentIngestRequest(BaseModel):
    """Request model for document ingestion."""
    chunk_size: Optional[int] = Field(None, description="Chunk size in characters", ge=100, le=10000)
    chunk_overlap: Optional[int] = Field(None, description="Chunk overlap in characters", ge=0, le=5000)
    chunking_strategy: Optional[str] = Field("recursive", description="Chunking strategy")


class DocumentIngestResponse(BaseModel):
    """Response model for document ingestion."""
    success: bool
    document_ids: List[str]
    total_chunks: int
    total_documents: int
    processing_time_seconds: float
    failed_files: List[str] = Field(default_factory=list)


class DocumentListResponse(BaseModel):
    """Response model for document listing."""
    documents: List[Dict[str, Any]]
    total: int


class DocumentInfo(BaseModel):
    """Document information model."""
    id: str
    name: str
    size_bytes: int
    file_type: str
    ingested_at: str
    chunk_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============== Helper Functions ==============

def get_state():
    """Get application state."""
    return get_app_state()


async def process_ingestion(
    files: List[UploadFile],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    chunking_strategy: str = "recursive"
) -> DocumentIngestResponse:
    """
    Process document ingestion.
    """
    state = get_state()
    loader = DocumentLoader()

    # Create temporary directory for uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        uploaded_files = []
        failed_files = []

        # Save uploaded files
        for file in files:
            file_path = Path(temp_dir) / file.filename

            # Validate file extension
            extension = file_path.suffix.lower()
            supported = state["config"].processing.supported_extensions
            if extension not in supported:
                failed_files.append(f"{file.filename} (unsupported format)")
                continue

            # Save file
            try:
                content = await file.read()
                with open(file_path, 'wb') as f:
                    f.write(content)
                uploaded_files.append(str(file_path))
            except Exception as e:
                logger.error(f"Failed to save {file.filename}: {e}")
                failed_files.append(f"{file.filename} ({str(e)})")

        if not uploaded_files:
            raise HTTPException(
                status_code=400,
                detail="No valid files uploaded"
            )

        # Load documents
        documents = []
        for file_path in uploaded_files:
            try:
                doc = loader.load_document(file_path)
                documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                failed_files.append(f"{Path(file_path).name} ({str(e)})")

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
            doc_chunks = chunking_pipeline.chunk_document(
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
        embedding_pipeline = EmbeddingGeneratorPipeline(
            model=state["config"].embedding.model,
            batch_size=state["config"].embedding.batch_size,
            use_cache=state["config"].embedding.cache_enabled
        )

        # Prepare chunks for embedding
        chunk_data = [
            {"text": chunk.text, "metadata": chunk.metadata}
            for chunk in chunks
        ]

        embeddings = embedding_pipeline.generate_embeddings(chunk_data)

        # Store in vector store
        vector_store = state["vector_store"]
        if not vector_store:
            raise HTTPException(
                status_code=503,
                detail="Vector store not initialized"
            )

        # Add embeddings
        embedding_vectors = [e.embedding for e in embeddings]
        texts = [e.text for e in embeddings]
        metadata_list = [e.metadata for e in embeddings]
        chunk_ids = [f"chunk_{i}" for i in range(len(embeddings))]

        indices = vector_store.add_embeddings(
            embeddings=embedding_vectors,
            texts=texts,
            metadata=metadata_list,
            chunk_ids=chunk_ids
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
            processing_time_seconds=0,  # Will be updated by caller
            failed_files=failed_files
        )


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
    """
    import time
    start_time = time.time()

    state = get_state()

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
        # Retrieve relevant documents
        retrieval_results = state["retriever"].retrieve(
            query=request.question,
            top_k=request.top_k
        )

        if not retrieval_results:
            return QueryResponse(
                answer="I couldn't find any relevant information in the documents to answer your question.",
                confidence=0.0,
                sources=[],
                processing_time_ms=(time.time() - start_time) * 1000,
                tokens_used=0
            )

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

        # Generate response with LLM
        llm_response = state["llm_interface"].generate_simple(
            prompt=prompt,
            system_prompt="You are a helpful assistant that answers questions based on provided documents."
        )

        # Post-process response
        processed_response = postprocess_response(
            response=llm_response,
            context=str(context_chunks[:3]),  # Use first 3 chunks as context
            aggressive_cleaning=True
        )

        # Prepare sources
        sources = []
        if request.include_sources:
            for r in retrieval_results[:5]:
                source = {
                    "text": r.text[:500] + "..." if len(r.text) > 500 else r.text,
                    "score": r.score,
                    "metadata": r.metadata
                }
                sources.append(source)

        return QueryResponse(
            answer=processed_response.cleaned_text,
            confidence=processed_response.confidence,
            sources=sources,
            processing_time_ms=(time.time() - start_time) * 1000,
            tokens_used=processed_response.tokens_used,
            has_hallucination=processed_response.has_hallucination
        )

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )


@router.post(
    "/documents/ingest",
    response_model=DocumentIngestResponse,
    summary="Ingest documents",
    description="Upload documents for ingestion into the system"
)
async def ingest_documents(
    files: List[UploadFile] = File(..., description="Documents to upload"),
    chunk_size: int = Form(1000, description="Chunk size in characters", ge=100, le=10000),
    chunk_overlap: int = Form(200, description="Chunk overlap in characters", ge=0, le=5000),
    chunking_strategy: str = Form("recursive", description="Chunking strategy")
):
    """
    Ingest documents into the system.
    """
    import time
    start_time = time.time()

    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files provided"
        )

    # Validate file sizes
    config = get_state()["config"]
    max_size = config.processing.max_file_size_mb * 1024 * 1024

    for file in files:
        file.file.seek(0, 2)
        size = file.file.tell()
        file.file.seek(0)

        if size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File {file.filename} exceeds maximum size of {config.processing.max_file_size_mb}MB"
            )

    try:
        # Process ingestion
        result = await process_ingestion(
            files=files,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunking_strategy=chunking_strategy
        )

        # Update processing time
        result.processing_time_seconds = time.time() - start_time

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )


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
    state = get_state()
    vector_store = state["vector_store"]

    if not vector_store:
        return DocumentListResponse(
            documents=[],
            total=0
        )

    # Get all documents from vector store
    # Note: This is simplified - in production, you'd maintain a separate document index
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
    state = get_state()
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
            # Try to match by filename
            if Path(metadata.get("file_path", "")).stem == document_id:
                indices_to_delete.append(i)
                doc_name = metadata.get("file_path")

    if not indices_to_delete:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{document_id}' not found"
        )

    # Delete from vector store
    vector_store.delete(indices_to_delete)

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
    state = get_state()
    vector_store = state["vector_store"]

    if not vector_store:
        raise HTTPException(
            status_code=404,
            detail="Vector store not initialized"
        )

    count = vector_store.get_size()
    vector_store.clear()

    return {
        "success": True,
        "message": f"Deleted all {count} documents",
        "chunks_deleted": count
    }


@router.get(
    "/documents/stats",
    summary="Get document statistics",
    description="Get statistics about ingested documents"
)
async def get_document_stats():
    """
    Get document statistics.
    """
    state = get_state()
    vector_store = state["vector_store"]

    if not vector_store:
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "file_types": {},
            "total_size_bytes": 0
        }

    stats = {
        "total_documents": 0,
        "total_chunks": vector_store.get_size(),
        "file_types": {},
        "total_size_bytes": 0,
        "documents": []
    }

    seen_docs = set()
    for metadata in vector_store.metadata:
        doc_name = metadata.get("file_path", metadata.get("file_name", ""))
        file_type = metadata.get("file_type", "unknown")

        if doc_name not in seen_docs:
            seen_docs.add(doc_name)
            stats["total_documents"] += 1
            stats["file_types"][file_type] = stats["file_types"].get(file_type, 0) + 1
            stats["total_size_bytes"] += metadata.get("file_size", 0)

    return stats


@router.get(
    "/health",
    summary="Health check",
    description="Check system health status"
)
async def health_check():
    """
    Health check endpoint.
    """
    state = get_state()

    vector_store_ready = state["vector_store"] is not None
    retriever_ready = state["retriever"] is not None
    llm_ready = state["llm_interface"] is not None

    status = "healthy"
    issues = []

    if not vector_store_ready:
        issues.append("Vector store not initialized")
        status = "degraded"

    if not retriever_ready:
        issues.append("Retriever not initialized")
        status = "degraded"

    if not llm_ready:
        issues.append("LLM interface not initialized")
        status = "degraded"

    return {
        "status": status,
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "vector_store": "ready" if vector_store_ready else "unavailable",
            "retriever": "ready" if retriever_ready else "unavailable",
            "llm_interface": "ready" if llm_ready else "unavailable"
        },
        "issues": issues,
        "vector_store_size": state["vector_store"].get_size() if vector_store_ready else 0,
        "uptime_seconds": (
            (datetime.now() - state["startup_time"]).total_seconds()
            if state["startup_time"] else 0
        )
    }


@router.get(
    "/metrics",
    summary="Get metrics",
    description="Get system metrics and performance data"
)
async def get_metrics():
    """
    Get system metrics.
    """
    state = get_state()
    vector_store = state["vector_store"]

    metrics = {
        "system": {
            "uptime_seconds": (
                (datetime.now() - state["startup_time"]).total_seconds()
                if state["startup_time"] else 0
            ),
            "request_count": state["request_count"],
            "active_connections": state["active_connections"]
        },
        "vector_store": {
            "size": vector_store.get_size() if vector_store else 0,
            "dimension": state["config"].vector_store.dimension if state["config"] else 0,
            "index_type": state["config"].vector_store.index_type if state["config"] else "unknown"
        },
        "config": {
            "model": state["config"].llm.model if state["config"] else "unknown",
            "embedding_model": state["config"].embedding.model if state["config"] else "unknown",
            "chunk_size": state["config"].processing.chunk_size if state["config"] else 0,
            "top_k": state["config"].retrieval.top_k if state["config"] else 0
        }
    }

    # Add memory usage if available
    try:
        import psutil
        process = psutil.Process()
        memory = process.memory_info()
        metrics["memory"] = {
            "rss_mb": memory.rss / 1024 / 1024,
            "vms_mb": memory.vms / 1024 / 1024,
            "percent": process.memory_percent()
        }
    except ImportError:
        pass

    return metrics


@router.get(
    "/config",
    summary="Get configuration",
    description="Get current system configuration (sensitive info redacted)"
)
async def get_config():
    """
    Get system configuration.
    """
    state = get_state()
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
