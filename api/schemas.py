"""
Pydantic schemas for API request/response validation.
Provides comprehensive validation, examples, and documentation for all API endpoints.
"""

from typing import List, Dict, Any, Optional, Union, Literal
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator, confloat, conint, conlist
import re


# ============================================================
# Enums
# ============================================================

class DocumentStatus(str, Enum):
    """Document status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DELETED = "deleted"


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ChunkingStrategy(str, Enum):
    """Chunking strategy enumeration."""
    FIXED_SIZE = "fixed_size"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    RECURSIVE = "recursive"
    SLIDING_WINDOW = "sliding_window"
    ADAPTIVE = "adaptive"
    MARKDOWN = "markdown"
    CODE = "code"


class LLMProvider(str, Enum):
    """LLM provider enumeration."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    GEMINI = "gemini"
    COHERE = "cohere"
    GROQ = "groq"
    LOCAL = "local"
    OLLAMA = "ollama"


class ResponseFormat(str, Enum):
    """Response format enumeration."""
    SSE = "sse"
    NDJSON = "ndjson"
    TEXT = "text"
    JSON = "json"


# ============================================================
# Base Models
# ============================================================

class ErrorDetail(BaseModel):
    """Error detail model."""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

    class Config:
        json_schema_extra = {
            "example": {
                "code": "RATE_LIMIT_EXCEEDED",
                "message": "Rate limit exceeded. Maximum 100 requests per 60 seconds.",
                "timestamp": "2026-06-19T10:30:00.000Z",
                "details": {"retry_after": 45}
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    error: ErrorDetail = Field(..., description="Error details")


class SuccessResponse(BaseModel):
    """Success response model."""
    success: bool = Field(True, description="Success status")
    message: str = Field(..., description="Success message")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Response timestamp")


# ============================================================
# Query Schemas
# ============================================================

class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str = Field(
        ...,
        description="Question to ask about your documents",
        min_length=1,
        max_length=1000,
        example="What is the main topic of the document?"
    )
    top_k: conint(ge=1, le=50) = Field(
        5,
        description="Number of documents to retrieve",
        example=5
    )
    temperature: Optional[confloat(ge=0, le=2)] = Field(
        None,
        description="LLM temperature (0-2). Higher = more creative",
        example=0.7
    )
    max_tokens: Optional[conint(ge=1, le=4096)] = Field(
        None,
        description="Maximum tokens in response",
        example=500
    )
    stream: bool = Field(
        False,
        description="Stream the response",
        example=False
    )
    include_sources: bool = Field(
        True,
        description="Include source citations in response",
        example=True
    )
    include_confidence: bool = Field(
        True,
        description="Include confidence score in response",
        example=True
    )
    session_id: Optional[str] = Field(
        None,
        description="Session ID for conversation context",
        example="session_123"
    )

    @validator('question')
    def validate_question(cls, v):
        """Validate question is not empty and contains meaningful content."""
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        if len(v.strip()) < 3:
            raise ValueError("Question must be at least 3 characters")
        return v.strip()

    @root_validator
    def validate_temperature_with_stream(cls, values):
        """Validate temperature is provided when streaming."""
        # No validation needed, temperature is optional
        return values

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is machine learning?",
                "top_k": 5,
                "temperature": 0.7,
                "max_tokens": 500,
                "stream": False,
                "include_sources": True,
                "include_confidence": True
            }
        }


class SourceDocument(BaseModel):
    """Source document model."""
    text: str = Field(..., description="Text excerpt from source document")
    score: confloat(ge=0, le=1) = Field(..., description="Relevance score")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Document metadata"
    )
    chunk_id: Optional[str] = Field(None, description="Chunk identifier")
    document_id: Optional[str] = Field(None, description="Document identifier")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Machine learning is a subset of artificial intelligence...",
                "score": 0.95,
                "metadata": {"file_path": "sample.pdf", "page": 5},
                "chunk_id": "chunk_123",
                "document_id": "doc_456"
            }
        }


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str = Field(..., description="Generated answer")
    confidence: confloat(ge=0, le=1) = Field(
        ...,
        description="Confidence score (0-1)"
    )
    sources: List[SourceDocument] = Field(
        default_factory=list,
        description="Source documents used for answer"
    )
    processing_time_ms: float = Field(
        ...,
        description="Total processing time in milliseconds"
    )
    tokens_used: int = Field(
        0,
        description="Total tokens used",
        example=150
    )
    has_hallucination: bool = Field(
        False,
        description="Whether hallucination was detected"
    )
    model: Optional[str] = Field(
        None,
        description="LLM model used",
        example="gpt-4"
    )
    session_id: Optional[str] = Field(
        None,
        description="Session ID for conversation context"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Response timestamp"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                "confidence": 0.92,
                "sources": [
                    {
                        "text": "Machine learning is a subset of artificial intelligence...",
                        "score": 0.95,
                        "metadata": {"file_path": "sample.pdf"}
                    }
                ],
                "processing_time_ms": 450.0,
                "tokens_used": 150,
                "has_hallucination": False,
                "model": "gpt-4",
                "session_id": "session_123"
            }
        }


# ============================================================
# Document Ingestion Schemas
# ============================================================

class DocumentIngestRequest(BaseModel):
    """Request model for document ingestion."""
    chunk_size: conint(ge=100, le=10000) = Field(
        800,
        description="Chunk size in characters",
        example=800
    )
    chunk_overlap: conint(ge=0, le=5000) = Field(
        150,
        description="Chunk overlap in characters",
        example=150
    )
    chunking_strategy: ChunkingStrategy = Field(
        ChunkingStrategy.ADAPTIVE,
        description="Chunking strategy"
    )
    extract_tables: bool = Field(
        True,
        description="Extract tables from documents",
        example=True
    )
    extract_metadata: bool = Field(
        True,
        description="Extract metadata from documents",
        example=True
    )
    preserve_formatting: bool = Field(
        True,
        description="Preserve formatting (bold, italic)",
        example=True
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Custom metadata to add to all documents"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_size": 800,
                "chunk_overlap": 150,
                "chunking_strategy": "adaptive",
                "extract_tables": True,
                "extract_metadata": True,
                "preserve_formatting": True
            }
        }


class DocumentIngestResponse(BaseModel):
    """Response model for document ingestion."""
    success: bool = Field(..., description="Success status")
    document_ids: List[str] = Field(
        default_factory=list,
        description="List of document IDs"
    )
    total_chunks: int = Field(
        ...,
        description="Total number of chunks created"
    )
    total_documents: int = Field(
        ...,
        description="Total number of documents processed"
    )
    processing_time_seconds: float = Field(
        ...,
        description="Total processing time in seconds"
    )
    failed_files: List[str] = Field(
        default_factory=list,
        description="List of failed files"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Response timestamp"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "document_ids": ["doc_20260101_001", "doc_20260101_002"],
                "total_chunks": 42,
                "total_documents": 2,
                "processing_time_seconds": 3.2,
                "failed_files": [],
                "warnings": ["Large document truncated"],
                "timestamp": "2026-06-19T10:30:00.000Z"
            }
        }


class DocumentUploadResponse(BaseModel):
    """Response model for document upload (background task)."""
    success: bool = Field(..., description="Success status")
    task_id: str = Field(..., description="Background task ID")
    message: str = Field(..., description="Status message")
    status_url: str = Field(..., description="URL to check task status")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "task_id": "task_123e4567-e89b-12d3-a456-426614174000",
                "message": "Document ingestion started as background task",
                "status_url": "/api/v1/tasks/task_123e4567-e89b-12d3-a456-426614174000"
            }
        }


# ============================================================
# Document Management Schemas
# ============================================================

class DocumentInfo(BaseModel):
    """Document information model."""
    id: str = Field(..., description="Document ID")
    name: str = Field(..., description="Document name")
    size_bytes: int = Field(..., description="File size in bytes")
    file_type: str = Field(..., description="File type")
    ingested_at: str = Field(..., description="Ingestion timestamp")
    chunk_count: int = Field(..., description="Number of chunks")
    status: DocumentStatus = Field(
        DocumentStatus.COMPLETED,
        description="Document status"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Document metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "doc_20260101_001",
                "name": "sample.pdf",
                "size_bytes": 245000,
                "file_type": "pdf",
                "ingested_at": "2026-01-15T10:30:00.000Z",
                "chunk_count": 15,
                "status": "completed",
                "metadata": {"author": "John Doe", "title": "Sample Document"}
            }
        }


class DocumentListResponse(BaseModel):
    """Response model for document listing."""
    documents: List[DocumentInfo] = Field(
        ...,
        description="List of documents"
    )
    total: int = Field(..., description="Total number of documents")
    page: int = Field(1, description="Current page number")
    page_size: int = Field(20, description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")

    class Config:
        json_schema_extra = {
            "example": {
                "documents": [
                    {
                        "id": "doc_20260101_001",
                        "name": "sample.pdf",
                        "size_bytes": 245000,
                        "file_type": "pdf",
                        "ingested_at": "2026-01-15T10:30:00.000Z",
                        "chunk_count": 15,
                        "status": "completed",
                        "metadata": {}
                    }
                ],
                "total": 1,
                "page": 1,
                "page_size": 20,
                "total_pages": 1
            }
        }


class DocumentStatsResponse(BaseModel):
    """Document statistics response."""
    total_documents: int = Field(0, description="Total number of documents")
    total_chunks: int = Field(0, description="Total number of chunks")
    total_size_bytes: int = Field(0, description="Total size in bytes")
    file_types: Dict[str, int] = Field(
        default_factory=dict,
        description="Breakdown by file type"
    )
    status_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Breakdown by status"
    )
    vector_store_size: int = Field(0, description="Vector store size")

    class Config:
        json_schema_extra = {
            "example": {
                "total_documents": 10,
                "total_chunks": 150,
                "total_size_bytes": 2500000,
                "file_types": {"pdf": 5, "docx": 3, "txt": 2},
                "status_counts": {"completed": 10},
                "vector_store_size": 150
            }
        }


class DeleteDocumentResponse(BaseModel):
    """Delete document response."""
    success: bool = Field(..., description="Success status")
    message: str = Field(..., description="Status message")
    chunks_deleted: int = Field(0, description="Number of chunks deleted")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Deleted document 'doc_20260101_001'",
                "chunks_deleted": 15
            }
        }


# ============================================================
# Task Management Schemas
# ============================================================

class TaskStatusResponse(BaseModel):
    """Task status response model."""
    id: str = Field(..., description="Task ID")
    name: str = Field(..., description="Task name")
    status: TaskStatus = Field(..., description="Task status")
    created_at: str = Field(..., description="Creation timestamp")
    started_at: Optional[str] = Field(None, description="Start timestamp")
    completed_at: Optional[str] = Field(None, description="Completion timestamp")
    progress: float = Field(0.0, description="Progress (0-100)")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result")
    error: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Task metadata"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "task_123e4567-e89b-12d3-a456-426614174000",
                "name": "ingest_documents",
                "status": "running",
                "created_at": "2026-06-19T10:30:00.000Z",
                "started_at": "2026-06-19T10:30:01.000Z",
                "completed_at": None,
                "progress": 45.0,
                "result": None,
                "error": None,
                "metadata": {"files": 5}
            }
        }


class TaskListResponse(BaseModel):
    """Task list response model."""
    tasks: List[TaskStatusResponse] = Field(..., description="List of tasks")
    total: int = Field(..., description="Total number of tasks")

    class Config:
        json_schema_extra = {
            "example": {
                "tasks": [
                    {
                        "id": "task_123",
                        "name": "ingest_documents",
                        "status": "running",
                        "created_at": "2026-06-19T10:30:00.000Z",
                        "started_at": "2026-06-19T10:30:01.000Z",
                        "completed_at": None,
                        "progress": 45.0,
                        "result": None,
                        "error": None,
                        "metadata": {}
                    }
                ],
                "total": 1
            }
        }


# ============================================================
# Health and Metrics Schemas
# ============================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ...,
        description="System health status"
    )
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    uptime_seconds: float = Field(0.0, description="System uptime in seconds")
    components: Dict[str, str] = Field(
        ...,
        description="Component status"
    )
    vector_store_size: int = Field(0, description="Number of vectors in store")
    active_tasks: int = Field(0, description="Number of active tasks")
    issues: List[str] = Field(
        default_factory=list,
        description="Any issues or warnings"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2026-06-19T10:30:00.000Z",
                "version": "1.0.0",
                "uptime_seconds": 86400.0,
                "components": {
                    "vector_store": "ready",
                    "retriever": "ready",
                    "llm_interface": "ready"
                },
                "vector_store_size": 1000,
                "active_tasks": 0,
                "issues": []
            }
        }


class MemoryMetrics(BaseModel):
    """Memory metrics model."""
    rss_mb: float = Field(..., description="RSS memory in MB")
    vms_mb: float = Field(..., description="VMS memory in MB")
    percent: float = Field(..., description="Memory usage percentage")

    class Config:
        json_schema_extra = {
            "example": {
                "rss_mb": 512.0,
                "vms_mb": 1024.0,
                "percent": 25.5
            }
        }


class RequestMetrics(BaseModel):
    """Request metrics model."""
    total: int = Field(..., description="Total requests")
    active_connections: int = Field(..., description="Active connections")

    class Config:
        json_schema_extra = {
            "example": {
                "total": 10000,
                "active_connections": 5
            }
        }


class CacheMetrics(BaseModel):
    """Cache metrics model."""
    hits: int = Field(..., description="Cache hits")
    misses: int = Field(..., description="Cache misses")
    hit_rate: float = Field(..., description="Cache hit rate")
    total_entries: int = Field(..., description="Total cache entries")
    size_mb: float = Field(..., description="Cache size in MB")

    class Config:
        json_schema_extra = {
            "example": {
                "hits": 1000,
                "misses": 200,
                "hit_rate": 0.83,
                "total_entries": 500,
                "size_mb": 50.0
            }
        }


class MetricsResponse(BaseModel):
    """Metrics response model."""
    requests: RequestMetrics = Field(..., description="Request metrics")
    vector_store: Dict[str, Any] = Field(..., description="Vector store metrics")
    cache: Optional[Dict[str, Any]] = Field(None, description="Cache metrics")
    tasks: Dict[str, Any] = Field(..., description="Task metrics")
    system: Dict[str, Any] = Field(..., description="System metrics")
    memory: Optional[MemoryMetrics] = Field(None, description="Memory metrics")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Response timestamp"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "requests": {"total": 10000, "active_connections": 5},
                "vector_store": {"size": 1000, "dimension": 1536},
                "cache": {"hits": 1000, "misses": 200, "hit_rate": 0.83},
                "tasks": {"total": 10, "running": 0, "pending": 0},
                "system": {"uptime_seconds": 86400},
                "memory": {"rss_mb": 512.0, "vms_mb": 1024.0, "percent": 25.5}
            }
        }


# ============================================================
# Rate Limit Schemas
# ============================================================

class RateLimitStatusResponse(BaseModel):
    """Rate limit status response."""
    identifier: str = Field(..., description="Client identifier")
    endpoint: str = Field(..., description="Endpoint")
    limit: int = Field(..., description="Rate limit")
    remaining: int = Field(..., description="Remaining requests")
    window: int = Field(..., description="Window size in seconds")
    count: int = Field(..., description="Requests in current window")
    allowed: bool = Field(..., description="Whether request is allowed")

    class Config:
        json_schema_extra = {
            "example": {
                "identifier": "ip:192.168.1.100",
                "endpoint": "/api/v1/query",
                "limit": 100,
                "remaining": 95,
                "window": 60,
                "count": 5,
                "allowed": True
            }
        }


# ============================================================
# Config Schemas
# ============================================================

class ConfigResponse(BaseModel):
    """Configuration response model (sensitive info redacted)."""
    environment: str = Field(..., description="Environment")
    debug: bool = Field(..., description="Debug mode")
    llm: Dict[str, Any] = Field(..., description="LLM configuration")
    embedding: Dict[str, Any] = Field(..., description="Embedding configuration")
    retrieval: Dict[str, Any] = Field(..., description="Retrieval configuration")
    vector_store: Dict[str, Any] = Field(..., description="Vector store configuration")
    processing: Dict[str, Any] = Field(..., description="Processing configuration")
    api: Dict[str, Any] = Field(..., description="API configuration")
    monitoring: Dict[str, Any] = Field(..., description="Monitoring configuration")

    class Config:
        json_schema_extra = {
            "example": {
                "environment": "production",
                "debug": False,
                "llm": {"provider": "openai", "model": "gpt-4", "temperature": 0.7},
                "embedding": {"model": "text-embedding-3-small", "dimension": 1536},
                "retrieval": {"top_k": 5, "score_threshold": 0.7},
                "vector_store": {"type": "faiss", "index_type": "HNSW64"},
                "processing": {"chunk_size": 800, "chunk_overlap": 150},
                "api": {"host": "0.0.0.0", "port": 8000},
                "monitoring": {"enabled": True}
            }
        }


# ============================================================
# Versioning Schemas
# ============================================================

class VersionInfo(BaseModel):
    """Version information model."""
    version_id: str = Field(..., description="Version ID")
    version_number: int = Field(..., description="Version number")
    created_at: str = Field(..., description="Creation timestamp")
    created_by: str = Field(..., description="Created by")
    action: str = Field(..., description="Action that created version")
    status: str = Field(..., description="Version status")
    description: str = Field(..., description="Version description")
    size_bytes: int = Field(..., description="Version size in bytes")
    hash: str = Field(..., description="Content hash")
    tags: List[str] = Field(default_factory=list, description="Version tags")

    class Config:
        json_schema_extra = {
            "example": {
                "version_id": "v1_a1b2c3d4",
                "version_number": 1,
                "created_at": "2026-06-19T10:30:00.000Z",
                "created_by": "user123",
                "action": "create",
                "status": "active",
                "description": "Initial document version",
                "size_bytes": 1024,
                "hash": "abc123def456",
                "tags": ["initial", "draft"]
            }
        }


class VersionHistoryResponse(BaseModel):
    """Version history response model."""
    document_id: str = Field(..., description="Document ID")
    versions: List[VersionInfo] = Field(..., description="Version list")
    total: int = Field(..., description="Total number of versions")
    current_version: int = Field(..., description="Current version number")

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_20260101_001",
                "versions": [
                    {
                        "version_id": "v2_e5f6g7h8",
                        "version_number": 2,
                        "created_at": "2026-06-19T10:30:00.000Z",
                        "created_by": "user123",
                        "action": "update",
                        "status": "active",
                        "description": "Updated content",
                        "size_bytes": 2048,
                        "hash": "def456ghi789",
                        "tags": ["updated"]
                    }
                ],
                "total": 2,
                "current_version": 2
            }
        }


class CompareVersionsResponse(BaseModel):
    """Compare versions response model."""
    version_1: Dict[str, Any] = Field(..., description="First version info")
    version_2: Dict[str, Any] = Field(..., description="Second version info")
    total_changes: int = Field(..., description="Total number of changes")
    added_lines: int = Field(..., description="Lines added")
    removed_lines: int = Field(..., description="Lines removed")
    unchanged_lines: int = Field(..., description="Lines unchanged")
    change_percentage: float = Field(..., description="Percentage of changes")
    summary: str = Field(..., description="Change summary")
    diff_lines: List[str] = Field(..., description="Diff lines")

    class Config:
        json_schema_extra = {
            "example": {
                "version_1": {"id": "v1", "number": 1, "created_at": "2026-06-19T10:30:00.000Z"},
                "version_2": {"id": "v2", "number": 2, "created_at": "2026-06-19T11:30:00.000Z"},
                "total_changes": 10,
                "added_lines": 3,
                "removed_lines": 2,
                "unchanged_lines": 5,
                "change_percentage": 50.0,
                "summary": "3 additions, 2 deletions, 5 unchanged",
                "diff_lines": ["-old line", "+new line"]
            }
        }


# ============================================================
# Streaming Schemas
# ============================================================

class StreamEvent(BaseModel):
    """Stream event model."""
    event: str = Field(..., description="Event type")
    id: str = Field(..., description="Event ID")
    timestamp: str = Field(..., description="Event timestamp")
    data: Dict[str, Any] = Field(..., description="Event data")

    class Config:
        json_schema_extra = {
            "example": {
                "event": "token",
                "id": "evt_123",
                "timestamp": "2026-06-19T10:30:00.000Z",
                "data": {"content": "The answer is..."}
            }
        }


class StreamStartEvent(BaseModel):
    """Stream start event."""
    event: Literal["start"] = "start"
    question: str = Field(..., description="User question")
    top_k: int = Field(..., description="Number of documents retrieved")
    model: str = Field(..., description="Model being used")
    timestamp: str = Field(..., description="Start timestamp")


class StreamTokenEvent(BaseModel):
    """Stream token event."""
    event: Literal["token"] = "token"
    content: str = Field(..., description="Token content")
    token_count: int = Field(..., description="Total tokens generated so far")


class StreamSourceEvent(BaseModel):
    """Stream source event."""
    event: Literal["source"] = "source"
    sources: List[SourceDocument] = Field(..., description="Retrieved sources")
    retrieval_time_ms: float = Field(..., description="Retrieval time in milliseconds")


class StreamProgressEvent(BaseModel):
    """Stream progress event."""
    event: Literal["progress"] = "progress"
    stage: str = Field(..., description="Current stage")
    message: str = Field(..., description="Progress message")
    progress: float = Field(..., description="Progress (0-100)")


class StreamFinalEvent(BaseModel):
    """Stream final event."""
    event: Literal["final"] = "final"
    answer: str = Field(..., description="Final answer")
    confidence: float = Field(..., description="Confidence score")
    sources: List[SourceDocument] = Field(..., description="Source documents")
    tokens_used: int = Field(..., description="Total tokens used")
    has_hallucination: bool = Field(..., description="Whether hallucination was detected")
    processing_time_ms: float = Field(..., description="Total processing time")
    retrieval_time_ms: float = Field(..., description="Retrieval time")
    generation_time_ms: float = Field(..., description="Generation time")


class StreamErrorEvent(BaseModel):
    """Stream error event."""
    event: Literal["error"] = "error"
    message: str = Field(..., description="Error message")
    type: str = Field(..., description="Error type")


class StreamDoneEvent(BaseModel):
    """Stream done event."""
    event: Literal["done"] = "done"
    timestamp: str = Field(..., description="Completion timestamp")


# ============================================================
# Pagination Schemas
# ============================================================

class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: conint(ge=1) = Field(1, description="Page number")
    page_size: conint(ge=1, le=100) = Field(20, description="Items per page")

    class Config:
        json_schema_extra = {
            "example": {"page": 1, "page_size": 20}
        }


class PaginatedResponse(BaseModel):
    """Generic paginated response."""
    items: List[Any] = Field(..., description="List of items")
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page")
    page_size: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total pages")


# ============================================================
# Export Schemas
# ============================================================

class ExportRequest(BaseModel):
    """Export request model."""
    format: Literal["json", "csv", "txt"] = Field(
        "json",
        description="Export format"
    )
    include_metadata: bool = Field(
        True,
        description="Include metadata in export"
    )
    include_embeddings: bool = Field(
        False,
        description="Include embeddings in export"
    )
    document_ids: Optional[List[str]] = Field(
        None,
        description="Specific documents to export"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "format": "json",
                "include_metadata": True,
                "include_embeddings": False,
                "document_ids": ["doc_123", "doc_456"]
            }
        }


# ============================================================
# Authentication Schemas
# ============================================================

class LoginRequest(BaseModel):
    """Login request model."""
    username: str = Field(..., description="Username", min_length=3, max_length=50)
    password: str = Field(..., description="Password", min_length=8)

    class Config:
        json_schema_extra = {
            "example": {"username": "admin", "password": "secure_password"}
        }


class LoginResponse(BaseModel):
    """Login response model."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("Bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration in seconds")
    user: Dict[str, Any] = Field(..., description="User information")


class ApiKeyRequest(BaseModel):
    """API key request model."""
    name: str = Field(..., description="API key name", min_length=1, max_length=100)
    expires_in_days: conint(ge=1, le=365) = Field(
        30,
        description="Expiration in days"
    )
    permissions: List[str] = Field(
        default_factory=list,
        description="Permissions granted"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Production API Key",
                "expires_in_days": 30,
                "permissions": ["query", "ingest"]
            }
        }


class ApiKeyResponse(BaseModel):
    """API key response model."""
    id: str = Field(..., description="API key ID")
    name: str = Field(..., description="API key name")
    key: str = Field(..., description="API key value")
    created_at: str = Field(..., description="Creation timestamp")
    expires_at: str = Field(..., description="Expiration timestamp")
    permissions: List[str] = Field(..., description="Permissions")
    last_used: Optional[str] = Field(None, description="Last usage timestamp")


# ============================================================
# Utility Functions
# ============================================================

def create_error_response(
    code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None
) -> ErrorResponse:
    """
    Create a standardized error response.

    Args:
        code: Error code
        message: Error message
        details: Additional error details

    Returns:
        ErrorResponse object
    """
    return ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            timestamp=datetime.now().isoformat(),
            details=details
        )
    )


def create_success_response(
    message: str,
    data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a standardized success response.

    Args:
        message: Success message
        data: Additional data

    Returns:
        Success response dictionary
    """
    response = {
        "success": True,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    if data:
        response.update(data)
    return response


# ============================================================
# Schema Exports
# ============================================================

__all__ = [
    # Enums
    "DocumentStatus",
    "TaskStatus",
    "ChunkingStrategy",
    "LLMProvider",
    "ResponseFormat",

    # Base Models
    "ErrorDetail",
    "ErrorResponse",
    "SuccessResponse",

    # Query Schemas
    "QueryRequest",
    "SourceDocument",
    "QueryResponse",

    # Document Schemas
    "DocumentIngestRequest",
    "DocumentIngestResponse",
    "DocumentUploadResponse",
    "DocumentInfo",
    "DocumentListResponse",
    "DocumentStatsResponse",
    "DeleteDocumentResponse",

    # Task Schemas
    "TaskStatusResponse",
    "TaskListResponse",

    # Health and Metrics
    "HealthResponse",
    "MemoryMetrics",
    "RequestMetrics",
    "CacheMetrics",
    "MetricsResponse",

    # Rate Limit
    "RateLimitStatusResponse",

    # Config
    "ConfigResponse",

    # Versioning
    "VersionInfo",
    "VersionHistoryResponse",
    "CompareVersionsResponse",

    # Streaming
    "StreamEvent",
    "StreamStartEvent",
    "StreamTokenEvent",
    "StreamSourceEvent",
    "StreamProgressEvent",
    "StreamFinalEvent",
    "StreamErrorEvent",
    "StreamDoneEvent",

    # Pagination
    "PaginationParams",
    "PaginatedResponse",

    # Export
    "ExportRequest",

    # Authentication
    "LoginRequest",
    "LoginResponse",
    "ApiKeyRequest",
    "ApiKeyResponse",

    # Utilities
    "create_error_response",
    "create_success_response",
]
