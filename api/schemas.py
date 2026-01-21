"""
Pydantic schemas for API request/response validation.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator


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
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()


class SourceDocument(BaseModel):
    """Source document model."""
    text: str = Field(..., description="Text excerpt")
    score: float = Field(..., description="Relevance score", ge=0, le=1)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str = Field(..., description="Generated answer")
    confidence: float = Field(..., description="Confidence score", ge=0, le=1)
    sources: List[SourceDocument] = Field(default_factory=list, description="Source documents")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    tokens_used: int = Field(0, description="Total tokens used")
    has_hallucination: bool = Field(False, description="Whether hallucination was detected")


class DocumentIngestRequest(BaseModel):
    """Request model for document ingestion."""
    chunk_size: Optional[int] = Field(1000, description="Chunk size in characters", ge=100, le=10000)
    chunk_overlap: Optional[int] = Field(200, description="Chunk overlap in characters", ge=0, le=5000)
    chunking_strategy: Optional[str] = Field("recursive", description="Chunking strategy")


class DocumentIngestResponse(BaseModel):
    """Response model for document ingestion."""
    success: bool
    document_ids: List[str]
    total_chunks: int
    total_documents: int
    processing_time_seconds: float
    failed_files: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class DocumentInfo(BaseModel):
    """Document information model."""
    id: str = Field(..., description="Document ID")
    name: str = Field(..., description="Document name")
    size_bytes: int = Field(0, description="File size in bytes")
    file_type: str = Field("unknown", description="File type")
    ingested_at: str = Field(..., description="Ingestion timestamp")
    chunk_count: int = Field(0, description="Number of chunks")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class DocumentListResponse(BaseModel):
    """Response model for document listing."""
    documents: List[DocumentInfo] = Field(..., description="List of documents")
    total: int = Field(..., description="Total number of documents")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="System status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp")
    components: Dict[str, str] = Field(..., description="Component status")
    issues: List[str] = Field(default_factory=list, description="Any issues")
    vector_store_size: int = Field(0, description="Number of vectors in store")
    uptime_seconds: float = Field(0, description="Uptime in seconds")


class MetricsResponse(BaseModel):
    """Metrics response."""
    system: Dict[str, Any] = Field(..., description="System metrics")
    vector_store: Dict[str, Any] = Field(..., description="Vector store metrics")
    config: Dict[str, Any] = Field(..., description="Configuration")
    memory: Optional[Dict[str, float]] = Field(None, description="Memory usage")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: Dict[str, Any] = Field(..., description="Error details")


class DeleteResponse(BaseModel):
    """Delete operation response."""
    success: bool
    message: str
    chunks_deleted: int = 0


class DocumentStatsResponse(BaseModel):
    """Document statistics response."""
    total_documents: int = 0
    total_chunks: int = 0
    file_types: Dict[str, int] = Field(default_factory=dict)
    total_size_bytes: int = 0
    documents: List[Dict[str, Any]] = Field(default_factory=list)


class ConfigResponse(BaseModel):
    """Configuration response."""
    environment: str
    debug: bool
    llm: Dict[str, Any]
    embedding: Dict[str, Any]
    retrieval: Dict[str, Any]
    vector_store: Dict[str, Any]
    processing: Dict[str, Any]
    api: Dict[str, Any]
