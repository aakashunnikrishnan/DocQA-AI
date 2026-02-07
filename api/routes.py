"""
API routes for DocQA AI system with async support.
Handles document ingestion, querying, and management endpoints.
ENHANCED: Full streaming support with multiple formats and real-time updates.
"""

import os
import json
import asyncio
import time
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, Request, status, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse, PlainTextResponse
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
from src.generation.llm_interface import LLMResponse

logger = get_logger(__name__)

# Create router
router = APIRouter(tags=["docqa"])


# ============================================================
# Streaming Response Models
# ============================================================

class StreamEventType:
    """Stream event types."""
    START = "start"
    TOKEN = "token"
    CHUNK = "chunk"
    SOURCE = "source"
    PROGRESS = "progress"
    THOUGHT = "thought"
    FINAL = "final"
    ERROR = "error"
    DONE = "done"


class StreamEvent:
    """Stream event for SSE and JSON streaming."""

    def __init__(self, event_type: str, data: Any, event_id: Optional[str] = None):
        self.event_type = event_type
        self.data = data
        self.event_id = event_id or str(time.time())
        self.timestamp = datetime.now().isoformat()

    def to_sse(self) -> str:
        """Convert to Server-Sent Events format."""
        lines = []
        lines.append(f"event: {self.event_type}")
        lines.append(f"id: {self.event_id}")
        lines.append(f"data: {json.dumps(self.data)}")
        lines.append("")
        return "\n".join(lines)

    def to_ndjson(self) -> str:
        """Convert to NDJSON format."""
        return json.dumps({
            "event": self.event_type,
            "id": self.event_id,
            "timestamp": self.timestamp,
            "data": self.data
        }) + "\n"


# ============================================================
# Streaming Helper Functions
# ============================================================

async def stream_query_response(
    question: str,
    top_k: int = 5,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    include_sources: bool = True,
    stream_format: str = "sse",  # sse, ndjson, text
    show_thoughts: bool = False
) -> AsyncGenerator[str, None]:
    """
    Stream query response with progressive updates.

    Args:
        question: User question
        top_k: Number of documents to retrieve
        temperature: LLM temperature
        max_tokens: Max tokens for response
        include_sources: Include source citations
        stream_format: Output format (sse, ndjson, text)
        show_thoughts: Show thought process

    Yields:
        Formatted stream events
    """
    state = get_app_state()
    start_time = time.time()

    # Validate state
    if not state["retriever"] or not state["llm_interface"]:
        yield StreamEvent(
            StreamEventType.ERROR,
            {"message": "System not fully initialized"}
        ).to_sse() if stream_format == "sse" else StreamEvent(
            StreamEventType.ERROR,
            {"message": "System not fully initialized"}
        ).to_ndjson()
        return

    if state["vector_store"].get_size() == 0:
        yield StreamEvent(
            StreamEventType.ERROR,
            {"message": "No documents ingested. Please upload documents first."}
        ).to_sse() if stream_format == "sse" else StreamEvent(
            StreamEventType.ERROR,
            {"message": "No documents ingested. Please upload documents first."}
        ).to_ndjson()
        return

    try:
        # Start event
        yield StreamEvent(
            StreamEventType.START,
            {
                "question": question,
                "top_k": top_k,
                "model": state["config"].llm.model,
                "timestamp": datetime.now().isoformat()
            }
        ).to_sse() if stream_format == "sse" else StreamEvent(
            StreamEventType.START,
            {
                "question": question,
                "top_k": top_k,
                "model": state["config"].llm.model,
                "timestamp": datetime.now().isoformat()
            }
        ).to_ndjson()

        # Progress: Retrieving documents
        yield StreamEvent(
            StreamEventType.PROGRESS,
            {"stage": "retrieving", "message": "Searching for relevant documents...", "progress": 0.2}
        ).to_sse() if stream_format == "sse" else StreamEvent(
            StreamEventType.PROGRESS,
            {"stage": "retrieving", "message": "Searching for relevant documents...", "progress": 0.2}
        ).to_ndjson()

        # Retrieve documents
        retrieval_start = time.time()
        retrieval_results = await run_in_threadpool(
            state["retriever"].retrieve,
            question,
            top_k=top_k
        )
        retrieval_time = (time.time() - retrieval_start) * 1000

        if not retrieval_results:
            yield StreamEvent(
                StreamEventType.FINAL,
                {
                    "answer": "I couldn't find any relevant information in the documents to answer your question.",
                    "confidence": 0.0,
                    "sources": [],
                    "tokens_used": 0,
                    "processing_time_ms": (time.time() - start_time) * 1000
                }
            ).to_sse() if stream_format == "sse" else StreamEvent(
                StreamEventType.FINAL,
                {
                    "answer": "I couldn't find any relevant information in the documents to answer your question.",
                    "confidence": 0.0,
                    "sources": [],
                    "tokens_used": 0,
                    "processing_time_ms": (time.time() - start_time) * 1000
                }
            ).to_ndjson()

            yield StreamEvent(
                StreamEventType.DONE,
                {"timestamp": datetime.now().isoformat()}
            ).to_sse() if stream_format == "sse" else StreamEvent(
                StreamEventType.DONE,
                {"timestamp": datetime.now().isoformat()}
            ).to_ndjson()
            return

        # Send sources
        if include_sources:
            sources = []
            for r in retrieval_results[:5]:
                sources.append({
                    "text": r.text[:300] + "..." if len(r.text) > 300 else r.text,
                    "score": r.score,
                    "metadata": r.metadata
                })

            yield StreamEvent(
                StreamEventType.SOURCE,
                {"sources": sources, "retrieval_time_ms": retrieval_time}
            ).to_sse() if stream_format == "sse" else StreamEvent(
                StreamEventType.SOURCE,
                {"sources": sources, "retrieval_time_ms": retrieval_time}
            ).to_ndjson()

        # Progress: Generating response
        yield StreamEvent(
            StreamEventType.PROGRESS,
            {"stage": "generating", "message": f"Generating response using {len(retrieval_results)} sources...", "progress": 0.5}
        ).to_sse() if stream_format == "sse" else StreamEvent(
            StreamEventType.PROGRESS,
            {"stage": "generating", "message": f"Generating response using {len(retrieval_results)} sources...", "progress": 0.5}
        ).to_ndjson()

        # Prepare context
        context_chunks = [
            {"text": r.text, "source": r.metadata.get("file_path", "Unknown")}
            for r in retrieval_results[:3]
        ]

        # Generate prompt
        prompt = get_rag_prompt(
            question=question,
            chunks=context_chunks
        )

        # Stream LLM response
        full_response = ""
        token_count = 0
        thought_parts = []

        if show_thoughts:
            yield StreamEvent(
                StreamEventType.THOUGHT,
                {"content": "Analyzing query and context...", "stage": "analysis"}
            ).to_sse() if stream_format == "sse" else StreamEvent(
                StreamEventType.THOUGHT,
                {"content": "Analyzing query and context...", "stage": "analysis"}
            ).to_ndjson()

        # Generate response with streaming
        llm_stream = await run_in_threadpool(
            state["llm_interface"].generate,
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )

        # Process stream
        for chunk in llm_stream:
            if isinstance(chunk, LLMResponse) and chunk.content:
                token_count += 1
                full_response += chunk.content

                # Send token chunk
                yield StreamEvent(
                    StreamEventType.TOKEN,
                    {"content": chunk.content, "token_count": token_count}
                ).to_sse() if stream_format == "sse" else StreamEvent(
                    StreamEventType.TOKEN,
                    {"content": chunk.content, "token_count": token_count}
                ).to_ndjson()

                # Update progress
                if token_count % 10 == 0:
                    progress = min(0.9, 0.5 + (token_count / 100) * 0.4)
                    yield StreamEvent(
                        StreamEventType.PROGRESS,
                        {"stage": "generating", "message": f"Generating... ({token_count} tokens)", "progress": progress}
                    ).to_sse() if stream_format == "sse" else StreamEvent(
                        StreamEventType.PROGRESS,
                        {"stage": "generating", "message": f"Generating... ({token_count} tokens)", "progress": progress}
                    ).to_ndjson()

        # Show thoughts about completion
        if show_thoughts:
            thought_parts.append("Response generated successfully")
            yield StreamEvent(
                StreamEventType.THOUGHT,
                {"content": "Response generated successfully", "stage": "completion"}
            ).to_sse() if stream_format == "sse" else StreamEvent(
                StreamEventType.THOUGHT,
                {"content": "Response generated successfully", "stage": "completion"}
            ).to_ndjson()

        # Post-process response
        processed_response = await run_in_threadpool(
            postprocess_response,
            full_response,
            str(context_chunks[:3]),
            aggressive_cleaning=True
        )

        # Prepare final response
        final_sources = []
        if include_sources:
            for r in retrieval_results[:5]:
                final_sources.append({
                    "text": r.text[:500] + "..." if len(r.text) > 500 else r.text,
                    "score": r.score,
                    "metadata": r.metadata
                })

        # Send final response
        yield StreamEvent(
            StreamEventType.FINAL,
            {
                "answer": processed_response.cleaned_text,
                "confidence": processed_response.confidence,
                "sources": final_sources,
                "tokens_used": token_count,
                "has_hallucination": processed_response.has_hallucination,
                "processing_time_ms": (time.time() - start_time) * 1000,
                "retrieval_time_ms": retrieval_time,
                "generation_time_ms": (time.time() - retrieval_start) * 1000
            }
        ).to_sse() if stream_format == "sse" else StreamEvent(
            StreamEventType.FINAL,
            {
                "answer": processed_response.cleaned_text,
                "confidence": processed_response.confidence,
                "sources": final_sources,
                "tokens_used": token_count,
                "has_hallucination": processed_response.has_hallucination,
                "processing_time_ms": (time.time() - start_time) * 1000,
                "retrieval_time_ms": retrieval_time,
                "generation_time_ms": (time.time() - retrieval_start) * 1000
            }
        ).to_ndjson()

        # Done event
        yield StreamEvent(
            StreamEventType.DONE,
            {"timestamp": datetime.now().isoformat()}
        ).to_sse() if stream_format == "sse" else StreamEvent(
            StreamEventType.DONE,
            {"timestamp": datetime.now().isoformat()}
        ).to_ndjson()

    except Exception as e:
        logger.error(f"Streaming query failed: {e}", exc_info=True)
        yield StreamEvent(
            StreamEventType.ERROR,
            {"message": str(e), "type": type(e).__name__}
        ).to_sse() if stream_format == "sse" else StreamEvent(
            StreamEventType.ERROR,
            {"message": str(e), "type": type(e).__name__}
        ).to_ndjson()


async def stream_document_ingestion(
    files: List[UploadFile],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    chunking_strategy: str = "adaptive"
) -> AsyncGenerator[str, None]:
    """
    Stream document ingestion with progress updates.

    Args:
        files: List of uploaded files
        chunk_size: Size of chunks
        chunk_overlap: Overlap between chunks
        chunking_strategy: Chunking strategy

    Yields:
        Stream events for ingestion progress
    """
    state = get_app_state()
    loader = DocumentLoader()

    try:
        # Start event
        yield StreamEvent(
            StreamEventType.START,
            {
                "files": [f.filename for f in files],
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "chunking_strategy": chunking_strategy
            }
        ).to_ndjson()

        # Validate files
        config = state["config"]
        max_size = config.processing.max_file_size_mb * 1024 * 1024

        for file in files:
            await file.seek(0, 2)
            size = await file.tell()
            await file.seek(0)

            if size > max_size:
                yield StreamEvent(
                    StreamEventType.ERROR,
                    {"file": file.filename, "message": f"File exceeds maximum size of {config.processing.max_file_size_mb}MB"}
                ).to_ndjson()
                return

        # Progress: Uploading files
        yield StreamEvent(
            StreamEventType.PROGRESS,
            {"stage": "uploading", "message": "Uploading files...", "progress": 0.1}
        ).to_ndjson()

        # Save uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            uploaded_files = []
            failed_files = []

            for file in files:
                file_path = Path(temp_dir) / file.filename
                try:
                    content = await file.read()
                    with open(file_path, 'wb') as f:
                        f.write(content)
                    uploaded_files.append(str(file_path))
                    yield StreamEvent(
                        StreamEventType.PROGRESS,
                        {"stage": "uploading", "file": file.filename, "message": f"Uploaded {file.filename}", "progress": 0.2}
                    ).to_ndjson()
                except Exception as e:
                    failed_files.append(file.filename)
                    yield StreamEvent(
                        StreamEventType.ERROR,
                        {"file": file.filename, "message": str(e)}
                    ).to_ndjson()

            if not uploaded_files:
                yield StreamEvent(
                    StreamEventType.ERROR,
                    {"message": "No valid files uploaded"}
                ).to_ndjson()
                yield StreamEvent(StreamEventType.DONE, {}).to_ndjson()
                return

            # Progress: Loading documents
            yield StreamEvent(
                StreamEventType.PROGRESS,
                {"stage": "loading", "message": "Loading documents...", "progress": 0.3}
            ).to_ndjson()

            # Load documents
            documents = []
            for file_path in uploaded_files:
                try:
                    doc = loader.load_document(file_path)
                    documents.append(doc)
                    yield StreamEvent(
                        StreamEventType.PROGRESS,
                        {"stage": "loading", "file": Path(file_path).name, "message": f"Loaded {Path(file_path).name}", "progress": 0.4}
                    ).to_ndjson()
                except Exception as e:
                    failed_files.append(Path(file_path).name)
                    yield StreamEvent(
                        StreamEventType.ERROR,
                        {"file": Path(file_path).name, "message": str(e)}
                    ).to_ndjson()

            if not documents:
                yield StreamEvent(
                    StreamEventType.ERROR,
                    {"message": "No documents could be loaded"}
                ).to_ndjson()
                yield StreamEvent(StreamEventType.DONE, {}).to_ndjson()
                return

            # Progress: Chunking documents
            yield StreamEvent(
                StreamEventType.PROGRESS,
                {"stage": "chunking", "message": "Chunking documents...", "progress": 0.5}
            ).to_ndjson()

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
                yield StreamEvent(
                    StreamEventType.ERROR,
                    {"message": "No chunks could be created"}
                ).to_ndjson()
                yield StreamEvent(StreamEventType.DONE, {}).to_ndjson()
                return

            yield StreamEvent(
                StreamEventType.PROGRESS,
                {"stage": "chunking", "message": f"Created {len(chunks)} chunks", "progress": 0.6}
            ).to_ndjson()

            # Progress: Generating embeddings
            yield StreamEvent(
                StreamEventType.PROGRESS,
                {"stage": "embedding", "message": "Generating embeddings...", "progress": 0.7}
            ).to_ndjson()

            # Generate embeddings
            embedding_generator = state["embedding_generator"]
            chunk_data = [
                {"text": chunk.text, "metadata": chunk.metadata}
                for chunk in chunks
            ]

            embeddings = await embedding_generator.generate_embeddings_async(chunk_data)

            yield StreamEvent(
                StreamEventType.PROGRESS,
                {"stage": "embedding", "message": f"Generated {len(embeddings)} embeddings", "progress": 0.85}
            ).to_ndjson()

            # Store in vector store
            vector_store = state["vector_store"]
            if not vector_store:
                yield StreamEvent(
                    StreamEventType.ERROR,
                    {"message": "Vector store not initialized"}
                ).to_ndjson()
                yield StreamEvent(StreamEventType.DONE, {}).to_ndjson()
                return

            # Add embeddings
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

            # Final event
            yield StreamEvent(
                StreamEventType.FINAL,
                {
                    "success": True,
                    "document_ids": document_ids,
                    "total_chunks": len(chunks),
                    "total_documents": len(documents),
                    "failed_files": failed_files,
                    "vector_store_size": vector_store.get_size()
                }
            ).to_ndjson()

            # Progress: Complete
            yield StreamEvent(
                StreamEventType.PROGRESS,
                {"stage": "complete", "message": "Ingestion complete!", "progress": 1.0}
            ).to_ndjson()

    except Exception as e:
        logger.error(f"Streaming ingestion failed: {e}", exc_info=True)
        yield StreamEvent(
            StreamEventType.ERROR,
            {"message": str(e), "type": type(e).__name__}
        ).to_ndjson()

    finally:
        yield StreamEvent(StreamEventType.DONE, {}).to_ndjson()


# ============================================================
# Streaming Endpoints
# ============================================================

@router.post(
    "/query/stream",
    summary="Ask a question with streaming",
    description="Ask a question and get streaming response with real-time updates"
)
async def query_stream(
    request: QueryRequest,
    format: str = Query("sse", description="Stream format: sse, ndjson, text"),
    show_thoughts: bool = Query(False, description="Show thought process")
):
    """
    Query with streaming response.
    Supports multiple formats: SSE, NDJSON, and plain text.
    """
    # Validate format
    if format not in ["sse", "ndjson", "text"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {format}. Supported: sse, ndjson, text"
        )

    # Determine media type
    media_types = {
        "sse": "text/event-stream",
        "ndjson": "application/x-ndjson",
        "text": "text/plain"
    }

    async def generate():
        if format == "text":
            # Plain text streaming
            async for event in stream_query_response(
                request.question,
                request.top_k,
                request.temperature,
                request.max_tokens,
                request.include_sources,
                stream_format="ndjson",  # Use NDJSON internally for parsing
                show_thoughts=show_thoughts
            ):
                try:
                    data = json.loads(event.split("data: ")[1].strip())
                    if data.get("event") == "token":
                        yield data["data"].get("content", "")
                    elif data.get("event") == "final":
                        yield "\n\n" + data["data"].get("answer", "")
                except Exception:
                    pass
        else:
            # SSE or NDJSON streaming
            async for event in stream_query_response(
                request.question,
                request.top_k,
                request.temperature,
                request.max_tokens,
                request.include_sources,
                stream_format=format,
                show_thoughts=show_thoughts
            ):
                yield event

    return StreamingResponse(
        generate(),
        media_type=media_types.get(format, "text/event-stream"),
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@router.post(
    "/documents/ingest/stream",
    summary="Ingest documents with streaming",
    description="Upload documents for ingestion with real-time progress"
)
async def ingest_documents_stream(
    files: List[UploadFile] = File(..., description="Documents to upload"),
    chunk_size: int = Form(800, description="Chunk size in characters", ge=100, le=10000),
    chunk_overlap: int = Form(150, description="Chunk overlap in characters", ge=0, le=5000),
    chunking_strategy: str = Form("adaptive", description="Chunking strategy")
):
    """
    Ingest documents with streaming progress updates.
    Returns NDJSON stream with progress events.
    """
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files provided"
        )

    async def generate():
        async for event in stream_document_ingestion(
            files,
            chunk_size,
            chunk_overlap,
            chunking_strategy
        ):
            yield event

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


@router.get(
    "/documents/export",
    summary="Export documents",
    description="Export all documents as a streaming download"
)
async def export_documents(
    format: str = Query("json", description="Export format: json, csv, txt")
):
    """
    Export documents as a streaming download.
    """
    state = get_app_state()
    vector_store = state["vector_store"]

    if not vector_store or vector_store.get_size() == 0:
        raise HTTPException(
            status_code=404,
            detail="No documents to export"
        )

    if format not in ["json", "csv", "txt"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {format}. Supported: json, csv, txt"
        )

    async def generate_json():
        yield '{"documents": ['
        docs = []
        for i, (text, metadata) in enumerate(zip(vector_store.texts, vector_store.metadata)):
            doc = {"id": i, "text": text, "metadata": metadata}
            docs.append(json.dumps(doc))
            if i % 10 == 0:
                yield ",".join(docs) + ("\n" if i < len(vector_store.texts) - 1 else "")
                docs = []
        if docs:
            yield ",".join(docs)
        yield "]}"

    async def generate_csv():
        # Header
        yield "id,text,metadata\n"
        for i, (text, metadata) in enumerate(zip(vector_store.texts, vector_store.metadata)):
            # Escape text for CSV
            escaped_text = text.replace('"', '""')
            yield f'"{i}","{escaped_text}","{json.dumps(metadata)}"\n'

    async def generate_txt():
        for i, (text, metadata) in enumerate(zip(vector_store.texts, vector_store.metadata)):
            yield f"=== Document {i} ===\n"
            if metadata:
                yield f"Metadata: {json.dumps(metadata, indent=2)}\n"
            yield f"Text: {text}\n\n"

    generators = {
        "json": generate_json,
        "csv": generate_csv,
        "txt": generate_txt
    }

    media_types = {
        "json": "application/json",
        "csv": "text/csv",
        "txt": "text/plain"
    }

    filenames = {
        "json": "documents.json",
        "csv": "documents.csv",
        "txt": "documents.txt"
    }

    return StreamingResponse(
        generators[format](),
        media_type=media_types[format],
        headers={
            "Content-Disposition": f'attachment; filename="{filenames[format]}"'
        }
    )


@router.post(
    "/query/chat-stream",
    summary="Chat with streaming",
    description="Chat with streaming response and conversation memory"
)
async def chat_stream(
    request: Request,
    question: str = Form(...),
    session_id: Optional[str] = Form(None),
    top_k: int = Form(5),
    temperature: Optional[float] = Form(None),
    max_tokens: Optional[int] = Form(None)
):
    """
    Chat with streaming response and conversation history.
    """
    # Use session_id to maintain conversation history
    # In production, store history in Redis or database

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

    async def generate():
        try:
            # Get conversation history for session
            # For demo, we'll just use a simple context
            history = []

            # Send initial acknowledgment
            yield f"data: {json.dumps({'event': 'start', 'data': {'session_id': session_id or 'new'}})}\n\n"

            # Retrieve documents
            retrieval_results = await run_in_threadpool(
                state["retriever"].retrieve,
                question,
                top_k=top_k
            )

            if not retrieval_results:
                yield f"data: {json.dumps({'event': 'error', 'data': {'message': 'No relevant documents found'}})}\n\n"
                return

            # Prepare context
            context_chunks = [
                {"text": r.text, "source": r.metadata.get("file_path", "Unknown")}
                for r in retrieval_results[:3]
            ]

            # Generate prompt with history
            prompt = get_rag_prompt(
                question=question,
                chunks=context_chunks,
                history=history
            )

            # Stream response
            llm_stream = await run_in_threadpool(
                state["llm_interface"].generate,
                [{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )

            full_response = ""
            for chunk in llm_stream:
                if isinstance(chunk, LLMResponse) and chunk.content:
                    full_response += chunk.content
                    yield f"data: {json.dumps({'event': 'token', 'data': {'content': chunk.content}})}\n\n"

            # Post-process
            processed = await run_in_threadpool(
                postprocess_response,
                full_response,
                str(context_chunks[:3]),
                True
            )

            # Send final response
            yield f"data: {json.dumps({'event': 'final', 'data': {'answer': processed.cleaned_text, 'confidence': processed.confidence}})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'data': {'message': str(e)}})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# ============================================================
# SSE Helper Endpoint
# ============================================================

@router.get(
    "/stream/health",
    summary="Stream health check",
    description="Stream health status for testing SSE connections"
)
async def stream_health():
    """
    Simple SSE health check endpoint for testing.
    """
    async def generate():
        for i in range(5):
            yield f"data: {json.dumps({'event': 'ping', 'data': {'count': i, 'timestamp': datetime.now().isoformat()}})}\n\n"
            await asyncio.sleep(1)
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )
