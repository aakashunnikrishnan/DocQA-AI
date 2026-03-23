"""
API routes for DocQA AI system with async support.
Handles document ingestion, querying, and management endpoints.
ENHANCED: Fixed streaming response encoding with proper Unicode handling.
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
import codecs
import sys

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
# Streaming Response Helpers
# ============================================================

class StreamingEncoder:
    """
    Handle encoding for streaming responses with proper Unicode support.
    """

    @staticmethod
    def ensure_unicode(text: str) -> str:
        """Ensure text is properly encoded as Unicode."""
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception:
                text = ""

        # Handle common Unicode issues
        try:
            # Replace invalid Unicode characters
            text = text.encode('utf-8', errors='replace').decode('utf-8')
            return text
        except Exception:
            return text

    @staticmethod
    def json_encode(data: Dict[str, Any], ensure_ascii: bool = False) -> str:
        """
        Encode data as JSON with proper Unicode handling.

        Args:
            data: Data to encode
            ensure_ascii: Whether to escape Unicode characters

        Returns:
            JSON string
        """
        try:
            # Custom encoder that handles non-serializable types
            def default_serializer(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                if isinstance(obj, (set, frozenset)):
                    return list(obj)
                if hasattr(obj, 'to_dict'):
                    return obj.to_dict()
                if hasattr(obj, '__dict__'):
                    return obj.__dict__
                return str(obj)

            # Encode with proper Unicode handling
            json_str = json.dumps(
                data,
                default=default_serializer,
                ensure_ascii=ensure_ascii,
                separators=(',', ':')
            )

            # Ensure Unicode characters are properly handled
            return StreamingEncoder.ensure_unicode(json_str)

        except Exception as e:
            logger.error(f"JSON encoding failed: {e}")
            return json.dumps({"error": "Encoding error", "message": str(e)})

    @staticmethod
    def format_sse_event(
        event_type: str,
        data: Dict[str, Any],
        event_id: Optional[str] = None,
        retry: int = 1000
    ) -> str:
        """
        Format a Server-Sent Events (SSE) message with proper encoding.

        Args:
            event_type: Event type
            data: Event data
            event_id: Event ID
            retry: Retry timeout in milliseconds

        Returns:
            Formatted SSE message
        """
        # Encode data as JSON with Unicode support
        json_data = StreamingEncoder.json_encode(data, ensure_ascii=False)

        # Build SSE message
        lines = []
        if event_type:
            lines.append(f"event: {event_type}")
        if event_id:
            lines.append(f"id: {event_id}")
        if retry:
            lines.append(f"retry: {retry}")

        # Split data into multiple lines if needed (SSE spec)
        # Each line should be prefixed with "data: "
        for line in json_data.split('\n'):
            lines.append(f"data: {line}")

        # Add empty line to signal end of event
        lines.append("")

        # Join with newlines and encode with UTF-8
        message = '\n'.join(lines)
        return StreamingEncoder.ensure_unicode(message)

    @staticmethod
    def format_ndjson_event(
        event_type: str,
        data: Dict[str, Any]
    ) -> str:
        """
        Format an NDJSON event with proper encoding.

        Args:
            event_type: Event type
            data: Event data

        Returns:
            Formatted NDJSON event
        """
        event_data = {
            "event": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }

        json_data = StreamingEncoder.json_encode(event_data, ensure_ascii=False)
        return StreamingEncoder.ensure_unicode(json_data + '\n')

    @staticmethod
    def format_plain_text(text: str) -> str:
        """
        Format plain text with proper encoding.

        Args:
            text: Text to format

        Returns:
            Formatted text
        """
        return StreamingEncoder.ensure_unicode(text)

    @staticmethod
    def create_text_iterator(
        text: str,
        chunk_size: int = 50,
        delay: float = 0.01
    ) -> AsyncGenerator[str, None]:
        """
        Create an async iterator that yields text in chunks.

        Args:
            text: Text to stream
            chunk_size: Size of chunks in characters
            delay: Delay between chunks in seconds

        Yields:
            Chunks of text
        """
        async def generator():
            # Ensure text is properly encoded
            text = StreamingEncoder.ensure_unicode(text)

            # Stream chunks
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                # Ensure chunk is properly encoded
                chunk = StreamingEncoder.ensure_unicode(chunk)
                yield chunk
                if delay > 0:
                    await asyncio.sleep(delay)

        return generator()


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

    # Additional headers for streaming
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # Disable nginx buffering
        "Content-Encoding": "identity",  # Prevent compression issues
        "Transfer-Encoding": "chunked"
    }

    # For SSE, add specific headers
    if format == "sse":
        headers["Content-Type"] = "text/event-stream; charset=utf-8"
    elif format == "ndjson":
        headers["Content-Type"] = "application/x-ndjson; charset=utf-8"
    else:
        headers["Content-Type"] = "text/plain; charset=utf-8"

    async def generate():
        try:
            if format == "text":
                # Plain text streaming with Unicode support
                async for event in stream_query_response(
                    request.question,
                    request.top_k,
                    request.temperature,
                    request.max_tokens,
                    request.include_sources,
                    stream_format="sse",  # Use SSE internally for parsing
                    show_thoughts=show_thoughts
                ):
                    try:
                        # Parse SSE event to extract content
                        if event.startswith("event: token"):
                            # Extract data from next line
                            lines = event.split('\n')
                            for line in lines:
                                if line.startswith("data: "):
                                    try:
                                        data = json.loads(line[6:])
                                        content = data.get("data", {}).get("content", "")
                                        if content:
                                            # Ensure proper Unicode encoding
                                            yield StreamingEncoder.ensure_unicode(content)
                                    except Exception:
                                        pass
                        elif event.startswith("event: final"):
                            # Final answer
                            lines = event.split('\n')
                            for line in lines:
                                if line.startswith("data: "):
                                    try:
                                        data = json.loads(line[6:])
                                        answer = data.get("data", {}).get("answer", "")
                                        if answer:
                                            # Ensure proper Unicode encoding
                                            yield "\n\n" + StreamingEncoder.ensure_unicode(answer)
                                    except Exception:
                                        pass
                    except Exception as e:
                        logger.warning(f"Text streaming parse error: {e}")
            else:
                # SSE or NDJSON streaming with proper encoding
                async for event in stream_query_response(
                    request.question,
                    request.top_k,
                    request.temperature,
                    request.max_tokens,
                    request.include_sources,
                    stream_format=format,
                    show_thoughts=show_thoughts
                ):
                    # Ensure event is properly encoded
                    yield StreamingEncoder.ensure_unicode(event)

        except Exception as e:
            # Send error event with proper encoding
            error_data = {"message": str(e), "type": type(e).__name__}
            if format == "text":
                yield f"\n\nError: {str(e)}"
            elif format == "sse":
                yield StreamingEncoder.format_sse_event("error", error_data)
            else:
                yield StreamingEncoder.format_ndjson_event("error", error_data)

    return StreamingResponse(
        generate(),
        media_type=media_types.get(format, "text/event-stream"),
        headers=headers
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
        try:
            async for event in stream_document_ingestion(
                files,
                chunk_size,
                chunk_overlap,
                chunking_strategy
            ):
                # Ensure event is properly encoded
                yield StreamingEncoder.ensure_unicode(event)

        except Exception as e:
            # Send error event with proper encoding
            error_data = {"message": str(e), "type": type(e).__name__}
            yield StreamingEncoder.format_ndjson_event("error", error_data)

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Encoding": "identity",
            "Transfer-Encoding": "chunked"
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
    Export documents as a streaming download with proper encoding.
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
        total = len(vector_store.texts)

        for i, (text, metadata) in enumerate(zip(vector_store.texts, vector_store.metadata)):
            # Ensure proper Unicode encoding
            text = StreamingEncoder.ensure_unicode(text)
            doc = {
                "id": i,
                "text": text,
                "metadata": metadata
            }
            docs.append(json.dumps(doc, ensure_ascii=False))

            if len(docs) >= 10:
                yield ",".join(docs) + ("\n" if i < total - 1 else "")
                docs = []

        if docs:
            yield ",".join(docs)
        yield "]}"

    async def generate_csv():
        # Proper CSV encoding with Unicode support
        yield "\uFEFF"  # BOM for UTF-8
        yield "id,text,metadata\n"

        for i, (text, metadata) in enumerate(zip(vector_store.texts, vector_store.metadata)):
            # Ensure proper Unicode encoding
            text = StreamingEncoder.ensure_unicode(text)
            # Escape quotes for CSV
            escaped_text = text.replace('"', '""')
            escaped_metadata = json.dumps(metadata, ensure_ascii=False).replace('"', '""')
            yield f'"{i}","{escaped_text}","{escaped_metadata}"\n'

    async def generate_txt():
        # Plain text with Unicode support
        yield "\uFEFF"  # BOM for UTF-8

        for i, (text, metadata) in enumerate(zip(vector_store.texts, vector_store.metadata)):
            # Ensure proper Unicode encoding
            text = StreamingEncoder.ensure_unicode(text)
            yield f"=== Document {i} ===\n"
            if metadata:
                yield f"Metadata: {json.dumps(metadata, ensure_ascii=False, indent=2)}\n"
            yield f"Text: {text}\n\n"

    generators = {
        "json": generate_json,
        "csv": generate_csv,
        "txt": generate_txt
    }

    media_types = {
        "json": "application/json; charset=utf-8",
        "csv": "text/csv; charset=utf-8",
        "txt": "text/plain; charset=utf-8"
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
            "Content-Disposition": f'attachment; filename="{filenames[format]}"',
            "Content-Encoding": "identity",
            "Transfer-Encoding": "chunked"
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
            # Sanitize question
            from src.utils.security import InputValidator
            question = InputValidator.sanitize_query(question)

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

            # Generate prompt
            from src.generation.prompt_templates import get_rag_prompt
            prompt = get_rag_prompt(
                question=question,
                chunks=context_chunks
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
                    # Ensure proper Unicode encoding
                    content = StreamingEncoder.ensure_unicode(chunk.content)
                    full_response += content
                    yield f"data: {json.dumps({'event': 'token', 'data': {'content': content}})}\n\n"

            # Post-process
            processed = await run_in_threadpool(
                postprocess_response,
                full_response,
                str(context_chunks[:3]),
                True
            )

            # Ensure answer is properly encoded
            answer = StreamingEncoder.ensure_unicode(processed.cleaned_text)

            # Send final response
            yield f"data: {json.dumps({'event': 'final', 'data': {'answer': answer, 'confidence': processed.confidence}})}\n\n"

        except Exception as e:
            error_msg = StreamingEncoder.ensure_unicode(str(e))
            yield f"data: {json.dumps({'event': 'error', 'data': {'message': error_msg}})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Encoding": "identity",
            "Transfer-Encoding": "chunked"
        }
    )


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
            # Ensure proper encoding
            data = {
                'event': 'ping',
                'data': {
                    'count': i,
                    'timestamp': datetime.now().isoformat(),
                    'message': f'Ping {i+1} of 5'
                }
            }
            json_data = json.dumps(data, ensure_ascii=False)
            yield f"data: {json_data}\n\n"
            await asyncio.sleep(1)
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Content-Encoding": "identity",
            "Transfer-Encoding": "chunked"
        }
    )


# ============================================================
# Helper Functions (Updated with Encoding Support)
# ============================================================

async def stream_query_response(
    question: str,
    top_k: int = 5,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    include_sources: bool = True,
    stream_format: str = "sse",
    show_thoughts: bool = False
) -> AsyncGenerator[str, None]:
    """
    Stream query response with progressive updates and proper encoding.
    """
    state = get_app_state()
    start_time = time.time()

    # Validate state
    if not state["retriever"] or not state["llm_interface"]:
        error_data = {"message": "System not fully initialized"}
        if stream_format == "sse":
            yield StreamingEncoder.format_sse_event("error", error_data)
        else:
            yield StreamingEncoder.format_ndjson_event("error", error_data)
        return

    if state["vector_store"].get_size() == 0:
        error_data = {"message": "No documents ingested. Please upload documents first."}
        if stream_format == "sse":
            yield StreamingEncoder.format_sse_event("error", error_data)
        else:
            yield StreamingEncoder.format_ndjson_event("error", error_data)
        return

    try:
        # Sanitize question
        from src.utils.security import InputValidator
        question = InputValidator.sanitize_query(question)

        # Start event
        start_data = {
            "question": StreamingEncoder.ensure_unicode(question),
            "top_k": top_k,
            "model": state["config"].llm.model,
            "timestamp": datetime.now().isoformat()
        }

        if stream_format == "sse":
            yield StreamingEncoder.format_sse_event("start", start_data)
        else:
            yield StreamingEncoder.format_ndjson_event("start", start_data)

        # Progress: Retrieving documents
        progress_data = {
            "stage": "retrieving",
            "message": "Searching for relevant documents...",
            "progress": 0.2
        }

        if stream_format == "sse":
            yield StreamingEncoder.format_sse_event("progress", progress_data)
        else:
            yield StreamingEncoder.format_ndjson_event("progress", progress_data)

        # Retrieve documents
        retrieval_start = time.time()
        retrieval_results = await run_in_threadpool(
            state["retriever"].retrieve,
            question,
            top_k=top_k
        )
        retrieval_time = (time.time() - retrieval_start) * 1000

        if not retrieval_results:
            final_data = {
                "answer": "I couldn't find any relevant information in the documents to answer your question.",
                "confidence": 0.0,
                "sources": [],
                "tokens_used": 0,
                "processing_time_ms": (time.time() - start_time) * 1000
            }

            if stream_format == "sse":
                yield StreamingEncoder.format_sse_event("final", final_data)
                yield StreamingEncoder.format_sse_event("done", {"timestamp": datetime.now().isoformat()})
            else:
                yield StreamingEncoder.format_ndjson_event("final", final_data)
                yield StreamingEncoder.format_ndjson_event("done", {})
            return

        # Send sources
        if include_sources:
            sources = []
            for r in retrieval_results[:5]:
                sources.append({
                    "text": StreamingEncoder.ensure_unicode(r.text[:300] + "..." if len(r.text) > 300 else r.text),
                    "score": r.score,
                    "metadata": r.metadata
                })

            source_data = {
                "sources": sources,
                "retrieval_time_ms": retrieval_time
            }

            if stream_format == "sse":
                yield StreamingEncoder.format_sse_event("source", source_data)
            else:
                yield StreamingEncoder.format_ndjson_event("source", source_data)

        # Progress: Generating response
        progress_data = {
            "stage": "generating",
            "message": f"Generating response using {len(retrieval_results)} sources...",
            "progress": 0.5
        }

        if stream_format == "sse":
            yield StreamingEncoder.format_sse_event("progress", progress_data)
        else:
            yield StreamingEncoder.format_ndjson_event("progress", progress_data)

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
                # Ensure content is properly encoded
                content = StreamingEncoder.ensure_unicode(chunk.content)
                full_response += content

                # Send token chunk
                token_data = {
                    "content": content,
                    "token_count": token_count
                }

                if stream_format == "sse":
                    yield StreamingEncoder.format_sse_event("token", token_data)
                else:
                    yield StreamingEncoder.format_ndjson_event("token", token_data)

                # Update progress
                if token_count % 10 == 0:
                    progress = min(0.9, 0.5 + (token_count / 100) * 0.4)
                    progress_data = {
                        "stage": "generating",
                        "message": f"Generating... ({token_count} tokens)",
                        "progress": progress
                    }

                    if stream_format == "sse":
                        yield StreamingEncoder.format_sse_event("progress", progress_data)
                    else:
                        yield StreamingEncoder.format_ndjson_event("progress", progress_data)

        # Post-process response
        processed_response = await run_in_threadpool(
            postprocess_response,
            full_response,
            str(context_chunks[:3]),
            aggressive_cleaning=True
        )

        # Ensure answer is properly encoded
        answer = StreamingEncoder.ensure_unicode(processed_response.cleaned_text)

        # Prepare final response
        final_sources = []
        if include_sources:
            for r in retrieval_results[:5]:
                final_sources.append({
                    "text": StreamingEncoder.ensure_unicode(r.text[:500] + "..." if len(r.text) > 500 else r.text),
                    "score": r.score,
                    "metadata": r.metadata
                })

        # Send final response
        final_data = {
            "answer": answer,
            "confidence": processed_response.confidence,
            "sources": final_sources,
            "tokens_used": token_count,
            "has_hallucination": processed_response.has_hallucination,
            "processing_time_ms": (time.time() - start_time) * 1000,
            "retrieval_time_ms": retrieval_time,
            "generation_time_ms": (time.time() - retrieval_start) * 1000
        }

        if stream_format == "sse":
            yield StreamingEncoder.format_sse_event("final", final_data)
            yield StreamingEncoder.format_sse_event("done", {"timestamp": datetime.now().isoformat()})
        else:
            yield StreamingEncoder.format_ndjson_event("final", final_data)
            yield StreamingEncoder.format_ndjson_event("done", {})

    except Exception as e:
        logger.error(f"Streaming query failed: {e}", exc_info=True)
        error_data = {
            "message": StreamingEncoder.ensure_unicode(str(e)),
            "type": type(e).__name__
        }

        if stream_format == "sse":
            yield StreamingEncoder.format_sse_event("error", error_data)
        else:
            yield StreamingEncoder.format_ndjson_event("error", error_data)
