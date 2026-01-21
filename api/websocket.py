"""
WebSocket endpoints for real-time communication.
"""

import json
import asyncio
from typing import Dict, Any, List
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.websockets import WebSocketState

from api.app import get_app_state
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()


class ConnectionManager:
    """Manage WebSocket connections."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_count: int = 0

    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_count += 1

        # Update app state
        state = get_app_state()
        state["active_connections"] = len(self.active_connections)

        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

        # Update app state
        state = get_app_state()
        state["active_connections"] = len(self.active_connections)

        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def send_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send a message to a specific client."""
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send message: {e}")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all clients."""
        for connection in self.active_connections:
            if connection.client_state == WebSocketState.CONNECTED:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Failed to broadcast: {e}")

    def get_connections_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)


manager = ConnectionManager()


@router.websocket("/query")
async def websocket_query(websocket: WebSocket):
    """
    WebSocket endpoint for real-time querying.
    """
    await manager.connect(websocket)

    try:
        # Send welcome message
        await manager.send_message({
            "type": "welcome",
            "message": "Connected to DocQA AI",
            "timestamp": datetime.now().isoformat()
        }, websocket)

        while True:
            # Receive message from client
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                query = message.get("question")

                if not query:
                    await manager.send_message({
                        "type": "error",
                        "message": "Missing 'question' field",
                        "timestamp": datetime.now().isoformat()
                    }, websocket)
                    continue

                # Process query
                await process_query(websocket, query, message.get("top_k", 5))

            except json.JSONDecodeError:
                await manager.send_message({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.now().isoformat()
                }, websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        manager.disconnect(websocket)


async def process_query(websocket: WebSocket, question: str, top_k: int):
    """
    Process a query and stream results back via WebSocket.
    """
    state = get_app_state()

    # Validate state
    if not state["retriever"] or not state["llm_interface"]:
        await manager.send_message({
            "type": "error",
            "message": "System not fully initialized",
            "timestamp": datetime.now().isoformat()
        }, websocket)
        return

    if state["vector_store"].get_size() == 0:
        await manager.send_message({
            "type": "error",
            "message": "No documents ingested. Please upload documents first.",
            "timestamp": datetime.now().isoformat()
        }, websocket)
        return

    try:
        # Send processing status
        await manager.send_message({
            "type": "status",
            "status": "retrieving",
            "message": "Searching for relevant documents...",
            "timestamp": datetime.now().isoformat()
        }, websocket)

        # Retrieve documents
        retrieval_results = state["retriever"].retrieve(
            query=question,
            top_k=top_k
        )

        if not retrieval_results:
            await manager.send_message({
                "type": "answer",
                "answer": "I couldn't find any relevant information in the documents to answer your question.",
                "confidence": 0.0,
                "sources": [],
                "timestamp": datetime.now().isoformat()
            }, websocket)
            return

        # Send retrieval complete
        await manager.send_message({
            "type": "status",
            "status": "generating",
            "message": f"Found {len(retrieval_results)} relevant documents. Generating answer...",
            "timestamp": datetime.now().isoformat()
        }, websocket)

        # Generate answer
        from src.generation.prompt_templates import get_rag_prompt
        from src.generation.response_postprocess import postprocess_response

        context_chunks = [
            {"text": r.text, "source": r.metadata.get("file_path", "Unknown")}
            for r in retrieval_results[:3]
        ]

        prompt = get_rag_prompt(
            question=question,
            chunks=context_chunks
        )

        # Use streaming for WebSocket
        answer = ""
        sources = []

        # Generate response with streaming
        llm_response = state["llm_interface"].generate(
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        # Send streaming response
        for chunk in llm_response:
            if chunk.content:
                answer += chunk.content
                await manager.send_message({
                    "type": "stream",
                    "chunk": chunk.content,
                    "timestamp": datetime.now().isoformat()
                }, websocket)

        # Post-process full response
        processed = postprocess_response(
            response=answer,
            context=str(context_chunks[:3]),
            aggressive_cleaning=True
        )

        # Prepare sources
        for r in retrieval_results[:5]:
            sources.append({
                "text": r.text[:500] + "..." if len(r.text) > 500 else r.text,
                "score": r.score,
                "metadata": r.metadata
            })

        # Send final answer
        await manager.send_message({
            "type": "answer",
            "answer": processed.cleaned_text,
            "confidence": processed.confidence,
            "sources": sources,
            "has_hallucination": processed.has_hallucination,
            "timestamp": datetime.now().isoformat()
        }, websocket)

    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        await manager.send_message({
            "type": "error",
            "message": f"Query processing failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }, websocket)
