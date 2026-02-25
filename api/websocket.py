"""
WebSocket support for real-time Q&A with streaming responses.
Provides real-time communication, conversation management, and multi-client handling.
"""

import json
import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional, Set, Union
from datetime import datetime
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect, APIRouter, HTTPException
from fastapi.websockets import WebSocketState

from src.utils.logger import get_logger
from src.utils.monitoring import get_performance_monitor, measure
from src.generation.llm_interface import LLMInterface, LLMResponse
from src.generation.prompt_templates import get_rag_prompt
from src.generation.response_postprocess import postprocess_response
from src.retrieval.retriever import BaseRetriever

logger = get_logger(__name__)

router = APIRouter()


# ============================================================
# WebSocket Message Types
# ============================================================

class MessageType(str, Enum):
    """WebSocket message types."""
    # Client -> Server
    QUERY = "query"
    CANCEL = "cancel"
    PING = "ping"
    SETTINGS = "settings"
    HISTORY = "history"

    # Server -> Client
    START = "start"
    STATUS = "status"
    THOUGHT = "thought"
    TOKEN = "token"
    SOURCE = "source"
    ANSWER = "answer"
    ERROR = "error"
    DONE = "done"
    PONG = "pong"
    WELCOME = "welcome"


@dataclass
class WebSocketMessage:
    """WebSocket message structure."""
    type: MessageType
    data: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps({
            "type": self.type.value,
            "data": self.data,
            "id": self.id,
            "timestamp": self.timestamp
        })

    @classmethod
    def from_json(cls, data: str) -> 'WebSocketMessage':
        """Create from JSON string."""
        parsed = json.loads(data)
        return cls(
            type=MessageType(parsed["type"]),
            data=parsed.get("data", {}),
            id=parsed.get("id", str(uuid.uuid4())),
            timestamp=parsed.get("timestamp", datetime.now().isoformat())
        )


@dataclass
class Conversation:
    """Conversation state for a WebSocket session."""
    id: str
    websocket: WebSocket
    history: List[Dict[str, str]]
    settings: Dict[str, Any]
    created_at: datetime
    last_active: datetime
    is_processing: bool = False
    current_query: Optional[str] = None
    cancelled: bool = False

    def add_message(self, role: str, content: str):
        """Add a message to conversation history."""
        self.history.append({"role": role, "content": content})
        self.last_active = datetime.now()

    def get_context(self, max_turns: int = 10) -> List[Dict[str, str]]:
        """Get conversation context."""
        return self.history[-max_turns * 2:]  # User + Assistant per turn


class ConnectionManager:
    """
    WebSocket connection manager for handling multiple clients.
    """

    def __init__(self):
        self.active_connections: Dict[str, Conversation] = {}
        self._lock = asyncio.Lock()
        self._monitor = get_performance_monitor()

        # Configuration
        self.max_conversation_history = 50
        self.default_settings = {
            "top_k": 5,
            "temperature": 0.7,
            "max_tokens": 500,
            "include_sources": True,
            "show_thoughts": False
        }

        logger.info("WebSocket ConnectionManager initialized")

    async def connect(self, websocket: WebSocket, session_id: Optional[str] = None) -> str:
        """
        Accept a WebSocket connection and create a conversation.

        Args:
            websocket: WebSocket connection
            session_id: Optional session ID

        Returns:
            Session ID
        """
        await websocket.accept()

        # Generate or use session ID
        if session_id:
            # Check if session already exists
            async with self._lock:
                if session_id in self.active_connections:
                    # Close existing connection
                    existing = self.active_connections[session_id]
                    try:
                        await existing.websocket.close()
                    except Exception:
                        pass
                    del self.active_connections[session_id]
        else:
            session_id = f"session_{uuid.uuid4().hex[:8]}"

        # Create conversation
        conversation = Conversation(
            id=session_id,
            websocket=websocket,
            history=[],
            settings=self.default_settings.copy(),
            created_at=datetime.now(),
            last_active=datetime.now()
        )

        async with self._lock:
            self.active_connections[session_id] = conversation

        # Send welcome message
        await self.send_message(session_id, WebSocketMessage(
            type=MessageType.WELCOME,
            data={
                "session_id": session_id,
                "message": "Connected to DocQA AI",
                "settings": conversation.settings,
                "timestamp": datetime.now().isoformat()
            }
        ))

        # Update metrics
        self._monitor.collector.set_gauge(
            "websocket_connections_total",
            len(self.active_connections)
        )

        logger.info(f"WebSocket connected: {session_id} (Total: {len(self.active_connections)})")
        return session_id

    async def disconnect(self, session_id: str):
        """
        Disconnect a WebSocket connection.

        Args:
            session_id: Session ID
        """
        async with self._lock:
            if session_id in self.active_connections:
                del self.active_connections[session_id]

        # Update metrics
        self._monitor.collector.set_gauge(
            "websocket_connections_total",
            len(self.active_connections)
        )

        logger.info(f"WebSocket disconnected: {session_id} (Total: {len(self.active_connections)})")

    async def send_message(self, session_id: str, message: WebSocketMessage) -> bool:
        """
        Send a message to a specific session.

        Args:
            session_id: Session ID
            message: Message to send

        Returns:
            Success status
        """
        async with self._lock:
            if session_id not in self.active_connections:
                return False

            conversation = self.active_connections[session_id]
            websocket = conversation.websocket

            if websocket.client_state != WebSocketState.CONNECTED:
                return False

        try:
            await websocket.send_text(message.to_json())
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {session_id}: {e}")
            return False

    async def broadcast(self, message: WebSocketMessage, exclude: Optional[List[str]] = None):
        """
        Broadcast a message to all connected clients.

        Args:
            message: Message to broadcast
            exclude: List of session IDs to exclude
        """
        exclude = exclude or []

        async with self._lock:
            sessions = list(self.active_connections.keys())

        for session_id in sessions:
            if session_id in exclude:
                continue
            await self.send_message(session_id, message)

    async def receive_messages(self, session_id: str, handler: Callable):
        """
        Receive and handle messages from a client.

        Args:
            session_id: Session ID
            handler: Message handler function
        """
        async with self._lock:
            if session_id not in self.active_connections:
                return

            conversation = self.active_connections[session_id]
            websocket = conversation.websocket

        try:
            while True:
                # Receive message
                data = await websocket.receive_text()

                try:
                    message = WebSocketMessage.from_json(data)

                    # Update last active
                    conversation.last_active = datetime.now()

                    # Handle message
                    await handler(session_id, message)

                except json.JSONDecodeError as e:
                    await self.send_message(session_id, WebSocketMessage(
                        type=MessageType.ERROR,
                        data={"message": f"Invalid JSON: {str(e)}"}
                    ))
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    await self.send_message(session_id, WebSocketMessage(
                        type=MessageType.ERROR,
                        data={"message": str(e)}
                    ))

        except WebSocketDisconnect:
            await self.disconnect(session_id)
        except Exception as e:
            logger.error(f"WebSocket receive error: {e}")
            await self.disconnect(session_id)

    def get_conversation(self, session_id: str) -> Optional[Conversation]:
        """Get a conversation by session ID."""
        return self.active_connections.get(session_id)

    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return list(self.active_connections.keys())

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "total_connections": len(self.active_connections),
            "sessions": [
                {
                    "id": conv.id,
                    "created_at": conv.created_at.isoformat(),
                    "last_active": conv.last_active.isoformat(),
                    "history_length": len(conv.history),
                    "is_processing": conv.is_processing
                }
                for conv in self.active_connections.values()
            ]
        }


# ============================================================
# WebSocket Query Handler
# ============================================================

class WebSocketQueryHandler:
    """
    Handle query processing through WebSocket connections.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm_interface: LLMInterface,
        connection_manager: ConnectionManager
    ):
        """
        Initialize WebSocket query handler.

        Args:
            retriever: Retriever for document retrieval
            llm_interface: LLM interface for generation
            connection_manager: Connection manager
        """
        self.retriever = retriever
        self.llm_interface = llm_interface
        self.connection_manager = connection_manager
        self._monitor = get_performance_monitor()

        logger.info("WebSocketQueryHandler initialized")

    async def handle_query(self, session_id: str, message: WebSocketMessage):
        """
        Handle a query message.

        Args:
            session_id: Session ID
            message: Query message
        """
        conversation = self.connection_manager.get_conversation(session_id)
        if not conversation:
            return

        # Check if already processing
        if conversation.is_processing:
            await self.connection_manager.send_message(session_id, WebSocketMessage(
                type=MessageType.ERROR,
                data={"message": "Already processing a query. Please wait."}
            ))
            return

        # Extract query
        query = message.data.get("query", "")
        if not query:
            await self.connection_manager.send_message(session_id, WebSocketMessage(
                type=MessageType.ERROR,
                data={"message": "Missing 'query' field"}
            ))
            return

        # Update settings if provided
        if "settings" in message.data:
            conversation.settings.update(message.data["settings"])

        # Reset cancelled flag
        conversation.cancelled = False

        # Set processing state
        conversation.is_processing = True
        conversation.current_query = query

        # Add user message to history
        conversation.add_message("user", query)

        # Send start message
        await self.connection_manager.send_message(session_id, WebSocketMessage(
            type=MessageType.START,
            data={
                "query": query,
                "session_id": session_id,
                "settings": conversation.settings,
                "timestamp": datetime.now().isoformat()
            }
        ))

        try:
            # Process query with streaming
            await self._process_query(session_id, query, conversation)

        except asyncio.CancelledError:
            logger.info(f"Query cancelled for session {session_id}")
            await self.connection_manager.send_message(session_id, WebSocketMessage(
                type=MessageType.ERROR,
                data={"message": "Query cancelled"}
            ))

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            await self.connection_manager.send_message(session_id, WebSocketMessage(
                type=MessageType.ERROR,
                data={"message": str(e)}
            ))

        finally:
            conversation.is_processing = False
            conversation.current_query = None

    async def _process_query(self, session_id: str, query: str, conversation: Conversation):
        """
        Process a query and stream results.

        Args:
            session_id: Session ID
            query: Query string
            conversation: Conversation object
        """
        with measure("websocket_query", {"session_id": session_id}):
            settings = conversation.settings

            # Send status: retrieving
            await self.connection_manager.send_message(session_id, WebSocketMessage(
                type=MessageType.STATUS,
                data={
                    "stage": "retrieving",
                    "message": "Searching for relevant documents...",
                    "progress": 0.2
                }
            ))

            # Retrieve documents
            retrieval_start = time.time()
            results = self.retriever.retrieve(
                query,
                top_k=settings.get("top_k", 5)
            )
            retrieval_time = (time.time() - retrieval_start) * 1000

            if not results:
                # No results found
                answer = "I couldn't find any relevant information in the documents to answer your question."

                # Add assistant response to history
                conversation.add_message("assistant", answer)

                await self.connection_manager.send_message(session_id, WebSocketMessage(
                    type=MessageType.ANSWER,
                    data={
                        "answer": answer,
                        "confidence": 0.0,
                        "sources": [],
                        "processing_time_ms": (time.time() - retrieval_start) * 1000
                    }
                ))

                await self.connection_manager.send_message(session_id, WebSocketMessage(
                    type=MessageType.DONE,
                    data={"timestamp": datetime.now().isoformat()}
                ))
                return

            # Check if cancelled
            if conversation.cancelled:
                raise asyncio.CancelledError()

            # Send sources
            if settings.get("include_sources", True):
                sources = []
                for r in results[:5]:
                    sources.append({
                        "text": r.text[:300] + "..." if len(r.text) > 300 else r.text,
                        "score": r.score,
                        "metadata": r.metadata
                    })

                await self.connection_manager.send_message(session_id, WebSocketMessage(
                    type=MessageType.SOURCE,
                    data={
                        "sources": sources,
                        "retrieval_time_ms": retrieval_time
                    }
                ))

            # Send status: generating
            await self.connection_manager.send_message(session_id, WebSocketMessage(
                type=MessageType.STATUS,
                data={
                    "stage": "generating",
                    "message": f"Generating response using {len(results)} sources...",
                    "progress": 0.5
                }
            ))

            # Check if cancelled
            if conversation.cancelled:
                raise asyncio.CancelledError()

            # Prepare context
            context_chunks = [
                {"text": r.text, "source": r.metadata.get("file_path", "Unknown")}
                for r in results[:3]
            ]

            # Generate prompt
            prompt = get_rag_prompt(
                question=query,
                chunks=context_chunks
            )

            # Stream LLM response
            full_response = ""
            token_count = 0

            # Get conversation context
            context = conversation.get_context(5)

            # Generate with streaming
            stream = self.llm_interface.generate(
                context + [{"role": "user", "content": prompt}],
                temperature=settings.get("temperature"),
                max_tokens=settings.get("max_tokens"),
                stream=True
            )

            # Process stream
            for chunk in stream:
                # Check if cancelled
                if conversation.cancelled:
                    raise asyncio.CancelledError()

                if isinstance(chunk, LLMResponse) and chunk.content:
                    token_count += 1
                    full_response += chunk.content

                    # Send token
                    await self.connection_manager.send_message(session_id, WebSocketMessage(
                        type=MessageType.TOKEN,
                        data={
                            "content": chunk.content,
                            "token_count": token_count
                        }
                    ))

                    # Update status periodically
                    if token_count % 10 == 0:
                        progress = min(0.9, 0.5 + (token_count / 100) * 0.4)
                        await self.connection_manager.send_message(session_id, WebSocketMessage(
                            type=MessageType.STATUS,
                            data={
                                "stage": "generating",
                                "message": f"Generating... ({token_count} tokens)",
                                "progress": progress
                            }
                        ))

            # Check if cancelled
            if conversation.cancelled:
                raise asyncio.CancelledError()

            # Post-process response
            processed = postprocess_response(
                full_response,
                str(context_chunks[:3]),
                aggressive_cleaning=True
            )

            # Prepare sources
            final_sources = []
            if settings.get("include_sources", True):
                for r in results[:5]:
                    final_sources.append({
                        "text": r.text[:500] + "..." if len(r.text) > 500 else r.text,
                        "score": r.score,
                        "metadata": r.metadata
                    })

            # Add assistant response to history
            conversation.add_message("assistant", processed.cleaned_text)

            # Send final answer
            await self.connection_manager.send_message(session_id, WebSocketMessage(
                type=MessageType.ANSWER,
                data={
                    "answer": processed.cleaned_text,
                    "confidence": processed.confidence,
                    "sources": final_sources,
                    "tokens_used": token_count,
                    "has_hallucination": processed.has_hallucination,
                    "processing_time_ms": (time.time() - retrieval_start) * 1000,
                    "retrieval_time_ms": retrieval_time
                }
            ))

            # Send done
            await self.connection_manager.send_message(session_id, WebSocketMessage(
                type=MessageType.DONE,
                data={"timestamp": datetime.now().isoformat()}
            ))

    async def handle_cancel(self, session_id: str, message: WebSocketMessage):
        """
        Handle a cancel message.

        Args:
            session_id: Session ID
            message: Cancel message
        """
        conversation = self.connection_manager.get_conversation(session_id)
        if not conversation:
            return

        conversation.cancelled = True

        await self.connection_manager.send_message(session_id, WebSocketMessage(
            type=MessageType.ANSWER,
            data={
                "answer": "Query cancelled by user.",
                "confidence": 0.0,
                "sources": []
            }
        ))

        await self.connection_manager.send_message(session_id, WebSocketMessage(
            type=MessageType.DONE,
            data={"timestamp": datetime.now().isoformat()}
        ))

    async def handle_settings(self, session_id: str, message: WebSocketMessage):
        """
        Handle a settings update message.

        Args:
            session_id: Session ID
            message: Settings message
        """
        conversation = self.connection_manager.get_conversation(session_id)
        if not conversation:
            return

        settings = message.data.get("settings", {})
        conversation.settings.update(settings)

        await self.connection_manager.send_message(session_id, WebSocketMessage(
            type=MessageType.WELCOME,
            data={
                "message": "Settings updated",
                "settings": conversation.settings
            }
        ))

    async def handle_history(self, session_id: str, message: WebSocketMessage):
        """
        Handle a history request.

        Args:
            session_id: Session ID
            message: History message
        """
        conversation = self.connection_manager.get_conversation(session_id)
        if not conversation:
            return

        await self.connection_manager.send_message(session_id, WebSocketMessage(
            type=MessageType.HISTORY,
            data={
                "history": conversation.history,
                "count": len(conversation.history)
            }
        ))

    async def handle_ping(self, session_id: str, message: WebSocketMessage):
        """
        Handle a ping message.

        Args:
            session_id: Session ID
            message: Ping message
        """
        await self.connection_manager.send_message(session_id, WebSocketMessage(
            type=MessageType.PONG,
            data={"timestamp": datetime.now().isoformat()}
        ))


# ============================================================
# WebSocket Endpoint
# ============================================================

# Global instances
_connection_manager: Optional[ConnectionManager] = None
_query_handler: Optional[WebSocketQueryHandler] = None


def initialize_websocket_handler(
    retriever: BaseRetriever,
    llm_interface: LLMInterface
):
    """
    Initialize WebSocket handler with required dependencies.

    Args:
        retriever: Retriever instance
        llm_interface: LLM interface instance
    """
    global _connection_manager, _query_handler

    _connection_manager = ConnectionManager()
    _query_handler = WebSocketQueryHandler(
        retriever,
        llm_interface,
        _connection_manager
    )

    logger.info("WebSocket handler initialized")


@router.websocket("/query")
async def websocket_query(
    websocket: WebSocket,
    session_id: Optional[str] = None
):
    """
    WebSocket endpoint for real-time Q&A.

    Args:
        websocket: WebSocket connection
        session_id: Optional session ID for resuming conversations
    """
    if not _connection_manager or not _query_handler:
        await websocket.close(code=1011, reason="WebSocket handler not initialized")
        return

    # Accept connection
    session_id = await _connection_manager.connect(websocket, session_id)

    try:
        # Message handler
        async def handle_message(sid: str, message: WebSocketMessage):
            if message.type == MessageType.QUERY:
                await _query_handler.handle_query(sid, message)
            elif message.type == MessageType.CANCEL:
                await _query_handler.handle_cancel(sid, message)
            elif message.type == MessageType.SETTINGS:
                await _query_handler.handle_settings(sid, message)
            elif message.type == MessageType.HISTORY:
                await _query_handler.handle_history(sid, message)
            elif message.type == MessageType.PING:
                await _query_handler.handle_ping(sid, message)
            else:
                await _connection_manager.send_message(sid, WebSocketMessage(
                    type=MessageType.ERROR,
                    data={"message": f"Unknown message type: {message.type.value}"}
                ))

        # Receive messages
        await _connection_manager.receive_messages(session_id, handle_message)

    except WebSocketDisconnect:
        await _connection_manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await _connection_manager.disconnect(session_id)


@router.get("/ws/stats")
async def get_websocket_stats():
    """
    Get WebSocket connection statistics.
    """
    if not _connection_manager:
        return {"error": "WebSocket handler not initialized"}

    return _connection_manager.get_stats()


# ============================================================
# Example Client
# ============================================================

"""
Example WebSocket client usage:
"""
import asyncio
import websockets
import json

async def websocket_client():
    uri = "ws://localhost:8000/ws/query"
    
    async with websockets.connect(uri) as websocket:
        # Receive welcome message
        welcome = await websocket.recv()
        print(f"Welcome: {welcome}")
        
        # Send query
        query = {
            "type": "query",
            "data": {
                "query": "What is machine learning?",
                "settings": {
                    "top_k": 5,
                    "temperature": 0.7,
                    "include_sources": True
                }
            }
        }
        await websocket.send(json.dumps(query))
        
        # Receive streaming response
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            
            if data["type"] == "token":
                print(data["data"]["content"], end="")
            elif data["type"] == "answer":
                print(f"\n\nAnswer: {data['data']['answer']}")
            elif data["type"] == "done":
                break
            elif data["type"] == "error":
                print(f"Error: {data['data']['message']}")
                break

asyncio.run(websocket_client())
