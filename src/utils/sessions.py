"""
Session management module with Redis support for DocQA AI system.
Provides session creation, management, persistence, and conversation history tracking.
"""

import os
import json
import time
import uuid
import hashlib
import logging
from typing import Dict, Any, Optional, List, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import threading
from contextlib import asynccontextmanager

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Try importing Redis
try:
    import redis.asyncio as aioredis
    from redis.exceptions import RedisError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis.asyncio not installed. Install with: pip install redis")


class SessionStatus(Enum):
    """Session status enumeration."""
    ACTIVE = "active"
    EXPIRED = "expired"
    TERMINATED = "terminated"
    SUSPENDED = "suspended"


class SessionConfig:
    """Session configuration settings."""

    # Default session TTL (seconds)
    DEFAULT_SESSION_TTL = 3600  # 1 hour

    # Maximum session TTL (seconds)
    MAX_SESSION_TTL = 86400 * 7  # 7 days

    # Conversation history limits
    MAX_HISTORY_LENGTH = 100
    MAX_HISTORY_TTL = 86400 * 30  # 30 days

    # Rate limiting per session
    MAX_REQUESTS_PER_MINUTE = 60
    MAX_QUERIES_PER_SESSION = 1000

    # Redis key prefixes
    SESSION_KEY_PREFIX = "session:"
    HISTORY_KEY_PREFIX = "history:"
    STATE_KEY_PREFIX = "state:"
    METADATA_KEY_PREFIX = "metadata:"
    RATE_KEY_PREFIX = "rate:"
    LOCK_KEY_PREFIX = "lock:"


@dataclass
class SessionData:
    """Session data container."""
    session_id: str
    user_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + SessionConfig.DEFAULT_SESSION_TTL)
    status: SessionStatus = SessionStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    request_count: int = 0
    query_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "expires_at": self.expires_at,
            "status": self.status.value,
            "metadata": self.metadata,
            "settings": self.settings,
            "request_count": self.request_count,
            "query_count": self.query_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionData':
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            user_id=data.get("user_id"),
            created_at=data.get("created_at", time.time()),
            last_accessed=data.get("last_accessed", time.time()),
            expires_at=data.get("expires_at", time.time() + SessionConfig.DEFAULT_SESSION_TTL),
            status=SessionStatus(data.get("status", "active")),
            metadata=data.get("metadata", {}),
            settings=data.get("settings", {}),
            request_count=data.get("request_count", 0),
            query_count=data.get("query_count", 0)
        )

    def is_expired(self) -> bool:
        """Check if session is expired."""
        return time.time() > self.expires_at

    def touch(self):
        """Update last accessed time."""
        self.last_accessed = time.time()
        self.expires_at = time.time() + SessionConfig.DEFAULT_SESSION_TTL


class ConversationMessage:
    """Conversation message model."""

    def __init__(
        self,
        role: str,
        content: str,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.role = role
        self.content = content
        self.timestamp = timestamp or time.time()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationMessage':
        """Create from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {})
        )


class RedisSessionManager:
    """
    Session manager using Redis for persistent, distributed session storage.
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        redis_password: Optional[str] = None,
        redis_db: int = 0,
        default_ttl: int = SessionConfig.DEFAULT_SESSION_TTL,
        max_history: int = SessionConfig.MAX_HISTORY_LENGTH,
        enable_locks: bool = True
    ):
        """
        Initialize Redis session manager.

        Args:
            redis_url: Redis connection URL
            redis_password: Redis password
            redis_db: Redis database number
            default_ttl: Default session TTL in seconds
            max_history: Maximum conversation history length
            enable_locks: Enable distributed locks
        """
        if not REDIS_AVAILABLE:
            raise ImportError("redis.asyncio not available. Install with: pip install redis")

        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_password = redis_password or os.getenv("REDIS_PASSWORD")
        self.redis_db = redis_db
        self.default_ttl = default_ttl
        self.max_history = max_history
        self.enable_locks = enable_locks

        # Initialize Redis client
        self._client = None
        self._async_client = None
        self._initialize_clients()

        # Statistics
        self.stats = {
            "total_sessions_created": 0,
            "active_sessions": 0,
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }

        # Background cleanup task
        self._cleanup_task = None
        self._running = False

        logger.info(f"RedisSessionManager initialized: redis_url={self.redis_url}, "
                   f"default_ttl={default_ttl}, max_history={max_history}")

    def _initialize_clients(self):
        """Initialize Redis clients."""
        # Async client
        self._async_client = aioredis.from_url(
            self.redis_url,
            password=self.redis_password,
            db=self.redis_db,
            decode_responses=True,
            encoding='utf-8'
        )

    @asynccontextmanager
    async def get_client(self):
        """Get Redis client context manager."""
        if not self._async_client:
            self._initialize_clients()

        try:
            yield self._async_client
        except Exception as e:
            logger.error(f"Redis client error: {e}")
            raise

    def _get_session_key(self, session_id: str) -> str:
        """Get Redis key for session data."""
        return f"{SessionConfig.SESSION_KEY_PREFIX}{session_id}"

    def _get_history_key(self, session_id: str) -> str:
        """Get Redis key for session history."""
        return f"{SessionConfig.HISTORY_KEY_PREFIX}{session_id}"

    def _get_metadata_key(self, session_id: str) -> str:
        """Get Redis key for session metadata."""
        return f"{SessionConfig.METADATA_KEY_PREFIX}{session_id}"

    def _get_rate_key(self, session_id: str) -> str:
        """Get Redis key for rate limiting."""
        return f"{SessionConfig.RATE_KEY_PREFIX}{session_id}"

    def _get_lock_key(self, session_id: str) -> str:
        """Get Redis key for distributed lock."""
        return f"{SessionConfig.LOCK_KEY_PREFIX}{session_id}"

    async def create_session(
        self,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None
    ) -> str:
        """
        Create a new session.

        Args:
            user_id: Optional user ID
            metadata: Optional session metadata
            ttl: Session TTL in seconds

        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        ttl = ttl or self.default_ttl

        # Create session data
        session = SessionData(
            session_id=session_id,
            user_id=user_id,
            expires_at=time.time() + ttl,
            metadata=metadata or {}
        )

        # Store in Redis
        async with self.get_client() as client:
            # Store session data
            await client.setex(
                self._get_session_key(session_id),
                ttl,
                json.dumps(session.to_dict())
            )

            # Store metadata separately for quick access
            if metadata:
                await client.setex(
                    self._get_metadata_key(session_id),
                    ttl,
                    json.dumps(metadata)
                )

        self.stats["total_sessions_created"] += 1
        self.stats["active_sessions"] += 1

        logger.info(f"Session created: {session_id} (user={user_id})")
        return session_id

    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """
        Get session data.

        Args:
            session_id: Session ID

        Returns:
            SessionData object or None
        """
        async with self.get_client() as client:
            data = await client.get(self._get_session_key(session_id))
            if not data:
                self.stats["cache_misses"] += 1
                return None

            self.stats["cache_hits"] += 1
            session = SessionData.from_dict(json.loads(data))

            # Check expiration
            if session.is_expired():
                await self.delete_session(session_id)
                return None

            return session

    async def update_session(
        self,
        session_id: str,
        updates: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Update session data.

        Args:
            session_id: Session ID
            updates: Updates to apply
            ttl: New TTL in seconds

        Returns:
            Success status
        """
        session = await self.get_session(session_id)
        if not session:
            return False

        # Apply updates
        for key, value in updates.items():
            if hasattr(session, key):
                setattr(session, key, value)

        # Touch session
        session.touch()

        # Store in Redis
        async with self.get_client() as client:
            ttl = ttl or self.default_ttl
            await client.setex(
                self._get_session_key(session_id),
                ttl,
                json.dumps(session.to_dict())
            )

        return True

    async def touch_session(self, session_id: str) -> bool:
        """
        Touch a session (update last accessed time).

        Args:
            session_id: Session ID

        Returns:
            Success status
        """
        return await self.update_session(session_id, {})

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id: Session ID

        Returns:
            Success status
        """
        async with self.get_client() as client:
            # Delete session data
            await client.delete(self._get_session_key(session_id))
            await client.delete(self._get_history_key(session_id))
            await client.delete(self._get_metadata_key(session_id))
            await client.delete(self._get_rate_key(session_id))
            await client.delete(self._get_lock_key(session_id))

        self.stats["active_sessions"] -= 1
        logger.info(f"Session deleted: {session_id}")
        return True

    async def add_conversation_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a message to conversation history.

        Args:
            session_id: Session ID
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional message metadata

        Returns:
            Success status
        """
        session = await self.get_session(session_id)
        if not session:
            return False

        # Create message
        message = ConversationMessage(role, content, metadata=metadata)

        # Add to history
        async with self.get_client() as client:
            key = self._get_history_key(session_id)

            # Add message to list (left push for latest first)
            await client.lpush(
                key,
                json.dumps(message.to_dict())
            )

            # Trim history to max length
            await client.ltrim(key, 0, self.max_history - 1)

            # Set TTL on history
            ttl = SessionConfig.MAX_HISTORY_TTL
            await client.expire(key, ttl)

        # Update session query count
        session.query_count += 1
        await self.update_session(session_id, {"query_count": session.query_count})

        self.stats["total_queries"] += 1

        return True

    async def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[ConversationMessage]:
        """
        Get conversation history.

        Args:
            session_id: Session ID
            limit: Maximum number of messages
            offset: Offset for pagination

        Returns:
            List of ConversationMessage objects
        """
        session = await self.get_session(session_id)
        if not session:
            return []

        limit = limit or self.max_history

        async with self.get_client() as client:
            key = self._get_history_key(session_id)

            # Get messages from Redis list (latest first)
            messages_data = await client.lrange(key, offset, offset + limit - 1)

            # Convert to ConversationMessage objects
            messages = []
            for msg_data in messages_data:
                try:
                    msg_dict = json.loads(msg_data)
                    messages.append(ConversationMessage.from_dict(msg_dict))
                except Exception as e:
                    logger.warning(f"Failed to parse message: {e}")

        return messages

    async def clear_conversation_history(self, session_id: str) -> bool:
        """
        Clear conversation history.

        Args:
            session_id: Session ID

        Returns:
            Success status
        """
        async with self.get_client() as client:
            key = self._get_history_key(session_id)
            await client.delete(key)

        return True

    async def set_session_metadata(
        self,
        session_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Set session metadata.

        Args:
            session_id: Session ID
            metadata: Metadata to set

        Returns:
            Success status
        """
        session = await self.get_session(session_id)
        if not session:
            return False

        session.metadata.update(metadata)
        return await self.update_session(session_id, {"metadata": session.metadata})

    async def get_session_metadata(self, session_id: str) -> Dict[str, Any]:
        """
        Get session metadata.

        Args:
            session_id: Session ID

        Returns:
            Session metadata dictionary
        """
        async with self.get_client() as client:
            data = await client.get(self._get_metadata_key(session_id))
            if data:
                try:
                    return json.loads(data)
                except Exception:
                    pass

        return {}

    async def check_rate_limit(
        self,
        session_id: str,
        max_requests: int = SessionConfig.MAX_REQUESTS_PER_MINUTE,
        window_seconds: int = 60
    ) -> Tuple[bool, int]:
        """
        Check rate limit for a session.

        Args:
            session_id: Session ID
            max_requests: Maximum requests per window
            window_seconds: Window size in seconds

        Returns:
            Tuple of (is_allowed, remaining_requests)
        """
        async with self.get_client() as client:
            key = self._get_rate_key(session_id)
            now = time.time()
            window_start = now - window_seconds

            # Remove old entries
            await client.zremrangebyscore(key, 0, window_start)

            # Get current count
            count = await client.zcard(key)

            # Check if allowed
            is_allowed = count < max_requests
            remaining = max_requests - count

            if is_allowed:
                # Add current request
                await client.zadd(key, {str(now): now})
                await client.expire(key, window_seconds)

            return is_allowed, max(0, remaining)

    async def acquire_lock(
        self,
        session_id: str,
        timeout: int = 10
    ) -> bool:
        """
        Acquire a distributed lock for a session.

        Args:
            session_id: Session ID
            timeout: Lock timeout in seconds

        Returns:
            True if lock acquired, False otherwise
        """
        if not self.enable_locks:
            return True

        async with self.get_client() as client:
            key = self._get_lock_key(session_id)
            lock_value = str(time.time())

            # Try to acquire lock
            acquired = await client.setnx(key, lock_value)

            if acquired:
                await client.expire(key, timeout)
                return True

            # Check if existing lock is expired
            existing_value = await client.get(key)
            if existing_value:
                try:
                    lock_time = float(existing_value)
                    if time.time() - lock_time > timeout:
                        # Lock expired, try to acquire
                        old_value = await client.getset(key, lock_value)
                        if old_value == existing_value:
                            await client.expire(key, timeout)
                            return True
                except Exception:
                    pass

            return False

    async def release_lock(self, session_id: str) -> bool:
        """
        Release a distributed lock.

        Args:
            session_id: Session ID

        Returns:
            Success status
        """
        if not self.enable_locks:
            return True

        async with self.get_client() as client:
            key = self._get_lock_key(session_id)
            await client.delete(key)
            return True

    async def get_active_sessions(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[SessionData]:
        """
        Get active sessions.

        Args:
            limit: Maximum number of sessions
            offset: Offset for pagination

        Returns:
            List of SessionData objects
        """
        async with self.get_client() as client:
            # Get all session keys
            pattern = f"{SessionConfig.SESSION_KEY_PREFIX}*"
            keys = await client.keys(pattern)
            keys = keys[offset:offset + limit]

            sessions = []
            for key in keys:
                data = await client.get(key)
                if data:
                    try:
                        session = SessionData.from_dict(json.loads(data))
                        if not session.is_expired():
                            sessions.append(session)
                    except Exception:
                        pass

            return sessions

    async def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        async with self.get_client() as client:
            # Get all session keys
            pattern = f"{SessionConfig.SESSION_KEY_PREFIX}*"
            keys = await client.keys(pattern)

            count = 0
            for key in keys:
                data = await client.get(key)
                if data:
                    try:
                        session = SessionData.from_dict(json.loads(data))
                        if session.is_expired():
                            session_id = session.session_id
                            await self.delete_session(session_id)
                            count += 1
                    except Exception:
                        # Delete invalid keys
                        await client.delete(key)
                        count += 1

            if count > 0:
                logger.info(f"Cleaned up {count} expired sessions")

            return count

    def get_stats(self) -> Dict[str, Any]:
        """Get session manager statistics."""
        return {
            **self.stats,
            "default_ttl": self.default_ttl,
            "max_history": self.max_history,
            "redis_url": self.redis_url
        }


class SessionContext:
    """
    Context manager for session operations with automatic cleanup.
    """

    def __init__(self, session_manager: RedisSessionManager, session_id: str):
        self.session_manager = session_manager
        self.session_id = session_id
        self.session = None

    async def __aenter__(self):
        self.session = await self.session_manager.get_session(self.session_id)
        if self.session:
            await self.session_manager.touch_session(self.session_id)
        return self.session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session_manager.touch_session(self.session_id)


# ============================================================
# Global Session Manager
# ============================================================

_session_manager: Optional[RedisSessionManager] = None
_manager_lock = threading.Lock()


def get_session_manager(
    redis_url: Optional[str] = None,
    redis_password: Optional[str] = None,
    default_ttl: int = SessionConfig.DEFAULT_SESSION_TTL
) -> RedisSessionManager:
    """
    Get or create global session manager.

    Args:
        redis_url: Redis connection URL
        redis_password: Redis password
        default_ttl: Default session TTL

    Returns:
        RedisSessionManager instance
    """
    global _session_manager

    if _session_manager is None:
        with _manager_lock:
            if _session_manager is None:
                _session_manager = RedisSessionManager(
                    redis_url=redis_url,
                    redis_password=redis_password,
                    default_ttl=default_ttl
                )

    return _session_manager


# ============================================================
# Convenience Functions
# ============================================================

async def create_session(
    user_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    ttl: Optional[int] = None
) -> str:
    """
    Create a new session.

    Args:
        user_id: Optional user ID
        metadata: Optional session metadata
        ttl: Session TTL in seconds

    Returns:
        Session ID
    """
    manager = get_session_manager()
    return await manager.create_session(user_id, metadata, ttl)


async def get_session(session_id: str) -> Optional[SessionData]:
    """
    Get session data.

    Args:
        session_id: Session ID

    Returns:
        SessionData object or None
    """
    manager = get_session_manager()
    return await manager.get_session(session_id)


async def update_session(session_id: str, updates: Dict[str, Any]) -> bool:
    """
    Update session data.

    Args:
        session_id: Session ID
        updates: Updates to apply

    Returns:
        Success status
    """
    manager = get_session_manager()
    return await manager.update_session(session_id, updates)


async def delete_session(session_id: str) -> bool:
    """
    Delete a session.

    Args:
        session_id: Session ID

    Returns:
        Success status
    """
    manager = get_session_manager()
    return await manager.delete_session(session_id)


async def add_conversation_message(
    session_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Add a message to conversation history.

    Args:
        session_id: Session ID
        role: Message role
        content: Message content
        metadata: Optional message metadata

    Returns:
        Success status
    """
    manager = get_session_manager()
    return await manager.add_conversation_message(session_id, role, content, metadata)


async def get_conversation_history(
    session_id: str,
    limit: Optional[int] = None
) -> List[ConversationMessage]:
    """
    Get conversation history.

    Args:
        session_id: Session ID
        limit: Maximum number of messages

    Returns:
        List of ConversationMessage objects
    """
    manager = get_session_manager()
    return await manager.get_conversation_history(session_id, limit)


if __name__ == "__main__":
    # Example usage
    import asyncio

    async def test_session_manager():
        """Test session manager functionality."""
        logging.basicConfig(level=logging.INFO)

        print("Testing Redis Session Manager...")

        # Create session manager
        manager = get_session_manager()

        # Create session
        session_id = await manager.create_session(
            user_id="test_user",
            metadata={"source": "test", "version": "1.0"}
        )
        print(f"✅ Session created: {session_id}")

        # Get session
        session = await manager.get_session(session_id)
        print(f"✅ Session retrieved: {session.session_id}")

        # Add conversation messages
        await manager.add_conversation_message(
            session_id,
            "user",
            "What is machine learning?"
        )
        await manager.add_conversation_message(
            session_id,
            "assistant",
            "Machine learning is a subset of AI..."
        )
        print("✅ Messages added to history")

        # Get conversation history
        history = await manager.get_conversation_history(session_id)
        print(f"✅ History retrieved: {len(history)} messages")
        for msg in history:
            print(f"  {msg.role}: {msg.content[:50]}...")

        # Check rate limit
        allowed, remaining = await manager.check_rate_limit(session_id)
        print(f"✅ Rate limit: allowed={allowed}, remaining={remaining}")

        # Get stats
        stats = manager.get_stats()
        print(f"✅ Stats: {stats}")

        # Cleanup
        await manager.delete_session(session_id)
        print("✅ Session deleted")

    # Run test
    asyncio.run(test_session_manager())
