"""
Caching module for DocQA AI system.
Provides multi-level caching for embeddings, API responses, and computed results.
Supports in-memory, disk, and Redis backends.
"""

import os
import json
import pickle
import hashlib
import logging
import time
import threading
from typing import Dict, Any, Optional, List, Union, Callable, TypeVar, Generic
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
from functools import wraps
import inspect

import numpy as np

logger = logging.getLogger(__name__)

# Try importing Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.debug("Redis not installed. Install with: pip install redis")

# Try importing diskcache
try:
    import diskcache
    DISKCACHE_AVAILABLE = True
except ImportError:
    DISKCACHE_AVAILABLE = False
    logger.debug("diskcache not installed. Install with: pip install diskcache")

T = TypeVar('T')


class CacheBackend(Enum):
    """Available cache backends."""
    MEMORY = "memory"
    DISK = "disk"
    REDIS = "redis"
    MULTI = "multi"  # Multi-level cache


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def touch(self):
        """Update access time and count."""
        self.last_accessed = time.time()
        self.access_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "size_bytes": self.size_bytes
        }


class BaseCache(ABC):
    """Abstract base class for cache backends."""

    def __init__(self, name: str = "docqa_cache"):
        self.name = name
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0
        }
        self._lock = threading.Lock()

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass

    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache entries."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass

    def _increment_hit(self):
        """Increment hit count."""
        with self._lock:
            self.stats["hits"] += 1

    def _increment_miss(self):
        """Increment miss count."""
        with self._lock:
            self.stats["misses"] += 1

    def _increment_set(self):
        """Increment set count."""
        with self._lock:
            self.stats["sets"] += 1

    def _increment_delete(self):
        """Increment delete count."""
        with self._lock:
            self.stats["deletes"] += 1


class MemoryCache(BaseCache):
    """In-memory cache using OrderedDict with LRU eviction."""

    def __init__(
        self,
        name: str = "docqa_cache",
        max_size: int = 10000,
        default_ttl: Optional[int] = 3600,
        eviction_policy: str = "lru"  # lru, fifo, ttl
    ):
        """
        Initialize memory cache.

        Args:
            name: Cache name
            max_size: Maximum number of items
            default_ttl: Default TTL in seconds
            eviction_policy: Eviction policy ('lru', 'fifo', 'ttl')
        """
        super().__init__(name)
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.eviction_policy = eviction_policy
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._size_bytes = 0

        logger.info(f"Initialized MemoryCache: max_size={max_size}, policy={eviction_policy}")

    def _get_size(self, value: Any) -> int:
        """Estimate size of value in bytes."""
        try:
            return len(pickle.dumps(value))
        except Exception:
            return len(str(value)) * 4  # Rough estimate

    def _evict(self):
        """Evict items based on policy."""
        if len(self._cache) < self.max_size:
            return

        if self.eviction_policy == "lru":
            # Remove least recently used
            oldest_key = next(iter(self._cache))
            self._evict_item(oldest_key)

        elif self.eviction_policy == "fifo":
            # Remove oldest (first in)
            oldest_key = next(iter(self._cache))
            self._evict_item(oldest_key)

        elif self.eviction_policy == "ttl":
            # Remove expired items first
            expired = [k for k, v in self._cache.items() if v.is_expired()]
            for key in expired[:2]:  # Remove up to 2 expired items
                self._evict_item(key)

            # If still over limit, remove oldest
            if len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                self._evict_item(oldest_key)

    def _evict_item(self, key: str):
        """Evict a single item."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._size_bytes -= entry.size_bytes
            self.stats["evictions"] += 1

    def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache."""
        with self._lock:
            if key not in self._cache:
                self._increment_miss()
                return None

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired():
                self._cache.pop(key)
                self._size_bytes -= entry.size_bytes
                self._increment_miss()
                return None

            # Update access time (LRU)
            if self.eviction_policy == "lru":
                self._cache.move_to_end(key)

            entry.touch()
            self._increment_hit()
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache."""
        with self._lock:
            # Calculate TTL
            expires_at = None
            if ttl is not None:
                expires_at = time.time() + ttl
            elif self.default_ttl is not None:
                expires_at = time.time() + self.default_ttl

            # Estimate size
            size = self._get_size(value)

            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                expires_at=expires_at,
                size_bytes=size
            )

            # If key exists, remove old entry
            if key in self._cache:
                old_entry = self._cache.pop(key)
                self._size_bytes -= old_entry.size_bytes

            # Evict if needed
            self._evict()

            # Store
            self._cache[key] = entry
            self._size_bytes += size

            # Move to end (LRU)
            if self.eviction_policy == "lru":
                self._cache.move_to_end(key)

            self._increment_set()
            return True

    def delete(self, key: str) -> bool:
        """Delete value from memory cache."""
        with self._lock:
            if key not in self._cache:
                return False

            entry = self._cache.pop(key)
            self._size_bytes -= entry.size_bytes
            self._increment_delete()
            return True

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        with self._lock:
            if key not in self._cache:
                return False

            entry = self._cache[key]
            if entry.is_expired():
                self._cache.pop(key)
                self._size_bytes -= entry.size_bytes
                return False

            return True

    def clear(self) -> bool:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._size_bytes = 0
            return True

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = len(self._cache)
            expired = sum(1 for v in self._cache.values() if v.is_expired())

            return {
                **self.stats,
                "total_entries": total,
                "expired_entries": expired,
                "size_bytes": self._size_bytes,
                "size_mb": self._size_bytes / (1024 * 1024),
                "max_size": self.max_size,
                "usage_percent": (total / self.max_size * 100) if self.max_size > 0 else 0,
                "policy": self.eviction_policy
            }


class DiskCache(BaseCache):
    """Disk-based cache using diskcache library."""

    def __init__(
        self,
        name: str = "docqa_cache",
        cache_dir: str = "./cache",
        max_size_mb: int = 1024,  # 1GB
        default_ttl: Optional[int] = 86400,  # 24 hours
        eviction_policy: str = "least-recently-stored"
    ):
        """
        Initialize disk cache.

        Args:
            name: Cache name
            cache_dir: Directory for cache files
            max_size_mb: Maximum cache size in MB
            default_ttl: Default TTL in seconds
            eviction_policy: Eviction policy
        """
        super().__init__(name)

        if not DISKCACHE_AVAILABLE:
            raise ImportError("diskcache is required for DiskCache. Install with: pip install diskcache")

        self.cache_dir = Path(cache_dir) / name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self.default_ttl = default_ttl

        # Initialize diskcache
        self._cache = diskcache.Cache(
            str(self.cache_dir),
            size_limit=max_size_mb * 1024 * 1024,
            eviction_policy=eviction_policy
        )

        logger.info(f"Initialized DiskCache: dir={self.cache_dir}, max_size={max_size_mb}MB")

    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        try:
            value = self._cache.get(key)
            if value is not None:
                self._increment_hit()
                return value
            self._increment_miss()
            return None
        except Exception as e:
            logger.warning(f"DiskCache get failed: {e}")
            self._increment_miss()
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in disk cache."""
        try:
            expire = ttl or self.default_ttl
            self._cache.set(key, value, expire=expire)
            self._increment_set()
            return True
        except Exception as e:
            logger.warning(f"DiskCache set failed: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete value from disk cache."""
        try:
            self._cache.delete(key)
            self._increment_delete()
            return True
        except Exception:
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in disk cache."""
        return key in self._cache

    def clear(self) -> bool:
        """Clear all cache entries."""
        try:
            self._cache.clear()
            return True
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            **self.stats,
            "total_entries": len(self._cache),
            "size_bytes": self._cache.volume(),
            "size_mb": self._cache.volume() / (1024 * 1024),
            "max_size_mb": self.max_size_mb,
            "cache_dir": str(self.cache_dir)
        }


class RedisCache(BaseCache):
    """Redis-based cache backend."""

    def __init__(
        self,
        name: str = "docqa_cache",
        redis_url: str = "redis://localhost:6379/0",
        default_ttl: Optional[int] = 3600,
        password: Optional[str] = None,
        decode_responses: bool = True
    ):
        """
        Initialize Redis cache.

        Args:
            name: Cache name (used as key prefix)
            redis_url: Redis connection URL
            default_ttl: Default TTL in seconds
            password: Redis password
            decode_responses: Decode responses to strings
        """
        super().__init__(name)

        if not REDIS_AVAILABLE:
            raise ImportError("redis is required for RedisCache. Install with: pip install redis")

        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.name = name

        # Initialize Redis client
        self._client = redis.Redis.from_url(
            redis_url,
            password=password,
            decode_responses=decode_responses,
            socket_connect_timeout=5,
            socket_timeout=5
        )

        # Test connection
        try:
            self._client.ping()
            logger.info(f"Initialized RedisCache: url={redis_url}, name={name}")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self._client = None

    def _get_key(self, key: str) -> str:
        """Get prefixed key."""
        return f"{self.name}:{key}"

    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if self._client is None:
            self._increment_miss()
            return None

        try:
            key = self._get_key(key)
            value = self._client.get(key)

            if value is not None:
                # Try to deserialize
                try:
                    value = pickle.loads(value)
                except Exception:
                    pass  # Keep as string

                self._increment_hit()
                return value

            self._increment_miss()
            return None

        except Exception as e:
            logger.warning(f"Redis get failed: {e}")
            self._increment_miss()
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        if self._client is None:
            return False

        try:
            key = self._get_key(key)

            # Serialize complex objects
            if not isinstance(value, (str, int, float, bool)):
                value = pickle.dumps(value)

            # Set with TTL
            expire = ttl or self.default_ttl
            if expire:
                self._client.setex(key, expire, value)
            else:
                self._client.set(key, value)

            self._increment_set()
            return True

        except Exception as e:
            logger.warning(f"Redis set failed: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete value from Redis cache."""
        if self._client is None:
            return False

        try:
            key = self._get_key(key)
            result = self._client.delete(key)
            self._increment_delete()
            return result > 0
        except Exception:
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in Redis cache."""
        if self._client is None:
            return False

        try:
            key = self._get_key(key)
            return self._client.exists(key) > 0
        except Exception:
            return False

    def clear(self) -> bool:
        """Clear all cache entries."""
        if self._client is None:
            return False

        try:
            pattern = f"{self.name}:*"
            keys = self._client.keys(pattern)
            if keys:
                self._client.delete(*keys)
            return True
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self._client is None:
            return {**self.stats, "connected": False}

        try:
            info = self._client.info()
            pattern = f"{self.name}:*"
            keys = self._client.keys(pattern)

            return {
                **self.stats,
                "connected": True,
                "total_entries": len(keys),
                "redis_version": info.get("redis_version", "unknown"),
                "used_memory_mb": info.get("used_memory", 0) / (1024 * 1024),
                "uptime_seconds": info.get("uptime_in_seconds", 0)
            }
        except Exception:
            return {**self.stats, "connected": False}


class MultiLevelCache(BaseCache):
    """
    Multi-level cache with fallback backends.
    Checks faster backends first (memory) before slower ones (disk, redis).
    """

    def __init__(
        self,
        backends: List[BaseCache],
        name: str = "docqa_cache",
        write_through: bool = True  # Write to all backends
    ):
        """
        Initialize multi-level cache.

        Args:
            backends: List of cache backends in order of preference
            name: Cache name
            write_through: Write to all backends on set
        """
        super().__init__(name)
        self.backends = backends
        self.write_through = write_through

        logger.info(f"Initialized MultiLevelCache with {len(backends)} backends")

    def get(self, key: str) -> Optional[Any]:
        """Get value from first available backend."""
        for backend in self.backends:
            value = backend.get(key)
            if value is not None:
                # Backfill faster backends
                for fill_backend in self.backends[:self.backends.index(backend)]:
                    fill_backend.set(key, value)

                self._increment_hit()
                return value

        self._increment_miss()
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in all backends."""
        success = True

        if self.write_through:
            # Write to all backends
            for backend in self.backends:
                if not backend.set(key, value, ttl):
                    success = False
        else:
            # Write only to first backend
            success = self.backends[0].set(key, value, ttl)

        self._increment_set()
        return success

    def delete(self, key: str) -> bool:
        """Delete from all backends."""
        success = True
        for backend in self.backends:
            if not backend.delete(key):
                success = False

        self._increment_delete()
        return success

    def exists(self, key: str) -> bool:
        """Check if key exists in any backend."""
        for backend in self.backends:
            if backend.exists(key):
                return True
        return False

    def clear(self) -> bool:
        """Clear all backends."""
        success = True
        for backend in self.backends:
            if not backend.clear():
                success = False
        return success

    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        stats = {**self.stats, "backends": []}
        for backend in self.backends:
            stats["backends"].append({
                "name": backend.name,
                "stats": backend.get_stats()
            })
        return stats


class CacheManager:
    """
    Centralized cache manager for DocQA AI.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if CacheManager._initialized:
            return

        self._caches: Dict[str, BaseCache] = {}
        self._default_cache: Optional[BaseCache] = None

        # Initialize default cache
        self._setup_default_cache()

        CacheManager._initialized = True
        logger.info("CacheManager initialized")

    def _setup_default_cache(self):
        """Setup default multi-level cache."""
        backends = []

        # Memory cache (fastest)
        memory_cache = MemoryCache(
            name="docqa_memory",
            max_size=10000,
            default_ttl=3600
        )
        backends.append(memory_cache)

        # Disk cache (slower but persistent)
        try:
            disk_cache = DiskCache(
                name="docqa_disk",
                cache_dir="./cache",
                max_size_mb=1024,
                default_ttl=86400
            )
            backends.append(disk_cache)
        except Exception as e:
            logger.debug(f"Disk cache not available: {e}")

        # Redis cache (shared, optional)
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        if redis_url:
            try:
                redis_cache = RedisCache(
                    name="docqa_redis",
                    redis_url=redis_url,
                    default_ttl=3600
                )
                backends.append(redis_cache)
            except Exception as e:
                logger.debug(f"Redis cache not available: {e}")

        # Create multi-level cache
        self._default_cache = MultiLevelCache(
            backends=backends,
            name="docqa_multi"
        )

        # Register caches
        self._caches["default"] = self._default_cache

    def get_cache(self, name: str = "default") -> BaseCache:
        """Get cache by name."""
        if name not in self._caches:
            if name == "memory":
                self._caches[name] = MemoryCache(name="docqa_memory")
            elif name == "disk":
                self._caches[name] = DiskCache(name="docqa_disk")
            elif name == "redis":
                self._caches[name] = RedisCache(name="docqa_redis")
            else:
                return self._default_cache

        return self._caches[name]

    def register_cache(self, name: str, cache: BaseCache):
        """Register a new cache."""
        self._caches[name] = cache

    def get(self, key: str, cache_name: str = "default") -> Optional[Any]:
        """Get value from cache."""
        cache = self.get_cache(cache_name)
        return cache.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None, cache_name: str = "default") -> bool:
        """Set value in cache."""
        cache = self.get_cache(cache_name)
        return cache.set(key, value, ttl)

    def delete(self, key: str, cache_name: str = "default") -> bool:
        """Delete value from cache."""
        cache = self.get_cache(cache_name)
        return cache.delete(key)

    def exists(self, key: str, cache_name: str = "default") -> bool:
        """Check if key exists in cache."""
        cache = self.get_cache(cache_name)
        return cache.exists(key)

    def clear(self, cache_name: Optional[str] = None) -> bool:
        """Clear cache or all caches."""
        if cache_name:
            cache = self.get_cache(cache_name)
            return cache.clear()
        else:
            success = True
            for cache in self._caches.values():
                if not cache.clear():
                    success = False
            return success

    def get_stats(self, cache_name: Optional[str] = None) -> Dict[str, Any]:
        """Get cache statistics."""
        if cache_name:
            cache = self.get_cache(cache_name)
            return cache.get_stats()
        else:
            stats = {}
            for name, cache in self._caches.items():
                stats[name] = cache.get_stats()
            return stats

    def get_embedding_cache(self) -> BaseCache:
        """Get cache for embeddings."""
        return self.get_cache("embedding") or self._default_cache


# Convenience functions for caching decorators
def cached(ttl: Optional[int] = None, cache_name: str = "default"):
    """
    Decorator for caching function results.

    Args:
        ttl: Time to live in seconds
        cache_name: Cache to use
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [func.__module__, func.__name__]

            # Add arguments
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            for param_name, param_value in bound_args.arguments.items():
                key_parts.append(f"{param_name}={param_value}")

            key = hashlib.md5(":".join(str(p) for p in key_parts).encode()).hexdigest()

            # Get from cache
            cache_manager = CacheManager()
            cached_value = cache_manager.get(key, cache_name)

            if cached_value is not None:
                return cached_value

            # Compute and cache
            result = func(*args, **kwargs)
            cache_manager.set(key, result, ttl, cache_name)
            return result

        return wrapper
    return decorator


def async_cached(ttl: Optional[int] = None, cache_name: str = "default"):
    """
    Decorator for caching async function results.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [func.__module__, func.__name__]

            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            for param_name, param_value in bound_args.arguments.items():
                key_parts.append(f"{param_name}={param_value}")

            key = hashlib.md5(":".join(str(p) for p in key_parts).encode()).hexdigest()

            # Get from cache
            cache_manager = CacheManager()
            cached_value = cache_manager.get(key, cache_name)

            if cached_value is not None:
                return cached_value

            # Compute and cache
            result = await func(*args, **kwargs)
            cache_manager.set(key, result, ttl, cache_name)
            return result

        return wrapper
    return decorator


# Convenience functions for embedding caching
def get_embedding_cache_key(text: str, model: str, dimension: Optional[int] = None) -> str:
    """Generate cache key for embeddings."""
    key_parts = [text, model]
    if dimension:
        key_parts.append(str(dimension))
    return hashlib.md5(":".join(key_parts).encode()).hexdigest()


def cache_embedding(text: str, model: str, embedding: List[float], ttl: Optional[int] = None):
    """Cache an embedding result."""
    key = get_embedding_cache_key(text, model)
    cache_manager = CacheManager()
    cache_manager.set(key, embedding, ttl or 86400 * 7, "embedding")


def get_cached_embedding(text: str, model: str) -> Optional[List[float]]:
    """Get cached embedding."""
    key = get_embedding_cache_key(text, model)
    cache_manager = CacheManager()
    return cache_manager.get(key, "embedding")


# Initialize global cache manager
cache_manager = CacheManager()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Test memory cache
    print("Testing MemoryCache...")
    memory_cache = MemoryCache(max_size=5)
    memory_cache.set("key1", "value1")
    memory_cache.set("key2", "value2", ttl=1)

    print(f"key1: {memory_cache.get('key1')}")
    print(f"key2: {memory_cache.get('key2')}")

    time.sleep(2)
    print(f"key2 (expired): {memory_cache.get('key2')}")

    print(f"Stats: {memory_cache.get_stats()}")

    # Test cache manager
    print("\nTesting CacheManager...")
    cache_mgr = CacheManager()
    cache_mgr.set("test_key", "test_value", ttl=60)
    print(f"test_key: {cache_mgr.get('test_key')}")
    print(f"Stats: {cache_mgr.get_stats()}")

    # Test decorator
    print("\nTesting cached decorator...")

    @cached(ttl=10)
    def expensive_function(x, y):
        print(f"Computing {x} + {y}...")
        return x + y

    result1 = expensive_function(5, 3)
    result2 = expensive_function(5, 3)  # Should be cached
    result3 = expensive_function(10, 20)  # Different args

    print(f"Result 1: {result1}")
    print(f"Result 2: {result2}")
    print(f"Result 3: {result3}")
