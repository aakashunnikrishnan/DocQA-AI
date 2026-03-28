"""
Tests for caching module.
"""

import pytest
import time
from pathlib import Path
from src.utils.cache import (
    MemoryCache, CacheManager, DiskCache, RedisCache,
    cached, async_cached, get_cache_manager
)


class TestMemoryCache:
    """Tests for MemoryCache."""

    def test_set_and_get(self):
        """Test setting and getting values."""
        cache = MemoryCache(max_size=10)
        cache.set("key1", "value1")

        assert cache.get("key1") == "value1"
        assert cache.get("nonexistent") is None

    def test_ttl(self):
        """Test TTL expiration."""
        cache = MemoryCache(default_ttl=1)
        cache.set("key1", "value1")

        assert cache.get("key1") == "value1"

        time.sleep(1.5)
        assert cache.get("key1") is None

    def test_max_size_eviction(self):
        """Test max size eviction."""
        cache = MemoryCache(max_size=3, eviction_policy="lru")
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")

        # key1 should be evicted (LRU)
        assert cache.get("key1") is None
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None
        assert cache.get("key4") is not None

    def test_delete(self):
        """Test deleting values."""
        cache = MemoryCache()
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        cache.delete("key1")
        assert cache.get("key1") is None

    def test_clear(self):
        """Test clearing cache."""
        cache = MemoryCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        assert cache.get_size() == 2
        cache.clear()
        assert cache.get_size() == 0

    def test_stats(self):
        """Test cache statistics."""
        cache = MemoryCache()
        cache.set("key1", "value1")
        cache.get("key1")
        cache.get("nonexistent")

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["sets"] == 1


class TestCacheManager:
    """Tests for CacheManager."""

    def test_singleton(self):
        """Test singleton pattern."""
        manager1 = get_cache_manager()
        manager2 = get_cache_manager()

        assert manager1 is manager2

    def test_get_default_cache(self):
        """Test getting default cache."""
        manager = get_cache_manager()
        cache = manager.get_cache("default")

        assert cache is not None

    def test_get_memory_cache(self):
        """Test getting memory cache."""
        manager = get_cache_manager()
        cache = manager.get_cache("memory")

        assert cache is not None
        assert isinstance(cache, MemoryCache)

    def test_set_and_get(self):
        """Test setting and getting values through manager."""
        manager = get_cache_manager()
        manager.set("test_key", "test_value")

        assert manager.get("test_key") == "test_value"


class TestCachedDecorator:
    """Tests for @cached decorator."""

    def test_cached_function(self):
        """Test caching function results."""
        call_count = 0

        @cached(ttl=10)
        def test_func(x, y):
            nonlocal call_count
            call_count += 1
            return x + y

        # First call should compute
        result1 = test_func(5, 3)
        assert result1 == 8
        assert call_count == 1

        # Second call should use cache
        result2 = test_func(5, 3)
        assert result2 == 8
        assert call_count == 1

        # Different arguments should compute
        result3 = test_func(10, 20)
        assert result3 == 30
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_cached(self):
        """Test caching async function results."""
        call_count = 0

        @async_cached(ttl=10)
        async def test_func(x, y):
            nonlocal call_count
            call_count += 1
            return x + y

        # First call should compute
        result1 = await test_func(5, 3)
        assert result1 == 8
        assert call_count == 1

        # Second call should use cache
        result2 = await test_func(5, 3)
        assert result2 == 8
        assert call_count == 1
