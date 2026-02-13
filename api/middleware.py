"""
API Middleware for DocQA AI system.
Provides rate limiting, authentication, logging, and security middleware.
"""

import time
import json
import logging
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timedelta
from collections import defaultdict
from functools import wraps
import asyncio
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import hashlib
import re

logger = logging.getLogger(__name__)

# Try importing Redis
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis.asyncio not installed. Redis rate limiting will not work.")


class RateLimitStrategy:
    """Rate limiting strategies."""
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


class RateLimiter:
    """
    Rate limiter with support for multiple strategies and backends.
    """

    def __init__(
        self,
        strategy: str = RateLimitStrategy.SLIDING_WINDOW,
        max_requests: int = 100,
        window_seconds: int = 60,
        redis_url: Optional[str] = None,
        redis_prefix: str = "rate_limit:",
        use_redis: bool = False
    ):
        """
        Initialize rate limiter.

        Args:
            strategy: Rate limiting strategy
            max_requests: Maximum requests per window
            window_seconds: Window size in seconds
            redis_url: Redis URL for distributed rate limiting
            redis_prefix: Redis key prefix
            use_redis: Whether to use Redis backend
        """
        self.strategy = strategy
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.redis_prefix = redis_prefix
        self.use_redis = use_redis and REDIS_AVAILABLE and redis_url

        # In-memory storage (fallback)
        self._memory_storage = defaultdict(list)

        # Redis client
        self._redis = None

        if self.use_redis:
            self._redis = aioredis.from_url(redis_url, decode_responses=True)
            logger.info(f"Rate limiter using Redis at {redis_url}")
        else:
            logger.info("Rate limiter using in-memory storage")

        logger.info(f"Rate limiter initialized: strategy={strategy}, max={max_requests}, window={window_seconds}s")

    def _get_key(self, identifier: str, endpoint: str) -> str:
        """Get cache key for rate limiting."""
        # Use consistent hashing for the key
        key = f"{self.redis_prefix}{endpoint}:{identifier}"
        return hashlib.md5(key.encode()).hexdigest()

    async def _get_redis_window(self, key: str) -> Dict[str, Any]:
        """Get or create sliding window in Redis."""
        now = time.time()
        window_start = now - self.window_seconds

        # Use Redis sorted set for sliding window
        async with self._redis as redis:
            # Remove old entries
            await redis.zremrangebyscore(key, 0, window_start)

            # Get count
            count = await redis.zcard(key)

            # Add current request
            await redis.zadd(key, {str(now): now})

            # Set expiry
            await redis.expire(key, self.window_seconds * 2)

            return {
                "count": count + 1,
                "limit": self.max_requests,
                "window": self.window_seconds
            }

    def _get_memory_window(self, key: str) -> Dict[str, Any]:
        """Get or create sliding window in memory."""
        now = time.time()
        window_start = now - self.window_seconds

        # Get stored timestamps
        timestamps = self._memory_storage.get(key, [])

        # Filter old entries
        timestamps = [t for t in timestamps if t > window_start]

        # Add current request
        timestamps.append(now)

        # Store updated timestamps
        self._memory_storage[key] = timestamps

        return {
            "count": len(timestamps),
            "limit": self.max_requests,
            "window": self.window_seconds
        }

    async def check_rate_limit(
        self,
        identifier: str,
        endpoint: str,
        increment: bool = True
    ) -> Dict[str, Any]:
        """
        Check if rate limit is exceeded.

        Args:
            identifier: Unique identifier (e.g., IP, API key)
            endpoint: Endpoint path
            increment: Whether to increment the counter

        Returns:
            Dictionary with rate limit status
        """
        key = self._get_key(identifier, endpoint)

        # Use Redis or memory
        if self.use_redis and self._redis:
            try:
                result = await self._get_redis_window(key)
            except Exception as e:
                logger.warning(f"Redis rate limit failed: {e}, falling back to memory")
                result = self._get_memory_window(key)
        else:
            result = self._get_memory_window(key)

        # Check if limit exceeded
        is_allowed = result["count"] <= result["limit"]

        return {
            "allowed": is_allowed,
            "remaining": max(0, result["limit"] - result["count"]),
            "limit": result["limit"],
            "window": result["window"],
            "count": result["count"]
        }

    async def get_remaining(self, identifier: str, endpoint: str) -> int:
        """Get remaining requests count."""
        result = await self.check_rate_limit(identifier, endpoint, increment=False)
        return result["remaining"]

    async def reset(self, identifier: str, endpoint: str):
        """Reset rate limit for a specific identifier and endpoint."""
        key = self._get_key(identifier, endpoint)

        if self.use_redis and self._redis:
            try:
                async with self._redis as redis:
                    await redis.delete(key)
            except Exception as e:
                logger.warning(f"Redis reset failed: {e}")

        if key in self._memory_storage:
            del self._memory_storage[key]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.
    Supports different rate limits per endpoint and user tier.
    """

    def __init__(
        self,
        app: ASGIApp,
        default_max_requests: int = 100,
        default_window_seconds: int = 60,
        redis_url: Optional[str] = None,
        use_redis: bool = False,
        rate_limit_by_ip: bool = True,
        rate_limit_by_api_key: bool = True,
        endpoints: Optional[Dict[str, Dict[str, int]]] = None,
        whitelist: Optional[List[str]] = None,
        blacklist: Optional[List[str]] = None
    ):
        """
        Initialize rate limit middleware.

        Args:
            app: ASGI application
            default_max_requests: Default max requests per window
            default_window_seconds: Default window size in seconds
            redis_url: Redis URL for distributed rate limiting
            use_redis: Whether to use Redis
            rate_limit_by_ip: Whether to rate limit by IP
            rate_limit_by_api_key: Whether to rate limit by API key
            endpoints: Custom rate limits per endpoint
            whitelist: IPs/API keys to whitelist (no rate limit)
            blacklist: IPs/API keys to blacklist (block requests)
        """
        super().__init__(app)

        self.default_max_requests = default_max_requests
        self.default_window_seconds = default_window_seconds
        self.rate_limit_by_ip = rate_limit_by_ip
        self.rate_limit_by_api_key = rate_limit_by_api_key
        self.whitelist = set(whitelist or [])
        self.blacklist = set(blacklist or [])

        # Create rate limiter
        self.rate_limiter = RateLimiter(
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            max_requests=default_max_requests,
            window_seconds=default_window_seconds,
            redis_url=redis_url,
            use_redis=use_redis
        )

        # Custom endpoint limits
        self.endpoint_limits = {}
        if endpoints:
            for endpoint, config in endpoints.items():
                self.endpoint_limits[endpoint] = {
                    "max_requests": config.get("max_requests", default_max_requests),
                    "window_seconds": config.get("window_seconds", default_window_seconds)
                }

        # Excluded endpoints (no rate limiting)
        self.excluded_endpoints = {
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/"
        }

        logger.info("Rate limit middleware initialized")

    async def dispatch(self, request: Request, call_next):
        """
        Process request and apply rate limiting.
        """
        # Get client identifier
        identifier = await self._get_identifier(request)

        # Get endpoint path
        endpoint = request.url.path

        # Check if endpoint is excluded
        if self._is_excluded(endpoint):
            return await call_next(request)

        # Check blacklist
        if identifier in self.blacklist:
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error": {
                        "code": "FORBIDDEN",
                        "message": "You have been blocked.",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            )

        # Check whitelist
        if identifier in self.whitelist:
            return await call_next(request)

        # Get rate limit for endpoint
        rate_limit = self._get_rate_limit(endpoint)

        # Update rate limiter with endpoint-specific limits
        if rate_limit != (self.default_max_requests, self.default_window_seconds):
            self.rate_limiter.max_requests = rate_limit[0]
            self.rate_limiter.window_seconds = rate_limit[1]

        # Check rate limit
        result = await self.rate_limiter.check_rate_limit(identifier, endpoint)

        # Add rate limit headers
        response = await call_next(request)

        # Add headers
        response.headers["X-RateLimit-Limit"] = str(result["limit"])
        response.headers["X-RateLimit-Remaining"] = str(result["remaining"])
        response.headers["X-RateLimit-Window"] = str(result["window"])

        # If rate limit exceeded
        if not result["allowed"]:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={
                    "Retry-After": str(result["window"]),
                    "X-RateLimit-Limit": str(result["limit"]),
                    "X-RateLimit-Remaining": "0"
                },
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": f"Rate limit exceeded. Maximum {result['limit']} requests per {result['window']} seconds.",
                        "timestamp": datetime.now().isoformat(),
                        "retry_after": result["window"]
                    }
                }
            )

        return response

    async def _get_identifier(self, request: Request) -> str:
        """
        Get unique identifier for rate limiting.
        Priority: API Key > IP Address > Session ID
        """
        # Check for API key
        if self.rate_limit_by_api_key:
            api_key = request.headers.get("Authorization")
            if api_key and api_key.startswith("Bearer "):
                return f"api_key:{api_key[7:]}"

        # Check for IP address
        if self.rate_limit_by_ip:
            # Get client IP from headers (behind proxy)
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                ip = forwarded.split(",")[0].strip()
            else:
                ip = request.client.host if request.client else "unknown"

            return f"ip:{ip}"

        # Fallback to session ID
        session_id = request.cookies.get("session_id")
        if session_id:
            return f"session:{session_id}"

        # Final fallback
        return f"anonymous:{request.client.host if request.client else 'unknown'}"

    def _is_excluded(self, endpoint: str) -> bool:
        """Check if endpoint is excluded from rate limiting."""
        # Exact match
        if endpoint in self.excluded_endpoints:
            return True

        # Pattern match for API docs
        if endpoint.startswith("/api/docs") or endpoint.startswith("/api/redoc"):
            return True

        # Pattern match for static files
        if endpoint.startswith("/static") or endpoint.startswith("/assets"):
            return True

        return False

    def _get_rate_limit(self, endpoint: str) -> Tuple[int, int]:
        """
        Get rate limit for endpoint.

        Returns:
            Tuple of (max_requests, window_seconds)
        """
        # Check for exact match
        if endpoint in self.endpoint_limits:
            config = self.endpoint_limits[endpoint]
            return config["max_requests"], config["window_seconds"]

        # Check for pattern match (e.g., /api/v1/query/*)
        for pattern, config in self.endpoint_limits.items():
            if pattern.endswith("*"):
                prefix = pattern[:-1]
                if endpoint.startswith(prefix):
                    return config["max_requests"], config["window_seconds"]

        return self.default_max_requests, self.default_window_seconds


class RateLimitDecorator:
    """
    Decorator for per-function rate limiting.
    """

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
        redis_url: Optional[str] = None,
        use_redis: bool = False
    ):
        """
        Initialize rate limit decorator.

        Args:
            max_requests: Maximum requests per window
            window_seconds: Window size in seconds
            redis_url: Redis URL for distributed rate limiting
            use_redis: Whether to use Redis
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.redis_url = redis_url
        self.use_redis = use_redis

        self.rate_limiter = RateLimiter(
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            max_requests=max_requests,
            window_seconds=window_seconds,
            redis_url=redis_url,
            use_redis=use_redis
        )

    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get identifier from request
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if not request:
                for arg in args:
                    if hasattr(arg, 'request') and isinstance(arg.request, Request):
                        request = arg.request
                        break

            if not request:
                # No request found, skip rate limiting
                return await func(*args, **kwargs)

            # Get identifier
            identifier = request.client.host if request.client else "unknown"
            if request.headers.get("Authorization"):
                api_key = request.headers.get("Authorization").replace("Bearer ", "")
                identifier = f"api_key:{api_key}"

            endpoint = request.url.path

            # Check rate limit
            result = await self.rate_limiter.check_rate_limit(identifier, endpoint)

            if not result["allowed"]:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Maximum {self.max_requests} requests per {self.window_seconds} seconds."
                )

            return await func(*args, **kwargs)

        return wrapper


class RateLimitService:
    """
    Service for managing rate limits and retrieving status.
    """

    def __init__(self, redis_url: Optional[str] = None, use_redis: bool = False):
        """
        Initialize rate limit service.

        Args:
            redis_url: Redis URL for distributed rate limiting
            use_redis: Whether to use Redis
        """
        self.redis_url = redis_url
        self.use_redis = use_redis
        self.rate_limiter = None

        if use_redis and redis_url:
            self.rate_limiter = RateLimiter(
                use_redis=True,
                redis_url=redis_url
            )
        else:
            self.rate_limiter = RateLimiter(use_redis=False)

    async def get_rate_limit_status(
        self,
        identifier: str,
        endpoint: str
    ) -> Dict[str, Any]:
        """
        Get rate limit status for a specific identifier and endpoint.

        Args:
            identifier: Unique identifier
            endpoint: Endpoint path

        Returns:
            Rate limit status
        """
        return await self.rate_limiter.check_rate_limit(identifier, endpoint, increment=False)

    async def reset_rate_limit(self, identifier: str, endpoint: str):
        """
        Reset rate limit for a specific identifier and endpoint.

        Args:
            identifier: Unique identifier
            endpoint: Endpoint path
        """
        await self.rate_limiter.reset(identifier, endpoint)

    async def add_to_whitelist(self, identifier: str):
        """
        Add identifier to whitelist.
        """
        # In production, this would update the database/Redis
        pass

    async def add_to_blacklist(self, identifier: str):
        """
        Add identifier to blacklist.
        """
        # In production, this would update the database/Redis
        pass


# ============================================================
# Middleware Factory
# ============================================================

def create_rate_limit_middleware(
    app: ASGIApp,
    default_max_requests: int = 100,
    default_window_seconds: int = 60,
    redis_url: Optional[str] = None,
    use_redis: bool = False,
    rate_limit_by_ip: bool = True,
    rate_limit_by_api_key: bool = True,
    endpoints: Optional[Dict[str, Dict[str, int]]] = None,
    whitelist: Optional[List[str]] = None,
    blacklist: Optional[List[str]] = None
) -> RateLimitMiddleware:
    """
    Factory function to create rate limit middleware with configuration.

    Args:
        app: ASGI application
        default_max_requests: Default max requests per window
        default_window_seconds: Default window size in seconds
        redis_url: Redis URL for distributed rate limiting
        use_redis: Whether to use Redis
        rate_limit_by_ip: Whether to rate limit by IP
        rate_limit_by_api_key: Whether to rate limit by API key
        endpoints: Custom rate limits per endpoint
        whitelist: IPs/API keys to whitelist
        blacklist: IPs/API keys to blacklist

    Returns:
        RateLimitMiddleware instance
    """
    return RateLimitMiddleware(
        app=app,
        default_max_requests=default_max_requests,
        default_window_seconds=default_window_seconds,
        redis_url=redis_url,
        use_redis=use_redis,
        rate_limit_by_ip=rate_limit_by_ip,
        rate_limit_by_api_key=rate_limit_by_api_key,
        endpoints=endpoints,
        whitelist=whitelist,
        blacklist=blacklist
    )


# ============================================================
# Default Configuration
# ============================================================

DEFAULT_RATE_LIMITS = {
    # Default limits
    "default": {"max_requests": 100, "window_seconds": 60},

    # Authentication endpoints (higher limits)
    "/api/v1/auth/login": {"max_requests": 20, "window_seconds": 60},
    "/api/v1/auth/register": {"max_requests": 10, "window_seconds": 60},

    # Query endpoints
    "/api/v1/query": {"max_requests": 50, "window_seconds": 60},
    "/api/v1/query/stream": {"max_requests": 20, "window_seconds": 60},

    # Document ingestion (lower limits)
    "/api/v1/documents/ingest": {"max_requests": 5, "window_seconds": 60},
    "/api/v1/documents/ingest/stream": {"max_requests": 3, "window_seconds": 60},

    # Document management
    "/api/v1/documents/*": {"max_requests": 30, "window_seconds": 60},

    # Admin endpoints
    "/api/v1/admin/*": {"max_requests": 10, "window_seconds": 60},

    # Metrics and health (high limits)
    "/metrics": {"max_requests": 1000, "window_seconds": 60},
    "/health": {"max_requests": 1000, "window_seconds": 60},
}

TIER_RATE_LIMITS = {
    "free": {"max_requests": 100, "window_seconds": 60},
    "pro": {"max_requests": 1000, "window_seconds": 60},
    "enterprise": {"max_requests": 10000, "window_seconds": 60},
}


if __name__ == "__main__":
    # Example usage
    import asyncio

    async def test_rate_limit():
        """Test rate limiter functionality."""
        print("Testing Rate Limiter...")

        # Create rate limiter
        limiter = RateLimiter(
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            max_requests=5,
            window_seconds=10
        )

        # Test requests
        identifier = "test_user"
        endpoint = "/api/v1/test"

        for i in range(10):
            result = await limiter.check_rate_limit(identifier, endpoint)
            print(f"Request {i+1}: allowed={result['allowed']}, remaining={result['remaining']}, count={result['count']}")

            if not result["allowed"]:
                print(f"Rate limit exceeded!")
                break

        # Test reset
        print("\nResetting rate limit...")
        await limiter.reset(identifier, endpoint)

        result = await limiter.check_rate_limit(identifier, endpoint, increment=False)
        print(f"After reset: count={result['count']}, remaining={result['remaining']}")

    # Run test
    asyncio.run(test_rate_limit())
