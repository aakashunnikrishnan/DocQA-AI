"""
API Middleware for DocQA AI system.
Provides rate limiting, authentication, logging, security, and CORS handling.
FIXED: CORS issues with proper configuration, preflight handling, and security headers.
"""

import os
import time
import json
import logging
from typing import Dict, Any, Optional, List, Callable, Union, Set
from datetime import datetime, timedelta
from collections import defaultdict
from functools import wraps
import asyncio
import re

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from starlette.middleware.cors import CORSMiddleware as StarletteCORSMiddleware
import hashlib

logger = logging.getLogger(__name__)

# Try importing Redis
try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis.asyncio not installed. Redis rate limiting will not work.")


# ============================================================
# CORS Configuration
# ============================================================

class CORSConfig:
    """CORS configuration settings."""

    # Default allowed origins (can be overridden by environment)
    DEFAULT_ALLOWED_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "https://docqa-ai.com",
        "https://api.docqa-ai.com",
        "https://*.docqa-ai.com"
    ]

    # Default allowed methods
    DEFAULT_ALLOWED_METHODS = [
        "GET",
        "POST",
        "PUT",
        "DELETE",
        "PATCH",
        "OPTIONS",
        "HEAD"
    ]

    # Default allowed headers
    DEFAULT_ALLOWED_HEADERS = [
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Content-Type",
        "Authorization",
        "X-API-Key",
        "X-Requested-With",
        "Origin",
        "Referer",
        "User-Agent",
        "Cache-Control",
        "Pragma",
        "Expires",
        "X-Request-ID",
        "X-Correlation-ID",
        "X-Forwarded-For",
        "X-Forwarded-Proto",
        "X-Forwarded-Host"
    ]

    # Default exposed headers
    DEFAULT_EXPOSED_HEADERS = [
        "Content-Disposition",
        "X-Response-Time",
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Window",
        "X-Request-ID",
        "X-Correlation-ID"
    ]

    # Default max age for preflight requests (in seconds)
    DEFAULT_MAX_AGE = 86400  # 24 hours


class CORSMiddleware(BaseHTTPMiddleware):
    """
    Enhanced CORS middleware with proper preflight handling and security headers.
    """

    def __init__(
        self,
        app: ASGIApp,
        allow_origins: Optional[List[str]] = None,
        allow_origin_regex: Optional[str] = None,
        allow_methods: Optional[List[str]] = None,
        allow_headers: Optional[List[str]] = None,
        allow_credentials: bool = True,
        expose_headers: Optional[List[str]] = None,
        max_age: int = 86400,
        allow_private_network: bool = True,
        preflight_continue: bool = False
    ):
        """
        Initialize CORS middleware.

        Args:
            app: ASGI application
            allow_origins: List of allowed origins
            allow_origin_regex: Regex pattern for allowed origins
            allow_methods: List of allowed HTTP methods
            allow_headers: List of allowed headers
            allow_credentials: Whether to allow credentials
            expose_headers: Headers to expose to the client
            max_age: Max age for preflight requests (seconds)
            allow_private_network: Allow private network access
            preflight_continue: Continue to next middleware on preflight
        """
        super().__init__(app)

        # Parse allowed origins from environment
        env_origins = os.getenv("CORS_ALLOWED_ORIGINS", "")
        if env_origins:
            self.allow_origins = [o.strip() for o in env_origins.split(",") if o.strip()]
        else:
            self.allow_origins = allow_origins or CORSConfig.DEFAULT_ALLOWED_ORIGINS

        self.allow_origin_regex = allow_origin_regex or os.getenv("CORS_ALLOWED_ORIGIN_REGEX")
        self.allow_methods = allow_methods or CORSConfig.DEFAULT_ALLOWED_METHODS
        self.allow_headers = allow_headers or CORSConfig.DEFAULT_ALLOWED_HEADERS
        self.allow_credentials = allow_credentials
        self.expose_headers = expose_headers or CORSConfig.DEFAULT_EXPOSED_HEADERS
        self.max_age = max_age or CORSConfig.DEFAULT_MAX_AGE
        self.allow_private_network = allow_private_network
        self.preflight_continue = preflight_continue

        # Compile regex if provided
        self.allow_origin_regex_compiled = None
        if self.allow_origin_regex:
            try:
                self.allow_origin_regex_compiled = re.compile(self.allow_origin_regex)
            except re.error as e:
                logger.warning(f"Invalid CORS origin regex: {e}")

        # Cache for allowed origins
        self._origin_cache: Dict[str, bool] = {}
        self._cache_lock = asyncio.Lock()

        # Additional security headers
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }

        # CSP header (optional)
        csp = os.getenv("CSP_POLICY")
        if csp:
            self.security_headers["Content-Security-Policy"] = csp

        logger.info(f"CORS middleware initialized with {len(self.allow_origins)} allowed origins")
        logger.info(f"Allowed methods: {self.allow_methods}")
        logger.info(f"Allow credentials: {self.allow_credentials}")
        logger.info(f"Max age: {self.max_age}s")

    def _is_origin_allowed(self, origin: str) -> bool:
        """
        Check if an origin is allowed.

        Args:
            origin: Origin URL

        Returns:
            True if allowed, False otherwise
        """
        if not origin:
            return False

        # Check cache
        if origin in self._origin_cache:
            return self._origin_cache[origin]

        # Check exact match
        if origin in self.allow_origins:
            self._origin_cache[origin] = True
            return True

        # Check wildcard patterns
        for allowed_origin in self.allow_origins:
            if allowed_origin.startswith("*."):
                # Wildcard subdomain: *.example.com
                domain = allowed_origin[2:]  # Remove "*."
                if origin.endswith(domain) and origin.count('.') >= domain.count('.'):
                    self._origin_cache[origin] = True
                    return True

        # Check regex pattern
        if self.allow_origin_regex_compiled:
            if self.allow_origin_regex_compiled.match(origin):
                self._origin_cache[origin] = True
                return True

        # Check for development origins (localhost, 127.0.0.1)
        if self._is_development_origin(origin):
            self._origin_cache[origin] = True
            return True

        self._origin_cache[origin] = False
        return False

    def _is_development_origin(self, origin: str) -> bool:
        """Check if origin is a development origin (localhost, 127.0.0.1)."""
        if not origin:
            return False

        # Parse origin
        try:
            import urllib.parse
            parsed = urllib.parse.urlparse(origin)
            hostname = parsed.hostname or ""
            port = parsed.port
        except Exception:
            return False

        # Check localhost
        if hostname in ["localhost", "127.0.0.1", "::1"]:
            # Allow any port in development
            return True

        # Check local IP patterns
        if hostname.startswith("192.168.") or hostname.startswith("10."):
            return True

        if hostname.startswith("172."):
            parts = hostname.split(".")
            if len(parts) >= 2:
                try:
                    second = int(parts[1])
                    if 16 <= second <= 31:
                        return True
                except ValueError:
                    pass

        return False

    def _get_cors_headers(
        self,
        origin: str,
        request_method: str,
        request_headers: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Generate CORS headers for a request.

        Args:
            origin: Request origin
            request_method: Request method
            request_headers: Request headers

        Returns:
            Dictionary of CORS headers
        """
        headers = {}

        # Check if origin is allowed
        is_allowed = self._is_origin_allowed(origin)

        if is_allowed:
            # Allow the origin
            headers["Access-Control-Allow-Origin"] = origin

            # Allow credentials if configured
            if self.allow_credentials:
                headers["Access-Control-Allow-Credentials"] = "true"

            # Expose headers
            if self.expose_headers:
                headers["Access-Control-Expose-Headers"] = ", ".join(self.expose_headers)

            # Handle preflight requests
            if request_method == "OPTIONS":
                # Allowed methods
                if self.allow_methods:
                    headers["Access-Control-Allow-Methods"] = ", ".join(self.allow_methods)

                # Allowed headers
                allowed_headers = self.allow_headers.copy()

                # Add requested headers
                requested_headers = request_headers.get("access-control-request-headers", "")
                if requested_headers:
                    requested_headers_list = [h.strip() for h in requested_headers.split(",")]
                    for h in requested_headers_list:
                        if h not in allowed_headers:
                            allowed_headers.append(h)

                headers["Access-Control-Allow-Headers"] = ", ".join(allowed_headers)

                # Max age
                headers["Access-Control-Max-Age"] = str(self.max_age)

                # Private network access
                if self.allow_private_network:
                    headers["Access-Control-Allow-Private-Network"] = "true"
        else:
            # Origin not allowed
            headers["Access-Control-Allow-Origin"] = "null"

        return headers

    async def dispatch(self, request: Request, call_next):
        """
        Process request and add CORS headers.
        """
        # Get request details
        origin = request.headers.get("origin")
        method = request.method
        request_headers = dict(request.headers)

        # Handle preflight requests
        if method == "OPTIONS":
            # Generate CORS headers
            cors_headers = self._get_cors_headers(origin, method, request_headers)

            # If preflight_continue is True, continue to next middleware
            if self.preflight_continue:
                response = await call_next(request)
                # Update response headers
                for key, value in cors_headers.items():
                    response.headers[key] = value
                return response

            # Return preflight response
            return JSONResponse(
                content={},
                status_code=status.HTTP_200_OK,
                headers=cors_headers
            )

        # Process actual request
        response = await call_next(request)

        # Add CORS headers to response
        if origin:
            cors_headers = self._get_cors_headers(origin, method, request_headers)
            for key, value in cors_headers.items():
                response.headers[key] = value

        # Add security headers (unless already set)
        for key, value in self.security_headers.items():
            if key not in response.headers:
                response.headers[key] = value

        return response


# ============================================================
# Enhanced CORS Middleware with Additional Features
# ============================================================

class EnhancedCORSMiddleware(CORSMiddleware):
    """
    Enhanced CORS middleware with additional features:
    - Request logging for CORS issues
    - Origin validation with logging
    - Custom error responses for CORS failures
    - Dynamic origin updates
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cors_error_count = 0
        self._cors_success_count = 0

    async def dispatch(self, request: Request, call_next):
        """
        Process request with enhanced CORS handling and logging.
        """
        origin = request.headers.get("origin")
        method = request.method

        # Log CORS requests in development
        if os.getenv("ENVIRONMENT") == "development":
            logger.debug(f"CORS request: origin={origin}, method={method}, path={request.url.path}")

        # Check if origin is allowed (for logging)
        if origin and not self._is_origin_allowed(origin):
            self._cors_error_count += 1
            logger.warning(f"CORS blocked: origin={origin} not allowed, path={request.url.path}")

            if self._cors_error_count % 10 == 0:
                logger.warning(f"CORS blocked {self._cors_error_count} requests since startup")

        # Process request
        response = await super().dispatch(request, call_next)

        # Log successful CORS responses
        if origin and "Access-Control-Allow-Origin" in response.headers:
            self._cors_success_count += 1

        return response

    def get_stats(self) -> Dict[str, Any]:
        """Get CORS middleware statistics."""
        return {
            "cors_success_count": self._cors_success_count,
            "cors_error_count": self._cors_error_count,
            "total_cors_requests": self._cors_success_count + self._cors_error_count,
            "allowed_origins_count": len(self.allow_origins),
            "allowed_origins": self.allow_origins
        }


# ============================================================
# CORS Utility Functions
# ============================================================

def create_cors_middleware(
    app: ASGIApp,
    allow_origins: Optional[List[str]] = None,
    allow_origin_regex: Optional[str] = None,
    allow_methods: Optional[List[str]] = None,
    allow_headers: Optional[List[str]] = None,
    allow_credentials: bool = True,
    expose_headers: Optional[List[str]] = None,
    max_age: int = 86400,
    allow_private_network: bool = True,
    enhanced: bool = True
) -> Union[CORSMiddleware, EnhancedCORSMiddleware]:
    """
    Factory function to create CORS middleware.

    Args:
        app: ASGI application
        allow_origins: List of allowed origins
        allow_origin_regex: Regex pattern for allowed origins
        allow_methods: List of allowed HTTP methods
        allow_headers: List of allowed headers
        allow_credentials: Whether to allow credentials
        expose_headers: Headers to expose to the client
        max_age: Max age for preflight requests
        allow_private_network: Allow private network access
        enhanced: Use enhanced CORS middleware

    Returns:
        CORS middleware instance
    """
    middleware_class = EnhancedCORSMiddleware if enhanced else CORSMiddleware

    return middleware_class(
        app=app,
        allow_origins=allow_origins,
        allow_origin_regex=allow_origin_regex,
        allow_methods=allow_methods,
        allow_headers=allow_headers,
        allow_credentials=allow_credentials,
        expose_headers=expose_headers,
        max_age=max_age,
        allow_private_network=allow_private_network
    )


# ============================================================
# CORS Configuration Helper
# ============================================================

def get_cors_config_from_env() -> Dict[str, Any]:
    """
    Get CORS configuration from environment variables.

    Returns:
        Dictionary of CORS configuration
    """
    config = {
        "allow_origins": [],
        "allow_origin_regex": None,
        "allow_methods": CORSConfig.DEFAULT_ALLOWED_METHODS.copy(),
        "allow_headers": CORSConfig.DEFAULT_ALLOWED_HEADERS.copy(),
        "allow_credentials": True,
        "expose_headers": CORSConfig.DEFAULT_EXPOSED_HEADERS.copy(),
        "max_age": int(os.getenv("CORS_MAX_AGE", CORSConfig.DEFAULT_MAX_AGE)),
        "allow_private_network": True
    }

    # Parse allowed origins
    origins_env = os.getenv("CORS_ALLOWED_ORIGINS", "")
    if origins_env:
        config["allow_origins"] = [o.strip() for o in origins_env.split(",") if o.strip()]
    else:
        # Use defaults
        config["allow_origins"] = CORSConfig.DEFAULT_ALLOWED_ORIGINS.copy()

        # Add environment-specific origins
        api_url = os.getenv("API_URL")
        if api_url:
            config["allow_origins"].append(api_url)

        frontend_url = os.getenv("FRONTEND_URL")
        if frontend_url:
            config["allow_origins"].append(frontend_url)

    # Parse regex
    regex_env = os.getenv("CORS_ALLOWED_ORIGIN_REGEX")
    if regex_env:
        config["allow_origin_regex"] = regex_env

    # Parse methods
    methods_env = os.getenv("CORS_ALLOWED_METHODS")
    if methods_env:
        config["allow_methods"] = [m.strip() for m in methods_env.split(",") if m.strip()]

    # Parse headers
    headers_env = os.getenv("CORS_ALLOWED_HEADERS")
    if headers_env:
        config["allow_headers"] = [h.strip() for h in headers_env.split(",") if h.strip()]

    # Parse credentials
    credentials_env = os.getenv("CORS_ALLOW_CREDENTIALS", "true")
    config["allow_credentials"] = credentials_env.lower() == "true"

    return config


# ============================================================
# Rate Limiting Classes (Existing)
# ============================================================

class RateLimitStrategy:
    """Rate limiting strategies."""
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


class RateLimiter:
    """Rate limiter with support for multiple strategies."""

    def __init__(
        self,
        strategy: str = RateLimitStrategy.SLIDING_WINDOW,
        max_requests: int = 100,
        window_seconds: int = 60,
        redis_url: Optional[str] = None,
        redis_prefix: str = "rate_limit:",
        use_redis: bool = False
    ):
        self.strategy = strategy
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.redis_prefix = redis_prefix
        self.use_redis = use_redis and REDIS_AVAILABLE and redis_url

        self._memory_storage = defaultdict(list)
        self._redis = None

        if self.use_redis:
            self._redis = aioredis.from_url(redis_url, decode_responses=True)
            logger.info(f"Rate limiter using Redis at {redis_url}")
        else:
            logger.info("Rate limiter using in-memory storage")

    def _get_key(self, identifier: str, endpoint: str) -> str:
        key = f"{self.redis_prefix}{endpoint}:{identifier}"
        return hashlib.md5(key.encode()).hexdigest()

    async def _get_redis_window(self, key: str) -> Dict[str, Any]:
        now = time.time()
        window_start = now - self.window_seconds

        async with self._redis as redis:
            await redis.zremrangebyscore(key, 0, window_start)
            count = await redis.zcard(key)
            await redis.zadd(key, {str(now): now})
            await redis.expire(key, self.window_seconds * 2)

            return {
                "count": count + 1,
                "limit": self.max_requests,
                "window": self.window_seconds
            }

    def _get_memory_window(self, key: str) -> Dict[str, Any]:
        now = time.time()
        window_start = now - self.window_seconds

        timestamps = self._memory_storage.get(key, [])
        timestamps = [t for t in timestamps if t > window_start]
        timestamps.append(now)
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
        key = self._get_key(identifier, endpoint)

        if self.use_redis and self._redis:
            try:
                result = await self._get_redis_window(key)
            except Exception as e:
                logger.warning(f"Redis rate limit failed: {e}, falling back to memory")
                result = self._get_memory_window(key)
        else:
            result = self._get_memory_window(key)

        is_allowed = result["count"] <= result["limit"]

        return {
            "allowed": is_allowed,
            "remaining": max(0, result["limit"] - result["count"]),
            "limit": result["limit"],
            "window": result["window"],
            "count": result["count"]
        }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""

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
        super().__init__(app)

        self.default_max_requests = default_max_requests
        self.default_window_seconds = default_window_seconds
        self.rate_limit_by_ip = rate_limit_by_ip
        self.rate_limit_by_api_key = rate_limit_by_api_key
        self.whitelist = set(whitelist or [])
        self.blacklist = set(blacklist or [])

        self.rate_limiter = RateLimiter(
            strategy=RateLimitStrategy.SLIDING_WINDOW,
            max_requests=default_max_requests,
            window_seconds=default_window_seconds,
            redis_url=redis_url,
            use_redis=use_redis
        )

        self.endpoint_limits = {}
        if endpoints:
            for endpoint, config in endpoints.items():
                self.endpoint_limits[endpoint] = {
                    "max_requests": config.get("max_requests", default_max_requests),
                    "window_seconds": config.get("window_seconds", default_window_seconds)
                }

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
        identifier = await self._get_identifier(request)
        endpoint = request.url.path

        if self._is_excluded(endpoint):
            return await call_next(request)

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

        if identifier in self.whitelist:
            return await call_next(request)

        rate_limit = self._get_rate_limit(endpoint)

        self.rate_limiter.max_requests = rate_limit[0]
        self.rate_limiter.window_seconds = rate_limit[1]

        result = await self.rate_limiter.check_rate_limit(identifier, endpoint)

        response = await call_next(request)

        response.headers["X-RateLimit-Limit"] = str(result["limit"])
        response.headers["X-RateLimit-Remaining"] = str(result["remaining"])
        response.headers["X-RateLimit-Window"] = str(result["window"])

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
        if self.rate_limit_by_api_key:
            api_key = request.headers.get("Authorization")
            if api_key and api_key.startswith("Bearer "):
                return f"api_key:{api_key[7:]}"

        if self.rate_limit_by_ip:
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                ip = forwarded.split(",")[0].strip()
            else:
                ip = request.client.host if request.client else "unknown"

            return f"ip:{ip}"

        session_id = request.cookies.get("session_id")
        if session_id:
            return f"session:{session_id}"

        return f"anonymous:{request.client.host if request.client else 'unknown'}"

    def _is_excluded(self, endpoint: str) -> bool:
        if endpoint in self.excluded_endpoints:
            return True

        if endpoint.startswith("/api/docs") or endpoint.startswith("/api/redoc"):
            return True

        if endpoint.startswith("/static") or endpoint.startswith("/assets"):
            return True

        return False

    def _get_rate_limit(self, endpoint: str) -> Tuple[int, int]:
        if endpoint in self.endpoint_limits:
            config = self.endpoint_limits[endpoint]
            return config["max_requests"], config["window_seconds"]

        for pattern, config in self.endpoint_limits.items():
            if pattern.endswith("*"):
                prefix = pattern[:-1]
                if endpoint.startswith(prefix):
                    return config["max_requests"], config["window_seconds"]

        return self.default_max_requests, self.default_window_seconds
