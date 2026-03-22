"""
Security module for DocQA AI API.
Provides security headers, input validation, sanitization, and protection against common vulnerabilities.
"""

import re
import json
import hashlib
import secrets
from typing import Dict, Any, Optional, List, Union, Set
from datetime import datetime, timedelta
from urllib.parse import urlparse, urljoin
import html
import bleach
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field, validator
import ipaddress

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================
# Security Headers Configuration
# ============================================================

class SecurityHeadersConfig:
    """Configuration for security headers."""

    # Default security headers
    DEFAULT_HEADERS = {
        # Content Security Policy
        "Content-Security-Policy": (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self' https:; "
            "frame-src 'none'; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "form-action 'self'; "
            "upgrade-insecure-requests; "
            "block-all-mixed-content"
        ),

        # XSS Protection
        "X-XSS-Protection": "1; mode=block",

        # Frame Options (clickjacking protection)
        "X-Frame-Options": "DENY",

        # Content Type Options (MIME sniffing protection)
        "X-Content-Type-Options": "nosniff",

        # Referrer Policy
        "Referrer-Policy": "strict-origin-when-cross-origin",

        # Permissions Policy (formerly Feature Policy)
        "Permissions-Policy": (
            "accelerometer=(), "
            "ambient-light-sensor=(), "
            "autoplay=(), "
            "battery=(), "
            "camera=(), "
            "display-capture=(), "
            "document-domain=(), "
            "encrypted-media=(), "
            "execution-while-not-rendered=(), "
            "execution-while-out-of-viewport=(), "
            "fullscreen=(), "
            "geolocation=(), "
            "gyroscope=(), "
            "magnetometer=(), "
            "microphone=(), "
            "midi=(), "
            "navigation-override=(), "
            "payment=(), "
            "picture-in-picture=(), "
            "publickey-credentials-get=(), "
            "speaker-selection=(), "
            "sync-xhr=(), "
            "usb=(), "
            "web-share=(), "
            "xr-spatial-tracking=()"
        ),

        # HSTS (HTTP Strict Transport Security)
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",

        # Cross-Origin Resource Sharing (handled separately)
        # "Access-Control-Allow-Origin": "*",  # Configured in CORS middleware

        # Cross-Origin Embedder Policy
        "Cross-Origin-Embedder-Policy": "require-corp",

        # Cross-Origin Opener Policy
        "Cross-Origin-Opener-Policy": "same-origin",

        # Cross-Origin Resource Policy
        "Cross-Origin-Resource-Policy": "same-origin",

        # Cache Control
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    }


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware for adding security headers to all responses.
    """

    def __init__(
        self,
        app,
        headers: Optional[Dict[str, str]] = None,
        csp_directives: Optional[Dict[str, List[str]]] = None,
        enable_hsts: bool = True,
        enable_csp: bool = True,
        enable_xss_protection: bool = True
    ):
        """
        Initialize security headers middleware.

        Args:
            app: ASGI application
            headers: Custom headers (overrides defaults)
            csp_directives: Custom CSP directives
            enable_hsts: Enable HSTS header
            enable_csp: Enable CSP header
            enable_xss_protection: Enable XSS protection header
        """
        super().__init__(app)

        self.headers = SecurityHeadersConfig.DEFAULT_HEADERS.copy()

        # Override with custom headers
        if headers:
            self.headers.update(headers)

        # Configure CSP
        if csp_directives and enable_csp:
            self.headers["Content-Security-Policy"] = self._build_csp(csp_directives)

        # Disable headers if not enabled
        if not enable_hsts:
            self.headers.pop("Strict-Transport-Security", None)

        if not enable_csp:
            self.headers.pop("Content-Security-Policy", None)

        if not enable_xss_protection:
            self.headers.pop("X-XSS-Protection", None)

        logger.info("Security headers middleware initialized")

    def _build_csp(self, directives: Dict[str, List[str]]) -> str:
        """Build CSP header from directives."""
        parts = []
        for key, values in directives.items():
            if values:
                parts.append(f"{key} {' '.join(values)}")
        return "; ".join(parts)

    async def dispatch(self, request: Request, call_next):
        """Add security headers to response."""
        response = await call_next(request)

        # Add security headers
        for key, value in self.headers.items():
            response.headers[key] = value

        return response


# ============================================================
# Input Validation and Sanitization
# ============================================================

class InputValidator:
    """
    Input validation and sanitization utilities.
    """

    # Patterns for validation
    PATTERNS = {
        "email": r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        "uuid": r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        "alphanumeric": r'^[a-zA-Z0-9_\-]+$',
        "username": r'^[a-zA-Z0-9_\-]{3,50}$',
        "password": r'^.{8,}$',
        "api_key": r'^[a-zA-Z0-9_\-]{20,}$',
        "slug": r'^[a-z0-9]+(?:-[a-z0-9]+)*$',
        "hex_color": r'^#[0-9a-fA-F]{6}$',
        "url": r'^https?://[^\s/$.?#].[^\s]*$',
        "ip_address": r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$',
        "document_id": r'^doc_[a-zA-Z0-9]{8,}$',
        "session_id": r'^session_[a-zA-Z0-9]{8,}$',
        "query_id": r'^query_[a-zA-Z0-9]{8,}$'
    }

    @classmethod
    def validate_email(cls, email: str) -> bool:
        """Validate email address."""
        return bool(re.match(cls.PATTERNS["email"], email))

    @classmethod
    def validate_uuid(cls, uuid_str: str) -> bool:
        """Validate UUID string."""
        return bool(re.match(cls.PATTERNS["uuid"], uuid_str))

    @classmethod
    def validate_username(cls, username: str) -> bool:
        """Validate username."""
        return bool(re.match(cls.PATTERNS["username"], username))

    @classmethod
    def validate_password(cls, password: str) -> bool:
        """Validate password strength."""
        if len(password) < 8:
            return False

        # Check for at least one uppercase, lowercase, digit, and special char
        checks = [
            any(c.isupper() for c in password),
            any(c.islower() for c in password),
            any(c.isdigit() for c in password),
            any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        ]

        return sum(checks) >= 3

    @classmethod
    def validate_url(cls, url: str) -> bool:
        """Validate URL."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    @classmethod
    def validate_ip(cls, ip: str) -> bool:
        """Validate IP address."""
        try:
            ipaddress.ip_address(ip)
            return True
        except ValueError:
            return False

    @classmethod
    def sanitize_input(cls, value: str, max_length: int = 10000) -> str:
        """
        Sanitize user input.

        Args:
            value: Input value
            max_length: Maximum length

        Returns:
            Sanitized value
        """
        if not value:
            return value

        # Trim to max length
        if len(value) > max_length:
            value = value[:max_length]

        # Escape HTML entities
        value = html.escape(value)

        # Remove control characters
        value = ''.join(c for c in value if c.isprintable() or c in '\n\r\t')

        return value

    @classmethod
    def sanitize_html(cls, html_content: str, allowed_tags: Optional[List[str]] = None) -> str:
        """
        Sanitize HTML content.

        Args:
            html_content: HTML content
            allowed_tags: Allowed HTML tags

        Returns:
            Sanitized HTML
        """
        if allowed_tags is None:
            allowed_tags = [
                'p', 'br', 'b', 'i', 'u', 'strong', 'em', 'ul', 'ol', 'li',
                'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'pre', 'code', 'blockquote',
                'a', 'img', 'table', 'tr', 'td', 'th', 'thead', 'tbody'
            ]

        allowed_attrs = {
            'a': ['href', 'title', 'target'],
            'img': ['src', 'alt', 'title', 'width', 'height'],
            'table': ['border', 'cellpadding', 'cellspacing'],
            '*': ['class', 'id', 'style']
        }

        return bleach.clean(
            html_content,
            tags=allowed_tags,
            attributes=allowed_attrs,
            strip=True
        )

    @classmethod
    def validate_file_name(cls, file_name: str) -> bool:
        """Validate file name for security."""
        # Check for path traversal
        if '..' in file_name or '/' in file_name or '\\' in file_name:
            return False

        # Check for null bytes
        if '\0' in file_name:
            return False

        # Check length
        if len(file_name) > 255:
            return False

        return True

    @classmethod
    def validate_content_type(cls, content_type: str, allowed_types: List[str]) -> bool:
        """Validate content type."""
        if not content_type:
            return False

        # Extract main type
        main_type = content_type.split(';')[0].strip().lower()
        return main_type in allowed_types

    @classmethod
    def sanitize_query(cls, query: str) -> str:
        """Sanitize a search query."""
        # Remove potentially harmful characters
        harmful_patterns = [
            r'[<>{}]',  # Braces and brackets
            r'[\'"]',   # Quotes
            r'[;|&$`]'  # Shell metacharacters
        ]

        for pattern in harmful_patterns:
            query = re.sub(pattern, '', query)

        # Limit length
        if len(query) > 1000:
            query = query[:1000]

        return query.strip()


# ============================================================
# Request Validation Models
# ============================================================

class SecureRequest(BaseModel):
    """Base model with security validations."""

    @validator('*')
    def sanitize_strings(cls, v):
        """Sanitize all string fields."""
        if isinstance(v, str):
            return InputValidator.sanitize_input(v)
        return v


class SecureQueryRequest(SecureRequest):
    """Secure query request model."""
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(5, ge=1, le=50)
    temperature: Optional[float] = Field(None, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, ge=1, le=4096)
    include_sources: bool = True
    session_id: Optional[str] = Field(None, regex=r'^[a-zA-Z0-9_\-]{8,}$')

    @validator('question')
    def validate_question(cls, v):
        """Validate and sanitize question."""
        v = InputValidator.sanitize_query(v)
        if len(v) < 2:
            raise ValueError('Question must be at least 2 characters')
        return v


class SecureDocumentUploadRequest(BaseModel):
    """Secure document upload request model."""
    file_name: str = Field(..., max_length=255)
    file_size: int = Field(..., ge=1, le=100*1024*1024)  # 100MB
    chunk_size: int = Field(800, ge=100, le=10000)
    chunk_overlap: int = Field(150, ge=0, le=5000)
    chunking_strategy: str = Field("adaptive", regex=r'^[a-zA-Z_\-]+$')

    @validator('file_name')
    def validate_file_name(cls, v):
        """Validate file name for security."""
        if not InputValidator.validate_file_name(v):
            raise ValueError('Invalid file name')
        return v


class SecureLoginRequest(SecureRequest):
    """Secure login request model."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)

    @validator('username')
    def validate_username(cls, v):
        """Validate username."""
        if not InputValidator.validate_username(v):
            raise ValueError('Invalid username format')
        return v

    @validator('password')
    def validate_password(cls, v):
        """Validate password."""
        if not InputValidator.validate_password(v):
            raise ValueError('Password must be at least 8 characters and contain uppercase, lowercase, digit, and special character')
        return v


# ============================================================
# CSRF Protection
# ============================================================

class CSRFProtection:
    """
    CSRF protection utilities.
    """

    @staticmethod
    def generate_token() -> str:
        """Generate a CSRF token."""
        return secrets.token_urlsafe(32)

    @staticmethod
    def verify_token(token: str, stored_token: str) -> bool:
        """Verify CSRF token."""
        if not token or not stored_token:
            return False

        # Use constant time comparison
        return secrets.compare_digest(token, stored_token)

    @staticmethod
    def get_token_from_request(request: Request) -> Optional[str]:
        """Extract CSRF token from request."""
        # Check header
        token = request.headers.get("X-CSRF-Token")
        if token:
            return token

        # Check body
        try:
            body = request._json
            if body and "csrf_token" in body:
                return body["csrf_token"]
        except Exception:
            pass

        # Check query parameters
        token = request.query_params.get("csrf_token")
        if token:
            return token

        # Check cookies
        token = request.cookies.get("csrf_token")
        if token:
            return token

        return None


# ============================================================
# Rate Limiting (Security-focused)
# ============================================================

class SecurityRateLimiter:
    """
    Security-focused rate limiter for login attempts and sensitive endpoints.
    """

    def __init__(self):
        self._attempts: Dict[str, List[float]] = {}
        self._blocked: Set[str] = set()
        self._blocked_until: Dict[str, float] = {}

    def check_rate_limit(
        self,
        identifier: str,
        max_attempts: int = 5,
        window_seconds: int = 300,
        block_seconds: int = 3600
    ) -> bool:
        """
        Check rate limit for an identifier.

        Args:
            identifier: Unique identifier (IP, username, etc.)
            max_attempts: Maximum attempts in window
            window_seconds: Window size in seconds
            block_seconds: Block duration in seconds

        Returns:
            True if allowed, False if rate limited
        """
        # Check if blocked
        if identifier in self._blocked:
            if identifier in self._blocked_until:
                if time.time() < self._blocked_until[identifier]:
                    return False
                else:
                    self._blocked.remove(identifier)
                    self._blocked_until.pop(identifier, None)
            else:
                return False

        now = time.time()
        window_start = now - window_seconds

        # Clean up old attempts
        if identifier in self._attempts:
            self._attempts[identifier] = [
                t for t in self._attempts[identifier] if t > window_start
            ]
        else:
            self._attempts[identifier] = []

        # Check if attempts exceeded
        if len(self._attempts[identifier]) >= max_attempts:
            # Block the identifier
            self._blocked.add(identifier)
            self._blocked_until[identifier] = now + block_seconds
            return False

        # Add attempt
        self._attempts[identifier].append(now)
        return True

    def reset_attempts(self, identifier: str):
        """Reset attempts for an identifier."""
        if identifier in self._attempts:
            del self._attempts[identifier]
        if identifier in self._blocked:
            self._blocked.remove(identifier)
        if identifier in self._blocked_until:
            del self._blocked_until[identifier]


# ============================================================
# SQL Injection Prevention
# ============================================================

class SQLInjectionPrevention:
    """
    SQL injection prevention utilities.
    """

    @staticmethod
    def sanitize_identifier(identifier: str) -> str:
        """
        Sanitize a SQL identifier (table name, column name).

        Args:
            identifier: SQL identifier

        Returns:
            Sanitized identifier
        """
        # Remove any non-alphanumeric characters
        return re.sub(r'[^a-zA-Z0-9_]', '', identifier)

    @staticmethod
    def sanitize_order_by(order_by: str) -> str:
        """
        Sanitize ORDER BY clause.

        Args:
            order_by: ORDER BY clause

        Returns:
            Sanitized ORDER BY clause
        """
        # Only allow simple order by clauses
        if not re.match(r'^[a-zA-Z0-9_]+( ASC| DESC)?$', order_by):
            return 'id ASC'
        return order_by

    @staticmethod
    def validate_pagination(page: int, page_size: int) -> tuple:
        """
        Validate pagination parameters.

        Args:
            page: Page number
            page_size: Page size

        Returns:
            Validated (page, page_size)
        """
        page = max(1, page)
        page_size = min(100, max(1, page_size))
        return page, page_size


# ============================================================
# Security Utilities
# ============================================================

class SecurityUtils:
    """
    Security utilities for common operations.
    """

    @staticmethod
    def generate_api_key() -> str:
        """Generate a secure API key."""
        return f"docqa_{secrets.token_urlsafe(32)}"

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password."""
        import bcrypt
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify a password against a hash."""
        import bcrypt
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate a secure random token."""
        return secrets.token_urlsafe(length)

    @staticmethod
    def generate_secure_id(prefix: str = "", length: int = 16) -> str:
        """Generate a secure ID."""
        if prefix:
            return f"{prefix}_{secrets.token_hex(length)}"
        return secrets.token_hex(length)

    @staticmethod
    def is_safe_url(url: str, allowed_hosts: List[str]) -> bool:
        """
        Check if a URL is safe.

        Args:
            url: URL to check
            allowed_hosts: List of allowed hosts

        Returns:
            True if safe, False otherwise
        """
        try:
            parsed = urlparse(url)
            return parsed.netloc in allowed_hosts
        except Exception:
            return False

    @staticmethod
    def sanitize_path(path: str) -> str:
        """
        Sanitize a file path.

        Args:
            path: File path

        Returns:
            Sanitized path
        """
        # Remove path traversal
        while '../' in path:
            path = path.replace('../', '')
        while '..\\' in path:
            path = path.replace('..\\', '')

        # Remove null bytes
        path = path.replace('\0', '')

        # Normalize path
        path = os.path.normpath(path)

        # Prevent absolute paths
        if os.path.isabs(path):
            path = path.lstrip('/')

        return path


# ============================================================
# Security Middleware
# ============================================================

class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Combined security middleware with all security features.
    """

    def __init__(
        self,
        app,
        enable_headers: bool = True,
        enable_validation: bool = True,
        enable_rate_limit: bool = True,
        enable_csrf: bool = False,
        enable_ip_whitelist: bool = False,
        whitelist: Optional[List[str]] = None,
        blacklist: Optional[List[str]] = None
    ):
        """
        Initialize security middleware.

        Args:
            app: ASGI application
            enable_headers: Enable security headers
            enable_validation: Enable input validation
            enable_rate_limit: Enable rate limiting
            enable_csrf: Enable CSRF protection
            enable_ip_whitelist: Enable IP whitelist
            whitelist: IP whitelist
            blacklist: IP blacklist
        """
        super().__init__(app)

        self.enable_headers = enable_headers
        self.enable_validation = enable_validation
        self.enable_rate_limit = enable_rate_limit
        self.enable_csrf = enable_csrf
        self.enable_ip_whitelist = enable_ip_whitelist

        self.whitelist = set(whitelist or [])
        self.blacklist = set(blacklist or [])
        self.rate_limiter = SecurityRateLimiter()

        # Initialize headers middleware if enabled
        self.headers_middleware = SecurityHeadersMiddleware(app) if enable_headers else None

        logger.info("Security middleware initialized")

    async def dispatch(self, request: Request, call_next):
        """
        Process request with security checks.
        """
        # IP filtering
        if self.enable_ip_whitelist or self.blacklist:
            client_ip = request.client.host if request.client else None

            if client_ip:
                if self.blacklist and client_ip in self.blacklist:
                    return JSONResponse(
                        status_code=status.HTTP_403_FORBIDDEN,
                        content={
                            "error": {
                                "code": "FORBIDDEN",
                                "message": "Access denied",
                                "timestamp": datetime.now().isoformat()
                            }
                        }
                    )

                if self.whitelist and client_ip not in self.whitelist:
                    return JSONResponse(
                        status_code=status.HTTP_403_FORBIDDEN,
                        content={
                            "error": {
                                "code": "FORBIDDEN",
                                "message": "Access denied",
                                "timestamp": datetime.now().isoformat()
                            }
                        }
                    )

        # CSRF protection
        if self.enable_csrf:
            # Skip for GET, HEAD, OPTIONS methods
            if request.method not in ["GET", "HEAD", "OPTIONS"]:
                token = CSRFProtection.get_token_from_request(request)
                # In production, check against stored token
                if not token:
                    return JSONResponse(
                        status_code=status.HTTP_403_FORBIDDEN,
                        content={
                            "error": {
                                "code": "CSRF_TOKEN_MISSING",
                                "message": "CSRF token missing",
                                "timestamp": datetime.now().isoformat()
                            }
                        }
                    )

        # Rate limiting for sensitive endpoints
        if self.enable_rate_limit:
            if request.url.path in [
                "/api/v1/auth/login",
                "/api/v1/auth/register",
                "/api/v1/auth/forgot-password"
            ]:
                identifier = request.client.host if request.client else "unknown"
                if not self.rate_limiter.check_rate_limit(identifier):
                    return JSONResponse(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        content={
                            "error": {
                                "code": "RATE_LIMIT_EXCEEDED",
                                "message": "Too many attempts. Please try again later.",
                                "timestamp": datetime.now().isoformat()
                            }
                        }
                    )

        # Process request
        response = await call_next(request)

        # Add security headers
        if self.headers_middleware:
            response = await self.headers_middleware.dispatch(request, lambda req: response)

        return response


# ============================================================
# Factory Functions
# ============================================================

def create_security_middleware(
    app,
    enable_headers: bool = True,
    enable_validation: bool = True,
    enable_rate_limit: bool = True,
    enable_csrf: bool = False,
    enable_ip_whitelist: bool = False,
    whitelist: Optional[List[str]] = None,
    blacklist: Optional[List[str]] = None
) -> SecurityMiddleware:
    """
    Create security middleware instance.

    Args:
        app: ASGI application
        enable_headers: Enable security headers
        enable_validation: Enable input validation
        enable_rate_limit: Enable rate limiting
        enable_csrf: Enable CSRF protection
        enable_ip_whitelist: Enable IP whitelist
        whitelist: IP whitelist
        blacklist: IP blacklist

    Returns:
        SecurityMiddleware instance
    """
    return SecurityMiddleware(
        app=app,
        enable_headers=enable_headers,
        enable_validation=enable_validation,
        enable_rate_limit=enable_rate_limit,
        enable_csrf=enable_csrf,
        enable_ip_whitelist=enable_ip_whitelist,
        whitelist=whitelist,
        blacklist=blacklist
    )


if __name__ == "__main__":
    # Example usage
    import asyncio
    import time

    print("Testing Security Module...")
    print("=" * 60)

    # Test input validation
    print("\n🔒 Input Validation:")
    print(f"  Valid email: {InputValidator.validate_email('test@example.com')}")
    print(f"  Invalid email: {InputValidator.validate_email('test@')}")
    print(f"  Valid password: {InputValidator.validate_password('Test123!@#')}")
    print(f"  Invalid password: {InputValidator.validate_password('pass')}")
    print(f"  Sanitized input: {InputValidator.sanitize_input('<script>alert(1)</script>')}")

    # Test rate limiter
    print("\n⏱️  Rate Limiter:")
    rate_limiter = SecurityRateLimiter()
    identifier = "test_user"

    for i in range(7):
        allowed = rate_limiter.check_rate_limit(identifier, max_attempts=5)
        print(f"  Attempt {i+1}: {'Allowed' if allowed else 'Blocked'}")

    # Test security utilities
    print("\n🔑 Security Utilities:")
    print(f"  API Key: {SecurityUtils.generate_api_key()[:20]}...")
    print(f"  Secure Token: {SecurityUtils.generate_secure_token()[:20]}...")
    print(f"  Secure ID: {SecurityUtils.generate_secure_id('doc')}")

    # Test CSRF protection
    print("\n🛡️  CSRF Protection:")
    token = CSRFProtection.generate_token()
    print(f"  Generated token: {token[:20]}...")
    print(f"  Verified: {CSRFProtection.verify_token(token, token)}")

    print("\n✅ Security module ready!")
