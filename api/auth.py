"""
Authentication and authorization module for DocQA AI API.
Provides JWT-based authentication, API key support, role-based access control,
and token management with refresh capabilities.
"""

import os
import time
import json
import hashlib
import secrets
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps

from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================
# Configuration
# ============================================================

class AuthConfig:
    """Authentication configuration."""

    # JWT settings
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7"))

    # API key settings
    API_KEY_HEADER: str = os.getenv("API_KEY_HEADER", "X-API-Key")

    # Password settings
    PASSWORD_CONTEXT = CryptContext(schemes=["bcrypt"], deprecated="auto")

    # Token blacklist (in-memory for demo, use Redis in production)
    _token_blacklist: set = set()
    _api_keys: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def add_to_blacklist(cls, token: str):
        """Add token to blacklist."""
        cls._token_blacklist.add(token)

    @classmethod
    def is_blacklisted(cls, token: str) -> bool:
        """Check if token is blacklisted."""
        return token in cls._token_blacklist

    @classmethod
    def clear_blacklist(cls):
        """Clear token blacklist."""
        cls._token_blacklist.clear()


# ============================================================
# Models
# ============================================================

class TokenType(str, Enum):
    """Token type enumeration."""
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"


class UserRole(str, Enum):
    """User role enumeration."""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"
    SYSTEM = "system"


class Permission(str, Enum):
    """Permission enumeration."""
    QUERY = "query"
    INGEST = "ingest"
    DELETE = "delete"
    MANAGE = "manage"
    ADMIN = "admin"
    VIEW_STATS = "view_stats"
    VIEW_CONFIG = "view_config"
    MANAGE_USERS = "manage_users"
    MANAGE_API_KEYS = "manage_api_keys"


class TokenData(BaseModel):
    """Token data model."""
    user_id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: Optional[str] = Field(None, description="User email")
    role: UserRole = Field(UserRole.USER, description="User role")
    permissions: List[Permission] = Field(
        default_factory=list,
        description="User permissions"
    )
    token_type: TokenType = Field(TokenType.ACCESS, description="Token type")
    expires_at: int = Field(..., description="Expiration timestamp")
    issued_at: int = Field(default_factory=lambda: int(time.time()), description="Issued timestamp")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "permissions": [p.value for p in self.permissions],
            "token_type": self.token_type.value,
            "expires_at": self.expires_at,
            "issued_at": self.issued_at
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TokenData':
        """Create from dictionary."""
        return cls(
            user_id=data["user_id"],
            username=data["username"],
            email=data.get("email"),
            role=UserRole(data.get("role", "user")),
            permissions=[Permission(p) for p in data.get("permissions", [])],
            token_type=TokenType(data.get("token_type", "access")),
            expires_at=data["expires_at"],
            issued_at=data.get("issued_at", int(time.time()))
        )


class User(BaseModel):
    """User model."""
    user_id: str = Field(..., description="User ID")
    username: str = Field(..., description="Username")
    email: Optional[str] = Field(None, description="User email")
    hashed_password: Optional[str] = Field(None, description="Hashed password")
    role: UserRole = Field(UserRole.USER, description="User role")
    permissions: List[Permission] = Field(
        default_factory=list,
        description="User permissions"
    )
    is_active: bool = Field(True, description="User active status")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    api_keys: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="API keys"
    )


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str = Field(..., description="Access token")
    refresh_token: str = Field(..., description="Refresh token")
    token_type: str = Field("Bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiration in seconds")
    refresh_expires_in: int = Field(..., description="Refresh token expiration in seconds")
    user: Dict[str, Any] = Field(..., description="User information")


class LoginRequest(BaseModel):
    """Login request model."""
    username: str = Field(..., description="Username")
    password: str = Field(..., description="Password")


class RefreshTokenRequest(BaseModel):
    """Refresh token request model."""
    refresh_token: str = Field(..., description="Refresh token")


class ChangePasswordRequest(BaseModel):
    """Change password request model."""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., description="New password", min_length=8)


# ============================================================
# Authentication Service
# ============================================================

class AuthService:
    """
    Authentication service for managing users, tokens, and API keys.
    """

    # In-memory user store (use database in production)
    _users: Dict[str, User] = {}

    @classmethod
    def initialize(cls):
        """Initialize with default admin user."""
        if not cls._users:
            # Create default admin user
            admin = User(
                user_id="admin_001",
                username="admin",
                email="admin@docqa-ai.com",
                hashed_password=AuthConfig.PASSWORD_CONTEXT.hash("admin123"),
                role=UserRole.ADMIN,
                permissions=[Permission.ADMIN, Permission.MANAGE, Permission.QUERY,
                           Permission.INGEST, Permission.DELETE, Permission.MANAGE_USERS,
                           Permission.MANAGE_API_KEYS, Permission.VIEW_STATS, Permission.VIEW_CONFIG]
            )
            cls._users["admin"] = admin

            # Create default user
            user = User(
                user_id="user_001",
                username="user",
                email="user@docqa-ai.com",
                hashed_password=AuthConfig.PASSWORD_CONTEXT.hash("user123"),
                role=UserRole.USER,
                permissions=[Permission.QUERY, Permission.VIEW_STATS]
            )
            cls._users["user"] = user

            logger.info("Initialized default admin and user accounts")

    @classmethod
    def authenticate_user(cls, username: str, password: str) -> Optional[User]:
        """
        Authenticate a user by username and password.

        Args:
            username: Username
            password: Password

        Returns:
            User object if authenticated, None otherwise
        """
        user = cls._users.get(username)
        if not user:
            return None

        if not user.is_active:
            return None

        if not user.hashed_password:
            return None

        if not AuthConfig.PASSWORD_CONTEXT.verify(password, user.hashed_password):
            return None

        # Update last login
        user.last_login = datetime.now()

        return user

    @classmethod
    def get_user(cls, username: str) -> Optional[User]:
        """Get user by username."""
        return cls._users.get(username)

    @classmethod
    def get_user_by_id(cls, user_id: str) -> Optional[User]:
        """Get user by ID."""
        for user in cls._users.values():
            if user.user_id == user_id:
                return user
        return None

    @classmethod
    def create_user(
        cls,
        username: str,
        password: str,
        email: Optional[str] = None,
        role: UserRole = UserRole.USER,
        permissions: Optional[List[Permission]] = None
    ) -> Optional[User]:
        """
        Create a new user.

        Args:
            username: Username
            password: Password
            email: User email
            role: User role
            permissions: User permissions

        Returns:
            Created User object or None
        """
        if username in cls._users:
            return None

        user = User(
            user_id=f"user_{len(cls._users) + 1:03d}",
            username=username,
            email=email,
            hashed_password=AuthConfig.PASSWORD_CONTEXT.hash(password),
            role=role,
            permissions=permissions or []
        )

        cls._users[username] = user
        return user

    @classmethod
    def update_user(cls, username: str, **kwargs) -> Optional[User]:
        """Update user information."""
        user = cls._users.get(username)
        if not user:
            return None

        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)

        user.updated_at = datetime.now()
        return user

    @classmethod
    def delete_user(cls, username: str) -> bool:
        """Delete a user."""
        if username in cls._users:
            del cls._users[username]
            return True
        return False

    @classmethod
    def list_users(cls) -> List[Dict[str, Any]]:
        """List all users."""
        return [
            {
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
                "permissions": [p.value for p in user.permissions],
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat(),
                "last_login": user.last_login.isoformat() if user.last_login else None
            }
            for user in cls._users.values()
        ]


# ============================================================
# Token Manager
# ============================================================

class TokenManager:
    """
    Token management for JWT and API keys.
    """

    @staticmethod
    def create_access_token(
        user: User,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a JWT access token.

        Args:
            user: User object
            expires_delta: Token expiration time

        Returns:
            JWT token string
        """
        if expires_delta is None:
            expires_delta = timedelta(minutes=AuthConfig.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)

        expires_at = int(time.time()) + int(expires_delta.total_seconds())

        token_data = TokenData(
            user_id=user.user_id,
            username=user.username,
            email=user.email,
            role=user.role,
            permissions=user.permissions,
            token_type=TokenType.ACCESS,
            expires_at=expires_at
        )

        return jwt.encode(
            token_data.to_dict(),
            AuthConfig.JWT_SECRET_KEY,
            algorithm=AuthConfig.JWT_ALGORITHM
        )

    @staticmethod
    def create_refresh_token(
        user: User,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a JWT refresh token.

        Args:
            user: User object
            expires_delta: Token expiration time

        Returns:
            JWT token string
        """
        if expires_delta is None:
            expires_delta = timedelta(days=AuthConfig.JWT_REFRESH_TOKEN_EXPIRE_DAYS)

        expires_at = int(time.time()) + int(expires_delta.total_seconds())

        token_data = TokenData(
            user_id=user.user_id,
            username=user.username,
            email=user.email,
            role=user.role,
            permissions=user.permissions,
            token_type=TokenType.REFRESH,
            expires_at=expires_at
        )

        return jwt.encode(
            token_data.to_dict(),
            AuthConfig.JWT_SECRET_KEY,
            algorithm=AuthConfig.JWT_ALGORITHM
        )

    @staticmethod
    def create_api_key(
        user: User,
        name: str,
        expires_in_days: int = 30
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Create an API key.

        Args:
            user: User object
            name: API key name
            expires_in_days: Expiration in days

        Returns:
            Tuple of (api_key, key_info)
        """
        # Generate API key
        key_prefix = "docqa_"
        key_suffix = secrets.token_urlsafe(32)
        api_key = f"{key_prefix}{key_suffix}"

        # Hash the key for storage
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        key_info = {
            "id": f"key_{len(user.api_keys) + 1}",
            "name": name,
            "hash": key_hash,
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(days=expires_in_days)).isoformat(),
            "permissions": [p.value for p in user.permissions],
            "last_used": None,
            "is_active": True
        }

        # Store in user's API keys
        user.api_keys.append(key_info)

        return api_key, key_info

    @staticmethod
    def verify_api_key(api_key: str) -> Optional[User]:
        """
        Verify an API key and return the associated user.

        Args:
            api_key: API key string

        Returns:
            User object if valid, None otherwise
        """
        # Check if key starts with prefix
        if not api_key.startswith("docqa_"):
            return None

        # Hash the key for lookup
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Search for user with this key
        for user in AuthService._users.values():
            for key_info in user.api_keys:
                if key_info.get("hash") == key_hash and key_info.get("is_active", True):
                    # Check expiration
                    expires_at = key_info.get("expires_at")
                    if expires_at:
                        if datetime.fromisoformat(expires_at) < datetime.now():
                            continue

                    # Update last used
                    key_info["last_used"] = datetime.now().isoformat()

                    return user

        return None

    @staticmethod
    def revoke_api_key(user: User, key_id: str) -> bool:
        """
        Revoke an API key.

        Args:
            user: User object
            key_id: API key ID

        Returns:
            True if revoked, False otherwise
        """
        for key_info in user.api_keys:
            if key_info.get("id") == key_id:
                key_info["is_active"] = False
                return True
        return False

    @staticmethod
    def decode_token(token: str) -> Optional[TokenData]:
        """
        Decode and validate a JWT token.

        Args:
            token: JWT token string

        Returns:
            TokenData object if valid, None otherwise
        """
        try:
            payload = jwt.decode(
                token,
                AuthConfig.JWT_SECRET_KEY,
                algorithms=[AuthConfig.JWT_ALGORITHM]
            )

            # Check blacklist
            if AuthConfig.is_blacklisted(token):
                return None

            # Check expiration
            if payload.get("expires_at", 0) < time.time():
                return None

            return TokenData.from_dict(payload)

        except JWTError:
            return None

    @staticmethod
    def refresh_access_token(refresh_token: str) -> Optional[str]:
        """
        Refresh an access token using a refresh token.

        Args:
            refresh_token: Refresh token

        Returns:
            New access token or None
        """
        # Decode refresh token
        token_data = TokenManager.decode_token(refresh_token)
        if not token_data:
            return None

        # Verify it's a refresh token
        if token_data.token_type != TokenType.REFRESH:
            return None

        # Get user
        user = AuthService.get_user(token_data.username)
        if not user:
            return None

        # Create new access token
        return TokenManager.create_access_token(user)

    @staticmethod
    def revoke_token(token: str):
        """Revoke a token (add to blacklist)."""
        AuthConfig.add_to_blacklist(token)


# ============================================================
# Permission Checker
# ============================================================

class PermissionChecker:
    """Permission checking utilities."""

    @staticmethod
    def has_permission(user: User, permission: Permission) -> bool:
        """
        Check if a user has a specific permission.

        Args:
            user: User object
            permission: Permission to check

        Returns:
            True if user has permission, False otherwise
        """
        if not user or not user.is_active:
            return False

        # Admin has all permissions
        if user.role == UserRole.ADMIN:
            return True

        return permission in user.permissions

    @staticmethod
    def has_any_permission(user: User, permissions: List[Permission]) -> bool:
        """
        Check if a user has any of the specified permissions.

        Args:
            user: User object
            permissions: List of permissions

        Returns:
            True if user has any permission, False otherwise
        """
        if not user or not user.is_active:
            return False

        if user.role == UserRole.ADMIN:
            return True

        return any(p in user.permissions for p in permissions)

    @staticmethod
    def has_all_permissions(user: User, permissions: List[Permission]) -> bool:
        """
        Check if a user has all of the specified permissions.

        Args:
            user: User object
            permissions: List of permissions

        Returns:
            True if user has all permissions, False otherwise
        """
        if not user or not user.is_active:
            return False

        if user.role == UserRole.ADMIN:
            return True

        return all(p in user.permissions for p in permissions)

    @staticmethod
    def require_permission(permission: Permission):
        """
        Decorator for requiring a specific permission.

        Args:
            permission: Required permission

        Returns:
            Decorator function
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Get user from context
                user = kwargs.get("current_user")
                if not user:
                    # Try to get from request
                    for arg in args:
                        if isinstance(arg, Request):
                            user = getattr(arg.state, "user", None)
                            break

                if not PermissionChecker.has_permission(user, permission):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Permission '{permission.value}' required"
                    )

                return await func(*args, **kwargs)
            return wrapper
        return decorator


# ============================================================
# FastAPI Dependencies
# ============================================================

class HTTPBearerWithAuth(HTTPBearer):
    """HTTP Bearer authentication with custom handling."""

    async def __call__(self, request: Request) -> Optional[User]:
        """
        Authenticate request using Bearer token.

        Args:
            request: FastAPI request

        Returns:
            User object if authenticated

        Raises:
            HTTPException: If authentication fails
        """
        # Try API key first
        api_key = request.headers.get(AuthConfig.API_KEY_HEADER)
        if api_key:
            user = TokenManager.verify_api_key(api_key)
            if user:
                request.state.user = user
                return user

        # Try Bearer token
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"}
            )

        token = credentials.credentials
        token_data = TokenManager.decode_token(token)

        if not token_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"}
            )

        user = AuthService.get_user(token_data.username)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"}
            )

        request.state.user = user
        request.state.token_data = token_data

        return user


# ============================================================
# Dependency Functions
# ============================================================

security = HTTPBearerWithAuth(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> User:
    """
    Get current authenticated user.

    Args:
        request: FastAPI request
        credentials: HTTP credentials

    Returns:
        User object

    Raises:
        HTTPException: If not authenticated
    """
    # Check if user is already in request state
    if hasattr(request.state, "user") and request.state.user:
        return request.state.user

    # Try API key
    api_key = request.headers.get(AuthConfig.API_KEY_HEADER)
    if api_key:
        user = TokenManager.verify_api_key(api_key)
        if user:
            request.state.user = user
            return user

    # Try Bearer token
    if credentials:
        token_data = TokenManager.decode_token(credentials.credentials)
        if token_data:
            user = AuthService.get_user(token_data.username)
            if user:
                request.state.user = user
                return user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"}
    )


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current active user.

    Args:
        current_user: Current user

    Returns:
        User object

    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User is inactive"
        )
    return current_user


async def get_current_admin_user(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """
    Get current admin user.

    Args:
        current_user: Current user

    Returns:
        User object

    Raises:
        HTTPException: If user is not admin
    """
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user


def require_permission(permission: Permission):
    """
    Dependency for requiring a specific permission.

    Args:
        permission: Required permission

    Returns:
        Dependency function
    """
    async def dependency(current_user: User = Depends(get_current_active_user)):
        if not PermissionChecker.has_permission(current_user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission.value}' required"
            )
        return current_user
    return dependency


def require_any_permission(permissions: List[Permission]):
    """
    Dependency for requiring any of the specified permissions.

    Args:
        permissions: List of permissions

    Returns:
        Dependency function
    """
    async def dependency(current_user: User = Depends(get_current_active_user)):
        if not PermissionChecker.has_any_permission(current_user, permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Any of permissions {[p.value for p in permissions]} required"
            )
        return current_user
    return dependency


# ============================================================
# Quick Initialization
# ============================================================

# Initialize with default users
AuthService.initialize()


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    # Test authentication
    print("Testing Authentication...")

    # Test login
    user = AuthService.authenticate_user("admin", "admin123")
    if user:
        print(f"✅ Authenticated user: {user.username} (role: {user.role.value})")

        # Create access token
        access_token = TokenManager.create_access_token(user)
        print(f"✅ Access token: {access_token[:20]}...")

        # Decode token
        token_data = TokenManager.decode_token(access_token)
        if token_data:
            print(f"✅ Token decoded: {token_data.username} ({token_data.token_type.value})")

        # Create API key
        api_key, key_info = TokenManager.create_api_key(user, "Test API Key")
        print(f"✅ API key created: {api_key[:20]}...")

        # Verify API key
        verified_user = TokenManager.verify_api_key(api_key)
        if verified_user:
            print(f"✅ API key verified: {verified_user.username}")
    else:
        print("❌ Authentication failed")
