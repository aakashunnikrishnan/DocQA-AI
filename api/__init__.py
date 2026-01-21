"""
API package for DocQA AI.
Exposes REST endpoints and WebSocket connections.
"""

from api.app import app, get_app_state
from api.routes import router
from api.schemas import (
    QueryRequest, QueryResponse, DocumentIngestRequest,
    DocumentIngestResponse, DocumentListResponse, DocumentInfo,
    HealthResponse, MetricsResponse, ErrorResponse
)

__all__ = [
    "app",
    "get_app_state",
    "router",
    "QueryRequest",
    "QueryResponse",
    "DocumentIngestRequest",
    "DocumentIngestResponse",
    "DocumentListResponse",
    "DocumentInfo",
    "HealthResponse",
    "MetricsResponse",
    "ErrorResponse"
]
