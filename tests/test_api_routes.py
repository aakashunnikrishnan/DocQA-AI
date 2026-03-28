"""
Tests for API routes.
"""

import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from api.app import app
from api.routes import router


client = TestClient(app)


class TestQueryEndpoints:
    """Tests for query endpoints."""

    def test_query_endpoint_validation(self):
        """Test query endpoint validation."""
        response = client.post(
            "/api/v1/query",
            json={"question": ""}
        )
        assert response.status_code == 422

    def test_query_endpoint_empty_question(self):
        """Test query with empty question."""
        response = client.post(
            "/api/v1/query",
            json={"question": "   "}
        )
        assert response.status_code == 422

    @patch('api.routes.get_app_state')
    def test_query_endpoint_no_documents(self, mock_get_state):
        """Test query when no documents are ingested."""
        mock_state = {
            "retriever": Mock(),
            "llm_interface": Mock(),
            "vector_store": Mock(),
            "config": Mock()
        }
        mock_state["vector_store"].get_size.return_value = 0
        mock_get_state.return_value = mock_state

        response = client.post(
            "/api/v1/query",
            json={"question": "What is AI?"}
        )
        assert response.status_code == 400
        assert "No documents ingested" in response.json()["detail"]


class TestDocumentEndpoints:
    """Tests for document endpoints."""

    def test_list_documents_empty(self):
        """Test listing documents when none exist."""
        response = client.get("/api/v1/documents")
        assert response.status_code == 200
        assert response.json()["total"] == 0

    def test_delete_nonexistent_document(self):
        """Test deleting non-existent document."""
        response = client.delete("/api/v1/documents/nonexistent")
        assert response.status_code == 404


class TestHealthEndpoint:
    """Tests for health endpoint."""

    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert "status" in response.json()

    def test_health_check_version(self):
        """Test health check returns version."""
        response = client.get("/health")
        assert "version" in response.json()
