"""
Tests for LLM interface module.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock

from src.generation.llm_interface import (
    LLMInterface, LLMProvider, Message, LLMResponse
)


class TestLLMInterface:
    """Tests for LLMInterface."""

    def test_init_openai(self, monkeypatch):
        """Test initialization with OpenAI provider."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Mock OpenAI import
        with patch('src.generation.llm_interface.OPENAI_AVAILABLE', True):
            with patch('src.generation.llm_interface.OpenAI') as mock_openai:
                with patch('src.generation.llm_interface.AsyncOpenAI') as mock_async:
                    llm = LLMInterface(
                        provider=LLMProvider.OPENAI,
                        model="gpt-4",
                        temperature=0.7
                    )

                    assert llm.provider == LLMProvider.OPENAI
                    assert llm.model == "gpt-4"
                    assert llm.temperature == 0.7

    def test_count_tokens(self, monkeypatch):
        """Test token counting."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        with patch('src.generation.llm_interface.OPENAI_AVAILABLE', True):
            with patch('src.generation.llm_interface.OpenAI'):
                with patch('src.generation.llm_interface.AsyncOpenAI'):
                    llm = LLMInterface(provider=LLMProvider.OPENAI, model="gpt-4")

                    # Simple token count test
                    text = "This is a test sentence."
                    tokens = llm.count_tokens(text)
                    assert tokens > 0

    def test_estimate_cost(self, monkeypatch):
        """Test cost estimation."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        with patch('src.generation.llm_interface.OPENAI_AVAILABLE', True):
            with patch('src.generation.llm_interface.OpenAI'):
                with patch('src.generation.llm_interface.AsyncOpenAI'):
                    llm = LLMInterface(provider=LLMProvider.OPENAI, model="gpt-4")

                    cost = llm.estimate_cost(100, 50)
                    assert cost > 0

    def test_generate_simple_with_mock(self, mock_llm_interface):
        """Test simple generation with mock."""
        response = mock_llm_interface.generate_simple("What is AI?")
        assert response is not None
        assert len(response) > 0


class TestMessage:
    """Tests for Message class."""

    def test_message_creation(self):
        """Test message creation."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.name is None

    def test_message_with_name(self):
        """Test message with name."""
        msg = Message(role="user", content="Hello", name="test_user")
        assert msg.name == "test_user"

    def test_message_to_dict(self):
        """Test conversion to dictionary."""
        msg = Message(role="user", content="Hello", name="test_user")
        msg_dict = msg.to_dict()

        assert msg_dict["role"] == "user"
        assert msg_dict["content"] == "Hello"
        assert msg_dict["name"] == "test_user"


class TestLLMResponse:
    """Tests for LLMResponse class."""

    def test_response_creation(self):
        """Test response creation."""
        response = LLMResponse(
            content="Test response",
            model="gpt-4",
            provider="openai"
        )

        assert response.content == "Test response"
        assert response.model == "gpt-4"
        assert response.provider == "openai"

    def test_cost_display(self):
        """Test cost display formatting."""
        response = LLMResponse(
            content="Test",
            model="gpt-4",
            provider="openai",
            cost=0.005
        )

        assert response.cost_display == "$0.005000"
