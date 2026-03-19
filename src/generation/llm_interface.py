"""
LLM interface for multiple providers including OpenAI, Anthropic, Azure, Google Gemini, Cohere, and local models.
ENHANCED: Automatic retries with exponential backoff, jitter, and intelligent error handling.
"""

import os
import json
import logging
import asyncio
import time
import random
from typing import List, Dict, Any, Optional, Union, AsyncIterator, Iterator, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
import math

import tiktoken
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    wait_random_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryError,
    retry_if_exception,
    Retrying,
    stop_after_delay,
    wait_fixed
)

# Try importing providers
try:
    from openai import OpenAI, AsyncOpenAI
    from openai.types.chat import ChatCompletion, ChatCompletionChunk
    from openai import RateLimitError, APIError, APIConnectionError, APITimeoutError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic, AsyncAnthropic
    from anthropic.types import Message, TextBlock
    from anthropic import APIError as AnthropicAPIError, RateLimitError as AnthropicRateLimitError
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerateContentResponse
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

try:
    import groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    GEMINI = "gemini"
    COHERE = "cohere"
    GROQ = "groq"
    LOCAL = "local"
    OLLAMA = "ollama"


@dataclass
class Message:
    """Chat message representation."""
    role: str  # "system", "user", "assistant"
    content: str
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        result = {"role": self.role, "content": self.content}
        if self.name:
            result["name"] = self.name
        return result


@dataclass
class LLMResponse:
    """LLM response with metadata."""
    content: str
    model: str
    provider: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    finish_reason: str = ""
    latency_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    raw_response: Any = None
    retry_count: int = 0
    success: bool = True

    @property
    def cost_display(self) -> str:
        """Format cost for display."""
        return f"${self.cost:.6f}"


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
        exponential_base: float = 2.0,
        retry_on_status_codes: List[int] = None,
        retry_on_exceptions: List[type] = None,
        retryable_error_messages: List[str] = None,
        stop_after_delay: Optional[float] = None
    ):
        """
        Initialize retry configuration.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            jitter: Whether to add jitter to delays
            exponential_base: Base for exponential backoff
            retry_on_status_codes: HTTP status codes to retry on
            retry_on_exceptions: Exception types to retry on
            retryable_error_messages: Error message patterns to retry on
            stop_after_delay: Maximum total retry time in seconds
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.exponential_base = exponential_base
        self.retry_on_status_codes = retry_on_status_codes or [429, 500, 502, 503, 504]
        self.retry_on_exceptions = retry_on_exceptions or [
            RateLimitError, APIError, APIConnectionError, APITimeoutError,
            ConnectionError, TimeoutError
        ]
        self.retryable_error_messages = retryable_error_messages or [
            "rate limit", "timeout", "connection", "unavailable", "overloaded",
            "server error", "internal error", "service unavailable",
            "too many requests", "throttling", "quota exceeded"
        ]
        self.stop_after_delay = stop_after_delay


class RetryManager:
    """
    Manages retry logic with exponential backoff and intelligent error handling.
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        """Initialize retry manager."""
        self.config = config or RetryConfig()
        self._retry_count = 0
        self._start_time = None
        self._last_error = None

        # Statistics
        self.stats = {
            "total_retries": 0,
            "successful_retries": 0,
            "failed_retries": 0,
            "errors_by_type": {},
            "total_retry_time_ms": 0
        }

    def should_retry(self, error: Exception) -> bool:
        """
        Determine if an error should be retried.

        Args:
            error: The exception that occurred

        Returns:
            True if should retry, False otherwise
        """
        # Check if max retries exceeded
        if self._retry_count >= self.config.max_retries:
            return False

        # Check stop after delay
        if self.config.stop_after_delay and self._start_time:
            elapsed = time.time() - self._start_time
            if elapsed >= self.config.stop_after_delay:
                return False

        # Check exception type
        error_type = type(error)
        if any(isinstance(error, exc_type) for exc_type in self.config.retry_on_exceptions):
            return True

        # Check error message
        error_str = str(error).lower()
        for pattern in self.config.retryable_error_messages:
            if pattern in error_str:
                return True

        # Check status codes (for HTTP errors)
        if hasattr(error, 'status_code'):
            if error.status_code in self.config.retry_on_status_codes:
                return True

        return False

    def get_delay(self) -> float:
        """
        Calculate the delay before the next retry using exponential backoff with jitter.

        Returns:
            Delay in seconds
        """
        # Exponential backoff
        delay = self.config.base_delay * (self.config.exponential_base ** self._retry_count)

        # Cap at max delay
        delay = min(delay, self.config.max_delay)

        # Add jitter
        if self.config.jitter:
            jitter = random.uniform(0.8, 1.2)
            delay = delay * jitter

        return delay

    def before_retry(self, error: Exception):
        """
        Called before each retry attempt.

        Args:
            error: The exception that triggered the retry
        """
        self._retry_count += 1
        self.stats["total_retries"] += 1

        error_type = type(error).__name__
        self.stats["errors_by_type"][error_type] = self.stats["errors_by_type"].get(error_type, 0) + 1

        delay = self.get_delay()

        logger.warning(
            f"Retry {self._retry_count}/{self.config.max_retries} after {delay:.2f}s "
            f"due to {error_type}: {str(error)[:100]}"
        )

        # Update stats
        if self._start_time is None:
            self._start_time = time.time()

        self._last_error = error

    def after_retry(self, success: bool):
        """
        Called after a retry attempt.

        Args:
            success: Whether the retry was successful
        """
        if success:
            self.stats["successful_retries"] += 1
            if self._start_time:
                self.stats["total_retry_time_ms"] += (time.time() - self._start_time) * 1000
        else:
            self.stats["failed_retries"] += 1

    def reset(self):
        """Reset retry state."""
        self._retry_count = 0
        self._start_time = None
        self._last_error = None

    def get_stats(self) -> Dict[str, Any]:
        """Get retry statistics."""
        return {
            **self.stats,
            "current_retry_count": self._retry_count,
            "max_retries": self.config.max_retries,
            "base_delay": self.config.base_delay,
            "max_delay": self.config.max_delay,
            "exponential_base": self.config.exponential_base,
            "jitter_enabled": self.config.jitter
        }


class RetryableLLMInterface:
    """
    Wrapper for LLM interface with automatic retry logic.
    """

    def __init__(
        self,
        llm_interface: 'LLMInterface',
        retry_config: Optional[RetryConfig] = None
    ):
        """
        Initialize retryable LLM interface.

        Args:
            llm_interface: LLM interface instance
            retry_config: Retry configuration
        """
        self.llm_interface = llm_interface
        self.retry_manager = RetryManager(retry_config)

        # Forward attributes
        self.provider = llm_interface.provider
        self.model = llm_interface.model

        logger.info(f"RetryableLLMInterface initialized for {llm_interface.provider.value}")

    def _execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with retry logic.

        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If all retries fail
        """
        self.retry_manager.reset()

        while True:
            try:
                # Execute function
                result = func(*args, **kwargs)

                # Update stats
                self.retry_manager.after_retry(True)

                # Add retry count to result if it's an LLMResponse
                if isinstance(result, LLMResponse):
                    result.retry_count = self.retry_manager._retry_count

                return result

            except Exception as e:
                # Check if should retry
                if not self.retry_manager.should_retry(e):
                    self.retry_manager.after_retry(False)
                    raise

                # Log and wait before retry
                self.retry_manager.before_retry(e)

                # Wait before retry
                delay = self.retry_manager.get_delay()
                time.sleep(delay)

                # Reset any state if needed
                # Some providers might need resetting
                if hasattr(self.llm_interface, 'reset'):
                    self.llm_interface.reset()

    async def _execute_with_retry_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute an async function with retry logic.

        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If all retries fail
        """
        self.retry_manager.reset()

        while True:
            try:
                # Execute function
                result = await func(*args, **kwargs)

                # Update stats
                self.retry_manager.after_retry(True)

                # Add retry count to result if it's an LLMResponse
                if isinstance(result, LLMResponse):
                    result.retry_count = self.retry_manager._retry_count

                return result

            except Exception as e:
                # Check if should retry
                if not self.retry_manager.should_retry(e):
                    self.retry_manager.after_retry(False)
                    raise

                # Log and wait before retry
                self.retry_manager.before_retry(e)

                # Wait before retry
                delay = self.retry_manager.get_delay()
                await asyncio.sleep(delay)

                # Reset any state if needed
                if hasattr(self.llm_interface, 'reset'):
                    await self.llm_interface.reset()

    def generate(
        self,
        messages: List[Union[Message, Dict[str, str]]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, Iterator[LLMResponse]]:
        """
        Generate response with automatic retries.

        Args:
            messages: List of messages
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling
            stream: Whether to stream
            **kwargs: Additional arguments

        Returns:
            LLMResponse or iterator of LLMResponse
        """
        if stream:
            # For streaming, we use a different approach
            # We need to handle retries for the stream creation
            stream_func = self.llm_interface.generate

            # Try to create the stream with retries
            stream_result = self._execute_with_retry(
                stream_func,
                messages,
                system_prompt,
                temperature,
                max_tokens,
                top_p,
                False,  # Don't stream internally
                **kwargs
            )

            # If we get a non-streaming response, yield it as a single chunk
            if isinstance(stream_result, LLMResponse):
                yield stream_result
            else:
                # It should be a stream, but we'll handle it generically
                yield from stream_result
        else:
            return self._execute_with_retry(
                self.llm_interface.generate,
                messages,
                system_prompt,
                temperature,
                max_tokens,
                top_p,
                False,
                **kwargs
            )

    async def generate_async(
        self,
        messages: List[Union[Message, Dict[str, str]]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, AsyncIterator[LLMResponse]]:
        """
        Generate response asynchronously with automatic retries.

        Args:
            messages: List of messages
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling
            stream: Whether to stream
            **kwargs: Additional arguments

        Returns:
            LLMResponse or async iterator of LLMResponse
        """
        if stream:
            # For streaming, create the stream with retries
            stream_func = self.llm_interface.generate_async

            # Try to create the stream with retries
            stream_result = await self._execute_with_retry_async(
                stream_func,
                messages,
                system_prompt,
                temperature,
                max_tokens,
                top_p,
                False,  # Don't stream internally
                **kwargs
            )

            # If we get a non-streaming response, yield it
            if isinstance(stream_result, LLMResponse):
                yield stream_result
            else:
                async for chunk in stream_result:
                    yield chunk
        else:
            return await self._execute_with_retry_async(
                self.llm_interface.generate_async,
                messages,
                system_prompt,
                temperature,
                max_tokens,
                top_p,
                False,
                **kwargs
            )

    def generate_simple(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Simple generation with retries.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Generated response text
        """
        messages = [Message(role="user", content=prompt)]
        response = self.generate(messages, system_prompt=system_prompt)
        return response.content

    async def generate_simple_async(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Async simple generation with retries.
        """
        messages = [Message(role="user", content=prompt)]
        response = await self.generate_async(messages, system_prompt=system_prompt)
        return response.content

    def get_retry_stats(self) -> Dict[str, Any]:
        """Get retry statistics."""
        return self.retry_manager.get_stats()

    def reset_retry_stats(self):
        """Reset retry statistics."""
        self.retry_manager.reset()
        self.retry_manager.stats = {
            "total_retries": 0,
            "successful_retries": 0,
            "failed_retries": 0,
            "errors_by_type": {},
            "total_retry_time_ms": 0
        }


# ============================================================
# Updated LLMInterface with Retry Support
# ============================================================

class LLMInterface:
    """
    Unified interface for multiple LLM providers with retry support.
    """

    # Cost per 1K tokens (USD)
    COSTS = {
        # OpenAI
        "gpt-4": {"prompt": 0.03, "completion": 0.06},
        "gpt-4-32k": {"prompt": 0.06, "completion": 0.12},
        "gpt-4-turbo-preview": {"prompt": 0.01, "completion": 0.03},
        "gpt-4o": {"prompt": 0.005, "completion": 0.015},
        "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
        "gpt-4o-2024-08-06": {"prompt": 0.0025, "completion": 0.01},
        "gpt-4o-mini-2024-07-18": {"prompt": 0.00015, "completion": 0.0006},
        "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
        "gpt-3.5-turbo-16k": {"prompt": 0.001, "completion": 0.002},
        "gpt-3.5-turbo-0125": {"prompt": 0.0005, "completion": 0.0015},
        "gpt-3.5-turbo-1106": {"prompt": 0.001, "completion": 0.002},

        # Anthropic
        "claude-3-opus-20240229": {"prompt": 0.015, "completion": 0.075},
        "claude-3-sonnet-20240229": {"prompt": 0.003, "completion": 0.015},
        "claude-3-haiku-20240307": {"prompt": 0.00025, "completion": 0.00125},
        "claude-3-5-sonnet-20241022": {"prompt": 0.003, "completion": 0.015},

        # Google Gemini
        "gemini-1.5-pro": {"prompt": 0.0025, "completion": 0.0075},
        "gemini-1.5-flash": {"prompt": 0.00035, "completion": 0.00105},
        "gemini-1.0-pro": {"prompt": 0.0005, "completion": 0.0015},

        # Cohere
        "command-r": {"prompt": 0.0005, "completion": 0.0015},
        "command-r-plus": {"prompt": 0.003, "completion": 0.015},
        "command": {"prompt": 0.0005, "completion": 0.0015},

        # Groq
        "llama-3.1-70b-versatile": {"prompt": 0.00059, "completion": 0.00079},
        "llama-3.1-8b-instant": {"prompt": 0.00005, "completion": 0.00008},
        "mixtral-8x7b-32768": {"prompt": 0.00024, "completion": 0.00024},
    }

    # Default retry configuration
    DEFAULT_RETRY_CONFIG = RetryConfig(
        max_retries=5,
        base_delay=1.0,
        max_delay=60.0,
        jitter=True,
        exponential_base=2.0,
        stop_after_delay=120.0  # Stop retrying after 2 minutes
    )

    def __init__(
        self,
        provider: Union[str, LLMProvider] = LLMProvider.OPENAI,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        timeout: int = 60,
        max_retries: int = 5,
        organization: Optional[str] = None,
        retry_config: Optional[RetryConfig] = None,
        enable_retries: bool = True,
        **kwargs
    ):
        """
        Initialize LLM interface.

        Args:
            provider: LLM provider
            model: Model name
            api_key: API key
            api_base: Custom API base URL
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            organization: Organization ID
            retry_config: Custom retry configuration
            enable_retries: Whether to enable retries
            **kwargs: Additional arguments
        """
        self.provider = LLMProvider(provider) if isinstance(provider, str) else provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.timeout = timeout
        self.max_retries = max_retries
        self.enable_retries = enable_retries

        # Initialize clients
        self.client = None
        self.async_client = None
        self._init_client(api_key, api_base, organization, **kwargs)

        # Tokenizer for cost estimation
        self.tokenizer = self._get_tokenizer()

        # Retry configuration
        self.retry_config = retry_config or self.DEFAULT_RETRY_CONFIG
        if max_retries != 5:
            self.retry_config.max_retries = max_retries

        # Create retryable wrapper if enabled
        self.retryable = None
        if self.enable_retries:
            self.retryable = RetryableLLMInterface(self, self.retry_config)

        logger.info(f"Initialized LLM interface: provider={self.provider.value}, model={model}, retries={enable_retries}")

    def _init_client(self, api_key, api_base, organization, **kwargs):
        """Initialize the appropriate client based on provider."""
        if self.provider == LLMProvider.OPENAI:
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package not installed. Install with: pip install openai")

            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not provided")

            self.client = OpenAI(
                api_key=api_key,
                base_url=api_base,
                organization=organization,
                timeout=self.timeout
            )
            self.async_client = AsyncOpenAI(
                api_key=api_key,
                base_url=api_base,
                organization=organization,
                timeout=self.timeout
            )

        elif self.provider == LLMProvider.ANTHROPIC:
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic package not installed. Install with: pip install anthropic")

            api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key not provided")

            self.client = Anthropic(api_key=api_key)
            self.async_client = AsyncAnthropic(api_key=api_key)

        elif self.provider == LLMProvider.AZURE:
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI package not installed")

            api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
            api_base = api_base or os.getenv("AZURE_OPENAI_ENDPOINT")
            api_version = kwargs.get("api_version", os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"))

            if not api_key or not api_base:
                raise ValueError("Azure OpenAI credentials not provided")

            self.client = OpenAI(
                api_key=api_key,
                base_url=f"{api_base}/openai/deployments/{self.model}",
                default_headers={"api-key": api_key},
                timeout=self.timeout
            )
            self.async_client = AsyncOpenAI(
                api_key=api_key,
                base_url=f"{api_base}/openai/deployments/{self.model}",
                default_headers={"api-key": api_key},
                timeout=self.timeout
            )

        elif self.provider == LLMProvider.GEMINI:
            if not GEMINI_AVAILABLE:
                raise ImportError("Google Generative AI package not installed. Install with: pip install google-generativeai")

            api_key = api_key or os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Google Gemini API key not provided")

            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.model)
            self.async_client = None

        elif self.provider == LLMProvider.COHERE:
            if not COHERE_AVAILABLE:
                raise ImportError("Cohere package not installed. Install with: pip install cohere")

            api_key = api_key or os.getenv("COHERE_API_KEY")
            if not api_key:
                raise ValueError("Cohere API key not provided")

            self.client = cohere.Client(api_key=api_key)
            self.async_client = cohere.AsyncClient(api_key=api_key)

        elif self.provider == LLMProvider.GROQ:
            if not GROQ_AVAILABLE:
                raise ImportError("Groq package not installed. Install with: pip install groq")

            api_key = api_key or os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("Groq API key not provided")

            self.client = groq.Groq(api_key=api_key)
            self.async_client = groq.AsyncGroq(api_key=api_key)

        elif self.provider == LLMProvider.OLLAMA:
            api_base = api_base or os.getenv("OLLAMA_URL", "http://localhost:11434/v1")

            self.client = OpenAI(
                api_key="ollama",
                base_url=api_base,
                timeout=self.timeout
            )
            self.async_client = AsyncOpenAI(
                api_key="ollama",
                base_url=api_base,
                timeout=self.timeout
            )

        elif self.provider == LLMProvider.LOCAL:
            api_base = api_base or os.getenv("LOCAL_LLM_URL", "http://localhost:8000/v1")

            self.client = OpenAI(
                api_key="not-needed",
                base_url=api_base,
                timeout=self.timeout
            )
            self.async_client = AsyncOpenAI(
                api_key="not-needed",
                base_url=api_base,
                timeout=self.timeout
            )

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _get_tokenizer(self):
        """Get tokenizer for the model."""
        try:
            if self.provider == LLMProvider.OPENAI:
                return tiktoken.encoding_for_model(self.model)
            elif self.provider in [LLMProvider.ANTHROPIC, LLMProvider.COHERE, LLMProvider.GROQ]:
                return tiktoken.get_encoding("cl100k_base")
            else:
                return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            return len(self.tokenizer.encode(text))
        except Exception:
            return len(text) // 4

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost based on tokens."""
        costs = self.COSTS.get(self.model, {"prompt": 0.001, "completion": 0.002})
        prompt_cost = (prompt_tokens / 1000) * costs["prompt"]
        completion_cost = (completion_tokens / 1000) * costs["completion"]
        return prompt_cost + completion_cost

    def _prepare_messages(
        self,
        messages: List[Union[Message, Dict[str, str]]],
        system_prompt: Optional[str]
    ) -> List[Dict[str, str]]:
        """Prepare messages for API call."""
        prepared = []

        if system_prompt:
            prepared.append({"role": "system", "content": system_prompt})

        for msg in messages:
            if isinstance(msg, Message):
                prepared.append(msg.to_dict())
            elif isinstance(msg, dict):
                prepared.append(msg)
            else:
                raise TypeError(f"Unsupported message type: {type(msg)}")

        return prepared

    def _prepare_anthropic_messages(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str]
    ) -> Tuple[Optional[str], List[Dict[str, str]]]:
        """Prepare messages for Anthropic API."""
        system = system_prompt
        conversation = []

        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                conversation.append(msg)

        return system, conversation

    def generate(
        self,
        messages: List[Union[Message, Dict[str, str]]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, Iterator[LLMResponse]]:
        """
        Generate response from LLM.
        """
        if self.retryable:
            return self.retryable.generate(
                messages, system_prompt, temperature, max_tokens, top_p, stream, **kwargs
            )

        # Original implementation (without retries)
        prepared_messages = self._prepare_messages(messages, system_prompt)

        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        top = top_p if top_p is not None else self.top_p

        if self.provider == LLMProvider.OPENAI:
            if stream:
                return self._stream_openai(prepared_messages, temp, max_tok, top, **kwargs)
            else:
                return self._generate_openai(prepared_messages, temp, max_tok, top, **kwargs)

        elif self.provider == LLMProvider.ANTHROPIC:
            system, conversation = self._prepare_anthropic_messages(prepared_messages, None)
            if stream:
                return self._stream_anthropic(conversation, system, temp, max_tok, top, **kwargs)
            else:
                return self._generate_anthropic(conversation, system, temp, max_tok, top, **kwargs)

        elif self.provider == LLMProvider.GEMINI:
            return self._generate_gemini(prepared_messages, temp, max_tok, top, **kwargs)

        elif self.provider == LLMProvider.COHERE:
            if stream:
                return self._stream_cohere(prepared_messages, temp, max_tok, top, **kwargs)
            else:
                return self._generate_cohere(prepared_messages, temp, max_tok, top, **kwargs)

        elif self.provider in [LLMProvider.GROQ, LLMProvider.AZURE, LLMProvider.OLLAMA, LLMProvider.LOCAL]:
            if stream:
                return self._stream_openai_compatible(prepared_messages, temp, max_tok, top, **kwargs)
            else:
                return self._generate_openai_compatible(prepared_messages, temp, max_tok, top, **kwargs)

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def generate_async(
        self,
        messages: List[Union[Message, Dict[str, str]]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, AsyncIterator[LLMResponse]]:
        """
        Async generate response from LLM.
        """
        if self.retryable:
            return await self.retryable.generate_async(
                messages, system_prompt, temperature, max_tokens, top_p, stream, **kwargs
            )

        prepared_messages = self._prepare_messages(messages, system_prompt)

        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        top = top_p if top_p is not None else self.top_p

        if self.provider == LLMProvider.OPENAI:
            if stream:
                return self._stream_openai_async(prepared_messages, temp, max_tok, top, **kwargs)
            else:
                return await self._generate_openai_async(prepared_messages, temp, max_tok, top, **kwargs)

        elif self.provider == LLMProvider.ANTHROPIC:
            system, conversation = self._prepare_anthropic_messages(prepared_messages, None)
            if stream:
                return self._stream_anthropic_async(conversation, system, temp, max_tok, top, **kwargs)
            else:
                return await self._generate_anthropic_async(conversation, system, temp, max_tok, top, **kwargs)

        elif self.provider == LLMProvider.GEMINI:
            return await self._generate_gemini_async(prepared_messages, temp, max_tok, top, **kwargs)

        elif self.provider == LLMProvider.COHERE:
            if stream:
                return self._stream_cohere_async(prepared_messages, temp, max_tok, top, **kwargs)
            else:
                return await self._generate_cohere_async(prepared_messages, temp, max_tok, top, **kwargs)

        elif self.provider in [LLMProvider.GROQ, LLMProvider.AZURE, LLMProvider.OLLAMA, LLMProvider.LOCAL]:
            if stream:
                return self._stream_openai_compatible_async(prepared_messages, temp, max_tok, top, **kwargs)
            else:
                return await self._generate_openai_compatible_async(prepared_messages, temp, max_tok, top, **kwargs)

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate_simple(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Simple generation."""
        if self.retryable:
            return self.retryable.generate_simple(prompt, system_prompt)

        messages = [Message(role="user", content=prompt)]
        response = self.generate(messages, system_prompt=system_prompt)
        return response.content

    async def generate_simple_async(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Async simple generation."""
        if self.retryable:
            return await self.retryable.generate_simple_async(prompt, system_prompt)

        messages = [Message(role="user", content=prompt)]
        response = await self.generate_async(messages, system_prompt=system_prompt)
        return response.content

    # ============================================================
    # Provider-specific methods (existing implementation)
    # ============================================================

    def _generate_openai(self, messages, temperature, max_tokens, top_p, **kwargs) -> LLMResponse:
        """Generate using OpenAI."""
        start_time = time.time()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            **kwargs
        )

        latency_ms = (time.time() - start_time) * 1000

        choice = response.choices[0]
        content = choice.message.content

        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        cost = self.estimate_cost(prompt_tokens, completion_tokens)

        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider.value,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=response.usage.total_tokens,
            cost=cost,
            finish_reason=choice.finish_reason,
            latency_ms=latency_ms,
            raw_response=response
        )

    # ... (rest of provider-specific methods remain the same)
    # Note: For brevity, the full implementation of all provider methods
    # would be included here. They are omitted for space but should be
    # included in the actual file.

    def get_retry_stats(self) -> Dict[str, Any]:
        """Get retry statistics."""
        if self.retryable:
            return self.retryable.get_retry_stats()
        return {}

    def reset_retry_stats(self):
        """Reset retry statistics."""
        if self.retryable:
            self.retryable.reset_retry_stats()


# ============================================================
# Factory Function with Retry Support
# ============================================================

def create_llm_interface(
    provider: str = "openai",
    model: Optional[str] = None,
    temperature: float = 0.7,
    enable_retries: bool = True,
    max_retries: int = 5,
    **kwargs
) -> LLMInterface:
    """
    Create LLM interface with retry support.

    Args:
        provider: Provider name
        model: Model name
        temperature: Sampling temperature
        enable_retries: Whether to enable retries
        max_retries: Maximum retry attempts
        **kwargs: Additional arguments

    Returns:
        LLMInterface instance
    """
    default_models = {
        "openai": "gpt-4",
        "anthropic": "claude-3-haiku-20240307",
        "azure": "gpt-4",
        "gemini": "gemini-1.5-pro",
        "cohere": "command-r",
        "groq": "llama-3.1-70b-versatile",
        "local": "local-model",
        "ollama": "llama2"
    }

    if model is None:
        model = default_models.get(provider, "gpt-4")

    return LLMInterface(
        provider=provider,
        model=model,
        temperature=temperature,
        enable_retries=enable_retries,
        max_retries=max_retries,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage with retries
    import sys
    logging.basicConfig(level=logging.INFO)

    if os.getenv("OPENAI_API_KEY"):
        print("Testing LLM Interface with Retries...")
        print("=" * 60)

        # Create interface with retries
        llm = create_llm_interface(
            "openai",
            "gpt-4",
            enable_retries=True,
            max_retries=3
        )

        try:
            # Test generation
            response = llm.generate_simple("What is the capital of France?")
            print(f"✅ Response: {response[:100]}...")
            print(f"Retry count: {response.retry_count if hasattr(response, 'retry_count') else 'N/A'}")

            # Get retry stats
            stats = llm.get_retry_stats()
            print(f"\n📊 Retry Stats:")
            for key, value in stats.items():
                if not isinstance(value, dict):
                    print(f"  {key}: {value}")

        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("⚠️  OPENAI_API_KEY not set. Skipping test.")
