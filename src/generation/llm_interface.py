"""
LLM interface for multiple providers including OpenAI, Anthropic, Azure, Google Gemini, Cohere, and local models.
Provides unified interface with streaming, retries, cost tracking, and provider-specific optimizations.
"""

import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, AsyncIterator, Iterator, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
import time

import tiktoken
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryError
)

# Try importing providers
try:
    from openai import OpenAI, AsyncOpenAI
    from openai.types.chat import ChatCompletion, ChatCompletionChunk
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic, AsyncAnthropic
    from anthropic.types import Message, TextBlock
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

    @property
    def cost_display(self) -> str:
        """Format cost for display."""
        return f"${self.cost:.6f}"


class LLMInterface:
    """
    Unified interface for multiple LLM providers.
    Supports: OpenAI, Anthropic, Azure, Google Gemini, Cohere, Groq, Local, Ollama
    """

    # Cost per 1K tokens (USD) - Updated with latest pricing
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
        max_retries: int = 3,
        organization: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize LLM interface.

        Args:
            provider: LLM provider ('openai', 'anthropic', 'azure', 'gemini', 'cohere', 'groq', 'local', 'ollama')
            model: Model name
            api_key: API key (defaults to environment variable)
            api_base: Custom API base URL
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty (-2 to 2)
            presence_penalty: Presence penalty (-2 to 2)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            organization: Organization ID (OpenAI)
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

        # Initialize clients based on provider
        self.client = None
        self.async_client = None
        self._init_client(api_key, api_base, organization, **kwargs)

        # Tokenizer for cost estimation
        self.tokenizer = self._get_tokenizer()

        logger.info(f"Initialized LLM interface: provider={self.provider.value}, model={model}")

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
            self.async_client = None  # Gemini doesn't have native async

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
            # Ollama uses OpenAI-compatible API
            api_base = api_base or os.getenv("OLLAMA_URL", "http://localhost:11434/v1")

            self.client = OpenAI(
                api_key="ollama",  # Ollama doesn't need API key
                base_url=api_base,
                timeout=self.timeout
            )
            self.async_client = AsyncOpenAI(
                api_key="ollama",
                base_url=api_base,
                timeout=self.timeout
            )
            logger.info(f"Using Ollama at {api_base}")

        elif self.provider == LLMProvider.LOCAL:
            # Local model support (vLLM, TGI, etc.)
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
            logger.info(f"Using local LLM at {api_base}")

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
            return len(text) // 4  # Rough estimate

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost based on tokens."""
        # Handle provider-specific model names
        model_key = self.model

        # Handle Azure deployments
        if self.provider == LLMProvider.AZURE:
            # Try to map Azure deployment to base model
            for key in self.COSTS:
                if key in model_key or model_key in key:
                    model_key = key
                    break

        costs = self.COSTS.get(model_key, {"prompt": 0.001, "completion": 0.002})
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

        # Add system prompt if provided
        if system_prompt:
            prepared.append({"role": "system", "content": system_prompt})

        # Convert messages to dict format
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
        """Prepare messages for Anthropic API (system separate)."""
        system = system_prompt
        conversation = []

        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                conversation.append(msg)

        return system, conversation

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
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

        Args:
            messages: List of messages (Message objects or dicts)
            system_prompt: Optional system prompt (prepended to messages)
            temperature: Override default temperature
            max_tokens: Override default max tokens
            top_p: Override default top_p
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse or iterator of LLMResponse for streaming
        """
        # Prepare messages
        prepared_messages = self._prepare_messages(messages, system_prompt)

        # Use provided parameters or defaults
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        top = top_p if top_p is not None else self.top_p

        # Route to appropriate provider
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
            # OpenAI-compatible endpoints
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
        Asynchronously generate response from LLM.
        """
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

    # ============================================================
    # OpenAI Methods
    # ============================================================

    def _generate_openai(self, messages, temperature, max_tokens, top_p, **kwargs) -> LLMResponse:
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

    async def _generate_openai_async(self, messages, temperature, max_tokens, top_p, **kwargs) -> LLMResponse:
        start_time = time.time()

        response = await self.async_client.chat.completions.create(
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

    def _stream_openai(self, messages, temperature, max_tokens, top_p, **kwargs) -> Iterator[LLMResponse]:
        start_time = time.time()

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stream=True,
            **kwargs
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield LLMResponse(
                    content=chunk.choices[0].delta.content,
                    model=self.model,
                    provider=self.provider.value,
                    latency_ms=(time.time() - start_time) * 1000,
                    raw_response=chunk
                )

    async def _stream_openai_async(self, messages, temperature, max_tokens, top_p, **kwargs) -> AsyncIterator[LLMResponse]:
        start_time = time.time()

        stream = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stream=True,
            **kwargs
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield LLMResponse(
                    content=chunk.choices[0].delta.content,
                    model=self.model,
                    provider=self.provider.value,
                    latency_ms=(time.time() - start_time) * 1000,
                    raw_response=chunk
                )

    # ============================================================
    # Anthropic Methods
    # ============================================================

    def _generate_anthropic(self, messages, system, temperature, max_tokens, top_p, **kwargs) -> LLMResponse:
        start_time = time.time()

        response = self.client.messages.create(
            model=self.model,
            system=system,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )

        latency_ms = (time.time() - start_time) * 1000

        content = response.content[0].text if response.content else ""

        # Estimate tokens
        prompt_tokens = self.count_tokens(str(messages) + (system or ""))
        completion_tokens = self.count_tokens(content)
        cost = self.estimate_cost(prompt_tokens, completion_tokens)

        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider.value,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=cost,
            finish_reason=response.stop_reason,
            latency_ms=latency_ms,
            raw_response=response
        )

    async def _generate_anthropic_async(self, messages, system, temperature, max_tokens, top_p, **kwargs) -> LLMResponse:
        start_time = time.time()

        response = await self.async_client.messages.create(
            model=self.model,
            system=system,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )

        latency_ms = (time.time() - start_time) * 1000

        content = response.content[0].text if response.content else ""

        prompt_tokens = self.count_tokens(str(messages) + (system or ""))
        completion_tokens = self.count_tokens(content)
        cost = self.estimate_cost(prompt_tokens, completion_tokens)

        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider.value,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=cost,
            finish_reason=response.stop_reason,
            latency_ms=latency_ms,
            raw_response=response
        )

    def _stream_anthropic(self, messages, system, temperature, max_tokens, top_p, **kwargs) -> Iterator[LLMResponse]:
        start_time = time.time()

        with self.client.messages.stream(
            model=self.model,
            system=system,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        ) as stream:
            for text in stream.text_stream:
                yield LLMResponse(
                    content=text,
                    model=self.model,
                    provider=self.provider.value,
                    latency_ms=(time.time() - start_time) * 1000
                )

    async def _stream_anthropic_async(self, messages, system, temperature, max_tokens, top_p, **kwargs) -> AsyncIterator[LLMResponse]:
        start_time = time.time()

        async with self.async_client.messages.stream(
            model=self.model,
            system=system,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        ) as stream:
            async for text in stream.text_stream:
                yield LLMResponse(
                    content=text,
                    model=self.model,
                    provider=self.provider.value,
                    latency_ms=(time.time() - start_time) * 1000
                )

    # ============================================================
    # Google Gemini Methods
    # ============================================================

    def _generate_gemini(self, messages, temperature, max_tokens, top_p, **kwargs) -> LLMResponse:
        start_time = time.time()

        # Convert messages to Gemini format
        gemini_messages = self._convert_to_gemini_format(messages)

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "top_p": top_p,
        }

        response = self.client.generate_content(
            gemini_messages,
            generation_config=generation_config,
            **kwargs
        )

        latency_ms = (time.time() - start_time) * 1000

        content = response.text if response.text else ""

        # Estimate tokens
        prompt_tokens = self.count_tokens(str(messages))
        completion_tokens = self.count_tokens(content)
        cost = self.estimate_cost(prompt_tokens, completion_tokens)

        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider.value,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=cost,
            finish_reason=str(response.candidates[0].finish_reason) if response.candidates else "",
            latency_ms=latency_ms,
            raw_response=response
        )

    async def _generate_gemini_async(self, messages, temperature, max_tokens, top_p, **kwargs) -> LLMResponse:
        # Gemini doesn't have native async, run in thread pool
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate_gemini,
            messages, temperature, max_tokens, top_p, **kwargs
        )

    def _convert_to_gemini_format(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to Gemini format."""
        # Gemini uses a simple prompt format
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        return "\n\n".join(prompt_parts)

    # ============================================================
    # Cohere Methods
    # ============================================================

    def _generate_cohere(self, messages, temperature, max_tokens, top_p, **kwargs) -> LLMResponse:
        start_time = time.time()

        # Extract the last user message as prompt
        prompt = messages[-1]["content"] if messages else ""

        response = self.client.chat(
            message=prompt,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        latency_ms = (time.time() - start_time) * 1000

        content = response.text

        # Estimate tokens
        prompt_tokens = self.count_tokens(prompt)
        completion_tokens = self.count_tokens(content)
        cost = self.estimate_cost(prompt_tokens, completion_tokens)

        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider.value,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=cost,
            finish_reason=response.finish_reason if hasattr(response, 'finish_reason') else "",
            latency_ms=latency_ms,
            raw_response=response
        )

    async def _generate_cohere_async(self, messages, temperature, max_tokens, top_p, **kwargs) -> LLMResponse:
        start_time = time.time()

        prompt = messages[-1]["content"] if messages else ""

        response = await self.async_client.chat(
            message=prompt,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        latency_ms = (time.time() - start_time) * 1000

        content = response.text

        prompt_tokens = self.count_tokens(prompt)
        completion_tokens = self.count_tokens(content)
        cost = self.estimate_cost(prompt_tokens, completion_tokens)

        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider.value,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=cost,
            finish_reason=response.finish_reason if hasattr(response, 'finish_reason') else "",
            latency_ms=latency_ms,
            raw_response=response
        )

    def _stream_cohere(self, messages, temperature, max_tokens, top_p, **kwargs) -> Iterator[LLMResponse]:
        start_time = time.time()
        prompt = messages[-1]["content"] if messages else ""

        stream = self.client.chat_stream(
            message=prompt,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        for event in stream:
            if event.event_type == "text-generation":
                yield LLMResponse(
                    content=event.text,
                    model=self.model,
                    provider=self.provider.value,
                    latency_ms=(time.time() - start_time) * 1000
                )

    async def _stream_cohere_async(self, messages, temperature, max_tokens, top_p, **kwargs) -> AsyncIterator[LLMResponse]:
        start_time = time.time()
        prompt = messages[-1]["content"] if messages else ""

        async_stream = self.async_client.chat_stream(
            message=prompt,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        async for event in async_stream:
            if event.event_type == "text-generation":
                yield LLMResponse(
                    content=event.text,
                    model=self.model,
                    provider=self.provider.value,
                    latency_ms=(time.time() - start_time) * 1000
                )

    # ============================================================
    # OpenAI-Compatible Methods (Groq, Azure, Ollama, Local)
    # ============================================================

    def _generate_openai_compatible(self, messages, temperature, max_tokens, top_p, **kwargs) -> LLMResponse:
        start_time = time.time()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )

        latency_ms = (time.time() - start_time) * 1000

        choice = response.choices[0]
        content = choice.message.content

        prompt_tokens = getattr(response.usage, 'prompt_tokens', self.count_tokens(str(messages)))
        completion_tokens = getattr(response.usage, 'completion_tokens', self.count_tokens(content))
        cost = self.estimate_cost(prompt_tokens, completion_tokens)

        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider.value,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=cost,
            finish_reason=choice.finish_reason,
            latency_ms=latency_ms,
            raw_response=response
        )

    async def _generate_openai_compatible_async(self, messages, temperature, max_tokens, top_p, **kwargs) -> LLMResponse:
        start_time = time.time()

        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            **kwargs
        )

        latency_ms = (time.time() - start_time) * 1000

        choice = response.choices[0]
        content = choice.message.content

        prompt_tokens = getattr(response.usage, 'prompt_tokens', self.count_tokens(str(messages)))
        completion_tokens = getattr(response.usage, 'completion_tokens', self.count_tokens(content))
        cost = self.estimate_cost(prompt_tokens, completion_tokens)

        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider.value,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=cost,
            finish_reason=choice.finish_reason,
            latency_ms=latency_ms,
            raw_response=response
        )

    def _stream_openai_compatible(self, messages, temperature, max_tokens, top_p, **kwargs) -> Iterator[LLMResponse]:
        start_time = time.time()

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=True,
            **kwargs
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield LLMResponse(
                    content=chunk.choices[0].delta.content,
                    model=self.model,
                    provider=self.provider.value,
                    latency_ms=(time.time() - start_time) * 1000,
                    raw_response=chunk
                )

    async def _stream_openai_compatible_async(self, messages, temperature, max_tokens, top_p, **kwargs) -> AsyncIterator[LLMResponse]:
        start_time = time.time()

        stream = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stream=True,
            **kwargs
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield LLMResponse(
                    content=chunk.choices[0].delta.content,
                    model=self.model,
                    provider=self.provider.value,
                    latency_ms=(time.time() - start_time) * 1000,
                    raw_response=chunk
                )

    # ============================================================
    # Convenience Methods
    # ============================================================

    def generate_simple(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Simple generation for a single prompt.

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
        """Async simple generation."""
        messages = [Message(role="user", content=prompt)]
        response = await self.generate_async(messages, system_prompt=system_prompt)
        return response.content

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "provider": self.provider.value,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "cost_info": self.COSTS.get(self.model, {"prompt": "unknown", "completion": "unknown"})
        }


# ============================================================
# Factory Function
# ============================================================

def create_llm_interface(
    provider: str = "openai",
    model: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs
) -> LLMInterface:
    """
    Factory function to create LLM interface with provider-specific defaults.

    Args:
        provider: Provider name ('openai', 'anthropic', 'azure', 'gemini', 'cohere', 'groq', 'local', 'ollama')
        model: Model name (uses provider default if not specified)
        temperature: Sampling temperature
        **kwargs: Additional arguments

    Returns:
        LLMInterface instance
    """
    # Default models by provider
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
        **kwargs
    )


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    # Test OpenAI
    if os.getenv("OPENAI_API_KEY"):
        print("Testing OpenAI...")
        llm = create_llm_interface("openai", "gpt-4")
        response = llm.generate_simple("What is the capital of France?")
        print(f"OpenAI: {response[:100]}...")

    # Test Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        print("Testing Anthropic...")
        llm = create_llm_interface("anthropic", "claude-3-haiku-20240307")
        response = llm.generate_simple("What is the capital of France?")
        print(f"Anthropic: {response[:100]}...")

    # Test Gemini
    if os.getenv("GEMINI_API_KEY"):
        print("Testing Gemini...")
        llm = create_llm_interface("gemini", "gemini-1.5-flash")
        response = llm.generate_simple("What is the capital of France?")
        print(f"Gemini: {response[:100]}...")

    print("\nLLM Interface ready with multiple providers!")
