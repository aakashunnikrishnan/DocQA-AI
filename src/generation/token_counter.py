"""
Token counting module for different LLM providers.
Provides accurate token counting for OpenAI, Anthropic, Cohere, Google Gemini, and local models.
Supports caching and estimation for cost tracking.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
import hashlib
import json

from src.utils.logger import get_logger
from src.utils.cache import CacheManager, cached

logger = get_logger(__name__)

# Try importing tiktoken for OpenAI tokenization
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logger.warning("tiktoken not installed. Install with: pip install tiktoken")

# Try importing transformers for local model tokenization
try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed. Token counting for local models will be limited.")


class TokenizerType(Enum):
    """Types of tokenizers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    GEMINI = "gemini"
    GROQ = "groq"
    LOCAL = "local"
    OLLAMA = "ollama"
    UNKNOWN = "unknown"


@dataclass
class TokenCount:
    """Token count result."""
    total_tokens: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost: float = 0.0
    model: str = ""
    provider: str = ""
    accuracy: float = 1.0  # 1.0 = exact, 0.0 = estimate

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "estimated_cost": self.estimated_cost,
            "model": self.model,
            "provider": self.provider,
            "accuracy": self.accuracy
        }


# ============================================================
# Provider Token Cost Configurations
# ============================================================

class TokenCosts:
    """Token costs per provider and model."""

    # Costs per 1K tokens (USD)
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
        "text-embedding-3-small": {"prompt": 0.00002, "completion": 0.0},
        "text-embedding-3-large": {"prompt": 0.00013, "completion": 0.0},
        "text-embedding-ada-002": {"prompt": 0.00010, "completion": 0.0},

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

    @classmethod
    def get_cost(cls, model: str, token_type: str = "prompt") -> float:
        """Get cost per 1K tokens for a model."""
        # Normalize model name
        model_lower = model.lower()

        # Check exact match
        if model in cls.COSTS:
            return cls.COSTS[model].get(token_type, 0.0)

        # Check partial match
        for key in cls.COSTS:
            if key in model_lower or model_lower in key:
                return cls.COSTS[key].get(token_type, 0.0)

        # Default cost (higher-end)
        return 0.01 if token_type == "prompt" else 0.02

    @classmethod
    def estimate_cost(cls, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost for token usage."""
        prompt_cost = cls.get_cost(model, "prompt") * (prompt_tokens / 1000)
        completion_cost = cls.get_cost(model, "completion") * (completion_tokens / 1000)
        return prompt_cost + completion_cost


# ============================================================
# Token Counter Base Class
# ============================================================

class TokenCounter:
    """
    Token counter for various LLM providers.
    """

    def __init__(self, cache_ttl: int = 3600):
        """
        Initialize token counter.

        Args:
            cache_ttl: Cache TTL in seconds
        """
        self.cache_ttl = cache_ttl
        self._cache = {}
        self.cache_manager = CacheManager()

        # Initialize provider-specific counters
        self._counters = {
            TokenizerType.OPENAI: OpenAITokenCounter(),
            TokenizerType.ANTHROPIC: AnthropicTokenCounter(),
            TokenizerType.COHERE: CohereTokenCounter(),
            TokenizerType.GEMINI: GeminiTokenCounter(),
            TokenizerType.GROQ: GroqTokenCounter(),
            TokenizerType.LOCAL: LocalTokenCounter(),
            TokenizerType.OLLAMA: OllamaTokenCounter(),
        }

        logger.info("TokenCounter initialized with caching")

    def get_counter(self, provider: Union[str, TokenizerType]) -> 'BaseTokenCounter':
        """Get the appropriate token counter for a provider."""
        if isinstance(provider, str):
            provider = TokenizerType(provider.lower())

        return self._counters.get(provider, self._counters[TokenizerType.UNKNOWN])

    @cached(ttl=3600)
    def count_tokens(
        self,
        text: str,
        model: str,
        provider: Union[str, TokenizerType]
    ) -> TokenCount:
        """
        Count tokens for a text using the appropriate provider.

        Args:
            text: Text to count tokens for
            model: Model name
            provider: Provider type

        Returns:
            TokenCount object
        """
        counter = self.get_counter(provider)
        return counter.count_tokens(text, model)

    def count_messages(
        self,
        messages: List[Dict[str, str]],
        model: str,
        provider: Union[str, TokenizerType]
    ) -> TokenCount:
        """
        Count tokens for a list of messages.

        Args:
            messages: List of message dictionaries
            model: Model name
            provider: Provider type

        Returns:
            TokenCount object
        """
        counter = self.get_counter(provider)
        return counter.count_messages(messages, model)

    def estimate_cost(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str
    ) -> float:
        """
        Estimate cost for token usage.

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            model: Model name

        Returns:
            Estimated cost in USD
        """
        return TokenCosts.estimate_cost(model, prompt_tokens, completion_tokens)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache_manager.get_stats()


# ============================================================
# Provider-Specific Token Counters
# ============================================================

class BaseTokenCounter:
    """Base token counter."""

    def count_tokens(self, text: str, model: str) -> TokenCount:
        """Count tokens for a text."""
        raise NotImplementedError

    def count_messages(self, messages: List[Dict[str, str]], model: str) -> TokenCount:
        """Count tokens for messages."""
        # Default: concatenate and count
        text = " ".join([f"{m.get('role', '')}: {m.get('content', '')}" for m in messages])
        return self.count_tokens(text, model)


class OpenAITokenCounter(BaseTokenCounter):
    """OpenAI token counter using tiktoken."""

    # Model to encoding mapping
    ENCODING_MAP = {
        "gpt-4": "cl100k_base",
        "gpt-4-32k": "cl100k_base",
        "gpt-4-turbo-preview": "cl100k_base",
        "gpt-4o": "o200k_base",
        "gpt-4o-mini": "o200k_base",
        "gpt-4o-2024-08-06": "o200k_base",
        "gpt-4o-mini-2024-07-18": "o200k_base",
        "gpt-3.5-turbo": "cl100k_base",
        "gpt-3.5-turbo-16k": "cl100k_base",
        "gpt-3.5-turbo-0125": "cl100k_base",
        "gpt-3.5-turbo-1106": "cl100k_base",
        "text-embedding-3-small": "cl100k_base",
        "text-embedding-3-large": "cl100k_base",
        "text-embedding-ada-002": "cl100k_base",
    }

    def __init__(self):
        self._encodings = {}

        if not TIKTOKEN_AVAILABLE:
            logger.warning("tiktoken not available, using fallback token counting")

    def _get_encoding(self, model: str) -> str:
        """Get encoding for a model."""
        model_lower = model.lower()

        # Check exact match
        if model in self.ENCODING_MAP:
            return self.ENCODING_MAP[model]

        # Check partial match
        for key in self.ENCODING_MAP:
            if key in model_lower or model_lower in key:
                return self.ENCODING_MAP[key]

        # Default to cl100k_base
        return "cl100k_base"

    def _get_tokenizer(self, model: str):
        """Get tiktoken tokenizer."""
        encoding_name = self._get_encoding(model)

        if encoding_name not in self._encodings:
            if TIKTOKEN_AVAILABLE:
                try:
                    self._encodings[encoding_name] = tiktoken.get_encoding(encoding_name)
                except Exception as e:
                    logger.warning(f"Failed to load encoding {encoding_name}: {e}")
                    return None
            else:
                return None

        return self._encodings.get(encoding_name)

    def count_tokens(self, text: str, model: str) -> TokenCount:
        """Count tokens using tiktoken."""
        tokenizer = self._get_tokenizer(model)

        if tokenizer:
            try:
                tokens = tokenizer.encode(text)
                token_count = len(tokens)
                return TokenCount(
                    total_tokens=token_count,
                    prompt_tokens=token_count,
                    model=model,
                    provider="openai",
                    accuracy=1.0
                )
            except Exception as e:
                logger.debug(f"Token counting failed: {e}")

        # Fallback: estimate
        char_count = len(text)
        estimated_tokens = int(char_count / 4)  # Rough estimate

        return TokenCount(
            total_tokens=estimated_tokens,
            prompt_tokens=estimated_tokens,
            model=model,
            provider="openai",
            accuracy=0.5
        )

    def count_messages(self, messages: List[Dict[str, str]], model: str) -> TokenCount:
        """Count tokens for messages with OpenAI-specific formatting."""
        if not messages:
            return TokenCount(
                total_tokens=0,
                prompt_tokens=0,
                model=model,
                provider="openai",
                accuracy=1.0
            )

        # OpenAI message format adds overhead
        # Each message has role and content, with formatting overhead
        text = ""
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            text += f"{role}: {content}\n"

        token_count = self.count_tokens(text, model)

        # Add overhead for message formatting (approximately 4 tokens per message)
        message_overhead = len(messages) * 4

        return TokenCount(
            total_tokens=token_count.total_tokens + message_overhead,
            prompt_tokens=token_count.prompt_tokens + message_overhead,
            model=model,
            provider="openai",
            accuracy=token_count.accuracy
        )


class AnthropicTokenCounter(BaseTokenCounter):
    """Anthropic token counter."""

    def __init__(self):
        self._tokenizer = None

        # Try to use Anthropic's tokenizer
        try:
            import anthropic
            self._tokenizer = anthropic.Anthropic().get_tokenizer()
        except (ImportError, AttributeError):
            pass

        if not self._tokenizer:
            logger.debug("Anthropic tokenizer not available, using fallback")

    def count_tokens(self, text: str, model: str) -> TokenCount:
        """Count tokens using Anthropic tokenizer."""
        if self._tokenizer:
            try:
                token_count = self._tokenizer.encode(text).token_count
                return TokenCount(
                    total_tokens=token_count,
                    prompt_tokens=token_count,
                    model=model,
                    provider="anthropic",
                    accuracy=1.0
                )
            except Exception as e:
                logger.debug(f"Anthropic token counting failed: {e}")

        # Fallback: use approximate counting
        char_count = len(text)
        estimated_tokens = int(char_count / 3.5)  # Anthropic uses ~3.5 chars per token

        return TokenCount(
            total_tokens=estimated_tokens,
            prompt_tokens=estimated_tokens,
            model=model,
            provider="anthropic",
            accuracy=0.6
        )


class CohereTokenCounter(BaseTokenCounter):
    """Cohere token counter."""

    def __init__(self):
        self._tokenizer = None

        # Try to use Cohere's tokenizer
        try:
            import cohere
            self._tokenizer = cohere.tokenizer.get_tokenizer()
        except (ImportError, AttributeError):
            pass

    def count_tokens(self, text: str, model: str) -> TokenCount:
        """Count tokens using Cohere tokenizer."""
        if self._tokenizer:
            try:
                token_count = len(self._tokenizer.encode(text).ids)
                return TokenCount(
                    total_tokens=token_count,
                    prompt_tokens=token_count,
                    model=model,
                    provider="cohere",
                    accuracy=1.0
                )
            except Exception as e:
                logger.debug(f"Cohere token counting failed: {e}")

        # Fallback
        char_count = len(text)
        estimated_tokens = int(char_count / 4)

        return TokenCount(
            total_tokens=estimated_tokens,
            prompt_tokens=estimated_tokens,
            model=model,
            provider="cohere",
            accuracy=0.5
        )


class GeminiTokenCounter(BaseTokenCounter):
    """Google Gemini token counter."""

    def __init__(self):
        self._tokenizer = None

        # Try to use Gemini's tokenizer
        try:
            import google.generativeai as genai
            self._tokenizer = genai.tokenizer.get_tokenizer()
        except (ImportError, AttributeError):
            pass

    def count_tokens(self, text: str, model: str) -> TokenCount:
        """Count tokens using Gemini tokenizer."""
        if self._tokenizer:
            try:
                token_count = self._tokenizer.count_tokens(text).total_tokens
                return TokenCount(
                    total_tokens=token_count,
                    prompt_tokens=token_count,
                    model=model,
                    provider="gemini",
                    accuracy=1.0
                )
            except Exception as e:
                logger.debug(f"Gemini token counting failed: {e}")

        # Fallback
        char_count = len(text)
        estimated_tokens = int(char_count / 4)

        return TokenCount(
            total_tokens=estimated_tokens,
            prompt_tokens=estimated_tokens,
            model=model,
            provider="gemini",
            accuracy=0.5
        )


class GroqTokenCounter(BaseTokenCounter):
    """Groq token counter (uses tiktoken for Llama models)."""

    def __init__(self):
        self._tokenizer = None

        if TIKTOKEN_AVAILABLE:
            try:
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception:
                pass

    def count_tokens(self, text: str, model: str) -> TokenCount:
        """Count tokens using tiktoken."""
        if self._tokenizer:
            try:
                tokens = self._tokenizer.encode(text)
                token_count = len(tokens)
                return TokenCount(
                    total_tokens=token_count,
                    prompt_tokens=token_count,
                    model=model,
                    provider="groq",
                    accuracy=0.95  # Groq uses Llama tokenizer, close to cl100k_base
                )
            except Exception as e:
                logger.debug(f"Groq token counting failed: {e}")

        # Fallback
        char_count = len(text)
        estimated_tokens = int(char_count / 4)

        return TokenCount(
            total_tokens=estimated_tokens,
            prompt_tokens=estimated_tokens,
            model=model,
            provider="groq",
            accuracy=0.5
        )


class LocalTokenCounter(BaseTokenCounter):
    """Local model token counter using transformers."""

    def __init__(self):
        self._tokenizers = {}

    def _get_tokenizer(self, model: str):
        """Get tokenizer for a local model."""
        if model not in self._tokenizers:
            if TRANSFORMERS_AVAILABLE:
                try:
                    self._tokenizers[model] = AutoTokenizer.from_pretrained(model)
                except Exception as e:
                    logger.debug(f"Failed to load tokenizer for {model}: {e}")
                    return None
            else:
                return None

        return self._tokenizers.get(model)

    def count_tokens(self, text: str, model: str) -> TokenCount:
        """Count tokens using transformers tokenizer."""
        tokenizer = self._get_tokenizer(model)

        if tokenizer:
            try:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                token_count = len(tokens)
                return TokenCount(
                    total_tokens=token_count,
                    prompt_tokens=token_count,
                    model=model,
                    provider="local",
                    accuracy=1.0
                )
            except Exception as e:
                logger.debug(f"Local token counting failed: {e}")

        # Fallback
        char_count = len(text)
        estimated_tokens = int(char_count / 4)

        return TokenCount(
            total_tokens=estimated_tokens,
            prompt_tokens=estimated_tokens,
            model=model,
            provider="local",
            accuracy=0.5
        )


class OllamaTokenCounter(BaseTokenCounter):
    """Ollama token counter."""

    def count_tokens(self, text: str, model: str) -> TokenCount:
        """Count tokens using Ollama API."""
        try:
            import ollama
            response = ollama.embeddings(model=model, prompt=text)

            # Ollama doesn't return token count directly, use embedding length as proxy
            if hasattr(response, 'embedding'):
                # Rough estimate: token count ~ embedding length / 2
                estimated_tokens = len(response.embedding) // 2
                return TokenCount(
                    total_tokens=estimated_tokens,
                    prompt_tokens=estimated_tokens,
                    model=model,
                    provider="ollama",
                    accuracy=0.7
                )
        except Exception as e:
            logger.debug(f"Ollama token counting failed: {e}")

        # Fallback
        char_count = len(text)
        estimated_tokens = int(char_count / 4)

        return TokenCount(
            total_tokens=estimated_tokens,
            prompt_tokens=estimated_tokens,
            model=model,
            provider="ollama",
            accuracy=0.5
        )


# ============================================================
# Convenience Functions
# ============================================================

_token_counter: Optional[TokenCounter] = None


def get_token_counter() -> TokenCounter:
    """Get global token counter instance."""
    global _token_counter
    if _token_counter is None:
        _token_counter = TokenCounter()
    return _token_counter


def count_tokens(text: str, model: str, provider: str) -> TokenCount:
    """
    Count tokens for a text.

    Args:
        text: Text to count tokens for
        model: Model name
        provider: Provider type ('openai', 'anthropic', 'cohere', 'gemini', 'groq', 'local', 'ollama')

    Returns:
        TokenCount object
    """
    counter = get_token_counter()
    return counter.count_tokens(text, model, provider)


def count_messages(messages: List[Dict[str, str]], model: str, provider: str) -> TokenCount:
    """
    Count tokens for messages.

    Args:
        messages: List of message dictionaries
        model: Model name
        provider: Provider type

    Returns:
        TokenCount object
    """
    counter = get_token_counter()
    return counter.count_messages(messages, model, provider)


def estimate_cost(prompt_tokens: int, completion_tokens: int, model: str) -> float:
    """
    Estimate cost for token usage.

    Args:
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        model: Model name

    Returns:
        Estimated cost in USD
    """
    counter = get_token_counter()
    return counter.estimate_cost(prompt_tokens, completion_tokens, model)


# ============================================================
# Utility Functions
# ============================================================

def truncate_to_token_limit(
    text: str,
    max_tokens: int,
    model: str,
    provider: str,
    truncation_side: str = "end"
) -> Tuple[str, int]:
    """
    Truncate text to a token limit.

    Args:
        text: Text to truncate
        max_tokens: Maximum tokens allowed
        model: Model name
        provider: Provider type
        truncation_side: 'start' or 'end'

    Returns:
        Tuple of (truncated_text, token_count)
    """
    token_count = count_tokens(text, model, provider)
    current_tokens = token_count.total_tokens

    if current_tokens <= max_tokens:
        return text, current_tokens

    # Need to truncate
    # Simple approach: truncate by characters
    # For production, use tokenizer-based truncation
    ratio = max_tokens / current_tokens
    new_length = int(len(text) * ratio * 0.95)  # Conservative truncation

    if truncation_side == "start":
        truncated = text[-new_length:]
    else:
        truncated = text[:new_length]

    # Recount tokens
    new_count = count_tokens(truncated, model, provider)

    # If still over limit, truncate more
    while new_count.total_tokens > max_tokens and len(truncated) > 10:
        if truncation_side == "start":
            truncated = truncated[10:]
        else:
            truncated = truncated[:-10]
        new_count = count_tokens(truncated, model, provider)

    return truncated, new_count.total_tokens


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Testing Token Counter...")
    print("=" * 60)

    # Test texts
    texts = [
        "Hello, world!",
        "This is a longer text with multiple words. It should have more tokens.",
        "The quick brown fox jumps over the lazy dog. " * 10,
    ]

    # Test different providers
    providers = ["openai", "anthropic", "cohere", "gemini", "groq", "local", "ollama"]
    models = [
        "gpt-4",
        "claude-3-haiku-20240307",
        "command-r",
        "gemini-1.5-pro",
        "llama-3.1-70b-versatile",
        "meta-llama/Llama-2-7b-hf",
        "llama2"
    ]

    counter = get_token_counter()

    for i, text in enumerate(texts):
        print(f"\nText {i+1} (length: {len(text)} chars):")
        print(f"  {text[:100]}...")

        for provider, model in zip(providers, models):
            try:
                token_count = counter.count_tokens(text, model, provider)
                print(f"  {provider:10} ({model:30}) -> {token_count.total_tokens:4} tokens "
                      f"(accuracy: {token_count.accuracy:.0%})")
            except Exception as e:
                print(f"  {provider:10} -> Error: {e}")

    # Test messages
    print("\n" + "=" * 60)
    print("Testing Message Counting:")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI..."}
    ]

    for provider, model in zip(providers, models):
        try:
            token_count = counter.count_messages(messages, model, provider)
            print(f"  {provider:10} ({model:30}) -> {token_count.total_tokens:4} tokens")
        except Exception as e:
            print(f"  {provider:10} -> Error: {e}")

    # Test cost estimation
    print("\n" + "=" * 60)
    print("Testing Cost Estimation:")

    prompt_tokens = 1000
    completion_tokens = 500

    for model in ["gpt-4", "gpt-4o-mini", "claude-3-haiku-20240307", "gemini-1.5-flash"]:
        cost = estimate_cost(prompt_tokens, completion_tokens, model)
        print(f"  {model:30} -> ${cost:.4f}")

    print("\n" + "=" * 60)
    print("Token Counter ready!")
