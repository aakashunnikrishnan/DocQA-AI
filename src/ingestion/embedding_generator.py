"""
OpenAI embeddings integration for generating vector representations of text chunks.
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import asyncio
from concurrent.futures import ThreadPoolExecutor

# OpenAI imports
try:
    from openai import OpenAI, AsyncOpenAI
    from openai.types import CreateEmbeddingResponse
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI package not installed. Install with: pip install openai")

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Represents an embedding result for a text chunk."""
    text: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0
    model: str = ""
    tokens_used: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "chunk_index": self.chunk_index,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "embedding_dimension": len(self.embedding)
        }

    def get_numpy_array(self) -> np.ndarray:
        """Get embedding as numpy array."""
        return np.array(self.embedding, dtype=np.float32)


class OpenAIEmbeddingGenerator:
    """
    Generate embeddings using OpenAI's API.

    Supports multiple models:
    - text-embedding-3-small (1536 dimensions, cheaper)
    - text-embedding-3-large (3072 dimensions, more accurate)
    - text-embedding-ada-002 (1536 dimensions, legacy)
    """

    # Model configurations
    MODEL_CONFIGS = {
        "text-embedding-3-small": {
            "dimension": 1536,
            "max_tokens": 8191,
            "cost_per_1k_tokens": 0.00002,
            "description": "Most cost-effective embedding model"
        },
        "text-embedding-3-large": {
            "dimension": 3072,
            "max_tokens": 8191,
            "cost_per_1k_tokens": 0.00013,
            "description": "Most powerful embedding model"
        },
        "text-embedding-ada-002": {
            "dimension": 1536,
            "max_tokens": 8191,
            "cost_per_1k_tokens": 0.00010,
            "description": "Legacy Ada v2 model"
        }
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        batch_size: int = 20,
        max_retries: int = 3,
        timeout: int = 60,
        dimensions: Optional[int] = None,
        organization: Optional[str] = None
    ):
        """
        Initialize OpenAI embedding generator.

        Args:
            model: OpenAI embedding model name
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            base_url: Custom API base URL
            batch_size: Number of texts to embed in one batch
            max_retries: Maximum number of retries for failed requests
            timeout: Request timeout in seconds
            dimensions: Output dimensions (for 3-small/3-large models)
            organization: OpenAI organization ID
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package is required. Install with: pip install openai"
            )

        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout

        # Validate model
        if model not in self.MODEL_CONFIGS:
            logger.warning(f"Unknown model {model}. Using default config.")
            self.model_config = {
                "dimension": dimensions or 1536,
                "max_tokens": 8191,
                "cost_per_1k_tokens": 0.00002,
                "description": "Custom model"
            }
        else:
            self.model_config = self.MODEL_CONFIGS[model].copy()

        # Override dimensions if specified
        if dimensions:
            if dimensions > self.model_config["dimension"]:
                logger.warning(
                    f"Requested dimensions {dimensions} > model max {self.model_config['dimension']}. "
                    f"Using {self.model_config['dimension']}."
                )
            else:
                self.model_config["dimension"] = dimensions

        # Initialize clients
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            timeout=timeout
        )

        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            timeout=timeout
        )

        logger.info(f"Initialized OpenAI embedding generator with model: {model}")
        logger.info(f"Embedding dimension: {self.model_config['dimension']}")

    def _truncate_text(self, text: str, max_tokens: int = 8191) -> str:
        """
        Truncate text to maximum token limit.
        Approximate truncation based on characters (rough estimate).
        """
        if len(text) <= max_tokens * 4:  # Rough estimate: 4 chars per token
            return text

        # More accurate truncation using tiktoken if available
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            tokens = encoding.encode(text)
            if len(tokens) > max_tokens:
                tokens = tokens[:max_tokens]
                return encoding.decode(tokens)
        except ImportError:
            # Fallback to character-based truncation
            logger.warning("tiktoken not installed. Using approximate truncation.")
            max_chars = max_tokens * 4
            if len(text) > max_chars:
                text = text[:max_chars]

        return text

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def _generate_embedding_single(self, text: str) -> Tuple[List[float], int]:
        """
        Generate embedding for a single text with retry logic.

        Returns:
            Tuple of (embedding, tokens_used)
        """
        text = self._truncate_text(text, self.model_config["max_tokens"])

        try:
            response: CreateEmbeddingResponse = self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.model_config.get("dimension")
            )

            embedding = response.data[0].embedding
            tokens_used = response.usage.total_tokens

            return embedding, tokens_used

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def _generate_embeddings_batch(self, texts: List[str]) -> Tuple[List[List[float]], int]:
        """
        Generate embeddings for multiple texts in a batch.

        Returns:
            Tuple of (embeddings list, total_tokens_used)
        """
        # Truncate each text
        truncated_texts = [
            self._truncate_text(t, self.model_config["max_tokens"])
            for t in texts
        ]

        try:
            response: CreateEmbeddingResponse = self.client.embeddings.create(
                model=self.model,
                input=truncated_texts,
                dimensions=self.model_config.get("dimension")
            )

            embeddings = [data.embedding for data in response.data]
            total_tokens = response.usage.total_tokens

            return embeddings, total_tokens

        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise

    def generate_embedding(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> EmbeddingResult:
        """
        Generate embedding for a single text chunk.

        Args:
            text: Text to embed
            metadata: Optional metadata to attach to result

        Returns:
            EmbeddingResult object
        """
        logger.debug(f"Generating embedding for text of length {len(text)}")

        embedding, tokens_used = self._generate_embedding_single(text)

        result = EmbeddingResult(
            text=text,
            embedding=embedding,
            metadata=metadata or {},
            model=self.model,
            tokens_used=tokens_used
        )

        logger.debug(f"Generated embedding with {len(embedding)} dimensions, {tokens_used} tokens")
        return result

    def generate_embeddings(
        self,
        chunks: List[Union[str, Dict[str, Any]]],
        show_progress: bool = True
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple chunks.

        Args:
            chunks: List of either text strings or dicts with 'text' and 'metadata' keys
            show_progress: Whether to show progress bar

        Returns:
            List of EmbeddingResult objects
        """
        # Parse chunks
        texts = []
        metadata_list = []

        for chunk in chunks:
            if isinstance(chunk, str):
                texts.append(chunk)
                metadata_list.append({})
            elif isinstance(chunk, dict):
                texts.append(chunk.get("text", ""))
                metadata_list.append(chunk.get("metadata", {}))
            else:
                raise TypeError(f"Unsupported chunk type: {type(chunk)}")

        results = []
        total_tokens = 0
        total_cost = 0

        # Process in batches
        batch_count = (len(texts) + self.batch_size - 1) // self.batch_size

        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=batch_count, desc="Generating embeddings")

        for i in iterator:
            batch_texts = texts[i:i + self.batch_size]
            batch_metadata = metadata_list[i:i + self.batch_size]

            try:
                if len(batch_texts) == 1:
                    embeddings, tokens = [self._generate_embedding_single(batch_texts[0])[0]], \
                                        self._generate_embedding_single(batch_texts[0])[1]
                else:
                    embeddings, tokens = self._generate_embeddings_batch(batch_texts)

                total_tokens += tokens

                # Create results
                for j, (embedding, text, metadata) in enumerate(
                    zip(embeddings, batch_texts, batch_metadata)
                ):
                    result = EmbeddingResult(
                        text=text,
                        embedding=embedding,
                        metadata=metadata,
                        chunk_index=i + j,
                        model=self.model,
                        tokens_used=tokens // len(batch_texts)  # Approximate per-chunk tokens
                    )
                    results.append(result)

                # Calculate cost
                cost_per_1k = self.model_config["cost_per_1k_tokens"]
                batch_cost = (tokens / 1000) * cost_per_1k
                total_cost += batch_cost

                # Rate limiting to avoid hitting API limits
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Failed to process batch starting at index {i}: {e}")
                # Add empty embeddings for failed batch
                for j, (text, metadata) in enumerate(zip(batch_texts, batch_metadata)):
                    results.append(EmbeddingResult(
                        text=text,
                        embedding=[0.0] * self.model_config["dimension"],
                        metadata=metadata,
                        chunk_index=i + j,
                        model=self.model,
                        tokens_used=0
                    ))

        logger.info(
            f"Generated {len(results)} embeddings. "
            f"Total tokens: {total_tokens}, "
            f"Estimated cost: ${total_cost:.6f}"
        )

        return results

    async def generate_embedding_async(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> EmbeddingResult:
        """Asynchronously generate embedding for a single text."""
        text = self._truncate_text(text, self.model_config["max_tokens"])

        try:
            response = await self.async_client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.model_config.get("dimension")
            )

            embedding = response.data[0].embedding
            tokens_used = response.usage.total_tokens

            return EmbeddingResult(
                text=text,
                embedding=embedding,
                metadata=metadata or {},
                model=self.model,
                tokens_used=tokens_used
            )

        except Exception as e:
            logger.error(f"Async embedding generation failed: {e}")
            raise

    async def generate_embeddings_async(
        self,
        chunks: List[Union[str, Dict[str, Any]]],
        max_concurrent: int = 10
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings asynchronously with concurrency control.

        Args:
            chunks: List of chunks to embed
            max_concurrent: Maximum number of concurrent requests

        Returns:
            List of EmbeddingResult objects
        """
        # Parse chunks
        texts = []
        metadata_list = []

        for chunk in chunks:
            if isinstance(chunk, str):
                texts.append(chunk)
                metadata_list.append({})
            else:
                texts.append(chunk.get("text", ""))
                metadata_list.append(chunk.get("metadata", {}))

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_one(text, metadata, idx):
            async with semaphore:
                return await self.generate_embedding_async(text, metadata)

        # Process all chunks
        tasks = [
            process_one(text, metadata, i)
            for i, (text, metadata) in enumerate(zip(texts, metadata_list))
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle failures
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to generate embedding for chunk {i}: {result}")
                final_results.append(EmbeddingResult(
                    text=texts[i],
                    embedding=[0.0] * self.model_config["dimension"],
                    metadata=metadata_list[i],
                    chunk_index=i,
                    model=self.model,
                    tokens_used=0
                ))
            else:
                result.chunk_index = i
                final_results.append(result)

        return final_results

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this generator."""
        return self.model_config["dimension"]

    def estimate_cost(self, num_tokens: int) -> float:
        """Estimate cost for embedding a given number of tokens."""
        return (num_tokens / 1000) * self.model_config["cost_per_1k_tokens"]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model": self.model,
            "dimension": self.model_config["dimension"],
            "max_tokens": self.model_config["max_tokens"],
            "cost_per_1k_tokens": self.model_config["cost_per_1k_tokens"],
            "description": self.model_config["description"]
        }


class EmbeddingCache:
    """
    Cache for embeddings to avoid redundant API calls.
    """

    def __init__(self, cache_dir: str = "./data/embeddings/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, EmbeddingResult] = {}
        self._load_cache()

    def _get_hash_key(self, text: str, model: str) -> str:
        """Generate hash key for text and model combination."""
        import hashlib
        content = f"{text}:{model}".encode('utf-8')
        return hashlib.md5(content).hexdigest()

    def _load_cache(self):
        """Load cached embeddings from disk."""
        cache_file = self.cache_dir / "embeddings_cache.npz"
        if cache_file.exists():
            try:
                data = np.load(cache_file, allow_pickle=True)
                keys = data['keys']
                embeddings = data['embeddings']
                for key, embedding in zip(keys, embeddings):
                    self._cache[key] = embedding
                logger.info(f"Loaded {len(self._cache)} cached embeddings")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

    def _save_cache(self):
        """Save cached embeddings to disk."""
        if not self._cache:
            return

        cache_file = self.cache_dir / "embeddings_cache.npz"
        try:
            keys = list(self._cache.keys())
            # Convert embeddings to numpy array if needed
            embeddings = [e.embedding if isinstance(e, EmbeddingResult) else e for e in self._cache.values()]
            np.savez_compressed(cache_file, keys=keys, embeddings=embeddings)
            logger.debug(f"Saved {len(self._cache)} embeddings to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def get(self, text: str, model: str) -> Optional[EmbeddingResult]:
        """Get cached embedding if available."""
        key = self._get_hash_key(text, model)
        return self._cache.get(key)

    def set(self, text: str, model: str, result: EmbeddingResult):
        """Cache an embedding result."""
        key = self._get_hash_key(text, model)
        self._cache[key] = result

        # Periodically save cache (every 100 items)
        if len(self._cache) % 100 == 0:
            self._save_cache()

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        cache_file = self.cache_dir / "embeddings_cache.npz"
        if cache_file.exists():
            cache_file.unlink()
        logger.info("Cache cleared")


class EmbeddingGeneratorPipeline:
    """
    Complete pipeline for generating embeddings with caching and batch processing.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        batch_size: int = 20,
        use_cache: bool = True,
        cache_dir: str = "./data/embeddings/cache"
    ):
        """
        Initialize embedding pipeline.

        Args:
            model: OpenAI model name
            api_key: OpenAI API key
            batch_size: Batch size for embedding generation
            use_cache: Whether to use embedding cache
            cache_dir: Directory for cache storage
        """
        self.generator = OpenAIEmbeddingGenerator(
            model=model,
            api_key=api_key,
            batch_size=batch_size
        )

        self.use_cache = use_cache
        self.cache = EmbeddingCache(cache_dir) if use_cache else None
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_embeddings": 0,
            "total_tokens": 0,
            "total_cost": 0.0
        }

    def generate_embeddings(
        self,
        chunks: List[Union[str, Dict[str, Any]]],
        show_progress: bool = True
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings with caching support.

        Args:
            chunks: List of chunks to embed
            show_progress: Whether to show progress bar

        Returns:
            List of embedding results
        """
        results = []
        chunks_to_embed = []

        # Check cache first
        if self.use_cache:
            for chunk in chunks:
                if isinstance(chunk, str):
                    text = chunk
                    metadata = {}
                else:
                    text = chunk.get("text", "")
                    metadata = chunk.get("metadata", {})

                cached = self.cache.get(text, self.generator.model)
                if cached:
                    # Update metadata
                    cached.metadata.update(metadata)
                    results.append(cached)
                    self.stats["cache_hits"] += 1
                else:
                    chunks_to_embed.append({"text": text, "metadata": metadata})
                    self.stats["cache_misses"] += 1
        else:
            chunks_to_embed = chunks

        # Generate embeddings for uncached chunks
        if chunks_to_embed:
            new_embeddings = self.generator.generate_embeddings(chunks_to_embed, show_progress)

            # Cache and add to results
            for emb in new_embeddings:
                if self.use_cache:
                    self.cache.set(emb.text, self.generator.model, emb)
                results.append(emb)

            # Update stats
            self.stats["total_embeddings"] += len(new_embeddings)
            self.stats["total_tokens"] += sum(emb.tokens_used for emb in new_embeddings)
            self.stats["total_cost"] += self.generator.estimate_cost(
                sum(emb.tokens_used for emb in new_embeddings)
            )

        logger.info(
            f"Embedding generation complete: {len(results)} total, "
            f"{self.stats['cache_hits']} cache hits, "
            f"{self.stats['cache_misses']} cache misses"
        )

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            **self.stats,
            "model_info": self.generator.get_model_info(),
            "cache_enabled": self.use_cache
        }

    def save_embeddings(self, embeddings: List[EmbeddingResult], filepath: str):
        """Save embeddings to disk."""
        import pickle

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(embeddings, f)

        logger.info(f"Saved {len(embeddings)} embeddings to {filepath}")

    def load_embeddings(self, filepath: str) -> List[EmbeddingResult]:
        """Load embeddings from disk."""
        import pickle

        with open(filepath, 'rb') as f:
            embeddings = pickle.load(f)

        logger.info(f"Loaded {len(embeddings)} embeddings from {filepath}")
        return embeddings


# Convenience function
def generate_embeddings(
    texts: List[str],
    model: str = "text-embedding-3-small",
    api_key: Optional[str] = None,
    batch_size: int = 20,
    use_cache: bool = True
) -> List[List[float]]:
    """
    Quick helper function to generate embeddings for a list of texts.

    Args:
        texts: List of text strings to embed
        model: OpenAI embedding model
        api_key: OpenAI API key
        batch_size: Batch size for processing
        use_cache: Whether to use cache

    Returns:
        List of embedding vectors
    """
    pipeline = EmbeddingGeneratorPipeline(
        model=model,
        api_key=api_key,
        batch_size=batch_size,
        use_cache=use_cache
    )

    chunks = [{"text": text, "metadata": {}} for text in texts]
    results = pipeline.generate_embeddings(chunks, show_progress=False)

    return [result.embedding for result in results]


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Test single embedding
    generator = OpenAIEmbeddingGenerator()
    result = generator.generate_embedding("Hello, world!")
    print(f"Embedding dimension: {len(result.embedding)}")
    print(f"Tokens used: {result.tokens_used}")

    # Test batch embedding
    texts = [
        "This is the first document.",
        "This is the second document.",
        "And a third one for good measure."
    ]
    results = generator.generate_embeddings(texts)
    for i, r in enumerate(results):
        print(f"Document {i}: {len(r.embedding)} dimensions")
