"""
Configuration management for DocQA AI system.
Handles loading, validation, and access to configuration from multiple sources.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field, asdict
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM settings."""
    provider: str = "openai"
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    timeout: int = 60
    max_retries: int = 3
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    organization: Optional[str] = None

    # Local LLM settings
    local_model_path: Optional[str] = None
    local_model_device: str = "cuda"
    quantize: bool = False


@dataclass
class EmbeddingConfig:
    """Configuration for embedding settings."""
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    dimension: int = 1536
    batch_size: int = 20
    max_tokens: int = 8191
    cache_enabled: bool = True
    cache_dir: str = "./data/embeddings/cache"
    api_key: Optional[str] = None
    api_base: Optional[str] = None


@dataclass
class VectorStoreConfig:
    """Configuration for vector store."""
    store_type: str = "faiss"  # faiss, chromadb, qdrant, pinecone
    index_type: str = "HNSW64"  # HNSW32, HNSW64, Flat, IVF
    index_path: str = "./data/embeddings/vector_index"
    dimension: int = 1536
    metric: str = "cosine"  # cosine, l2, ip
    ef_search: int = 100
    ef_construction: int = 200
    m: int = 16

    # For ChromaDB
    chroma_persist_dir: str = "./data/chromadb"

    # For Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "documents"

    # For Pinecone
    pinecone_api_key: Optional[str] = None
    pinecone_environment: str = "us-west1-gcp"
    pinecone_index: str = "doc-qa"


@dataclass
class RetrievalConfig:
    """Configuration for retrieval settings."""
    top_k: int = 5
    score_threshold: float = 0.7
    enable_hybrid_search: bool = True
    enable_reranking: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    similarity_weight: float = 0.7  # Weight for vector similarity vs keyword
    keyword_weight: float = 0.3
    mmr_diversity: float = 0.5  # For MMR reranking
    use_query_expansion: bool = False
    expansion_terms: int = 3


@dataclass
class APIConfig:
    """Configuration for API server."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    log_level: str = "INFO"
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_credentials: bool = True
    rate_limit: str = "100/minute"
    request_timeout: int = 30
    max_request_size: int = 100 * 1024 * 1024  # 100MB


@dataclass
class AuthConfig:
    """Configuration for authentication."""
    enabled: bool = False
    jwt_secret_key: Optional[str] = None
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    api_keys: List[str] = field(default_factory=list)


@dataclass
class DatabaseConfig:
    """Configuration for database."""
    url: str = "sqlite:///./docqa.db"
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False


@dataclass
class CacheConfig:
    """Configuration for caching."""
    enabled: bool = True
    redis_url: str = "redis://localhost:6379/0"
    ttl_seconds: int = 3600
    max_size: int = 1000


@dataclass
class MonitoringConfig:
    """Configuration for monitoring."""
    enabled: bool = True
    metrics_port: int = 9090
    enable_tracing: bool = False
    jaeger_agent_host: str = "localhost"
    jaeger_agent_port: int = 6831
    log_requests: bool = True
    log_responses: bool = False


@dataclass
class ProcessingConfig:
    """Configuration for document processing."""
    max_file_size_mb: int = 100
    supported_extensions: List[str] = field(default_factory=lambda: [
        ".pdf", ".docx", ".txt", ".md", ".html", ".csv", ".json"
    ])
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunking_strategy: str = "recursive"
    language: str = "en"
    preserve_formatting: bool = True
    extract_tables: bool = True
    extract_images: bool = False


@dataclass
class Config:
    """Main configuration class containing all sub-configurations."""
    # Environment
    environment: str = "development"  # development, staging, production
    debug: bool = True

    # Sub-configurations
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    api: APIConfig = field(default_factory=APIConfig)
    auth: AuthConfig = field(default_factory=AuthConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)

    # Additional custom settings
    custom: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire config to nested dictionary."""
        result = {
            "environment": self.environment,
            "debug": self.debug,
        }

        # Add all dataclass fields
        for field_name in dir(self):
            if field_name.startswith('_') or field_name in ['environment', 'debug', 'to_dict', 'from_dict']:
                continue

            value = getattr(self, field_name)
            if hasattr(value, '__dataclass_fields__'):
                result[field_name] = asdict(value)
            elif isinstance(value, dict):
                result[field_name] = value
            else:
                result[field_name] = value

        return result

    def from_dict(self, data: Dict[str, Any]) -> 'Config':
        """Update config from dictionary."""
        for key, value in data.items():
            if hasattr(self, key):
                current = getattr(self, key)
                if hasattr(current, '__dataclass_fields__') and isinstance(value, dict):
                    # Recursively update sub-config
                    for sub_key, sub_value in value.items():
                        if hasattr(current, sub_key):
                            setattr(current, sub_key, sub_value)
                else:
                    setattr(self, key, value)
            elif key in self.custom:
                self.custom[key] = value
            else:
                self.custom[key] = value

        return self

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Validate environment
        if self.environment not in ["development", "staging", "production"]:
            issues.append(f"Invalid environment: {self.environment}")

        # Validate LLM settings
        if self.llm.provider not in ["openai", "anthropic", "local", "azure"]:
            issues.append(f"Invalid LLM provider: {self.llm.provider}")

        if self.llm.temperature < 0 or self.llm.temperature > 2:
            issues.append(f"Temperature should be between 0 and 2, got {self.llm.temperature}")

        # Validate embedding settings
        if self.embedding.provider not in ["openai", "sentence-transformers", "local"]:
            issues.append(f"Invalid embedding provider: {self.embedding.provider}")

        if self.embedding.batch_size < 1 or self.embedding.batch_size > 100:
            issues.append(f"Batch size should be 1-100, got {self.embedding.batch_size}")

        # Validate vector store
        if self.vector_store.store_type not in ["faiss", "chromadb", "qdrant", "pinecone"]:
            issues.append(f"Invalid vector store type: {self.vector_store.store_type}")

        # Validate retrieval settings
        if self.retrieval.top_k < 1 or self.retrieval.top_k > 50:
            issues.append(f"top_k should be 1-50, got {self.retrieval.top_k}")

        if self.retrieval.score_threshold < 0 or self.retrieval.score_threshold > 1:
            issues.append(f"score_threshold should be 0-1, got {self.retrieval.score_threshold}")

        # Validate API settings
        if self.api.port < 1 or self.api.port > 65535:
            issues.append(f"Invalid port: {self.api.port}")

        if self.api.workers < 1 or self.api.workers > 32:
            issues.append(f"Workers should be 1-32, got {self.api.workers}")

        # Validate auth settings
        if self.auth.enabled:
            if not self.auth.jwt_secret_key and not self.auth.api_keys:
                issues.append("Authentication enabled but no JWT secret or API keys provided")

        return issues

    def get_llm_api_key(self) -> Optional[str]:
        """Get LLM API key from config or environment."""
        if self.llm.api_key:
            return self.llm.api_key

        # Check environment variables
        env_var_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "azure": "AZURE_OPENAI_API_KEY",
        }

        env_var = env_var_map.get(self.llm.provider)
        if env_var:
            return os.getenv(env_var)

        return None

    def get_embedding_api_key(self) -> Optional[str]:
        """Get embedding API key from config or environment."""
        if self.embedding.api_key:
            return self.embedding.api_key

        if self.embedding.provider == "openai":
            return os.getenv("OPENAI_API_KEY")

        return None


class ConfigManager:
    """
    Manages configuration loading from multiple sources with priority:
    1. Environment variables (DOCQA_*)
    2. Config files (YAML/JSON)
    3. Default values
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to config file (YAML or JSON)
        """
        self.config = Config()
        self.config_path = config_path
        self._loaded_from_file = False

        # Load configuration
        self.load()

    def load(self) -> Config:
        """
        Load configuration from all sources.

        Priority: Environment > Config file > Defaults
        """
        # Start with defaults (already set in Config)

        # Load from config file if provided
        if self.config_path:
            self._load_from_file(self.config_path)

        # Override with environment variables
        self._load_from_env()

        # Validate configuration
        issues = self.config.validate()
        if issues:
            logger.warning(f"Configuration validation issues: {issues}")
            if self.config.environment == "production":
                for issue in issues:
                    logger.error(f"Production config issue: {issue}")

        logger.info(f"Configuration loaded. Environment: {self.config.environment}")
        return self.config

    def _load_from_file(self, config_path: str):
        """Load configuration from YAML or JSON file."""
        path = Path(config_path)

        if not path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif path.suffix == '.json':
                    data = json.load(f)
                else:
                    logger.warning(f"Unsupported config file type: {path.suffix}")
                    return

            if data:
                self.config.from_dict(data)
                self._loaded_from_file = True
                logger.info(f"Loaded configuration from {config_path}")

        except Exception as e:
            logger.error(f"Failed to load config file {config_path}: {e}")

    def _load_from_env(self):
        """Load configuration from environment variables."""
        env_prefix = "DOCQA_"

        # Map environment variables to config paths
        env_mappings = {
            "ENVIRONMENT": ("environment", None),
            "DEBUG": ("debug", lambda x: x.lower() == 'true'),
            "LLM_PROVIDER": ("llm.provider", None),
            "LLM_MODEL": ("llm.model", None),
            "LLM_TEMPERATURE": ("llm.temperature", float),
            "LLM_MAX_TOKENS": ("llm.max_tokens", int),
            "EMBEDDING_MODEL": ("embedding.model", None),
            "EMBEDDING_DIMENSION": ("embedding.dimension", int),
            "EMBEDDING_BATCH_SIZE": ("embedding.batch_size", int),
            "VECTOR_STORE_TYPE": ("vector_store.store_type", None),
            "VECTOR_INDEX_PATH": ("vector_store.index_path", None),
            "RETRIEVAL_TOP_K": ("retrieval.top_k", int),
            "RETRIEVAL_SCORE_THRESHOLD": ("retrieval.score_threshold", float),
            "API_HOST": ("api.host", None),
            "API_PORT": ("api.port", int),
            "API_WORKERS": ("api.workers", int),
            "API_RATE_LIMIT": ("api.rate_limit", None),
            "CACHE_ENABLED": ("cache.enabled", lambda x: x.lower() == 'true'),
            "REDIS_URL": ("cache.redis_url", None),
            "LOG_LEVEL": ("api.log_level", None),
            "CHUNK_SIZE": ("processing.chunk_size", int),
            "CHUNK_OVERLAP": ("processing.chunk_overlap", int),
            "CHUNKING_STRATEGY": ("processing.chunking_strategy", None),
        }

        for env_var, (config_path, converter) in env_mappings.items():
            value = os.getenv(f"{env_prefix}{env_var}") or os.getenv(env_var)

            if value is not None:
                # Apply converter if provided
                if converter:
                    try:
                        value = converter(value)
                    except Exception as e:
                        logger.warning(f"Failed to convert {env_var}={value}: {e}")
                        continue

                # Set configuration value
                self._set_nested_config(config_path, value)

        # Handle API keys separately
        if os.getenv("OPENAI_API_KEY"):
            self.config.llm.api_key = os.getenv("OPENAI_API_KEY")
            if not self.config.embedding.api_key:
                self.config.embedding.api_key = os.getenv("OPENAI_API_KEY")

        if os.getenv("ANTHROPIC_API_KEY"):
            self.config.llm.api_key = os.getenv("ANTHROPIC_API_KEY")

    def _set_nested_config(self, path: str, value: Any):
        """Set nested configuration value using dot notation."""
        parts = path.split('.')
        obj = self.config

        for part in parts[:-1]:
            obj = getattr(obj, part, None)
            if obj is None:
                return

        if obj and hasattr(obj, parts[-1]):
            setattr(obj, parts[-1], value)

    def reload(self, config_path: Optional[str] = None):
        """Reload configuration."""
        if config_path:
            self.config_path = config_path

        self.load()

    def save(self, config_path: Optional[str] = None, format: str = "yaml"):
        """
        Save current configuration to file.

        Args:
            config_path: Path to save config file
            format: 'yaml' or 'json'
        """
        save_path = config_path or self.config_path

        if not save_path:
            raise ValueError("No config path specified for saving")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = self.config.to_dict()

        try:
            if format == "yaml":
                with open(save_path, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            elif format == "json":
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Configuration saved to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save config to {save_path}: {e}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-notated key.

        Args:
            key: Dot-notated key (e.g., 'llm.model')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        parts = key.split('.')
        obj = self.config

        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                return default

        return obj

    def set(self, key: str, value: Any):
        """Set configuration value by dot-notated key."""
        self._set_nested_config(key, value)

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.config.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.config.environment == "development"

    def get_env_specific_config(self) -> Dict[str, Any]:
        """Get environment-specific configuration overrides."""
        env_configs = {
            "development": {
                "debug": True,
                "api": {"reload": True, "log_level": "DEBUG"},
                "cache": {"enabled": False},
                "monitoring": {"enabled": False}
            },
            "staging": {
                "debug": False,
                "api": {"reload": False, "log_level": "INFO", "workers": 2},
                "cache": {"enabled": True, "ttl_seconds": 1800}
            },
            "production": {
                "debug": False,
                "api": {"reload": False, "log_level": "WARNING", "workers": 8},
                "cache": {"enabled": True, "ttl_seconds": 7200},
                "monitoring": {"enabled": True}
            }
        }

        return env_configs.get(self.config.environment, {})


# Global config instance
_config_manager: Optional[ConfigManager] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get global configuration instance (singleton).

    Args:
        config_path: Path to config file (only used on first call)

    Returns:
        Config instance
    """
    global _config_manager

    if _config_manager is None:
        _config_manager = ConfigManager(config_path)

    return _config_manager.config


def reload_config(config_path: Optional[str] = None):
    """Reload global configuration."""
    global _config_manager

    if _config_manager:
        _config_manager.reload(config_path)
    else:
        _config_manager = ConfigManager(config_path)


def get_config_manager() -> ConfigManager:
    """Get configuration manager instance."""
    global _config_manager

    if _config_manager is None:
        _config_manager = ConfigManager()

    return _config_manager


# Convenience functions
def get_llm_config() -> LLMConfig:
    """Get LLM configuration."""
    return get_config().llm


def get_embedding_config() -> EmbeddingConfig:
    """Get embedding configuration."""
    return get_config().embedding


def get_vector_store_config() -> VectorStoreConfig:
    """Get vector store configuration."""
    return get_config().vector_store


def get_retrieval_config() -> RetrievalConfig:
    """Get retrieval configuration."""
    return get_config().retrieval


def get_api_config() -> APIConfig:
    """Get API configuration."""
    return get_config().api


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Load default config
    config = get_config()
    print("Default configuration:")
    print(f"  Environment: {config.environment}")
    print(f"  LLM Model: {config.llm.model}")
    print(f"  Embedding Model: {config.embedding.model}")
    print(f"  Vector Store: {config.vector_store.store_type}")
    print(f"  Top K: {config.retrieval.top_k}")

    # Load from file
    config_manager = get_config_manager()

    # Access nested config
    print(f"\nLLM temperature: {config_manager.get('llm.temperature')}")

    # Validate
    issues = config.validate()
    if issues:
        print(f"\nValidation issues: {issues}")

    # Save example config
    # config_manager.save("config/example.yaml")

    print("\nConfiguration loaded successfully!")
