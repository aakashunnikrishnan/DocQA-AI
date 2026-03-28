"""
Tests for configuration module.
"""

import pytest
import os
import yaml
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from src.utils.config import (
    Config, ConfigManager, get_config, get_config_manager,
    LLMConfig, EmbeddingConfig, VectorStoreConfig, RetrievalConfig
)


class TestConfig:
    """Tests for Config class."""

    def test_default_config(self):
        """Test default configuration."""
        config = Config()
        assert config.environment == "development"
        assert config.debug is True
        assert config.llm.model == "gpt-4"
        assert config.embedding.model == "text-embedding-3-small"
        assert config.retrieval.top_k == 5

    def test_config_to_dict(self):
        """Test conversion to dictionary."""
        config = Config()
        config_dict = config.to_dict()

        assert "environment" in config_dict
        assert "llm" in config_dict
        assert "embedding" in config_dict
        assert config_dict["environment"] == "development"

    def test_config_from_dict(self):
        """Test updating from dictionary."""
        config = Config()
        config.from_dict({
            "environment": "production",
            "debug": False,
            "llm": {"model": "gpt-4o"}
        })

        assert config.environment == "production"
        assert config.debug is False
        assert config.llm.model == "gpt-4o"

    def test_validate_config(self):
        """Test configuration validation."""
        config = Config()
        issues = config.validate()

        # Default config should be valid
        assert len(issues) == 0

    def test_validate_invalid_environment(self):
        """Test validation with invalid environment."""
        config = Config()
        config.environment = "invalid"

        issues = config.validate()
        assert len(issues) > 0
        assert "Invalid environment" in issues[0]


class TestConfigManager:
    """Tests for ConfigManager."""

    def test_init(self):
        """Test config manager initialization."""
        manager = ConfigManager()
        assert manager.config is not None

    def test_load_from_file_yaml(self, temp_dir):
        """Test loading from YAML file."""
        config_path = temp_dir / "config.yaml"
        config_data = {
            "environment": "staging",
            "llm": {"model": "gpt-4o-mini", "temperature": 0.5}
        }

        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)

        manager = ConfigManager(str(config_path))
        assert manager.config.environment == "staging"
        assert manager.config.llm.model == "gpt-4o-mini"
        assert manager.config.llm.temperature == 0.5

    def test_load_from_file_json(self, temp_dir):
        """Test loading from JSON file."""
        config_path = temp_dir / "config.json"
        config_data = {
            "environment": "staging",
            "llm": {"model": "gpt-4o-mini", "temperature": 0.5}
        }

        with open(config_path, 'w') as f:
            json.dump(config_data, f)

        manager = ConfigManager(str(config_path))
        assert manager.config.environment == "staging"

    def test_load_from_env(self, monkeypatch):
        """Test loading from environment variables."""
        monkeypatch.setenv("DOCQA_ENVIRONMENT", "production")
        monkeypatch.setenv("DOCQA_LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("DOCQA_RETRIEVAL_TOP_K", "10")

        manager = ConfigManager()
        manager._load_from_env()

        assert manager.config.environment == "production"
        assert manager.config.llm.model == "gpt-4o"
        assert manager.config.retrieval.top_k == 10

    def test_get(self):
        """Test getting configuration values."""
        manager = ConfigManager()

        assert manager.get("environment") == "development"
        assert manager.get("llm.model") == "gpt-4"
        assert manager.get("nonexistent", "default") == "default"

    def test_set(self):
        """Test setting configuration values."""
        manager = ConfigManager()
        manager.set("llm.model", "gpt-4o")

        assert manager.config.llm.model == "gpt-4o"


class TestLLMConfig:
    """Tests for LLMConfig."""

    def test_defaults(self):
        """Test default LLM configuration."""
        config = LLMConfig()
        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.temperature == 0.7
        assert config.max_tokens == 2000

    def test_custom_values(self):
        """Test custom LLM configuration."""
        config = LLMConfig(
            provider="anthropic",
            model="claude-3-haiku",
            temperature=0.3,
            max_tokens=1000
        )

        assert config.provider == "anthropic"
        assert config.model == "claude-3-haiku"
        assert config.temperature == 0.3
        assert config.max_tokens == 1000
