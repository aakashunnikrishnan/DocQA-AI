#!/usr/bin/env python3
"""
Model download script for DocQA AI system.
Supports downloading:
- LLM models (Llama 2, Mistral, Phi, etc.) from Hugging Face
- Embedding models (sentence-transformers, OpenAI compatible)
- GGUF quantized models for llama-cpp
- Models from Ollama
- Custom model configurations
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


# ============================================================
# Model Definitions
# ============================================================

@dataclass
class ModelDefinition:
    """Model definition for download."""
    name: str
    source: str  # huggingface, ollama, local
    model_id: str
    filename: Optional[str] = None
    description: str = ""
    size_gb: float = 0.0
    requires_auth: bool = False
    default_quantization: Optional[str] = None
    additional_files: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    category: str = "llm"  # llm, embedding, reranker

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "source": self.source,
            "model_id": self.model_id,
            "filename": self.filename,
            "description": self.description,
            "size_gb": self.size_gb,
            "requires_auth": self.requires_auth,
            "default_quantization": self.default_quantization,
            "additional_files": self.additional_files,
            "tags": self.tags,
            "category": self.category
        }


# ============================================================
# Model Catalogs
# ============================================================

# LLM Models
LLM_MODELS = [
    # Llama 2 models
    ModelDefinition(
        name="llama-2-7b-chat",
        source="huggingface",
        model_id="meta-llama/Llama-2-7b-chat-hf",
        description="Llama 2 7B chat model (requires auth)",
        size_gb=13.5,
        requires_auth=True,
        category="llm",
        tags=["llama2", "chat", "7b"]
    ),
    ModelDefinition(
        name="llama-2-13b-chat",
        source="huggingface",
        model_id="meta-llama/Llama-2-13b-chat-hf",
        description="Llama 2 13B chat model (requires auth)",
        size_gb=26.0,
        requires_auth=True,
        category="llm",
        tags=["llama2", "chat", "13b"]
    ),
    ModelDefinition(
        name="llama-2-70b-chat",
        source="huggingface",
        model_id="meta-llama/Llama-2-70b-chat-hf",
        description="Llama 2 70B chat model (requires auth)",
        size_gb=140.0,
        requires_auth=True,
        category="llm",
        tags=["llama2", "chat", "70b"]
    ),

    # Llama 2 GGUF (for llama-cpp)
    ModelDefinition(
        name="llama-2-7b-chat-gguf",
        source="huggingface",
        model_id="TheBloke/Llama-2-7B-Chat-GGUF",
        filename="llama-2-7b-chat.Q4_K_M.gguf",
        description="Llama 2 7B chat GGUF (4-bit quantized)",
        size_gb=4.0,
        default_quantization="q4_k_m",
        category="llm",
        tags=["llama2", "chat", "7b", "gguf", "quantized"]
    ),
    ModelDefinition(
        name="llama-2-13b-chat-gguf",
        source="huggingface",
        model_id="TheBloke/Llama-2-13B-Chat-GGUF",
        filename="llama-2-13b-chat.Q4_K_M.gguf",
        description="Llama 2 13B chat GGUF (4-bit quantized)",
        size_gb=7.5,
        default_quantization="q4_k_m",
        category="llm",
        tags=["llama2", "chat", "13b", "gguf", "quantized"]
    ),

    # Mistral models
    ModelDefinition(
        name="mistral-7b-instruct",
        source="huggingface",
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
        description="Mistral 7B Instruct v0.2",
        size_gb=15.0,
        category="llm",
        tags=["mistral", "instruct", "7b"]
    ),
    ModelDefinition(
        name="mistral-7b-instruct-gguf",
        source="huggingface",
        model_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        description="Mistral 7B Instruct GGUF (4-bit quantized)",
        size_gb=4.2,
        default_quantization="q4_k_m",
        category="llm",
        tags=["mistral", "instruct", "7b", "gguf", "quantized"]
    ),

    # Phi models
    ModelDefinition(
        name="phi-2",
        source="huggingface",
        model_id="microsoft/phi-2",
        description="Microsoft Phi-2 (2.7B parameters)",
        size_gb=5.0,
        category="llm",
        tags=["phi", "microsoft", "2.7b"]
    ),
    ModelDefinition(
        name="phi-3-mini",
        source="huggingface",
        model_id="microsoft/Phi-3-mini-4k-instruct",
        description="Microsoft Phi-3 Mini (3.8B parameters)",
        size_gb=7.0,
        category="llm",
        tags=["phi3", "microsoft", "3.8b"]
    ),

    # Gemma models
    ModelDefinition(
        name="gemma-2b",
        source="huggingface",
        model_id="google/gemma-2b-it",
        description="Google Gemma 2B Instruct",
        size_gb=5.0,
        category="llm",
        tags=["gemma", "google", "2b"]
    ),
    ModelDefinition(
        name="gemma-7b",
        source="huggingface",
        model_id="google/gemma-7b-it",
        description="Google Gemma 7B Instruct",
        size_gb=16.0,
        category="llm",
        tags=["gemma", "google", "7b"]
    ),

    # Zephyr models
    ModelDefinition(
        name="zephyr-7b-beta",
        source="huggingface",
        model_id="HuggingFaceH4/zephyr-7b-beta",
        description="Zephyr 7B Beta (fine-tuned Mistral)",
        size_gb=14.0,
        category="llm",
        tags=["zephyr", "mistral", "7b"]
    ),
]

# Embedding Models
EMBEDDING_MODELS = [
    ModelDefinition(
        name="all-MiniLM-L6-v2",
        source="huggingface",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        description="MiniLM L6 v2 (384 dims, fast)",
        size_gb=0.5,
        category="embedding",
        tags=["sentence-transformers", "mini", "384d"]
    ),
    ModelDefinition(
        name="all-mpnet-base-v2",
        source="huggingface",
        model_id="sentence-transformers/all-mpnet-base-v2",
        description="MPNet base v2 (768 dims, good quality)",
        size_gb=1.2,
        category="embedding",
        tags=["sentence-transformers", "mpnet", "768d"]
    ),
    ModelDefinition(
        name="all-distilroberta-v1",
        source="huggingface",
        model_id="sentence-transformers/all-distilroberta-v1",
        description="DistilRoBERTa v1 (768 dims)",
        size_gb=1.0,
        category="embedding",
        tags=["sentence-transformers", "distilroberta", "768d"]
    ),
    ModelDefinition(
        name="gte-small",
        source="huggingface",
        model_id="thenlper/gte-small",
        description="GTE Small (384 dims, excellent quality)",
        size_gb=0.8,
        category="embedding",
        tags=["gte", "384d"]
    ),
    ModelDefinition(
        name="gte-base",
        source="huggingface",
        model_id="thenlper/gte-base",
        description="GTE Base (768 dims)",
        size_gb=1.5,
        category="embedding",
        tags=["gte", "768d"]
    ),
    ModelDefinition(
        name="text-embedding-3-small",
        source="openai",
        model_id="text-embedding-3-small",
        description="OpenAI text-embedding-3-small (1536 dims)",
        size_gb=0.0,  # API model
        category="embedding",
        tags=["openai", "api"]
    ),
    ModelDefinition(
        name="text-embedding-3-large",
        source="openai",
        model_id="text-embedding-3-large",
        description="OpenAI text-embedding-3-large (3072 dims)",
        size_gb=0.0,  # API model
        category="embedding",
        tags=["openai", "api"]
    ),
]

# Reranker Models
RERANKER_MODELS = [
    ModelDefinition(
        name="ms-marco-MiniLM-L-6-v2",
        source="huggingface",
        model_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="MiniLM cross-encoder reranker",
        size_gb=0.5,
        category="reranker",
        tags=["cross-encoder", "minilm", "marco"]
    ),
    ModelDefinition(
        name="ms-marco-MiniLM-L-12-v2",
        source="huggingface",
        model_id="cross-encoder/ms-marco-MiniLM-L-12-v2",
        description="MiniLM L12 cross-encoder reranker",
        size_gb=1.0,
        category="reranker",
        tags=["cross-encoder", "minilm", "marco"]
    ),
    ModelDefinition(
        name="ms-marco-bert-base-v2",
        source="huggingface",
        model_id="cross-encoder/ms-marco-bert-base-v2",
        description="BERT base cross-encoder reranker",
        size_gb=1.5,
        category="reranker",
        tags=["cross-encoder", "bert", "marco"]
    ),
]

# Ollama Models
OLLAMA_MODELS = [
    ModelDefinition(
        name="llama2",
        source="ollama",
        model_id="llama2:7b-chat",
        description="Llama 2 7B chat (Ollama)",
        size_gb=3.8,
        category="llm",
        tags=["llama2", "ollama", "7b"]
    ),
    ModelDefinition(
        name="llama2:13b",
        source="ollama",
        model_id="llama2:13b-chat",
        description="Llama 2 13B chat (Ollama)",
        size_gb=7.3,
        category="llm",
        tags=["llama2", "ollama", "13b"]
    ),
    ModelDefinition(
        name="mistral",
        source="ollama",
        model_id="mistral:7b-instruct",
        description="Mistral 7B instruct (Ollama)",
        size_gb=4.1,
        category="llm",
        tags=["mistral", "ollama", "7b"]
    ),
    ModelDefinition(
        name="phi",
        source="ollama",
        model_id="phi:2.7b",
        description="Phi 2.7B (Ollama)",
        size_gb=2.0,
        category="llm",
        tags=["phi", "ollama"]
    ),
    ModelDefinition(
        name="gemma:2b",
        source="ollama",
        model_id="gemma:2b-instruct",
        description="Gemma 2B instruct (Ollama)",
        size_gb=1.8,
        category="llm",
        tags=["gemma", "ollama", "2b"]
    ),
    ModelDefinition(
        name="nomic-embed-text",
        source="ollama",
        model_id="nomic-embed-text",
        description="Nomic embedding model (Ollama)",
        size_gb=0.5,
        category="embedding",
        tags=["nomic", "ollama", "embedding"]
    ),
]

# All model catalog
MODEL_CATALOG = {
    "llm": LLM_MODELS,
    "embedding": EMBEDDING_MODELS,
    "reranker": RERANKER_MODELS,
    "ollama": OLLAMA_MODELS,
}

# All models for quick lookup
ALL_MODELS = []
for models in MODEL_CATALOG.values():
    ALL_MODELS.extend(models)


# ============================================================
# Model Downloader
# ============================================================

class ModelDownloader:
    """
    Download and manage models from various sources.
    """

    def __init__(
        self,
        models_dir: str = "./models",
        hf_token: Optional[str] = None,
        use_ollama: bool = True
    ):
        """
        Initialize model downloader.

        Args:
            models_dir: Directory to store models
            hf_token: Hugging Face token for private models
            use_ollama: Whether to use Ollama
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.hf_token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
        self.use_ollama = use_ollama

        # Check dependencies
        self._check_dependencies()

        logger.info(f"ModelDownloader initialized: models_dir={models_dir}")

    def _check_dependencies(self):
        """Check if required dependencies are installed."""
        self.hf_available = False
        self.ollama_available = False

        try:
            import huggingface_hub
            self.hf_available = True
        except ImportError:
            logger.warning("huggingface-hub not installed. Install with: pip install huggingface-hub")

        try:
            import ollama
            self.ollama_available = True
        except ImportError:
            logger.warning("ollama not installed. Install with: pip install ollama")

    def download_huggingface_model(
        self,
        model_id: str,
        filename: Optional[str] = None,
        revision: Optional[str] = None,
        token: Optional[str] = None
    ) -> bool:
        """
        Download model from Hugging Face.

        Args:
            model_id: Hugging Face model ID
            filename: Specific filename to download (for GGUF)
            revision: Git revision
            token: Hugging Face token

        Returns:
            Success status
        """
        if not self.hf_available:
            logger.error("huggingface-hub not available")
            return False

        try:
            from huggingface_hub import snapshot_download, hf_hub_download, login

            # Login if token provided
            if token or self.hf_token:
                login(token or self.hf_token)

            # Determine download path
            model_name = model_id.replace('/', '_')
            download_path = self.models_dir / model_name

            # Check if already downloaded
            if download_path.exists():
                logger.info(f"Model already exists: {download_path}")
                return True

            # Download specific file or full model
            if filename:
                logger.info(f"Downloading {filename} from {model_id}...")
                hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    local_dir=download_path,
                    revision=revision,
                    token=token or self.hf_token
                )
            else:
                logger.info(f"Downloading full model {model_id}...")
                snapshot_download(
                    repo_id=model_id,
                    local_dir=download_path,
                    revision=revision,
                    token=token or self.hf_token,
                    ignore_patterns=["*.safetensors", "*.bin"]  # Download all
                )

            logger.info(f"Successfully downloaded model: {download_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {e}")
            return False

    def download_ollama_model(self, model_id: str) -> bool:
        """
        Download model using Ollama.

        Args:
            model_id: Ollama model ID

        Returns:
            Success status
        """
        if not self.ollama_available:
            logger.error("Ollama not available")
            return False

        try:
            import ollama

            # Check if already downloaded
            response = ollama.list()
            existing_models = [m['name'] for m in response.get('models', [])]

            # Check if model exists
            model_name = model_id.split(':')[0] if ':' in model_id else model_id
            model_tag = model_id.split(':')[1] if ':' in model_id else 'latest'

            for existing in existing_models:
                if existing.startswith(model_name):
                    logger.info(f"Model {model_id} already exists in Ollama")
                    return True

            logger.info(f"Pulling model {model_id} from Ollama...")
            stream = ollama.pull(model_id, stream=True)

            for chunk in stream:
                if 'status' in chunk:
                    logger.info(f"  {chunk['status']}")

            logger.info(f"Successfully pulled model: {model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to pull model {model_id} from Ollama: {e}")
            return False

    def download_model(self, model_def: ModelDefinition) -> bool:
        """
        Download a model based on its definition.

        Args:
            model_def: ModelDefinition object

        Returns:
            Success status
        """
        logger.info(f"\nDownloading model: {model_def.name}")
        logger.info(f"  Source: {model_def.source}")
        logger.info(f"  Model ID: {model_def.model_id}")
        logger.info(f"  Size: {model_def.size_gb} GB")

        if model_def.source == "huggingface":
            return self.download_huggingface_model(
                model_def.model_id,
                model_def.filename,
                token=self.hf_token
            )
        elif model_def.source == "ollama":
            if not self.use_ollama:
                logger.warning("Ollama downloads disabled")
                return False
            return self.download_ollama_model(model_def.model_id)
        elif model_def.source == "openai":
            logger.info("OpenAI models are API-based, no download needed")
            return True
        else:
            logger.error(f"Unknown source: {model_def.source}")
            return False

    def download_models(
        self,
        model_names: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        skip_existing: bool = True
    ) -> Dict[str, bool]:
        """
        Download multiple models.

        Args:
            model_names: List of model names to download
            categories: Filter by categories
            tags: Filter by tags
            skip_existing: Skip already downloaded models

        Returns:
            Dictionary of model -> success status
        """
        # Select models
        models = ALL_MODELS

        if model_names:
            models = [m for m in models if m.name in model_names]

        if categories:
            models = [m for m in models if m.category in categories]

        if tags:
            models = [m for m in models if any(tag in m.tags for tag in tags)]

        if not models:
            logger.warning("No models selected for download")
            return {}

        logger.info(f"Selected {len(models)} models for download")

        results = {}
        for model in models:
            # Skip if already exists
            if skip_existing and self._model_exists(model):
                logger.info(f"Model {model.name} already exists, skipping")
                results[model.name] = True
                continue

            success = self.download_model(model)
            results[model.name] = success

            if not success:
                logger.warning(f"Failed to download model: {model.name}")

        return results

    def _model_exists(self, model_def: ModelDefinition) -> bool:
        """Check if a model already exists."""
        if model_def.source == "ollama":
            # Check Ollama
            try:
                import ollama
                response = ollama.list()
                existing_models = [m['name'] for m in response.get('models', [])]
                model_name = model_def.model_id.split(':')[0] if ':' in model_def.model_id else model_def.model_id
                return any(m.startswith(model_name) for m in existing_models)
            except:
                return False

        elif model_def.source == "huggingface":
            # Check local files
            model_name = model_def.model_id.replace('/', '_')
            model_path = self.models_dir / model_name

            if model_def.filename:
                return (model_path / model_def.filename).exists()
            else:
                return model_path.exists()

        elif model_def.source == "openai":
            # OpenAI models are always available
            return True

        return False

    def list_models(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available models.

        Args:
            category: Filter by category

        Returns:
            List of model dictionaries
        """
        models = ALL_MODELS
        if category:
            models = [m for m in models if m.category == category]

        return [m.to_dict() for m in models]

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        for model in ALL_MODELS:
            if model.name == model_name:
                return model.to_dict()
        return None

    def get_download_status(self) -> Dict[str, Any]:
        """Get download status of all models."""
        status = {}
        for model in ALL_MODELS:
            status[model.name] = {
                "exists": self._model_exists(model),
                "source": model.source,
                "size_gb": model.size_gb
            }
        return status


# ============================================================
# CLI Interface
# ============================================================

def main():
    """Main entry point for model download script."""
    parser = argparse.ArgumentParser(
        description="Download models for DocQA AI system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available models
  python download_models.py --list

  # Download a specific model
  python download_models.py --model llama-2-7b-chat-gguf

  # Download all LLM models
  python download_models.py --category llm

  # Download all embedding models
  python download_models.py --category embedding

  # Download models with specific tags
  python download_models.py --tags llama2,gguf

  # Download from Ollama
  python download_models.py --ollama --model llama2

  # Download all models
  python download_models.py --all
        """
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        action="append",
        help="Model name to download (can be used multiple times)"
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["llm", "embedding", "reranker", "ollama"],
        action="append",
        help="Category of models to download"
    )
    parser.add_argument(
        "--tags",
        type=str,
        help="Comma-separated tags to filter models"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available models"
    )
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Include Ollama models"
    )

    # Options
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Check download status of models"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
        help="Directory to store models (default: ./models)"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="Hugging Face token for private models"
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Don't skip already downloaded models"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if exists"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level, log_dir="./logs")

    # Initialize downloader
    downloader = ModelDownloader(
        models_dir=args.models_dir,
        hf_token=args.hf_token,
        use_ollama=args.ollama
    )

    # List models
    if args.list:
        print("\nAvailable Models:")
        print("=" * 60)

        for category in ["llm", "embedding", "reranker"]:
            print(f"\n{category.upper()} Models:")
            print("-" * 40)
            models = downloader.list_models(category)
            for model in models:
                status = "✓" if downloader._model_exists(
                    next(m for m in ALL_MODELS if m.name == model["name"])
                ) else "✗"
                print(f"  {status} {model['name']} ({model['size_gb']:.1f}GB) - {model['description']}")

        if args.ollama:
            print("\nOLLAMA Models:")
            print("-" * 40)
            for model in downloader.list_models("ollama"):
                status = "✓" if downloader._model_exists(
                    next(m for m in ALL_MODELS if m.name == model["name"])
                ) else "✗"
                print(f"  {status} {model['name']} ({model['size_gb']:.1f}GB) - {model['description']}")

        return

    # Check status
    if args.status:
        status = downloader.get_download_status()
        print("\nModel Download Status:")
        print("=" * 60)
        for name, info in status.items():
            icon = "✓" if info["exists"] else "✗"
            print(f"  {icon} {name}: {info['source']} ({info['size_gb']:.1f}GB)")
        return

    # Select models to download
    model_names = args.model or []
    categories = args.category or []

    if args.all:
        categories = ["llm", "embedding", "reranker"]
        if args.ollama:
            categories.append("ollama")

    if not model_names and not categories and not args.all:
        parser.error("Please specify models to download (--model, --category, --all, or --ollama)")

    # Parse tags
    tags = args.tags.split(",") if args.tags else []

    # Download models
    skip_existing = not args.no_skip and not args.force

    results = downloader.download_models(
        model_names=model_names,
        categories=categories,
        tags=tags,
        skip_existing=skip_existing
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Download Summary:")
    print("=" * 60)

    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)

    for name, success in results.items():
        icon = "✓" if success else "✗"
        print(f"  {icon} {name}")

    print(f"\nSuccessfully downloaded: {success_count}/{total_count}")

    if success_count < total_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
