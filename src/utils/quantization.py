"""
Model quantization support for DocQA AI system.
Provides quantization utilities for LLMs, embeddings, and other models.
Supports multiple quantization methods: GPTQ, AWQ, GGUF, BitsAndBytes, etc.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import subprocess
import tempfile
import shutil

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Try importing quantization libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("torch not installed. Install with: pip install torch")

try:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed. Install with: pip install transformers")

try:
    import bitsandbytes as bnb
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    logger.warning("bitsandbytes not installed. Install with: pip install bitsandbytes")

try:
    from optimum.gptq import GPTQQuantizer, load_quantized_model
    OPTIMUM_AVAILABLE = True
except ImportError:
    OPTIMUM_AVAILABLE = False
    logger.warning("optimum not installed. Install with: pip install optimum")

try:
    from awq import AutoAWQForCausalLM
    from awq.quantize import quantize
    AWQ_AVAILABLE = True
except ImportError:
    AWQ_AVAILABLE = False
    logger.warning("awq not installed. Install with: pip install awq")

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not installed. Install with: pip install llama-cpp-python")

try:
    import gguf
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False
    logger.warning("gguf not installed. Install with: pip install gguf")


class QuantizationMethod(Enum):
    """Quantization methods."""
    NONE = "none"
    BITS_AND_BYTES_8BIT = "bnb_8bit"
    BITS_AND_BYTES_4BIT = "bnb_4bit"
    GPTQ = "gptq"
    AWQ = "awq"
    GGUF = "gguf"
    GGML = "ggml"
    Q4_0 = "q4_0"
    Q4_K_M = "q4_k_m"
    Q5_0 = "q5_0"
    Q5_K_M = "q5_k_m"
    Q6_K = "q6_k"
    Q8_0 = "q8_0"


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    method: QuantizationMethod
    bits: int = 4
    group_size: int = 128
    desc_act: bool = False
    sym: bool = True
    true_sequential: bool = True
    use_cuda_fp16: bool = True
    use_cpu: bool = False
    model_name: Optional[str] = None
    output_dir: Optional[str] = None
    calibration_dataset: Optional[List[str]] = None
    calibration_samples: int = 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method.value,
            "bits": self.bits,
            "group_size": self.group_size,
            "desc_act": self.desc_act,
            "sym": self.sym,
            "true_sequential": self.true_sequential,
            "use_cuda_fp16": self.use_cuda_fp16,
            "use_cpu": self.use_cpu,
            "model_name": self.model_name,
            "output_dir": self.output_dir,
            "calibration_samples": self.calibration_samples
        }


@dataclass
class QuantizedModel:
    """Quantized model result."""
    model: Any
    tokenizer: Any
    config: QuantizationConfig
    size_mb: float
    original_size_mb: float
    compression_ratio: float
    method: str
    success: bool
    error: Optional[str] = None


class QuantizationManager:
    """
    Manager for quantizing models using various methods.
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize quantization manager.

        Args:
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device
        self._cache = {}

        # Check available methods
        self.available_methods = []
        if BITSANDBYTES_AVAILABLE and TORCH_AVAILABLE:
            self.available_methods.extend([
                QuantizationMethod.BITS_AND_BYTES_8BIT,
                QuantizationMethod.BITS_AND_BYTES_4BIT
            ])
        if OPTIMUM_AVAILABLE:
            self.available_methods.append(QuantizationMethod.GPTQ)
        if AWQ_AVAILABLE:
            self.available_methods.append(QuantizationMethod.AWQ)
        if GGUF_AVAILABLE or LLAMA_CPP_AVAILABLE:
            self.available_methods.extend([
                QuantizationMethod.GGUF,
                QuantizationMethod.Q4_0,
                QuantizationMethod.Q4_K_M,
                QuantizationMethod.Q5_0,
                QuantizationMethod.Q5_K_M,
                QuantizationMethod.Q6_K,
                QuantizationMethod.Q8_0
            ])

        logger.info(f"QuantizationManager initialized with methods: {[m.value for m in self.available_methods]}")

    def can_quantize(self, method: QuantizationMethod) -> bool:
        """Check if a quantization method is available."""
        return method in self.available_methods

    def quantize_with_bitsandbytes(
        self,
        model_name: str,
        config: QuantizationConfig,
        **kwargs
    ) -> QuantizedModel:
        """
        Quantize model using bitsandbytes.

        Args:
            model_name: Model name or path
            config: Quantization configuration
            **kwargs: Additional arguments

        Returns:
            QuantizedModel object
        """
        if not BITSANDBYTES_AVAILABLE or not TRANSFORMERS_AVAILABLE:
            return QuantizedModel(
                model=None,
                tokenizer=None,
                config=config,
                size_mb=0,
                original_size_mb=0,
                compression_ratio=0,
                method=config.method.value,
                success=False,
                error="bitsandbytes or transformers not available"
            )

        try:
            logger.info(f"Quantizing {model_name} with bitsandbytes (bits={config.bits})")

            # Configure quantization
            if config.method == QuantizationMethod.BITS_AND_BYTES_8BIT:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=True
                )
            else:  # 4-bit
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )

            # Load quantized model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )

            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

            # Calculate size
            size_mb = self._estimate_model_size(model)
            original_size_mb = size_mb / (config.bits / 16)  # Estimate original size

            return QuantizedModel(
                model=model,
                tokenizer=tokenizer,
                config=config,
                size_mb=size_mb,
                original_size_mb=original_size_mb,
                compression_ratio=original_size_mb / size_mb if size_mb > 0 else 0,
                method=config.method.value,
                success=True
            )

        except Exception as e:
            logger.error(f"BitsAndBytes quantization failed: {e}")
            return QuantizedModel(
                model=None,
                tokenizer=None,
                config=config,
                size_mb=0,
                original_size_mb=0,
                compression_ratio=0,
                method=config.method.value,
                success=False,
                error=str(e)
            )

    def quantize_with_gptq(
        self,
        model_name: str,
        config: QuantizationConfig,
        **kwargs
    ) -> QuantizedModel:
        """
        Quantize model using GPTQ.

        Args:
            model_name: Model name or path
            config: Quantization configuration
            **kwargs: Additional arguments

        Returns:
            QuantizedModel object
        """
        if not OPTIMUM_AVAILABLE:
            return QuantizedModel(
                model=None,
                tokenizer=None,
                config=config,
                size_mb=0,
                original_size_mb=0,
                compression_ratio=0,
                method=config.method.value,
                success=False,
                error="optimum not available"
            )

        try:
            logger.info(f"Quantizing {model_name} with GPTQ (bits={config.bits})")

            # Create output directory
            output_dir = config.output_dir or f"./models/quantized/{model_name.replace('/', '_')}_gptq"
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

            # Prepare calibration dataset
            calibration_data = self._prepare_calibration_data(
                config.calibration_dataset,
                tokenizer,
                config.calibration_samples
            )

            # Configure GPTQ
            from optimum.gptq import GPTQQuantizer

            quantizer = GPTQQuantizer(
                bits=config.bits,
                group_size=config.group_size,
                desc_act=config.desc_act,
                sym=config.sym,
                true_sequential=config.true_sequential,
                use_cuda_fp16=config.use_cuda_fp16
            )

            # Quantize model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )

            # Quantize
            quantized_model = quantizer.quantize_model(
                model,
                tokenizer,
                calibration_data
            )

            # Save quantized model
            quantizer.save(quantized_model, output_dir)

            # Calculate size
            size_mb = self._get_directory_size(output_dir)
            original_size_mb = self._get_original_size(model_name)

            return QuantizedModel(
                model=quantized_model,
                tokenizer=tokenizer,
                config=config,
                size_mb=size_mb,
                original_size_mb=original_size_mb,
                compression_ratio=original_size_mb / size_mb if size_mb > 0 else 0,
                method=config.method.value,
                success=True
            )

        except Exception as e:
            logger.error(f"GPTQ quantization failed: {e}")
            return QuantizedModel(
                model=None,
                tokenizer=None,
                config=config,
                size_mb=0,
                original_size_mb=0,
                compression_ratio=0,
                method=config.method.value,
                success=False,
                error=str(e)
            )

    def quantize_with_awq(
        self,
        model_name: str,
        config: QuantizationConfig,
        **kwargs
    ) -> QuantizedModel:
        """
        Quantize model using AWQ.

        Args:
            model_name: Model name or path
            config: Quantization configuration
            **kwargs: Additional arguments

        Returns:
            QuantizedModel object
        """
        if not AWQ_AVAILABLE:
            return QuantizedModel(
                model=None,
                tokenizer=None,
                config=config,
                size_mb=0,
                original_size_mb=0,
                compression_ratio=0,
                method=config.method.value,
                success=False,
                error="awq not available"
            )

        try:
            logger.info(f"Quantizing {model_name} with AWQ (bits={config.bits})")

            # Create output directory
            output_dir = config.output_dir or f"./models/quantized/{model_name.replace('/', '_')}_awq"
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Load model
            model = AutoAWQForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

            # Quantize
            quantize(
                model=model,
                tokenizer=tokenizer,
                quant_config={
                    "zero_point": True,
                    "q_group_size": config.group_size,
                    "w_bit": config.bits,
                    "version": "GEMM"
                },
                save_dir=output_dir
            )

            # Calculate size
            size_mb = self._get_directory_size(output_dir)
            original_size_mb = self._get_original_size(model_name)

            return QuantizedModel(
                model=model,
                tokenizer=tokenizer,
                config=config,
                size_mb=size_mb,
                original_size_mb=original_size_mb,
                compression_ratio=original_size_mb / size_mb if size_mb > 0 else 0,
                method=config.method.value,
                success=True
            )

        except Exception as e:
            logger.error(f"AWQ quantization failed: {e}")
            return QuantizedModel(
                model=None,
                tokenizer=None,
                config=config,
                size_mb=0,
                original_size_mb=0,
                compression_ratio=0,
                method=config.method.value,
                success=False,
                error=str(e)
            )

    def quantize_with_gguf(
        self,
        model_path: str,
        config: QuantizationConfig,
        **kwargs
    ) -> QuantizedModel:
        """
        Convert model to GGUF format.

        Args:
            model_path: Path to model (GGML or safetensors)
            config: Quantization configuration
            **kwargs: Additional arguments

        Returns:
            QuantizedModel object
        """
        if not LLAMA_CPP_AVAILABLE:
            return QuantizedModel(
                model=None,
                tokenizer=None,
                config=config,
                size_mb=0,
                original_size_mb=0,
                compression_ratio=0,
                method=config.method.value,
                success=False,
                error="llama-cpp-python not available"
            )

        try:
            logger.info(f"Converting {model_path} to GGUF (method={config.method.value})")

            # Determine quantization type
            qtype_map = {
                QuantizationMethod.Q4_0: "q4_0",
                QuantizationMethod.Q4_K_M: "q4_k_m",
                QuantizationMethod.Q5_0: "q5_0",
                QuantizationMethod.Q5_K_M: "q5_k_m",
                QuantizationMethod.Q6_K: "q6_k",
                QuantizationMethod.Q8_0: "q8_0",
                QuantizationMethod.GGUF: "q4_k_m"  # Default
            }

            qtype = qtype_map.get(config.method, "q4_k_m")

            # Use llama-cpp's convert script
            output_path = config.output_dir or f"./models/quantized/{Path(model_path).stem}_{qtype}.gguf"
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert using llama-cpp
            self._run_llama_convert(model_path, str(output_path), qtype)

            # Load quantized model
            model = Llama(
                model_path=str(output_path),
                n_ctx=4096,
                n_gpu_layers=-1 if self.device == "cuda" else 0,
                verbose=False
            )

            # Calculate size
            size_mb = output_path.stat().st_size / (1024 * 1024)
            original_size_mb = self._get_original_size(model_path)

            return QuantizedModel(
                model=model,
                tokenizer=None,  # llama-cpp handles tokenization internally
                config=config,
                size_mb=size_mb,
                original_size_mb=original_size_mb,
                compression_ratio=original_size_mb / size_mb if size_mb > 0 else 0,
                method=config.method.value,
                success=True
            )

        except Exception as e:
            logger.error(f"GGUF conversion failed: {e}")
            return QuantizedModel(
                model=None,
                tokenizer=None,
                config=config,
                size_mb=0,
                original_size_mb=0,
                compression_ratio=0,
                method=config.method.value,
                success=False,
                error=str(e)
            )

    def quantize_model(
        self,
        model_name: str,
        method: Union[str, QuantizationMethod],
        config: Optional[QuantizationConfig] = None,
        **kwargs
    ) -> QuantizedModel:
        """
        Quantize a model using the specified method.

        Args:
            model_name: Model name or path
            method: Quantization method
            config: Quantization configuration
            **kwargs: Additional arguments

        Returns:
            QuantizedModel object
        """
        if isinstance(method, str):
            method = QuantizationMethod(method)

        if not self.can_quantize(method):
            return QuantizedModel(
                model=None,
                tokenizer=None,
                config=config or QuantizationConfig(method=method),
                size_mb=0,
                original_size_mb=0,
                compression_ratio=0,
                method=method.value,
                success=False,
                error=f"Quantization method {method.value} not available"
            )

        # Create default config if not provided
        if config is None:
            config = QuantizationConfig(method=method, model_name=model_name, **kwargs)

        # Route to appropriate quantization method
        if method in [QuantizationMethod.BITS_AND_BYTES_8BIT, QuantizationMethod.BITS_AND_BYTES_4BIT]:
            return self.quantize_with_bitsandbytes(model_name, config, **kwargs)
        elif method == QuantizationMethod.GPTQ:
            return self.quantize_with_gptq(model_name, config, **kwargs)
        elif method == QuantizationMethod.AWQ:
            return self.quantize_with_awq(model_name, config, **kwargs)
        elif method in [QuantizationMethod.GGUF, QuantizationMethod.Q4_0, QuantizationMethod.Q4_K_M,
                        QuantizationMethod.Q5_0, QuantizationMethod.Q5_K_M, QuantizationMethod.Q6_K,
                        QuantizationMethod.Q8_0]:
            return self.quantize_with_gguf(model_name, config, **kwargs)
        else:
            return QuantizedModel(
                model=None,
                tokenizer=None,
                config=config,
                size_mb=0,
                original_size_mb=0,
                compression_ratio=0,
                method=method.value,
                success=False,
                error=f"Unsupported quantization method: {method.value}"
            )

    def _prepare_calibration_data(
        self,
        dataset: Optional[List[str]],
        tokenizer: Any,
        num_samples: int = 100
    ) -> List[str]:
        """
        Prepare calibration data for quantization.

        Args:
            dataset: Optional dataset
            tokenizer: Tokenizer for the model
            num_samples: Number of samples to use

        Returns:
            List of calibration texts
        """
        if dataset:
            return dataset[:num_samples]

        # Default calibration data
        default_samples = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing deals with text and language.",
            "Computer vision enables machines to understand images.",
            "Reinforcement learning involves agents learning through interaction.",
            "The transformer architecture uses attention mechanisms.",
            "Large language models are trained on massive text corpora.",
            "Fine-tuning adapts pre-trained models to specific tasks.",
            "Transfer learning applies knowledge from one domain to another.",
            "Generative AI creates new content based on training data."
        ]

        # Tokenize and return
        return default_samples * (num_samples // len(default_samples) + 1)

    def _estimate_model_size(self, model: Any) -> float:
        """Estimate model size in MB."""
        try:
            if hasattr(model, 'parameters'):
                param_count = sum(p.numel() for p in model.parameters())
                # Estimate size: 2 bytes per parameter for fp16, 4 for fp32
                bytes_per_param = 2 if model.dtype == torch.float16 else 4
                return (param_count * bytes_per_param) / (1024 * 1024)
        except Exception:
            pass

        return 0

    def _get_directory_size(self, path: str) -> float:
        """Get directory size in MB."""
        path = Path(path)
        total = 0
        for file in path.rglob('*'):
            if file.is_file():
                total += file.stat().st_size
        return total / (1024 * 1024)

    def _get_original_size(self, model_path: str) -> float:
        """Get original model size in MB."""
        if Path(model_path).exists():
            if Path(model_path).is_dir():
                return self._get_directory_size(model_path)
            else:
                return Path(model_path).stat().st_size / (1024 * 1024)
        return 0

    def _run_llama_convert(self, input_path: str, output_path: str, qtype: str):
        """Run llama-cpp conversion script."""
        # Check if convert script exists
        convert_script = shutil.which('convert.py')
        if not convert_script:
            # Try to find in common locations
            possible_paths = [
                '/usr/local/bin/convert.py',
                '/usr/bin/convert.py',
                './convert.py',
                './scripts/convert.py'
            ]
            for path in possible_paths:
                if Path(path).exists():
                    convert_script = path
                    break

        if not convert_script:
            # Use the llama-cpp Python API
            from llama_cpp import Llama
            # Load and save as GGUF
            model = Llama(
                model_path=input_path,
                n_ctx=4096,
                n_gpu_layers=-1 if self.device == "cuda" else 0,
                verbose=False
            )
            model.save(output_path)
            return

        # Run conversion script
        cmd = [
            'python', convert_script,
            '--input', input_path,
            '--output', output_path,
            '--quantize', qtype
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Conversion failed: {result.stderr}")


# ============================================================
# Convenience Functions
# ============================================================

def get_quantization_manager(device: str = "cuda") -> QuantizationManager:
    """Get quantization manager instance."""
    return QuantizationManager(device=device)


def quantize_model(
    model_name: str,
    method: str,
    bits: int = 4,
    output_dir: Optional[str] = None,
    device: str = "cuda",
    **kwargs
) -> QuantizedModel:
    """
    Quick function to quantize a model.

    Args:
        model_name: Model name or path
        method: Quantization method ('bnb_4bit', 'gptq', 'awq', 'gguf', 'q4_k_m', etc.)
        bits: Number of bits (for applicable methods)
        output_dir: Output directory
        device: Device to use
        **kwargs: Additional arguments

    Returns:
        QuantizedModel object
    """
    manager = QuantizationManager(device=device)
    config = QuantizationConfig(
        method=QuantizationMethod(method),
        bits=bits,
        output_dir=output_dir,
        **kwargs
    )
    return manager.quantize_model(model_name, method, config)


def get_available_quantization_methods() -> List[str]:
    """Get list of available quantization methods."""
    manager = QuantizationManager()
    return [m.value for m in manager.available_methods]


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Testing Quantization...")
    print("=" * 60)
    print(f"Available methods: {get_available_quantization_methods()}")

    # Example: Quantize with bitsandbytes
    # result = quantize_model(
    #     model_name="meta-llama/Llama-2-7b-chat-hf",
    #     method="bnb_4bit",
    #     bits=4
    # )
    # print(f"Quantization result: {result.success}")
    # print(f"Size: {result.size_mb:.2f} MB")
    # print(f"Compression ratio: {result.compression_ratio:.2f}x")

    print("\nQuantization module ready!")
