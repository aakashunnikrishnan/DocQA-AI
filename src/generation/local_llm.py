"""
Local LLM support for DocQA AI system.
Supports multiple local inference backends:
- Transformers (Hugging Face)
- llama-cpp-python (GGUF models)
- vLLM (high-performance inference)
- Ollama (local API)
- ExLlamaV2 (GPTQ models)
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional, Union, Iterator, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
import gc

import numpy as np

from src.utils.logger import get_logger
from src.generation.llm_interface import LLMResponse, Message

logger = get_logger(__name__)

# Try importing backends
try:
    import torch
    import transformers
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoConfig,
        pipeline,
        TextStreamer,
        GenerationConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed. Install with: pip install transformers torch")

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not installed. Install with: pip install llama-cpp-python")

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not installed. Install with: pip install vllm")

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("ollama not installed. Install with: pip install ollama")

try:
    import exllamav2
    EXLLAMA_AVAILABLE = True
except ImportError:
    EXLLAMA_AVAILABLE = False
    logger.warning("exllamav2 not installed")


class LocalLLMBackend(Enum):
    """Local LLM backends."""
    TRANSFORMERS = "transformers"
    LLAMA_CPP = "llama_cpp"
    VLLM = "vllm"
    OLLAMA = "ollama"
    EXLLAMA = "exllama"


class LocalLLMQuantization(Enum):
    """Quantization types for local LLMs."""
    NONE = "none"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    GGUF_Q4_0 = "q4_0"
    GGUF_Q4_K_M = "q4_k_m"
    GGUF_Q5_0 = "q5_0"
    GGUF_Q5_K_M = "q5_k_m"
    GGUF_Q6_K = "q6_k"
    GGUF_Q8_0 = "q8_0"
    GPTQ = "gptq"


@dataclass
class LocalLLMConfig:
    """Configuration for local LLM."""
    model_path: str
    backend: LocalLLMBackend
    model_name: str = ""
    quantization: LocalLLMQuantization = LocalLLMQuantization.NONE
    device: str = "cuda"
    device_map: Optional[str] = None
    max_memory: Optional[Dict[str, str]] = None
    max_length: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    num_beams: int = 1
    do_sample: bool = True
    batch_size: int = 8
    max_batch_size: int = 32
    trust_remote_code: bool = False
    use_cache: bool = True
    offload_folder: Optional[str] = None
    rope_scaling: Optional[Dict[str, Any]] = None

    # llama-cpp specific
    n_ctx: int = 4096
    n_threads: int = 4
    n_gpu_layers: int = -1  # -1 = all layers
    f16_kv: bool = True

    # vLLM specific
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_num_batched_tokens: int = 4096

    # Ollama specific
    ollama_host: str = "http://localhost:11434"


class LocalLLMInterface:
    """
    Unified interface for local LLM inference with multiple backends.
    """

    def __init__(self, config: LocalLLMConfig):
        """
        Initialize local LLM interface.

        Args:
            config: LocalLLMConfig object
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.backend = config.backend
        self._is_initialized = False
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)

        # Initialize backend
        self._init_backend()

        logger.info(f"LocalLLMInterface initialized: backend={config.backend.value}, "
                   f"model={config.model_path}")

    def _init_backend(self):
        """Initialize the selected backend."""
        if self.backend == LocalLLMBackend.TRANSFORMERS:
            self._init_transformers()
        elif self.backend == LocalLLMBackend.LLAMA_CPP:
            self._init_llama_cpp()
        elif self.backend == LocalLLMBackend.VLLM:
            self._init_vllm()
        elif self.backend == LocalLLMBackend.OLLAMA:
            self._init_ollama()
        elif self.backend == LocalLLMBackend.EXLLAMA:
            self._init_exllama()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        self._is_initialized = True

    def _init_transformers(self):
        """Initialize transformers backend."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not installed")

        try:
            # Device setup
            if self.config.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self.config.device = "cpu"

            # Load tokenizer
            logger.info(f"Loading tokenizer from {self.config.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=self.config.trust_remote_code,
                use_fast=True
            )

            # Add padding token if missing
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            logger.info(f"Loading model from {self.config.model_path}")

            # Determine device map
            if self.config.device_map is None:
                self.config.device_map = "auto" if self.config.device == "cuda" else None

            # Quantization config
            quantization_config = None
            if self.config.quantization == LocalLLMQuantization.INT8:
                quantization_config = transformers.BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
            elif self.config.quantization == LocalLLMQuantization.INT4:
                quantization_config = transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                trust_remote_code=self.config.trust_remote_code,
                device_map=self.config.device_map,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
                quantization_config=quantization_config,
                offload_folder=self.config.offload_folder,
                use_cache=self.config.use_cache
            )

            # Move to device if needed
            if self.config.device_map is None and self.config.device != "cpu":
                self.model = self.model.to(self.config.device)

            # Set model to evaluation mode
            self.model.eval()

            logger.info("Transformers model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to initialize transformers: {e}")
            raise

    def _init_llama_cpp(self):
        """Initialize llama-cpp-python backend."""
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python not installed")

        try:
            model_path = Path(self.config.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Determine GPU layers
            if self.config.device == "cuda" and torch.cuda.is_available():
                n_gpu_layers = self.config.n_gpu_layers
            else:
                n_gpu_layers = 0

            # Determine model format
            model_path_str = str(model_path)

            logger.info(f"Loading model from {model_path_str}")

            self.model = Llama(
                model_path=model_path_str,
                n_ctx=self.config.n_ctx,
                n_threads=self.config.n_threads,
                n_gpu_layers=n_gpu_layers,
                f16_kv=self.config.f16_kv,
                verbose=False,
                use_mlock=False,
                use_mmap=True
            )

            logger.info("llama-cpp model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to initialize llama-cpp: {e}")
            raise

    def _init_vllm(self):
        """Initialize vLLM backend."""
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM not installed")

        try:
            # vLLM expects model name or path
            model_name = self.config.model_path

            # Determine quantization
            quantization = None
            if self.config.quantization == LocalLLMQuantization.GPTQ:
                quantization = "gptq"

            logger.info(f"Loading vLLM model: {model_name}")

            from vllm import LLM, SamplingParams

            self.model = LLM(
                model=model_name,
                trust_remote_code=self.config.trust_remote_code,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_num_batched_tokens=self.config.max_num_batched_tokens,
                quantization=quantization,
                max_model_len=self.config.max_length
            )

            logger.info("vLLM model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to initialize vLLM: {e}")
            raise

    def _init_ollama(self):
        """Initialize Ollama backend."""
        if not OLLAMA_AVAILABLE:
            raise ImportError("ollama not installed")

        try:
            # Check if Ollama is running
            import requests
            try:
                response = requests.get(f"{self.config.ollama_host}/api/tags")
                if response.status_code == 200:
                    logger.info(f"Connected to Ollama at {self.config.ollama_host}")
                else:
                    logger.warning(f"Ollama server returned status {response.status_code}")
            except Exception as e:
                logger.warning(f"Failed to connect to Ollama: {e}")

            # Set model name
            self.model_name = self.config.model_path

            logger.info(f"Ollama initialized with model: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            raise

    def _init_exllama(self):
        """Initialize ExLlamaV2 backend."""
        if not EXLLAMA_AVAILABLE:
            raise ImportError("exllamav2 not installed")

        try:
            # ExLlamaV2 initialization
            # This is a simplified placeholder
            logger.info("ExLlamaV2 backend not fully implemented")

        except Exception as e:
            logger.error(f"Failed to initialize ExLlamaV2: {e}")
            raise

    def _prepare_messages_transformers(self, messages: List[Dict[str, str]]) -> str:
        """Prepare messages for transformers backend."""
        # Simple conversation formatting
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")

        return "\n".join(formatted) + "\nAssistant:"

    def _prepare_messages_llama_cpp(self, messages: List[Dict[str, str]]) -> str:
        """Prepare messages for llama-cpp backend."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                formatted.append(f"<<SYS>>\n{content}\n<</SYS>>")
            elif role == "user":
                formatted.append(f"[INST] {content} [/INST]")
            elif role == "assistant":
                formatted.append(content)

        return "\n".join(formatted)

    def _prepare_messages_vllm(self, messages: List[Dict[str, str]]) -> str:
        """Prepare messages for vLLM backend."""
        # vLLM uses similar format to transformers
        return self._prepare_messages_transformers(messages)

    def generate(
        self,
        messages: List[Union[Message, Dict[str, str]]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[LLMResponse, Iterator[LLMResponse]]:
        """
        Generate response from local LLM.

        Args:
            messages: List of messages
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling
            top_k: Top-k sampling
            stream: Whether to stream
            **kwargs: Additional arguments

        Returns:
            LLMResponse or iterator of LLMResponse
        """
        # Convert messages to dict format
        dict_messages = []
        for msg in messages:
            if isinstance(msg, Message):
                dict_messages.append(msg.to_dict())
            else:
                dict_messages.append(msg)

        # Add system prompt
        if system_prompt:
            dict_messages.insert(0, {"role": "system", "content": system_prompt})

        # Use default values
        temp = temperature if temperature is not None else self.config.temperature
        max_tok = max_tokens if max_tokens is not None else self.config.max_length
        top = top_p if top_p is not None else self.config.top_p
        tk = top_k if top_k is not None else self.config.top_k

        # Route to backend
        if self.backend == LocalLLMBackend.TRANSFORMERS:
            if stream:
                return self._stream_transformers(dict_messages, temp, max_tok, top, tk, **kwargs)
            else:
                return self._generate_transformers(dict_messages, temp, max_tok, top, tk, **kwargs)

        elif self.backend == LocalLLMBackend.LLAMA_CPP:
            if stream:
                return self._stream_llama_cpp(dict_messages, temp, max_tok, top, tk, **kwargs)
            else:
                return self._generate_llama_cpp(dict_messages, temp, max_tok, top, tk, **kwargs)

        elif self.backend == LocalLLMBackend.VLLM:
            if stream:
                return self._stream_vllm(dict_messages, temp, max_tok, top, tk, **kwargs)
            else:
                return self._generate_vllm(dict_messages, temp, max_tok, top, tk, **kwargs)

        elif self.backend == LocalLLMBackend.OLLAMA:
            if stream:
                return self._stream_ollama(dict_messages, temp, max_tok, top, tk, **kwargs)
            else:
                return self._generate_ollama(dict_messages, temp, max_tok, top, tk, **kwargs)

        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _generate_transformers(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        top_k: int,
        **kwargs
    ) -> LLMResponse:
        """Generate using transformers backend."""
        start_time = time.time()

        # Prepare prompt
        prompt = self._prepare_messages_transformers(messages)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length
        )

        if self.config.device == "cuda":
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=self.config.repetition_penalty,
                **kwargs
            )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):]

        # Clean response
        response = response.strip()

        # Count tokens
        prompt_tokens = inputs['input_ids'].shape[1]
        completion_tokens = len(self.tokenizer.encode(response))

        latency_ms = (time.time() - start_time) * 1000

        return LLMResponse(
            content=response,
            model=self.config.model_path,
            provider="local",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=0.0,  # Local models are free
            finish_reason="stop",
            latency_ms=latency_ms
        )

    def _stream_transformers(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        top_k: int,
        **kwargs
    ) -> Iterator[LLMResponse]:
        """Stream using transformers backend."""
        start_time = time.time()

        prompt = self._prepare_messages_transformers(messages)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length
        )

        if self.config.device == "cuda":
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

        # Streamer
        output_text = ""

        with torch.no_grad():
            for output in self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=self.config.repetition_penalty,
                stream=True,
                **kwargs
            ):
                # Decode new tokens
                if hasattr(output, 'tokens'):
                    for token_id in output.tokens:
                        token_text = self.tokenizer.decode(token_id, skip_special_tokens=True)
                        output_text += token_text
                        yield LLMResponse(
                            content=token_text,
                            model=self.config.model_path,
                            provider="local",
                            latency_ms=(time.time() - start_time) * 1000
                        )

    def _generate_llama_cpp(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        top_k: int,
        **kwargs
    ) -> LLMResponse:
        """Generate using llama-cpp backend."""
        start_time = time.time()

        prompt = self._prepare_messages_llama_cpp(messages)

        # Generate
        output = self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=self.config.repetition_penalty,
            echo=False,
            **kwargs
        )

        response = output['choices'][0]['text'].strip()

        latency_ms = (time.time() - start_time) * 1000

        # Estimate tokens
        prompt_tokens = len(self.model.tokenize(prompt.encode()))
        completion_tokens = len(self.model.tokenize(response.encode()))

        return LLMResponse(
            content=response,
            model=self.config.model_path,
            provider="local",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=0.0,
            finish_reason=output['choices'][0].get('finish_reason', 'stop'),
            latency_ms=latency_ms
        )

    def _stream_llama_cpp(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        top_k: int,
        **kwargs
    ) -> Iterator[LLMResponse]:
        """Stream using llama-cpp backend."""
        start_time = time.time()

        prompt = self._prepare_messages_llama_cpp(messages)

        # Stream generation
        for output in self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=self.config.repetition_penalty,
            stream=True,
            echo=False,
            **kwargs
        ):
            if 'choices' in output and output['choices']:
                token_text = output['choices'][0].get('text', '')
                if token_text:
                    yield LLMResponse(
                        content=token_text,
                        model=self.config.model_path,
                        provider="local",
                        latency_ms=(time.time() - start_time) * 1000
                    )

    def _generate_vllm(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        top_k: int,
        **kwargs
    ) -> LLMResponse:
        """Generate using vLLM backend."""
        start_time = time.time()

        prompt = self._prepare_messages_vllm(messages)

        from vllm import SamplingParams

        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            repetition_penalty=self.config.repetition_penalty
        )

        # Generate
        outputs = self.model.generate([prompt], sampling_params)

        response = outputs[0].outputs[0].text.strip()

        latency_ms = (time.time() - start_time) * 1000

        # Get token counts
        prompt_tokens = len(outputs[0].prompt_token_ids)
        completion_tokens = len(outputs[0].outputs[0].token_ids)

        return LLMResponse(
            content=response,
            model=self.config.model_path,
            provider="local",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost=0.0,
            finish_reason=outputs[0].outputs[0].finish_reason,
            latency_ms=latency_ms
        )

    def _stream_vllm(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        top_k: int,
        **kwargs
    ) -> Iterator[LLMResponse]:
        """Stream using vLLM backend."""
        # vLLM doesn't support streaming in the same way
        # Return full response as a single chunk
        response = self._generate_vllm(
            messages, temperature, max_tokens, top_p, top_k, **kwargs
        )
        yield response

    def _generate_ollama(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        top_k: int,
        **kwargs
    ) -> LLMResponse:
        """Generate using Ollama backend."""
        start_time = time.time()

        # Convert messages to Ollama format
        ollama_messages = []
        for msg in messages:
            ollama_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })

        # Generate
        response = ollama.chat(
            model=self.model_name,
            messages=ollama_messages,
            options={
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_predict": max_tokens
            }
        )

        content = response['message']['content']

        latency_ms = (time.time() - start_time) * 1000

        # Estimate tokens
        prompt_tokens = len(content.split()) * 1.3  # Rough estimate
        completion_tokens = len(content.split()) * 1.3

        return LLMResponse(
            content=content,
            model=self.model_name,
            provider="ollama",
            prompt_tokens=int(prompt_tokens),
            completion_tokens=int(completion_tokens),
            total_tokens=int(prompt_tokens + completion_tokens),
            cost=0.0,
            finish_reason="stop",
            latency_ms=latency_ms
        )

    def _stream_ollama(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        top_k: int,
        **kwargs
    ) -> Iterator[LLMResponse]:
        """Stream using Ollama backend."""
        start_time = time.time()

        ollama_messages = []
        for msg in messages:
            ollama_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })

        stream = ollama.chat(
            model=self.model_name,
            messages=ollama_messages,
            options={
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_predict": max_tokens
            },
            stream=True
        )

        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                content = chunk['message']['content']
                yield LLMResponse(
                    content=content,
                    model=self.model_name,
                    provider="ollama",
                    latency_ms=(time.time() - start_time) * 1000
                )

    async def generate_async(
        self,
        messages: List[Union[Message, Dict[str, str]]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Async generation (runs in thread pool).
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.generate,
            messages,
            system_prompt,
            temperature,
            max_tokens,
            top_p,
            top_k,
            False,
            **kwargs
        )

    def generate_simple(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Simple generation for a single prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Generated response
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.generate(messages, system_prompt=system_prompt)
        return response.content

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "backend": self.backend.value,
            "model_path": self.config.model_path,
            "model_name": self.config.model_name or self.config.model_path,
            "device": self.config.device,
            "quantization": self.config.quantization.value,
            "max_length": self.config.max_length,
            "is_initialized": self._is_initialized
        }


# ============================================================
# Factory Functions
# ============================================================

def create_local_llm(
    model_path: str,
    backend: Union[str, LocalLLMBackend] = "transformers",
    quantization: Union[str, LocalLLMQuantization] = "none",
    device: str = "cuda",
    **kwargs
) -> LocalLLMInterface:
    """
    Create a local LLM interface with default configuration.

    Args:
        model_path: Path to model
        backend: Backend to use
        quantization: Quantization type
        device: Device to use
        **kwargs: Additional configuration

    Returns:
        LocalLLMInterface instance
    """
    if isinstance(backend, str):
        backend = LocalLLMBackend(backend)

    if isinstance(quantization, str):
        quantization = LocalLLMQuantization(quantization)

    # Default configurations for different backends
    if backend == LocalLLMBackend.TRANSFORMERS:
        default_kwargs = {
            "device_map": "auto" if device == "cuda" else None,
            "trust_remote_code": True
        }
    elif backend == LocalLLMBackend.LLAMA_CPP:
        default_kwargs = {
            "n_ctx": kwargs.get("n_ctx", 4096),
            "n_gpu_layers": -1 if device == "cuda" else 0,
            "n_threads": kwargs.get("n_threads", 4)
        }
    elif backend == LocalLLMBackend.VLLM:
        default_kwargs = {
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9
        }
    elif backend == LocalLLMBackend.OLLAMA:
        default_kwargs = {
            "ollama_host": kwargs.get("ollama_host", "http://localhost:11434")
        }
    else:
        default_kwargs = {}

    # Merge with provided kwargs
    config_kwargs = {**default_kwargs, **kwargs}

    config = LocalLLMConfig(
        model_path=model_path,
        backend=backend,
        model_name=kwargs.get("model_name", ""),
        quantization=quantization,
        device=device,
        **config_kwargs
    )

    return LocalLLMInterface(config)


def load_llama2(
    model_path: str,
    quantization: str = "q4_k_m",
    device: str = "cuda",
    **kwargs
) -> LocalLLMInterface:
    """
    Convenience function to load Llama 2 models.

    Args:
        model_path: Path to model (GGUF file for llama-cpp, or HF path)
        quantization: Quantization type
        device: Device to use
        **kwargs: Additional arguments

    Returns:
        LocalLLMInterface instance
    """
    # Auto-detect backend based on file extension
    if model_path.endswith('.gguf'):
        backend = LocalLLMBackend.LLAMA_CPP
    elif model_path.endswith('.safetensors') or 'llama' in model_path.lower():
        backend = LocalLLMBackend.TRANSFORMERS
    else:
        backend = LocalLLMBackend.TRANSFORMERS

    return create_local_llm(
        model_path=model_path,
        backend=backend,
        quantization=quantization,
        device=device,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Example: Load local model
    # model_path = "./models/llama-2-7b-chat.Q4_K_M.gguf"
    # llm = load_llama2(model_path, quantization="q4_k_m")

    # Example: Load with transformers
    # llm = create_local_llm("meta-llama/Llama-2-7b-chat-hf", backend="transformers")

    # Example: Load with Ollama
    # llm = create_local_llm("llama2", backend="ollama")

    print("Local LLM module ready")
    print(f"Available backends:")
    print(f"  Transformers: {TRANSFORMERS_AVAILABLE}")
    print(f"  llama-cpp: {LLAMA_CPP_AVAILABLE}")
    print(f"  vLLM: {VLLM_AVAILABLE}")
    print(f"  Ollama: {OLLAMA_AVAILABLE}")
    print(f"  ExLlamaV2: {EXLLAMA_AVAILABLE}")
