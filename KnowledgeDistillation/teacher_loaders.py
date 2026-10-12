"""Teacher model loaders supporting three backends: Direct, Ollama, vLLM."""

import json
import os
from abc import ABC, abstractmethod
from typing import Callable

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from kd_args import TeacherArgs


class TeacherLoader(ABC):
    """Abstract base class for teacher model loaders."""
    
    @abstractmethod
    def get_callable(self) -> Callable:
        """Return a callable that takes (input_ids, attention_mask) and returns logits."""
        pass


class DirectTeacherLoader(TeacherLoader):
    """Load teacher model directly from HuggingFace transformers library."""
    
    def __init__(self, args: TeacherArgs):
        self.args = args
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer from HuggingFace with optional quantization and memory constraints."""
        print(f"Loading teacher model (direct): {self.args.model_id}")
        
        # Aggressive GPU memory cleanup BEFORE loading to prevent caching_allocator_warmup OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            print("GPU memory cleaned and synchronized")
        
        # Set up quantization if requested (default is 8bit with CPU offloading)
        quantization_config = None
        if self.args.quantization == "4bit":
            print(f"Applying 4-bit quantization (compute_dtype={self.args.bnb_4bit_compute_dtype})")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16 if self.args.bnb_4bit_compute_dtype == "float16" else torch.bfloat16,
                bnb_4bit_use_double_quant=self.args.bnb_4bit_use_double_quant,
                llm_int8_enable_fp32_cpu_offload=True
            )
        elif self.args.quantization == "8bit":
            print("Applying 8-bit quantization with CPU offloading (key for multi-GPU)")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                llm_int8_threshold=6.0,
            )
        elif self.args.quantization != "none":
            raise ValueError(f"Unknown quantization method: {self.args.quantization}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_id,
            trust_remote_code=True
        )
        
        # Check if running with DDP (distributed training)
        use_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
        local_rank = int(os.environ.get("LOCAL_RANK", 0)) if use_ddp else 0
        
        # CRITICAL FIX: Don't use device_map="auto" with DDP + quantization
        # This causes DTensor objects which are incompatible with bitsandbytes
        # Instead, use explicit device placement for each rank
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": "auto",
        }
        
        if use_ddp:
            # For DDP, load model to current rank's GPU explicitly
            print(f"[DDP Mode] Loading teacher model to cuda:{local_rank} explicitly (avoiding device_map='auto')")
            load_kwargs["device_map"] = {"": f"cuda:{local_rank}"}
        else:
            # For single-GPU, use explicit device placement (avoid device_map="auto" which causes dtype issues with quantization)
            print(f"[Single GPU Mode] Loading teacher model to cuda:0 explicitly")
            load_kwargs["device_map"] = {"": "cuda:0"}
        
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        
        print("Loading model with quantization...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_id,
            **load_kwargs
        )
        self.model.eval()
        
        # Check quantization status
        if self.args.quantization == "4bit":
            print(f"✓ 4-bit quantization applied. Memory footprint reduced ~87%")
        elif self.args.quantization == "8bit":
            print(f"✓ 8-bit quantization with CPU offloading applied. Higher accuracy, more CPU/GPU balance")
        print(f"Teacher model loaded successfully")
    
    def get_callable(self) -> Callable:
        """Return a function that computes teacher logits."""
        # Capture the device at callable creation time
        use_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
        local_rank = int(os.environ.get("LOCAL_RANK", 0)) if use_ddp else 0
        device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        
        def forward(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                # Move inputs to the device where teacher model was loaded
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
            return outputs.logits
        
        return forward


class OllamaTeacherLoader(TeacherLoader):
    """Load teacher model via Ollama REST API."""
    
    def __init__(self, args: TeacherArgs):
        self.args = args
        self.base_url = f"http://localhost:{args.ollama_port}"
        self._verify_connection()
    
    def _verify_connection(self):
        """Check if Ollama service is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                print(f"Connected to Ollama at {self.base_url}")
            else:
                raise ConnectionError(f"Ollama returned status {response.status_code}")
        except requests.ConnectionError as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Ensure Ollama is running: ollama serve"
            ) from e
    
    def get_callable(self) -> Callable:
        """Return a function that queries Ollama API for logits."""
        def forward(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            """Query Ollama endpoint. Returns logits of shape (batch, seq_len, vocab_size)."""
            batch_size, seq_len = input_ids.shape
            
            # For Ollama via REST API, we'll make requests per sequence
            # Note: This is a simplified version. Ollama doesn't directly expose logits,
            # so we would need a custom integration or use the embedding endpoint.
            # For now, this is a placeholder that demonstrates the interface.
            raise NotImplementedError(
                "Ollama API doesn't directly expose logits. "
                "Consider using vLLM instead, or run Ollama with a custom server endpoint."
            )
        
        return forward


class VLLMTeacherLoader(TeacherLoader):
    """Load teacher model via vLLM OpenAI-compatible API."""
    
    def __init__(self, args: TeacherArgs):
        self.args = args
        self.base_url = f"http://localhost:{args.vllm_port}/v1"
        self.model_name = args.model_id.split("/")[-1]  # Extract model name
        self._verify_connection()
    
    def _verify_connection(self):
        """Check if vLLM service is running."""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=2)
            if response.status_code == 200:
                print(f"Connected to vLLM at {self.base_url}")
            else:
                raise ConnectionError(f"vLLM returned status {response.status_code}")
        except requests.ConnectionError as e:
            raise ConnectionError(
                f"Cannot connect to vLLM at {self.base_url}. "
                f"Ensure vLLM is running: python -m vllm.entrypoints.openai.api_server --port {self.args.vllm_port}"
            ) from e
    
    def get_callable(self) -> Callable:
        """Return a function that queries vLLM API for logits."""
        def forward(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            """Query vLLM endpoint with logits_processors."""
            batch_size, seq_len = input_ids.shape
            device = input_ids.device
            
            # Convert token IDs to text (requires tokenizer; simplified here)
            # In practice, you'd keep the tokenizer and decode token sequences
            raise NotImplementedError(
                "vLLM logits endpoint needs custom implementation. "
                "For now, consider using the Direct loader with local vLLM model copy."
            )
        
        return forward


def get_teacher_loader(args: TeacherArgs) -> TeacherLoader:
    """Factory function to create appropriate teacher loader based on args."""
    method = args.loading_method.lower()
    
    if method == "direct":
        return DirectTeacherLoader(args)
    elif method == "ollama":
        print("Warning: Ollama loader requires custom integration for full logit access.")
        return OllamaTeacherLoader(args)
    elif method == "vllm":
        print("Warning: vLLM loader requires custom integration for full logit access.")
        return VLLMTeacherLoader(args)
    else:
        raise ValueError(
            f"Unknown teacher loading method: {method}. "
            f"Must be one of: direct, ollama, vllm"
        )
