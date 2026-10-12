"""Argument definitions for Knowledge Distillation training.

Mirrors LLaMA-Factory pattern using @dataclass for argument groups.
"""

import argparse
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DistillationArgs:
    """Knowledge distillation specific hyperparameters."""
    
    temperature: float = field(
        default=3.0,
        metadata={"help": "Temperature for softmax scaling of KL divergence loss."}
    )
    kd_alpha: float = field(
        default=0.5,
        metadata={"help": "Weight for KD loss: total_loss = alpha * CE + (1 - alpha) * KD."}
    )
    num_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs."}
    )
    batch_size: int = field(
        default=2,
        metadata={"help": "Training batch size (reduced for memory efficiency)."}
    )
    eval_batch_size: int = field(
        default=4,
        metadata={"help": "Evaluation batch size (reduced for memory efficiency)."}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "Number of gradient accumulation steps (increases effective batch size without memory cost)."}
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "Learning rate for optimizer."}
    )
    warmup_steps: int = field(
        default=500,
        metadata={"help": "Number of warmup steps for learning rate scheduler."}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay for AdamW optimizer."}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log metrics every N steps."}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every N steps."}
    )


@dataclass
class TeacherArgs:
    """Teacher model configuration."""
    
    model_id: str = field(
        default="meta-llama/Meta-Llama-3.1-8B",
        metadata={"help": "HuggingFace model ID or local path for teacher model."}
    )
    loading_method: str = field(
        default="direct",
        metadata={
            "help": "How to load teacher model: 'direct' (transformers), 'ollama', or 'vllm'."
        }
    )
    device: str = field(
        default="cuda",
        metadata={"help": "Device to load teacher on: 'cuda' or 'cpu'."}
    )
    ollama_port: int = field(
        default=11434,
        metadata={"help": "Port for Ollama service (used if loading_method='ollama')."}
    )
    vllm_port: int = field(
        default=8009,
        metadata={"help": "Port for vLLM service (used if loading_method='vllm')."}
    )
    api_key: str = field(
        default="0",
        metadata={"help": "API key for remote endpoints (Ollama/vLLM)."}
    )
    quantization: str = field(
        default="4bit",
        metadata={
            "help": "Quantization for teacher model: 'none', '4bit', or '8bit'. "
                    "4bit reduces 8B model ~15GB -> ~2GB; 8bit ~15GB -> ~4GB. "
                    "Applies only to direct loading method."
        }
    )
    bnb_4bit_compute_dtype: str = field(
        default="float16",
        metadata={
            "help": "Compute dtype for 4-bit quantization: 'float16', 'bfloat16', or 'float32'."
        }
    )
    bnb_4bit_use_double_quant: bool = field(
        default=True,
        metadata={"help": "Use double quantization (nested quantization) for 4-bit. Saves ~0.4GB more."}
    )


@dataclass
class StudentArgs:
    """Student model configuration."""
    
    model_id: str = field(
        default="meta-llama/Llama-3.2-1B",
        metadata={"help": "HuggingFace model ID or local path for student model."}
    )
    device: str = field(
        default="cuda",
        metadata={"help": "Device to load student on: 'cuda' or 'cpu'."}
    )


@dataclass
class DataArgs:
    """Data loading configuration."""
    
    train_data_path: str = field(
        default=None,
        metadata={"help": "Path to training data (JSONL format with 'text' field)."}
    )
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to evaluation data (optional, JSONL format)."}
    )
    max_seq_length: int = field(
        default=256,
        metadata={"help": "Maximum sequence length for tokenization (reduced to save memory)."}
    )
    num_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Limit number of training samples (None for all)."}
    )
    num_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Limit number of eval samples (None for all)."}
    )


@dataclass
class OutputArgs:
    """Output and checkpointing configuration."""
    
    save_dir: str = field(
        default="KnowledgeDistillation/",
        metadata={"help": "Base directory to save models and checkpoints."}
    )
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Auto-generated checkpoint dir (model_name_timestamp/). Set at runtime."}
    )
    save_hf_format: bool = field(
        default=True,
        metadata={"help": "Save student model in HuggingFace format."}
    )
    save_pytorch_format: bool = field(
        default=True,
        metadata={"help": "Save student model as PyTorch checkpoint."}
    )
    save_trainer_state: bool = field(
        default=True,
        metadata={"help": "Save trainer state (loss history) as JSON."}
    )


def parse_args() -> tuple[DistillationArgs, TeacherArgs, StudentArgs, DataArgs, OutputArgs]:
    """Parse command-line arguments and return argument objects."""
    parser = argparse.ArgumentParser(
        description="Knowledge Distillation for Large Language Models"
    )
    
    def _add_arguments(arg_class, prefix=""):
        """Helper to add arguments from dataclass, handling booleans properly."""
        for field_obj in arg_class.__dataclass_fields__.values():
            arg_name = f"--{prefix}{field_obj.name}" if prefix else f"--{field_obj.name}"
            
            if field_obj.type == bool:
                # Handle bool fields as store_true action
                parser.add_argument(
                    arg_name,
                    action="store_true" if field_obj.default is False else "store_false",
                    default=field_obj.default,
                    help=field_obj.metadata.get("help", "")
                )
            else:
                parser.add_argument(
                    arg_name,
                    type=field_obj.type,
                    default=field_obj.default,
                    help=field_obj.metadata.get("help", "")
                )
    
    # Add arguments from each dataclass
    _add_arguments(DistillationArgs)
    _add_arguments(TeacherArgs, prefix="teacher_")
    _add_arguments(StudentArgs, prefix="student_")
    _add_arguments(DataArgs)
    _add_arguments(OutputArgs)
    
    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        default="live",
        choices=["live", "sandbox"],
        help="Run mode: 'live' uses real models, 'sandbox' uses stubs for testing."
    )
    
    # Alias for convenience: --quantization maps to --teacher_quantization
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="(Alias for --teacher_quantization) Quantization for teacher model: 'none', '4bit', or '8bit'."
    )
    
    args = parser.parse_args()
    
    # Handle --quantization alias
    if args.quantization is not None:
        args.teacher_quantization = args.quantization
    
    # Construct dataclass instances from parsed args
    distill_args = DistillationArgs(
        temperature=args.temperature,
        kd_alpha=args.kd_alpha,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
    )
    
    teacher_args = TeacherArgs(
        model_id=args.teacher_model_id,
        loading_method=args.teacher_loading_method,
        device=args.teacher_device,
        ollama_port=args.teacher_ollama_port,
        vllm_port=args.teacher_vllm_port,
        api_key=args.teacher_api_key,
        quantization=args.teacher_quantization,
        bnb_4bit_compute_dtype=args.teacher_bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=args.teacher_bnb_4bit_use_double_quant,
    )
    
    student_args = StudentArgs(
        model_id=args.student_model_id,
        device=args.student_device,
    )
    
    data_args = DataArgs(
        train_data_path=args.train_data_path,
        eval_data_path=args.eval_data_path,
        max_seq_length=args.max_seq_length,
        num_train_samples=args.num_train_samples,
        num_eval_samples=args.num_eval_samples,
    )
    
    output_args = OutputArgs(
        save_dir=args.save_dir,
        save_hf_format=args.save_hf_format,
        save_pytorch_format=args.save_pytorch_format,
        save_trainer_state=args.save_trainer_state,
    )
    
    return distill_args, teacher_args, student_args, data_args, output_args
