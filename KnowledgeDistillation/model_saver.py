"""Model saving utilities for HuggingFace and PyTorch formats."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import save_file
from peft import get_peft_model_state_dict

from kd_args import DistillationArgs, OutputArgs, StudentArgs, TeacherArgs
from student_model import StudentModelWrapper


class ModelSaver:
    """Handles saving student model in multiple formats."""
    
    def __init__(
        self,
        student_wrapper: StudentModelWrapper,
        teacher_args: TeacherArgs,
        student_args: StudentArgs,
        distill_args: DistillationArgs,
        output_args: OutputArgs,
    ):
        self.student_wrapper = student_wrapper
        self.teacher_args = teacher_args
        self.student_args = student_args
        self.distill_args = distill_args
        self.output_args = output_args
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self.checkpoint_dir = self._create_checkpoint_dir()
    
    def _create_checkpoint_dir(self) -> Path:
        """Create and return checkpoint directory with timestamp and epochs."""
        # Check if checkpoint_dir is already set with epochs (from kd.py)
        if self.output_args.checkpoint_dir and "_epochs" in self.output_args.checkpoint_dir:
            checkpoint_dir = Path(self.output_args.checkpoint_dir)
        else:
            # Fall back to creating new folder with model names, timestamp, and epochs
            teacher_name = self.teacher_args.model_id.split("/")[-1]
            student_name = self.student_args.model_id.split("/")[-1]
            
            dir_name = f"{student_name}_from_{teacher_name}_epochs{self.distill_args.num_epochs}_{self.timestamp}"
            checkpoint_dir = Path(self.output_args.save_dir) / dir_name
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Checkpoint directory: {checkpoint_dir}")
        return checkpoint_dir
    
    def save_hf_format(self):
        """Save student model in HuggingFace format."""
        if not self.output_args.save_hf_format:
            return
        
        hf_dir = self.checkpoint_dir / "hf_model"
        hf_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving model artifacts to {hf_dir}")
        
        # Extract base model from DDP wrapper if needed
        model = self.student_wrapper.model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        
        # For 8-bit quantized models, saving in HF format has compatibility issues
        # Skip the model and just save tokenizer + config
        # The state_dict is saved in pytorch_format() instead
        try:
            print("Note: Quantized models cannot be fully saved in HuggingFace format due to torch compatibility.")
            print("Saving config and tokenizer only. Use pytorch_format for model weights.")
            
            # Save config
            if hasattr(model, 'config'):
                model.config.save_pretrained(str(hf_dir))
        except Exception as e:
            print(f"Warning: Could not save config: {e}")
        
        # Always try to save tokenizer
        self.student_wrapper.tokenizer.save_pretrained(str(hf_dir))
        
        print(f"Tokenizer and config saved to {hf_dir}")
    
    def save_pytorch_format(self):
        """Save student model as PyTorch checkpoint."""
        if not self.output_args.save_pytorch_format:
            return
        
        pytorch_path = self.checkpoint_dir / "pytorch_model.bin"
        
        print(f"Saving model in PyTorch format to {pytorch_path}")
        
        # Extract base model from DDP wrapper if needed
        model = self.student_wrapper.model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        
        torch.save(
            model.state_dict(),
            str(pytorch_path)
        )
        
        print(f"PyTorch model saved to {pytorch_path}")
    
    def save_safetensors_format(self):
        """Save LoRA adapters in safetensors format for ollama/llamafactory.
        
        This saves only the LoRA adapter weights (lora_A, lora_B), which can be:
        - Loaded with peft for inference
        - Used with llamafactory for further fine-tuning
        - Used with ollama if GGUF conversion is done
        """
        print("Saving model in safetensors format (LoRA adapters for ollama/llamafactory)...")
        
        # Create adapter directory
        adapter_dir = self.checkpoint_dir / "adapter_safetensors"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        
        model = self.student_wrapper.model
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        
        try:
            # Extract LoRA adapter state dict (only lora_A and lora_B weights, not base model)
            adapter_state = get_peft_model_state_dict(model)
            
            # Convert to CPU and detach to ensure compatibility
            adapter_state_cpu = {
                k: v.cpu() if isinstance(v, torch.Tensor) else v 
                for k, v in adapter_state.items()
            }
            
            # Save adapters in safetensors format
            safetensors_path = adapter_dir / "adapter_model.safetensors"
            save_file(adapter_state_cpu, str(safetensors_path))
            
            print(f"✓ LoRA adapters saved to {safetensors_path}")
            
            # Also save adapter config for PEFT
            try:
                adapter_config_path = adapter_dir / "adapter_config.json"
                
                # peft_config is a dict with config name as key, LoraConfig as value
                # Extract the LoraConfig object (first/default config)
                if hasattr(model, 'peft_config'):
                    peft_configs = model.peft_config
                    
                    # Handle both dict and single config
                    if isinstance(peft_configs, dict):
                        # Get the first config
                        first_config = next(iter(peft_configs.values())) if peft_configs else None
                    else:
                        first_config = peft_configs
                    
                    if first_config is not None:
                        # Convert LoraConfig to dict using to_dict() method
                        if hasattr(first_config, 'to_dict'):
                            config_dict = first_config.to_dict()
                        else:
                            config_dict = vars(first_config) if hasattr(first_config, '__dict__') else {}
                        
                        with open(str(adapter_config_path), 'w') as f:
                            json.dump(config_dict, f, indent=2, default=str)
                        print(f"✓ Adapter config saved to {adapter_config_path}")
            except Exception as e:
                print(f"Warning: Could not save adapter config: {e}")
            
            # Save tokenizer to adapter dir for convenience
            self.student_wrapper.tokenizer.save_pretrained(str(adapter_dir))
            print(f"✓ Tokenizer saved to {adapter_dir}")
            
            # Create README with instructions
            readme_path = adapter_dir / "README.md"
            readme_content = f"""# LoRA Adapter for {self.student_args.model_id.split('/')[-1]}

## Overview
This directory contains LoRA (Low-Rank Adaptation) adapters fine-tuned via knowledge distillation from {self.teacher_args.model_id.split('/')[-1]}.

## Files
- `adapter_model.safetensors`: LoRA adapter weights (lora_A, lora_B)
- `adapter_config.json`: LoRA configuration
- `tokenizer_config.json`: Tokenizer configuration
- `special_tokens_map.json`: Special tokens mapping

## Usage

### With Hugging Face Transformers + PEFT
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "{self.student_args.model_id}",
    load_in_4bit=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, ".")
```

### With LLaMA-Factory
Place this adapter directory in `data/lora_model/` and reference it in your config:
```yaml
adapter_name_or_path: ./data/lora_model/adapter_dir
```

### Converting to GGUF for Ollama
1. First merge the adapter with base model (if supported)
2. Convert merged model to GGUF using `llama.cpp`
3. Load in Ollama

## Training Details
- Temperature: {self.distill_args.temperature}
- KD Alpha: {self.distill_args.kd_alpha}
- Epochs: {self.distill_args.num_epochs}
- Learning Rate: {self.distill_args.learning_rate}
- Batch Size: {self.distill_args.batch_size}
"""
            with open(str(readme_path), 'w') as f:
                f.write(readme_content)
            print(f"✓ README saved to {readme_path}")
            
        except Exception as e:
            print(f"Warning: Could not save safetensors format: {e}")
            import traceback
            traceback.print_exc()
    
    def save_config(self):
        """Save configuration snapshot as JSON."""
        config = {
            "timestamp": self.timestamp,
            "teacher": {
                "model_id": self.teacher_args.model_id,
                "loading_method": self.teacher_args.loading_method,
                "device": self.teacher_args.device,
            },
            "student": {
                "model_id": self.student_args.model_id,
                "device": self.student_args.device,
            },
            "distillation": {
                "temperature": self.distill_args.temperature,
                "kd_alpha": self.distill_args.kd_alpha,
                "num_epochs": self.distill_args.num_epochs,
                "batch_size": self.distill_args.batch_size,
                "learning_rate": self.distill_args.learning_rate,
                "warmup_steps": self.distill_args.warmup_steps,
            },
        }
        
        config_path = self.checkpoint_dir / "config_snapshot.json"
        with open(str(config_path), "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"Config snapshot saved to {config_path}")
    
    def save_training_log(self, training_log: list):
        """Save training log as JSON."""
        log_path = self.checkpoint_dir / "trainer_state.json"
        
        with open(str(log_path), "w") as f:
            json.dump(training_log, f, indent=2)
        
        print(f"Training log saved to {log_path}")
    
    def save_all(self, training_log: list = None):
        """Save model in all formats and configs."""
        print("\n" + "="*50)
        print("Saving models and checkpoints...")
        print("="*50)
        
        self.save_hf_format()
        self.save_pytorch_format()
        self.save_safetensors_format()
        self.save_config()
        
        if training_log is not None:
            self.save_training_log(training_log)
        
        print("="*50)
        print(f"All models saved to: {self.checkpoint_dir}")
        print("="*50 + "\n")
    
    def get_checkpoint_dir(self) -> Path:
        """Return checkpoint directory path."""
        return self.checkpoint_dir
