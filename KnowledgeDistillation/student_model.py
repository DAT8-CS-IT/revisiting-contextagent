"""Student model initialization and optimizer setup with DDP support."""

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

from kd_args import DistillationArgs, StudentArgs


class StudentModelWrapper:
    """Wraps student model, tokenizer, optimizer, and scheduler with DDP support."""
    
    def __init__(
        self,
        student_args: StudentArgs,
        distill_args: DistillationArgs,
        num_training_steps: Optional[int] = None,
        use_ddp: bool = False,
    ):
        self.student_args = student_args
        self.distill_args = distill_args
        self.use_ddp = use_ddp
        
        # Get distributed training info
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0)) if use_ddp else 0
        self.world_size = int(os.environ.get("WORLD_SIZE", 1)) if use_ddp else 1
        self.rank = int(os.environ.get("RANK", 0)) if use_ddp else 0
        
        # Set device properly for DDP
        if use_ddp:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device(student_args.device)
        
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        
        self._load_model()
        self._setup_optimizer(num_training_steps)
    
    def _log(self, msg: str):
        """Log only from rank 0 to avoid duplicate messages."""
        if self.rank == 0:
            print(msg)
    
    def _load_model(self):
        """Load student model and tokenizer with memory optimizations."""
        # Aggressive GPU memory cleanup BEFORE loading to prevent OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            self._log("GPU memory cleaned before student loading")
        
        self._log(f"Loading student model: {self.student_args.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.student_args.model_id,
            trust_remote_code=True,
            padding_side="left"
        )
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure 4-bit quantization 
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        
        # Load model with explicit device placement (not auto) for proper 4-bit initialization
        # The 1B student model fits entirely on one GPU, so avoid device_map complexity
        self._log(f"Loading student model with 4-bit quantization")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.student_args.model_id,
            quantization_config=bnb_config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            use_cache=False,  # Incompatible with gradient checkpointing (even when disabled)
            device_map=self.device,  # Explicit device placement
        )
        
        # CRITICAL: Prepare model for k-bit training (enables NF4 quant gradients)
        self._log("Preparing model for 4-bit training...")
        self.model = prepare_model_for_kbit_training(self.model)
        
        # CRITICAL: 4-bit quantized models must use LoRA for training
        # LoRA adapts frozen weights without updating base model
        self._log("Setting up LoRA adapter for 4-bit quantized model...")
        lora_config = LoraConfig(
            r=16,  # Increased rank for better learning capacity
            lora_alpha=32,  # Scaling factor
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],  # More projections
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # Enable training mode but NOT gradient checkpointing with LoRA
        self.model.train()
        
        self._log(f"Student model loaded with 4-bit quantization + LoRA on {self.device}")
    
    def _setup_optimizer(self, num_training_steps: Optional[int] = None):
        """Setup AdamW optimizer and learning rate scheduler."""
        # Get base model for parameter access
        base_model = self.model
        
        # Verify LoRA parameters are actually trainable
        trainable_params = [p for p in base_model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            self._log("ERROR: No trainable parameters found! LoRA setup may have failed.")
            raise RuntimeError("LoRA did not create any trainable parameters")
        
        self._log(f"Found {len(trainable_params)} trainable parameter groups")
        
        # Separate parameters for weight decay
        no_decay = ["bias", "LayerNorm.weight", "lora", "norm"]  # Include lora in no_decay name check
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in base_model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.distill_args.weight_decay,
            },
            {
                "params": [
                    p for n, p in base_model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        
        # Filter out empty groups
        optimizer_grouped_parameters = [
            group for group in optimizer_grouped_parameters if len(group["params"]) > 0
        ]
        
        if len(optimizer_grouped_parameters) == 0:
            raise RuntimeError("No optimizer parameter groups created")
        
        self.optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.distill_args.learning_rate
        )
        
        self._log(f"Optimizer created with {len(optimizer_grouped_parameters)} parameter groups")
        
        # Setup learning rate scheduler (linear warmup)
        if num_training_steps is not None and self.distill_args.warmup_steps > 0:
            # LinearLR with start_factor multiplies the base LR, so we need a reasonable value
            # start_factor=1.0 means no warmup scaling (recommended for fine-tuning)
            # start_factor=0.1 would warm up from 10% to 100% of base LR
            self._log(f"Setting up LR scheduler with warmup_steps={self.distill_args.warmup_steps}")
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,  # Start at full learning rate (no aggressive scaling)
                end_factor=1.0,    # End at full learning rate
                total_iters=self.distill_args.warmup_steps
            )
        else:
            self._log("No warmup steps configured, training without LR scheduler")
            self.scheduler = None
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through student model.
        
        Returns:
            student_logits: Model output logits (batch, seq_len, vocab_size)
        """
        # Move inputs to model's device (device_map auto may place model on different device)
        model_device = next(self.model.parameters()).device
        input_ids = input_ids.to(model_device)
        attention_mask = attention_mask.to(model_device)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        return outputs.logits
    
    def train_mode(self):
        """Set model to training mode."""
        self.model.train()
    
    def eval_mode(self):
        """Set model to evaluation mode."""
        self.model.eval()
    
    def zero_grad(self):
        """Zero out gradients."""
        self.optimizer.zero_grad()
    
    def backward(self, loss: torch.Tensor):
        """Backward pass with safety checks."""
        # Ensure loss requires grad for backward
        if not loss.requires_grad:
            raise RuntimeError(f"Loss tensor does not require grad! grad_fn={loss.grad_fn}")
        if loss.grad_fn is None:
            raise RuntimeError("Loss tensor has no grad_fn - computation graph is broken!")
        loss.backward()
    
    def step(self):
        """Optimizer step."""
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]
    
    def is_rank_0(self) -> bool:
        """Check if current process is rank 0."""
        return self.rank == 0
    
    def get_base_model(self):
        """Get unwrapped base model (for saving)."""
        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            return self.model.module
        return self.model
