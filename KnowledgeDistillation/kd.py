"""Main entry point for knowledge distillation training.

Usage:
    python kd.py \
        --teacher_model_id meta-llama/Meta-Llama-3.1-8B \
        --student_model_id meta-llama/Llama-3.2-1B \
        --teacher_loading_method direct \
        --train_data_path /path/to/data.jsonl \
        --num_epochs 3 \
        --batch_size 8 \
        --temperature 3.0 \
        --kd_alpha 0.5
"""

import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# Suppress known deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*TRANSFORMERS_CACHE.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*use_reentrant.*")

import kd_config
from kd_args import parse_args
from distillation_trainer import DistillationTrainer
from model_saver import ModelSaver
from student_model import StudentModelWrapper
from teacher_loaders import get_teacher_loader
from utils import plotting


def cleanup_memory(rank: Optional[int] = None):
    """Aggressively clean up GPU memory before critical operations like distributed barriers."""
    try:
        if torch.cuda.is_available():
            if rank is not None:
                torch.cuda.set_device(rank)
            # Empty cache and perform garbage collection
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            import gc
            gc.collect()
            if rank == 0:
                print("[Memory] GPU cache cleared before barrier")
    except Exception as e:
        if rank == 0:
            print(f"[Warning] Error during memory cleanup: {e}")


class TextDataset(Dataset):
    """JSONL dataset for knowledge distillation training.
    
    Handles multiple data formats:
    1. JSONL (one JSON object per line)
    2. Nested JSON (single file with dict of examples)
    3. Plain text (one line per sample)
    
    Example formats:
      JSONL: {"text": "..."} per line
      Nested JSON: {"example-1": {...}, "example-2": {...}}
      With fields: Category, Vision, Context information, Response, Action, Tools, Thoughts
    """
    
    def __init__(self, file_path: str, tokenizer, max_seq_length: int = 512, num_samples: Optional[int] = None):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.texts = []
        
        # Load texts from file (auto-detect format)
        with open(file_path, "r") as f:
            # Read entire file to detect format
            content = f.read()
        
        # Try to parse as JSON (could be nested or JSONL)
        try:
            # Check if it's JSONL by trying to parse as standalone JSON first
            full_json = json.loads(content)
            
            # If we get here, it's valid JSON - check if it's nested (dict of examples)
            if isinstance(full_json, dict) and len(full_json) > 0:
                first_key = next(iter(full_json.keys()))
                first_value = full_json[first_key]
                
                # Check if values are example objects (dicts with text/context fields)
                if isinstance(first_value, dict) and any(
                    k in first_value for k in ["Vision", "Context information", "Response", "Category"]
                ):
                    # This is nested JSON format: {"example-1": {...}, "example-2": {...}}
                    print(f"Detected nested JSON format with {len(full_json)} examples")
                    for idx, (example_id, example_data) in enumerate(full_json.items()):
                        if num_samples is not None and idx >= num_samples:
                            break
                        if isinstance(example_data, dict):
                            text = self._extract_text_from_example(example_data)
                            if text and text.strip():
                                self.texts.append(text)
                    
                elif isinstance(full_json, list):
                    # This is a JSON list format: [{...}, {...}]
                    print(f"Detected JSON list format with {len(full_json)} examples")
                    for idx, example_data in enumerate(full_json):
                        if num_samples is not None and idx >= num_samples:
                            break
                        if isinstance(example_data, dict):
                            text = self._extract_text_from_example(example_data)
                            if text and text.strip():
                                self.texts.append(text)
                else:
                    # Single-object JSON, treat as one sample
                    print("Detected single JSON object format")
                    text = self._extract_text_from_example(full_json)
                    if text and text.strip():
                        self.texts.append(text)
        except json.JSONDecodeError:
            # Not valid JSON - try JSONL format (one JSON per line)
            print("Not valid JSON, trying JSONL format...")
            lines = content.strip().split('\n')
            for idx, line in enumerate(lines):
                if num_samples is not None and idx >= num_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if isinstance(data, dict):
                        text = self._extract_text_from_example(data)
                        if text and text.strip():
                            self.texts.append(text)
                except json.JSONDecodeError:
                    # Fall back to treating as plain text
                    self.texts.append(line)
        
        if not self.texts:
            raise ValueError(f"No valid texts found in {file_path}")
        
        print(f"Loaded {len(self.texts)} samples from {file_path}")
    
    @staticmethod
    def _extract_text_from_example(data: dict) -> str:
        """Extract text from a single example dictionary."""
        if isinstance(data, str):
            return data
        
        if not isinstance(data, dict):
            return ""
        
        # Check for explicit "text" field
        if "text" in data and isinstance(data["text"], str):
            return data["text"]
        
        # Handle nested structure with semantic fields
        if any(k in data for k in ["Vision", "Context information", "Response", "Category", "Action", "Thoughts"]):
            parts = []
            
            # Add category if present
            if "Category" in data and data["Category"] and data["Category"] != "None":
                parts.append(f"Category: {data['Category']}")
            
            # Add vision (scene description)
            if "Vision" in data and data["Vision"] and data["Vision"] != "None":
                parts.append(f"Vision: {data['Vision']}")
            
            # Add context information
            if "Context information" in data and data["Context information"] and data["Context information"] != "None":
                parts.append(f"Context: {data['Context information']}")
            
            # Add thoughts (agent reasoning)
            if "Thoughts" in data and data["Thoughts"] and data["Thoughts"] != "None":
                parts.append(f"Thoughts: {data['Thoughts']}")
            
            # Add response
            if "Response" in data and data["Response"] and data["Response"] != "None":
                parts.append(f"Response: {data['Response']}")
            
            # Add action
            if "Action" in data and data["Action"] and data["Action"] != "None":
                action = data["Action"]
                if isinstance(action, list):
                    action_text = " ".join(str(a) for a in action if a)
                else:
                    action_text = str(action)
                if action_text:
                    parts.append(f"Action: {action_text}")
            
            if parts:
                return " ".join(parts)
        
        # Fallback: concatenate all non-empty string values
        values = []
        for key, value in data.items():
            if value and value != "None":
                if isinstance(value, (list, dict)):
                    continue  # Skip complex types
                values.append(str(value))
        
        if values:
            return " ".join(values)
        
        return ""
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": encoding["input_ids"].squeeze(0).clone(),  # Same as input for CLM
        }


def create_dataloaders(
    train_data_path: Optional[str],
    eval_data_path: Optional[str],
    tokenizer,
    batch_size: int,
    eval_batch_size: int,
    max_seq_length: int,
    num_train_samples: Optional[int],
    num_eval_samples: Optional[int],
    use_ddp: bool,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, Optional[DataLoader]]:
    """Create train and eval dataloaders."""
    
    if train_data_path is None:
        if kd_config.is_sandbox():
            # Create dummy data for sandbox mode
            print("Creating dummy data for sandbox mode...")
            class DummyDataset(Dataset):
                def __init__(self, size=100):
                    self.size = size
                
                def __len__(self):
                    return self.size
                
                def __getitem__(self, idx):
                    seq_len = 128
                    return {
                        "input_ids": torch.randint(0, 30000, (seq_len,)),
                        "attention_mask": torch.ones(seq_len),
                        "labels": torch.randint(0, 30000, (seq_len,)),
                    }
            
            train_dataset = DummyDataset(size=20)
            eval_dataset = DummyDataset(size=5)
        else:
            raise ValueError("--train_data_path is required in live mode")
    else:
        train_dataset = TextDataset(
            train_data_path,
            tokenizer,
            max_seq_length,
            num_samples=num_train_samples
        )
        
        if eval_data_path:
            eval_dataset = TextDataset(
                eval_data_path,
                tokenizer,
                max_seq_length,
                num_samples=num_eval_samples
            )
        else:
            eval_dataset = None
    
    # Create samplers for DDP
    if use_ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=42,
        )
        eval_sampler = None
        if eval_dataset is not None:
            eval_sampler = DistributedSampler(
                eval_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                seed=42,
            )
    else:
        train_sampler = None
        eval_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # Only shuffle if not using sampler
        num_workers=0,
        collate_fn=collate_fn,
    )
    
    eval_loader = None
    if eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            sampler=eval_sampler,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
    
    return train_loader, eval_loader


def collate_fn(batch):
    """Custom collate function for batching."""
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }


def init_distributed_training():
    """Initialize distributed training if RANK is set (by torchrun).
    
    Returns:
        Tuple of (use_ddp, rank, world_size)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend="nccl")
        if rank == 0:
            print(f"Initialized DDP: rank {rank}/{world_size}")
        return True, rank, world_size
    return False, 0, 1


def main():
    """Main training function."""
    
    # Initialize distributed training
    use_ddp, rank, world_size = init_distributed_training()
    
    # DEBUG: Log DDP initialization
    print(f"\n{'='*60}")
    print(f"[INIT] DDP Status: use_ddp={use_ddp}, rank={rank}, world_size={world_size}")
    print(f"[INIT] Environment: RANK={os.environ.get('RANK', 'not set')}, WORLD_SIZE={os.environ.get('WORLD_SIZE', 'not set')}")
    print(f"{'='*60}\n")
    
    # Parse arguments
    distill_args, teacher_args, student_args, data_args, output_args = parse_args()
    
    # Set mode
    mode = None
    # Check if --mode was passed via sys.argv
    import sys
    if "--mode" in sys.argv:
        mode_idx = sys.argv.index("--mode") + 1
        if mode_idx < len(sys.argv):
            mode = sys.argv[mode_idx]
    
    if mode:
        kd_config.set_mode(mode)
    
    print(f"Mode: {'SANDBOX' if kd_config.is_sandbox() else 'LIVE'}")
    print(f"Device: {student_args.device}")
    
    # Set up device
    device = torch.device(student_args.device)
    if device.type == "cuda":
        # Check if CUDA is actually available before setting device
        if not torch.cuda.is_available():
            print("ERROR: CUDA device requested but not available!")
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
            raise RuntimeError(
                "CUDA not available. A CUDA-enabled PyTorch installation and GPU are required."
            )
        
        # Set device to 0 (first available device in CUDA_VISIBLE_DEVICES)
        try:
            torch.cuda.set_device(0)
            print(f"CUDA initialized on device 0 ({torch.cuda.get_device_name(0)})")
        except RuntimeError as e:
            print(f"ERROR: Failed to set CUDA device 0: {e}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            raise
    
    # Stagger model loading across ranks to reduce peak memory usage
    # Critical: teacher model takes ~3 minutes to load per rank
    # Use shorter stagger delay (90s) and synchronize before training
    print(f"[Stagger] Checking delay: rank={rank}, world_size={world_size}, condition: {rank is not None and world_size is not None and world_size > 1}")
    
    if rank is not None and world_size is not None and world_size > 1:
        # 90-second delay per rank = rank 0 loads immediately, rank 1 waits 90s, rank 2 waits 180s, etc.
        # This prevents simultaneous loading (OOM) while minimizing straggler effect
        # At 3 min per load, this keeps ranks reasonably aligned
        delay = rank * 90.0
        if rank > 0:
            print(f"[Rank {rank}] Waiting {delay:.0f}s ({delay/60:.1f}min) for previous ranks to load teacher model...")
            time.sleep(delay)
        else:
            print(f"[Rank {rank}] No delay (rank 0 loads immediately)")
    else:
        print(f"[Stagger] Skipping delay: rank is not None? {rank is not None}, world_size is not None? {world_size is not None}, world_size > 1? {(world_size or 0) > 1}")
    
    # Aggressive memory cleanup BEFORE loading models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        print(f"[Rank {rank}] GPU memory cleaned before model loading")
    
    # Load teacher model
    print("\n" + "="*50)
    print("Loading teacher model...")
    print("="*50)
    teacher_loader = get_teacher_loader(teacher_args)
    teacher_forward = teacher_loader.get_callable()
    
    # Load student model
    print("\n" + "="*50)
    print("Loading student model...")
    print("="*50)
    
    # Barrier to synchronize after student initialization
    # REMOVED: This barrier caused OOM with NCCL during model loading
    # device_map="auto" already handles synchronization via CPU offloading
    
    # Compute number of training steps for scheduler
    if data_args.train_data_path is not None:
        # This is approximate; in practice calculate from dataset size
        num_train_samples = data_args.num_train_samples or 1000
        num_steps_per_epoch = num_train_samples // distill_args.batch_size
        num_training_steps = num_steps_per_epoch * distill_args.num_epochs
    else:
        num_training_steps = None
    
    student_wrapper = StudentModelWrapper(
        student_args=student_args,
        distill_args=distill_args,
        num_training_steps=num_training_steps,
        use_ddp=use_ddp,
    )
    
    # Create dataloaders
    print("\n" + "="*50)
    print("Preparing data...")
    print("="*50)
    train_loader, eval_loader = create_dataloaders(
        train_data_path=data_args.train_data_path,
        eval_data_path=data_args.eval_data_path,
        tokenizer=student_wrapper.tokenizer,
        batch_size=distill_args.batch_size,
        eval_batch_size=distill_args.eval_batch_size,
        max_seq_length=data_args.max_seq_length,
        num_train_samples=data_args.num_train_samples,
        num_eval_samples=data_args.num_eval_samples,
        use_ddp=use_ddp,
        rank=rank,
        world_size=world_size,
    )
    
    # Synchronization barrier: Wait for all ranks to finish loading models before training
    # This ensures training starts simultaneously across all ranks (prevents stragglers)
    if use_ddp:
        print(f"\n[Rank {rank}] Waiting for all ranks to reach training synchronization point...")
        cleanup_memory(rank)  # Clean GPU memory before barrier to prevent OOM
        dist.barrier()
        if rank == 0:
            print("[Sync] All ranks synchronized - starting training together")
    
    # Initialize trainer
    print("\n" + "="*50)
    print("Initializing trainer...")
    print("="*50)
    
    # Don't set checkpoint_dir here - let model_saver construct the full name
    # with student/teacher models and epochs included
    trainer = DistillationTrainer(
        student_wrapper=student_wrapper,
        teacher_forward=teacher_forward,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        distill_args=distill_args,
        output_dir=output_args.checkpoint_dir if output_args.checkpoint_dir else str(output_args.save_dir),
    )
    
    # Run training
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    
    history = trainer.train()
    
    # Save model and checkpoints (only from rank 0)
    if rank == 0:
        print("\n" + "="*50)
        print("Saving model...")
        print("="*50)
        
        saver = ModelSaver(
            student_wrapper=student_wrapper,
            teacher_args=teacher_args,
            student_args=student_args,
            distill_args=distill_args,
            output_args=output_args,
        )
        
        # Get unwrapped model for saving
        student_wrapper.model = student_wrapper.get_base_model()
        saver.save_all(training_log=trainer.training_log)
        
        # Generate loss plot
        print("\n" + "="*50)
        print("Generating loss plot...")
        print("="*50 + "\n")
        
        checkpoint_dir = saver.get_checkpoint_dir()
        output_filename = f"{student_args.model_id.split('/')[-1]}_from_{teacher_args.model_id.split('/')[-1]}_loss.png"
        plotting.plot_from_checkpoint(str(checkpoint_dir), output_filename)
        
        print("\n" + "="*50)
        print("Training complete!")
        print(f"Models saved to: {checkpoint_dir}")
        print("="*50 + "\n")
    
    # Clean up DDP
    if use_ddp:
        cleanup_memory(rank)  # Clean before final barrier operation
        dist.destroy_process_group()
    
    # Final cleanup
    cleanup_memory(rank)


if __name__ == "__main__":
    main()
