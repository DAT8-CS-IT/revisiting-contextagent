"""Distillation training loop with KL divergence loss."""

import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from kd_args import DistillationArgs, StudentArgs
from student_model import StudentModelWrapper


def print_gpu_memory(rank: int, stage: str):
    """Print GPU memory usage for debugging OOM issues."""
    if rank == 0 and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[Memory] {stage}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


class DistillationTrainer:
    """Handles knowledge distillation training."""
    
    def __init__(
        self,
        student_wrapper: StudentModelWrapper,
        teacher_forward: Callable,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader],
        distill_args: DistillationArgs,
        output_dir: str,
    ):
        self.student_wrapper = student_wrapper
        self.teacher_forward = teacher_forward
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.distill_args = distill_args
        self.output_dir = Path(output_dir)
        self.device = student_wrapper.device
        
        # Distributed training info
        self.rank = student_wrapper.rank
        self.world_size = student_wrapper.world_size
        self.use_ddp = student_wrapper.use_ddp
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.training_log: List[Dict] = []
        self.best_eval_loss = float("inf")
    
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute KL divergence loss + optional CE loss with numerical stability.
        
        Args:
            student_logits: (batch, seq_len, student_vocab_size)
            teacher_logits: (batch, seq_len, teacher_vocab_size)
            labels: (batch, seq_len) - target token IDs
        
        Returns:
            total_loss, kd_loss, ce_loss
        """
        T = self.distill_args.temperature
        alpha = self.distill_args.kd_alpha
        
        # Handle different vocab sizes between teacher and student
        batch_size, seq_len, student_vocab_size = student_logits.shape
        teacher_batch, teacher_seq, teacher_vocab_size = teacher_logits.shape
        
        # If vocab sizes differ, align to the smaller vocabulary (use common tokens)
        if student_vocab_size != teacher_vocab_size:
            common_vocab_size = min(student_vocab_size, teacher_vocab_size)
            if self.rank == 0:
                self._log(f"[WARNING] Vocab size mismatch: student={student_vocab_size}, teacher={teacher_vocab_size}. Using common vocab size={common_vocab_size}")
            student_logits = student_logits[:, :, :common_vocab_size]
            teacher_logits = teacher_logits[:, :, :common_vocab_size]
            vocab_size = common_vocab_size
        else:
            vocab_size = student_vocab_size
        
        # Flatten logits for more efficient memory usage
        student_logits_flat = student_logits.reshape(-1, vocab_size)
        teacher_logits_flat = teacher_logits.reshape(-1, vocab_size)
        
        # Normalize logits by subtracting max for numerical stability (log-sum-exp trick)
        student_logits_flat = student_logits_flat - student_logits_flat.detach().max(dim=-1, keepdim=True)[0]
        teacher_logits_flat = teacher_logits_flat - teacher_logits_flat.detach().max(dim=-1, keepdim=True)[0]
        
        # Temperature-scaled log probabilities
        student_log_probs = F.log_softmax(student_logits_flat / T, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits_flat / T, dim=-1)
        
        # Compute probabilities from log_probs (this is numerically stable)
        teacher_probs = torch.exp(teacher_log_probs)
        
        # KL divergence: KL(P||Q) = sum(P * (log(P) - log(Q)))
        # Rewrite as: sum(P * log(P)) - sum(P * log(Q))
        # This avoids computing log(P) twice and is more numerically stable
        kd_loss_per_sample = torch.sum(teacher_probs * (teacher_log_probs - student_log_probs), dim=-1)
        kd_loss = kd_loss_per_sample.mean()
        
        # Clamp to catch any remaining NaN issues
        kd_loss = torch.clamp(kd_loss, min=0.0, max=100.0)
        
        # CRITICAL: Do NOT create new tensors detached from computation graph
        # If NaN occurs, log warning but continue with KD loss (CE loss is fallback only)
        if torch.isnan(kd_loss):
            self._log(f"WARNING: KD loss is NaN! Using teacher_probs and logits separately...")
            # Recompute with safer numerics - use KL divergence directly without probs
            kd_loss = -torch.sum(torch.softmax(teacher_logits_flat / T, dim=-1) * F.log_softmax(student_logits_flat / T, dim=-1), dim=-1).mean()
            kd_loss = torch.clamp(kd_loss, min=0.0, max=100.0)
        
        # Cross-entropy loss (if labels provided and not all padding)
        ce_loss = torch.tensor(0.0, device=self.device, dtype=student_logits.dtype)
        if labels is not None and vocab_size == student_vocab_size:
            # Only compute CE loss if vocab sizes matched (labels are guaranteed to be valid)
            labels_flat = labels.reshape(-1)
            
            # Check if there are any non-padding labels
            valid_mask = labels_flat != -100
            num_valid_labels = valid_mask.sum().item()
            
            if num_valid_labels > 0:
                # Only compute CE loss if there are non-padding labels
                try:
                    ce_loss = F.cross_entropy(
                        student_logits_flat,
                        labels_flat,
                        ignore_index=-100,
                        reduction='mean'
                    )
                    # Replace NaN with 0 (happens when all predictions for valid labels are wrong)
                    if torch.isnan(ce_loss):
                        ce_loss = torch.tensor(0.0, device=self.device, dtype=student_logits.dtype)
                except Exception as e:
                    self._log(f"WARNING: Error computing CE loss: {e}")
                    ce_loss = torch.tensor(0.0, device=self.device, dtype=student_logits.dtype)
            else:
                # All labels are padding, set CE loss to 0
                ce_loss = torch.tensor(0.0, device=self.device, dtype=student_logits.dtype)
        
        # Weighted combination
        total_loss = alpha * ce_loss + (1 - alpha) * kd_loss
        
        # Final safety check for NaN with diagnostic info
        if torch.isnan(total_loss):
            self._log(f"WARNING: Total loss is NaN! CE_loss={ce_loss.item()}, KD_loss={kd_loss.item()}, alpha={alpha}")
            # Use KD loss only (CE is fallback)
            total_loss = kd_loss
        
        return total_loss, kd_loss, ce_loss
    
    def _log(self, msg: str):
        """Log only from rank 0 to avoid duplicate messages."""
        if self.rank == 0:
            print(msg)
    
    def train_epoch(self) -> Dict[str, float]:
        """Run one training epoch with gradient accumulation.
        
        Returns:
            Dictionary with epoch statistics.
        """
        self.student_wrapper.train_mode()
        
        # DEBUG: Check that we have trainable parameters
        total_params = sum(p.numel() for p in self.student_wrapper.model.parameters())
        trainable_params = sum(p.numel() for p in self.student_wrapper.model.parameters() if p.requires_grad)
        if self.rank == 0:
            self._log(f"[DEBUG] Total params: {total_params}, Trainable: {trainable_params}")
            if trainable_params == 0:
                self._log("WARNING: No trainable parameters found! LoRA may not be initialized.")
        
        total_loss = 0.0
        total_kd_loss = 0.0
        total_ce_loss = 0.0
        num_batches = 0
        accumulation_steps = self.distill_args.gradient_accumulation_steps
        
        # Only show progress bar on rank 0
        if self.rank == 0:
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {self.current_epoch}",
                leave=False
            )
        else:
            progress_bar = self.train_dataloader
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
            labels = batch.get("labels", None)
            if labels is not None:
                labels = labels.to(self.device)
            
            # Student forward pass
            student_logits = self.student_wrapper.forward(input_ids, attention_mask)
            
            # Teacher forward pass (no grad)
            with torch.no_grad():
                teacher_logits = self.teacher_forward(input_ids, attention_mask)
            
            # DEBUG: Check logits have gradients on first batch
            if batch_idx == 0 and self.rank == 0:
                self._log(f"[DEBUG] Student logits grad_fn: {student_logits.grad_fn}, requires_grad: {student_logits.requires_grad}")
                self._log(f"[DEBUG] Teacher logits grad_fn: {teacher_logits.grad_fn}, requires_grad: {teacher_logits.requires_grad}")
            
            # Compute loss
            total, kd, ce = self.compute_distillation_loss(
                student_logits, teacher_logits, labels
            )
            
            # Scale loss by accumulation steps
            scaled_loss = total / accumulation_steps
            
            # Backward pass
            if batch_idx % accumulation_steps == 0:
                self.student_wrapper.zero_grad()
            
            # DEBUG: Check loss tensor graph before backward
            if batch_idx < 1 and self.rank == 0:
                self._log(f"[DEBUG] Loss tensor requires_grad: {scaled_loss.requires_grad}, grad_fn: {scaled_loss.grad_fn}")
                # Check if model has trainable params with gradients enabled
                model = self.student_wrapper.model
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        self._log(f"[DEBUG] Trainable param: {name} shape={param.shape}")
                        break
            
            self.student_wrapper.backward(scaled_loss)
            
            # DEBUG: Check if gradients are flowing on first few batches
            if batch_idx < 3 and self.rank == 0:
                total_grad_norm = 0.0
                has_grads = False
                for name, param in self.student_wrapper.model.named_parameters():
                    if param.grad is not None:
                        has_grads = True
                        total_grad_norm += param.grad.norm().item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                self._log(f"[DEBUG] Batch {batch_idx}: Loss={total.item():.4f}, Has grads: {has_grads}, Grad Norm={total_grad_norm:.6f}")
            
            # Accumulate statistics (before deleting tensors)
            loss_scalar = total.item()
            kd_loss_scalar = kd.item()
            ce_loss_scalar = ce.item() if ce.item() > 0 else 0.0
            
            total_loss += loss_scalar
            total_kd_loss += kd_loss_scalar
            total_ce_loss += ce_loss_scalar
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar (only on rank 0 where progress_bar is tqdm)
            current_lr = self.student_wrapper.get_lr()
            if self.rank == 0:
                progress_bar.set_postfix({
                    "loss": loss_scalar,
                    "lr": current_lr
                })
            
            # Logging (before tensor deletion)
            if self.global_step % self.distill_args.logging_steps == 0:
                self.training_log.append({
                    "step": self.global_step,
                    "epoch": self.current_epoch,
                    "loss": loss_scalar,
                    "kd_loss": kd_loss_scalar,
                    "ce_loss": ce_loss_scalar,
                    "lr": current_lr,
                })
            
            # Update optimizer only every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                self.student_wrapper.step()
                
                # Clean up intermediate tensors to prevent memory buildup
                del student_logits, teacher_logits, total, kd, ce
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches
        avg_kd_loss = total_kd_loss / num_batches
        avg_ce_loss = total_ce_loss / num_batches if num_batches > 0 else 0.0
        
        # Synchronize metrics across ranks if using DDP
        if self.use_ddp:
            # Clean GPU memory before barrier to prevent OOM during synchronization
            torch.cuda.empty_cache()
            
            avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
            avg_kd_loss_tensor = torch.tensor(avg_kd_loss, device=self.device)
            avg_ce_loss_tensor = torch.tensor(avg_ce_loss, device=self.device)
            dist.all_reduce(avg_loss_tensor)
            dist.all_reduce(avg_kd_loss_tensor)
            dist.all_reduce(avg_ce_loss_tensor)
            avg_loss = (avg_loss_tensor / self.world_size).item()
            avg_kd_loss = (avg_kd_loss_tensor / self.world_size).item()
            avg_ce_loss = (avg_ce_loss_tensor / self.world_size).item()
            
            # Clean up after sync
            del avg_loss_tensor, avg_kd_loss_tensor, avg_ce_loss_tensor
            torch.cuda.empty_cache()
        
        return {
            "train_loss": avg_loss,
            "train_kd_loss": avg_kd_loss,
            "train_ce_loss": avg_ce_loss,
        }
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on the eval dataset.
        
        Returns:
            Dictionary with evaluation statistics.
        """
        if self.eval_dataloader is None:
            return {}
        
        self.student_wrapper.eval_mode()
        
        total_loss = 0.0
        num_batches = 0
        
        # Only show progress bar on rank 0
        if self.rank == 0:
            progress_bar = tqdm(
                self.eval_dataloader,
                desc="Evaluating",
                leave=False
            )
        else:
            progress_bar = self.eval_dataloader
        
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask", torch.ones_like(input_ids)).to(self.device)
            labels = batch.get("labels", None)
            if labels is not None:
                labels = labels.to(self.device)
            
            # Forward passes
            student_logits = self.student_wrapper.forward(input_ids, attention_mask)
            teacher_logits = self.teacher_forward(input_ids, attention_mask)
            
            # Compute loss
            total, _, _ = self.compute_distillation_loss(
                student_logits, teacher_logits, labels
            )
            
            total_loss += total.item()
            num_batches += 1
        
        eval_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Synchronize eval_loss across ranks if using DDP
        if self.use_ddp:
            # Clean GPU memory before barrier to prevent OOM during synchronization
            torch.cuda.empty_cache()
            
            eval_loss_tensor = torch.tensor(eval_loss, device=self.device)
            dist.all_reduce(eval_loss_tensor)
            eval_loss = (eval_loss_tensor / self.world_size).item()
            
            # Clean up after sync
            del eval_loss_tensor
            torch.cuda.empty_cache()
        
        return {"eval_loss": eval_loss}
    
    def train(self) -> Dict[str, List]:
        """Run full training loop.
        
        Returns:
            Dictionary with training history.
        """
        self._log(f"Starting training for {self.distill_args.num_epochs} epochs...")
        
        all_train_losses = []
        all_eval_losses = []
        
        for epoch in range(self.distill_args.num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            all_train_losses.append(train_metrics["train_loss"])
            self._log(f"Epoch {epoch} - Train Loss: {train_metrics['train_loss']:.4f}")
            
            # Evaluation
            if self.eval_dataloader is not None:
                eval_metrics = self.evaluate()
                all_eval_losses.append(eval_metrics["eval_loss"])
                self.training_log.append({
                    "step": self.global_step,
                    "epoch": epoch,
                    "eval_loss": eval_metrics["eval_loss"],
                })
                self._log(f"Epoch {epoch} - Eval Loss: {eval_metrics['eval_loss']:.4f}")
                
                # Track best model
                if eval_metrics["eval_loss"] < self.best_eval_loss:
                    self.best_eval_loss = eval_metrics["eval_loss"]
                    self._log(f"  -> Best eval loss so far!")
        
        history = {
            "train_losses": all_train_losses,
            "eval_losses": all_eval_losses,
            "training_log": self.training_log,
        }
        
        return history
    
    def save_training_log(self, path: str):
        """Save training log as JSON (only from rank 0)."""
        if not self.rank == 0:
            return
        with open(path, "w") as f:
            json.dump(self.training_log, f, indent=2)
        print(f"Training log saved to {path}")
