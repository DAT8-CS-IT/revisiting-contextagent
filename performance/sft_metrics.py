"""
SFT-specific metrics collection for supervised fine-tuning.
Wraps LLaMA-Factory inference/training to collect performance metrics.

Tracks both training phase (loss, learning rate, epochs) and 
evaluation phase (inference latency, accuracy, throughput).
"""

from .metrics_collector import MetricsCollector
from typing import Dict, Any, Optional, List


class SFTMetricsCollector(MetricsCollector):
    """
    Specialized collector for SFT training and evaluation.
    Tracks training/evaluation time, GPU memory, loss progression, accuracy metrics.
    
    Supports two phases:
    - Training phase: loss, learning rate, step/epoch metrics
    - Evaluation phase: inference latency, accuracy, throughput
    """
    
    def __init__(self, model_name: str, experiment_conf: str, 
                 base_model_value: str, think: str = "w_t", personas: str = "w_p"):
        """
        Initialize SFT metrics collector with experiment parameters.
        
        Args:
            model_name: SFT model identifier
            experiment_conf: Configuration name (e.g., "baseline", "pruned_5percent")
            base_model_value: Base model used (e.g., "qwen2.5")
            think: Chain-of-thought condition ("w_t" or "wo_t")
            personas: Persona flag ("w_p" or "wo_p")
        """
        # Build experiment name from parameters
        exp_name = f"sft_{experiment_conf}_{think}_{personas}"
        super().__init__(exp_name, model_name)
        
        self.experiment_conf = experiment_conf
        self.base_model_value = base_model_value
        self.think = think
        self.personas = personas
        
        self.training_metrics = []
        self.evaluation_metrics = []
        
        # Add experiment details to metrics
        self.metrics["experiment_params"] = {
            "experiment_conf": experiment_conf,
            "base_model_value": base_model_value,
            "think": think,
            "personas": personas,
        }
        self.metrics["phase"] = None  # Will be "training" or "evaluation"
    
    def record_training_step(self, step: int, loss: float, 
                            learning_rate: Optional[float] = None,
                            duration_seconds: Optional[float] = None,
                            gpu_memory_gb: Optional[float] = None) -> Dict[str, Any]:
        """
        Record metrics for a training step.
        
        Args:
            step: Training step number
            loss: Loss value for this step
            learning_rate: Current learning rate (optional)
            duration_seconds: Time taken for this step (optional)
            gpu_memory_gb: GPU memory used in GB (optional, auto-captured if not provided)
        
        Returns:
            Dictionary of recorded metrics
        """
        # Capture GPU memory if not provided
        if gpu_memory_gb is None:
            try:
                gpu_info = self.get_gpu_memory_from_nvidia_smi()
                if gpu_info and "gpu_memory_used_gb" in gpu_info:
                    gpu_memory_gb = gpu_info["gpu_memory_used_gb"]
            except Exception as e:
                # Silently skip if GPU info unavailable
                pass
        
        metric = {
            "step": step,
            "loss": loss,
            "timestamp": self.metrics["timestamp"]
        }
        
        if learning_rate is not None:
            metric["learning_rate"] = learning_rate
        if duration_seconds is not None:
            metric["duration_seconds"] = duration_seconds
        if gpu_memory_gb is not None:
            metric["gpu_memory_gb"] = gpu_memory_gb
            # Track peak GPU memory
            if "peak_gpu_memory_gb" not in self.metrics or self.metrics["peak_gpu_memory_gb"] < gpu_memory_gb:
                self.metrics["peak_gpu_memory_gb"] = gpu_memory_gb
        
        self.training_metrics.append(metric)
        self.metrics["phase"] = "training"
        
        return metric
    
    def record_training_epoch(self, epoch: int, avg_loss: float,
                             epoch_duration_seconds: float,
                             gpu_memory_gb: Optional[float] = None) -> Dict[str, Any]:
        """
        Record metrics for a complete training epoch.
        
        Args:
            epoch: Epoch number
            avg_loss: Average loss across epoch
            epoch_duration_seconds: Total time for epoch
            gpu_memory_gb: Peak GPU memory during epoch (optional)
        
        Returns:
            Dictionary of recorded epoch metrics
        """
        # Capture GPU memory if not provided
        if gpu_memory_gb is None:
            try:
                gpu_info = self.get_gpu_memory_from_nvidia_smi()
                if gpu_info and "gpu_memory_used_gb" in gpu_info:
                    print(f"[DEBUG]: GPU memory info")
                    gpu_memory_gb = gpu_info["gpu_memory_used_gb"]
            except Exception as e:
                # Silently skip if GPU info unavailable
                pass
        
        metric = {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "duration_seconds": epoch_duration_seconds,
        }
        
        if gpu_memory_gb is not None:
            metric["gpu_memory_gb"] = gpu_memory_gb
            # Track peak GPU memory
            if "peak_gpu_memory_gb" not in self.metrics or self.metrics["peak_gpu_memory_gb"] < gpu_memory_gb:
                self.metrics["peak_gpu_memory_gb"] = gpu_memory_gb
        
        self.training_metrics.append(metric)
        self.metrics["phase"] = "training"
        return metric
    
    def record_inference(self, start_time: float, response_text: str,
                        num_input_tokens: Optional[int] = None,
                        num_output_tokens: Optional[int] = None,
                        gpu_memory_gb: Optional[float] = None) -> Dict[str, Any]:
        """
        Record metrics for a single inference call during SFT evaluation.
        Mirrors ICL's per-sample inference recording pattern.
        
        Args:
            start_time: Start time from start_timing()
            response_text: Generated response
            num_input_tokens: Number of input tokens (optional)
            num_output_tokens: Number of output tokens (optional)
            gpu_memory_gb: GPU memory used in GB (optional, auto-captured if not provided)
        
        Returns:
            Dictionary of recorded inference metrics
        """
        # Capture GPU memory if not provided
        if gpu_memory_gb is None:
            try:
                gpu_info = self.get_gpu_memory_from_nvidia_smi()
                if gpu_info and "gpu_memory_used_gb" in gpu_info:
                    gpu_memory_gb = gpu_info["gpu_memory_used_gb"]
            except Exception as e:
                # Silently skip if GPU info unavailable
                pass
        
        additional = {}
        
        # Estimate tokens if not provided
        if num_input_tokens is None and response_text:
            num_output_words = len(response_text.split())
            additional["numbers_of_words"] = num_output_words
        
        if num_input_tokens is not None:
            additional["input_tokens"] = num_input_tokens
        if num_output_tokens is not None:
            additional["output_tokens"] = num_output_tokens
        if gpu_memory_gb is not None:
            additional["gpu_memory_gb"] = gpu_memory_gb
            # Track peak GPU memory
            if "peak_gpu_memory_gb" not in self.metrics or self.metrics["peak_gpu_memory_gb"] < gpu_memory_gb:
                self.metrics["peak_gpu_memory_gb"] = gpu_memory_gb
        
        self.metrics["phase"] = "evaluation"
        return self.stop_timing(start_time, batch_size=1, additional_metrics=additional)
    
    def record_evaluation(self, eval_metric_name: str, eval_metric_value: float,
                         num_samples: int,
                         eval_duration_seconds: Optional[float] = None) -> Dict[str, Any]:
        """
        Record evaluation metrics (accuracy, BLEU, etc.).
        
        Args:
            eval_metric_name: Name of metric (e.g., "accuracy", "f1")
            eval_metric_value: Metric value
            num_samples: Number of samples evaluated
            eval_duration_seconds: Time taken for evaluation
        
        Returns:
            Dictionary of recorded metrics
        """
        metric = {
            "metric": eval_metric_name,
            "value": eval_metric_value,
            "num_samples": num_samples,
            "timestamp": self.metrics["timestamp"]
        }
        
        if eval_duration_seconds is not None:
            metric["duration_seconds"] = eval_duration_seconds
        
        self.evaluation_metrics.append(metric)
        self.metrics["phase"] = "evaluation"
        
        return metric
    
    def record_inference_batch(self, batch_size: int, 
                              avg_latency_seconds: float,
                              num_output_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Record metrics for inference on a batch during evaluation.
        
        Args:
            batch_size: Number of samples in batch
            avg_latency_seconds: Average latency per sample
            num_output_tokens: Total output tokens (optional)
        
        Returns:
            Dictionary of recorded metrics
        """
        total_time = avg_latency_seconds * batch_size
        additional = {}
        
        if num_output_tokens is not None:
            additional["total_output_tokens"] = num_output_tokens
            additional["tokens_per_sample"] = num_output_tokens / batch_size
        
        return self.stop_timing(time.perf_counter() - total_time,
                               batch_size=batch_size,
                               additional_metrics=additional)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for training phase.
        
        Returns:
            Dictionary with training summary
        """
        if not self.training_metrics:
            return {}
        
        losses = [m.get("loss") or m.get("avg_loss") for m in self.training_metrics]
        durations = [m.get("duration_seconds", 0) for m in self.training_metrics]
        
        return {
            "num_training_steps": len(self.training_metrics),
            "final_loss": losses[-1] if losses else None,
            "min_loss": min(losses) if losses else None,
            "avg_loss": sum(losses) / len(losses) if losses else None,
            "total_training_time": sum(durations),
        }
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for evaluation phase.
        
        Returns:
            Dictionary with evaluation summary
        """
        if not self.evaluation_metrics:
            return {}
        
        durations = [m.get("duration_seconds", 0) for m in self.evaluation_metrics]
        
        return {
            "num_evaluations": len(self.evaluation_metrics),
            "total_eval_time": sum(durations),
            "avg_eval_time": sum(durations) / len(durations) if durations else 0,
            "metrics": [
                {
                    "name": m.get("metric", "unknown"),
                    "value": m.get("value"),
                    "num_samples": m.get("num_samples", 0)
                }
                for m in self.evaluation_metrics
            ]
        }
    
    def save_results(self, output_dir: str = "performance/results",
                    organize_by_type: bool = True,
                    include_summary: bool = True) -> str:
        """
        Save training and evaluation metrics to organized directory structure.
        
        Results are saved to organized paths:
        - Training: performance/results/sft_training/{YYYY-MM-DD}/
        - Evaluation: performance/results/sft_evaluation/{YYYY-MM-DD}/
        
        Args:
            output_dir: Base results directory
            organize_by_type: Whether to use organized subdirectories
            include_summary: Whether to include summaries
        
        Returns:
            Path to saved results file
        """
        # Add phase-specific metrics
        self.metrics["training_metrics"] = self.training_metrics
        self.metrics["evaluation_metrics"] = self.evaluation_metrics
        
        if include_summary:
            self.metrics["training_summary"] = self.get_training_summary()
            self.metrics["evaluation_summary"] = self.get_evaluation_summary()
        
        # For training phase, calculate total_runtime from training_metrics
        phase = self.metrics.get("phase", "unknown")
        if phase and "training" in phase.lower():
            if self.training_metrics:
                training_summary = self.get_training_summary()
                self.metrics["total_runtime_seconds"] = training_summary.get("total_training_time", 0)
        
        # For evaluation phase, calculate from evaluation_metrics
        elif phase and "evaluation" in phase.lower():
            if self.evaluation_metrics:
                eval_summary = self.get_evaluation_summary()
                self.metrics["total_runtime_seconds"] = eval_summary.get("total_eval_time", 0)
        
        # Determine result type based on phase
        phase = self.metrics.get("phase", "unknown")
        if phase is None:
            phase = "unknown"
        if "training" in phase.lower():
            result_type = "sft_training"
        elif "evaluation" in phase.lower():
            result_type = "sft_evaluation"
        else:
            result_type = "sft_unknown"
        
        return super().save_results(
            output_dir=output_dir,
            result_type=result_type,
            organize_by_type=organize_by_type,
            include_summary=include_summary
        )


# For time.perf_counter() call in record_inference_batch
import time
