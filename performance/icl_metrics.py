"""
ICL-specific metrics collection for in-context learning inference.
Wraps icl/inference.py to collect performance metrics during Ollama inference.
"""

from .metrics_collector import MetricsCollector
from typing import Dict, Any, Optional


class ICLMetricsCollector(MetricsCollector):
    """
    Specialized collector for ICL inference tasks.
    Tracks inference latency, GPU memory during Ollama calls.
    """
    
    def __init__(self, model_name: str, zs: str, think: str, personas: str, 
                 dataset: str = "cab"):
        """
        Initialize ICL metrics collector with experiment parameters.
        
        Args:
            model_name: Ollama model (e.g., "deepseek-r1:7b")
            zs: Zero-shot flag ("true" or "false")
            think: Chain-of-thought condition ("w_t" or "wo_t")
            personas: Persona flag ("w_p" or "wo_p")
            dataset: Dataset name (default: "cab")
        """
        # Build experiment name from parameters
        exp_name = f"icl_{dataset}_{zs}_{think}_{personas}"
        super().__init__(exp_name, model_name)
        
        self.zs = zs
        self.think = think
        self.personas = personas
        self.dataset = dataset
        
        # Add experiment details to metrics
        self.metrics["experiment_params"] = {
            "zs": zs,
            "think": think,
            "personas": personas,
            "dataset": dataset,
        }
    
    def record_inference(self, start_time: float, response_text: str,
                        num_input_tokens: Optional[int] = None,
                        num_output_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Record metrics for a single inference call to Ollama.
        
        Args:
            start_time: Start time from start_timing()
            response_text: Generated response (for token counting)
            num_input_tokens: Number of input tokens (if available)
            num_output_tokens: Number of output tokens (if available)
        
        Returns:
            Dictionary of recorded metrics for this inference
        """
        additional = {}
        
        # Estimate tokens if not provided
        if num_input_tokens is None and response_text:
            # Rough estimate: ~4 chars per token
            num_output_tokens = len(response_text) // 4
            additional["estimated_output_tokens"] = num_output_tokens
        
        if num_input_tokens is not None:
            additional["input_tokens"] = num_input_tokens
        if num_output_tokens is not None:
            additional["output_tokens"] = num_output_tokens
        
        return self.stop_timing(start_time, batch_size=1, 
                               additional_metrics=additional)
    
    def record_batch_inference(self, start_time: float, batch_size: int,
                              num_output_tokens_total: Optional[int] = None) -> Dict[str, Any]:
        """
        Record metrics for batch inference (multiple samples).
        
        Args:
            start_time: Start time from start_timing()
            batch_size: Number of samples processed
            num_output_tokens_total: Total output tokens for batch
        
        Returns:
            Dictionary of recorded metrics
        """
        additional = {}
        if num_output_tokens_total is not None:
            additional["total_output_tokens"] = num_output_tokens_total
            additional["avg_tokens_per_sample"] = num_output_tokens_total / batch_size
        
        return self.stop_timing(start_time, batch_size=batch_size,
                               additional_metrics=additional)
    
    def save_results(self, output_dir: str = "performance/results",
                    organize_by_type: bool = True,
                    include_summary: bool = True):
        """
        Save ICL results to organized directory structure.
        
        Results are saved to: performance/results/icl/{YYYY-MM-DD}/
        
        Args:
            output_dir: Base results directory
            organize_by_type: Whether to use organized subdirectories
            include_summary: Include summary statistics
        
        Returns:
            Path to saved results file
        """
        return super().save_results(
            output_dir=output_dir,
            result_type="icl",
            organize_by_type=organize_by_type,
            include_summary=include_summary
        )
