"""
Shared metrics collection utilities for ICL and SFT experiments.
Collects: inference time, GPU memory, throughput, and other performance indicators.
"""

import time
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class MetricsCollector:
    """
    Base class for collecting performance metrics during inference/training.
    
    Attributes:
        experiment_name: Identifier for the experiment (e.g., "icl_false_w_t_w_p")
        model_name: Model identifier (e.g., "deepseek-r1:7b")
        metrics: Dictionary storing all collected metrics
    """
    
    def __init__(self, experiment_name: str, model_name: str):
        """
        Initialize metrics collector.
        
        Args:
            experiment_name: Experiment identifier
            model_name: Model name/path
        """
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.start_time = None
        self.batch_start_time = None
        
        self.metrics = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "experiment": experiment_name,
            "inference_times": [],
            "gpu_memory_history": [],
            "peak_gpu_memory_gb": 0,
            "total_samples": 0,
            "total_runtime_seconds": 0,
        }
        
        # Reset GPU stats if available
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
    
    def get_allocated_gpus(self) -> List[int]:
        """
        Detect which GPUs are allocated to this job.
        
        Checks in order:
        1. CUDA_VISIBLE_DEVICES
        2. NVIDIA_VISIBLE_DEVICES
        3. Default to GPU 0
        
        Returns:
            List of GPU indices allocated to this job
        """
        import os
        
        # Check CUDA_VISIBLE_DEVICES.
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
        if cuda_visible and cuda_visible != '-1':
            try:
                gpu_list = [int(x.strip()) for x in cuda_visible.split(',')]
                return gpu_list
            except (ValueError, AttributeError):
                pass
        
        # Check NVIDIA_VISIBLE_DEVICES
        nvidia_visible = os.environ.get('NVIDIA_VISIBLE_DEVICES', '').strip()
        if nvidia_visible and nvidia_visible != '-1':
            try:
                gpu_list = [int(x.strip()) for x in nvidia_visible.split(',')]
                return gpu_list
            except (ValueError, AttributeError):
                pass
        
        # Default to GPU 0
        return [0]
    
    def get_gpu_memory_from_nvidia_smi(self, verbose: bool = True) -> Dict[str, float]:
        """
        Get current GPU memory usage via nvidia-smi for allocated GPUs (for external processes like Ollama).
        
        For multi-GPU jobs, sums memory across all allocated GPUs. Detects GPU allocation via
        CUDA_VISIBLE_DEVICES and NVIDIA_VISIBLE_DEVICES.
        
        Args:
            verbose: If True, print debug information about GPU queries
        
        Returns:
            Dictionary with current and peak memory in GB (aggregated across allocated GPUs),
            or empty dict if nvidia-smi unavailable
        """
        try:
            import os
            
            # Get list of GPUs allocated to this job
            allocated_gpus = self.get_allocated_gpus()
            if verbose:
                print(f"[DEBUG] Allocated GPUs: {allocated_gpus}")
                print(f"[DEBUG] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
            
            # Query memory for all GPUs (use GPU indices as shown by nvidia-smi)
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,nounits,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if verbose:
                print(f"[DEBUG] nvidia-smi return code: {result.returncode}")
                print(f"[DEBUG] nvidia-smi stdout:\n{result.stdout}")
                if result.stderr:
                    print(f"[DEBUG] nvidia-smi stderr:\n{result.stderr}")
            
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                if lines:
                    total_used_mb = 0
                    total_capacity_mb = 0
                    gpus_found = 0
                    
                    # Parse output: each line has "gpu_idx,used_mb,total_mb"
                    # When CUDA_VISIBLE_DEVICES is set, nvidia-smi still shows physical GPU indices
                    for line in lines:
                        parts = line.split(',')
                        if len(parts) >= 3:
                            try:
                                gpu_idx = int(parts[0].strip())
                                used_mb = float(parts[1].strip())
                                total_mb = float(parts[2].strip())
                                
                                # Only aggregate if this GPU is in allocated list
                                if gpu_idx in allocated_gpus:
                                    total_used_mb += used_mb
                                    total_capacity_mb += total_mb
                                    gpus_found += 1
                                    if verbose:
                                        print(f"[DEBUG] GPU {gpu_idx}: {used_mb} MB / {total_mb} MB")
                            except (ValueError, IndexError) as e:
                                if verbose:
                                    print(f"[DEBUG] Failed to parse line: {line}, error: {e}")
                                continue
                    
                    if gpus_found > 0:
                        result_dict = {
                            "gpu_memory_used_gb": total_used_mb / 1024,
                            "gpu_memory_total_gb": total_capacity_mb / 1024,
                            "gpu_memory_percent": (total_used_mb / total_capacity_mb * 100) if total_capacity_mb > 0 else 0,
                            "num_gpus": gpus_found
                        }
                        if verbose:
                            print(f"[DEBUG] Final result: {result_dict}")
                        return result_dict
                    elif verbose:
                        print(f"[DEBUG] No matching GPUs found in allocated list")
            elif verbose:
                print(f"[DEBUG] nvidia-smi returned empty output")
                
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
            if verbose:
                print(f"[DEBUG] Exception in get_gpu_memory_from_nvidia_smi: {e}")
        
        return {}
    
    def start_timing(self) -> float:
        """
        Start timing a batch inference. Resets GPU peak memory stats.
        
        Returns:
            Start time (perf_counter value)
        """
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        self.batch_start_time = time.perf_counter()
        if self.start_time is None:
            self.start_time = self.batch_start_time
        
        return self.batch_start_time
    
    def stop_timing(self, start_time: float, batch_size: int = 1, 
                   additional_metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Stop timing and record metrics for a batch.
        
        Args:
            start_time: Start time from start_timing()
            batch_size: Number of samples in batch (default: 1)
            additional_metrics: Extra metrics to attach (e.g., token count, loss)
        
        Returns:
            Dictionary of recorded metrics for this batch
        """
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        duration = time.perf_counter() - start_time
        
        batch_metrics = {
            "duration_seconds": duration,
            "batch_size": batch_size,
            "throughput_samples_per_sec": batch_size / duration if duration > 0 else 0,
        }
        
        # Collect GPU memory - try nvidia-smi first (for Ollama/external processes)
        gpu_info = self.get_gpu_memory_from_nvidia_smi()
        if gpu_info:
            batch_metrics.update(gpu_info)
            self.metrics["peak_gpu_memory_gb"] = max(
                self.metrics["peak_gpu_memory_gb"],
                gpu_info.get("gpu_memory_used_gb", 0)
            )
            self.metrics["gpu_memory_history"].append({
                "timestamp": datetime.now().isoformat(),
                **gpu_info
            })
        # Fall back to torch if available
        elif HAS_TORCH and torch.cuda.is_available():
            peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            current_mem_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            
            batch_metrics["gpu_peak_memory_gb"] = peak_mem_gb
            batch_metrics["gpu_current_memory_gb"] = current_mem_gb
            
            self.metrics["peak_gpu_memory_gb"] = max(
                self.metrics["peak_gpu_memory_gb"], 
                peak_mem_gb
            )
            self.metrics["gpu_memory_history"].append({
                "timestamp": datetime.now().isoformat(),
                "memory_gb": current_mem_gb,
                "peak_memory_gb": peak_mem_gb
            })
        
        # Add any additional metrics
        if additional_metrics:
            batch_metrics.update(additional_metrics)
        
        self.metrics["inference_times"].append(batch_metrics)
        self.metrics["total_samples"] += batch_size
        
        return batch_metrics
    
    def get_gpu_device_info(self) -> Dict[str, Any]:
        """
        Get GPU device information via nvidia-smi.
        
        Returns:
            Dictionary with GPU name and CUDA info
        """
        info = {}
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    parts = lines[0].split(',')
                    info["gpu_device"] = parts[0].strip()
                    if len(parts) > 1:
                        info["driver_version"] = parts[1].strip()
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass
        
        if HAS_TORCH:
            try:
                info["cuda_version"] = torch.version.cuda
                info["pytorch_version"] = torch.__version__
            except:
                pass
        
        return info
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Calculate summary statistics from collected metrics.
        
        Returns:
            Dictionary with aggregated statistics
        """
        times = [t["duration_seconds"] for t in self.metrics["inference_times"]]
        
        if not times:
            return {}
        
        total_time = sum(times)
        
        return {
            "total_runtime_seconds": total_time,
            "num_batches": len(times),
            "avg_batch_time": total_time / len(times),
            "min_batch_time": min(times),
            "max_batch_time": max(times),
            "total_samples": self.metrics["total_samples"],
            "avg_samples_per_sec": self.metrics["total_samples"] / total_time if total_time > 0 else 0,
            "peak_gpu_memory_gb": self.metrics["peak_gpu_memory_gb"],
        }
    
    def save_results(self, output_dir: str = "performance/results",
                    result_type: str = "unknown",
                    organize_by_type: bool = False,
                    include_summary: bool = True) -> Path:
        """
        Save collected metrics to JSON file.
        
        Args:
            output_dir: Base directory to save results
            result_type: Type of result (icl, sft_training, sft_evaluation)
            organize_by_type: If True, organize as {type}/{date}/ subdirectories
            include_summary: Whether to include summary statistics
        
        Returns:
            Path to saved results file
        """
        # Use organized path if requested
        if organize_by_type and result_type != "unknown":
            from .results_manager import organize_result_path
            output_path = Path(organize_result_path(output_dir, result_type))
        else:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.experiment_name}_{self.model_name.replace('/', '_')}_{timestamp}.json"
        filepath = output_path / filename
        
        # Prepare output
        output_data = self.metrics.copy()
        
        if include_summary:
            output_data["summary"] = self.get_summary()
        
        # Add hardware metadata
        output_data["hardware"] = self.get_gpu_device_info()
        
        # Save to file
        with open(filepath, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"[INFO] Performance metrics saved to {filepath}")
        return filepath
    
    def print_summary(self):
        """Print summary statistics to console."""
        summary = self.get_summary()
        if not summary:
            print("[WARNING] No metrics collected yet")
            return
        
        print("\n" + "=" * 60)
        print(f"Performance Summary: {self.experiment_name} ({self.model_name})")
        print("=" * 60)
        print(f"Total runtime:        {summary['total_runtime_seconds']:.2f}s")
        print(f"Number of batches:    {summary['num_batches']}")
        print(f"Avg batch time:       {summary['avg_batch_time']:.4f}s")
        print(f"Min/Max batch time:   {summary['min_batch_time']:.4f}s / {summary['max_batch_time']:.4f}s")
        print(f"Total samples:        {summary['total_samples']}")
        print(f"Throughput:           {summary['avg_samples_per_sec']:.2f} samples/sec")
        if summary['peak_gpu_memory_gb'] > 0:
            print(f"Peak GPU memory:      {summary['peak_gpu_memory_gb']:.2f} GB")
        print("=" * 60 + "\n")
