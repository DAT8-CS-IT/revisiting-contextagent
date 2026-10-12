"""
Wrapper for SFT training that adds performance metrics collection.
This script instruments LLaMA-Factory training runs to collect performance data.

This script can be layered on top of existing SFT training setup or
called as a subprocess wrapper to "llamafactory-cli train".
"""

import sys
import json
import time
import re
import subprocess
from pathlib import Path
from typing import Optional

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent))

from performance.sft_metrics import SFTMetricsCollector


def _load_json_file(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as file_handle:
            return json.load(file_handle)
    except Exception as exc:
        print(f"[WARNING] Failed to load JSON artifact {path}: {exc}")
        return None


def _load_jsonl_file(path: Path):
    records = []
    if not path.exists():
        return records
    try:
        with open(path, "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
    except Exception as exc:
        print(f"[WARNING] Failed to load JSONL artifact {path}: {exc}")
    return records


def _find_sft_artifact_dir(factory_dir: Path, base_model_value: str) -> Path:
    preferred = factory_dir / "saves" / "cab" / base_model_value / "sft"
    if preferred.exists():
        return preferred

    saves_root = factory_dir / "saves" / "cab"
    candidates = []
    if saves_root.exists():
        for trainer_state_path in saves_root.rglob("trainer_state.json"):
            artifact_dir = trainer_state_path.parent
            if artifact_dir.name != "sft":
                continue
            if base_model_value.lower() in str(artifact_dir).lower():
                candidates.append(artifact_dir)

    if candidates:
        return max(candidates, key=lambda path: (path / "trainer_state.json").stat().st_mtime)

    return preferred


def run_sft_training_with_metrics(
    experiment_conf: str,
    base_model_value: str,
    think: str = "w_t",
    personas: str = "w_p",
    config_path: Optional[str] = None,
):
    """
    Run SFT training with metrics collection.
    
    Args:
        experiment_conf: Configuration name (e.g., "baseline")
        base_model_value: Base model (e.g., "qwen2.5")
        think: Chain-of-thought condition
        personas: Persona condition
        config_path: Path to training config YAML (optional override)
    """
    
    # Initialize metrics collector
    collector = SFTMetricsCollector(
        model_name=f"sft_{experiment_conf}",
        experiment_conf=experiment_conf,
        base_model_value=base_model_value,
        think=think,
        personas=personas,
    )
    
    print(f"[METRICS] Initialized SFT metrics for: {collector.experiment_name}")
    
    # Build training command with absolute paths
    script_dir = Path(__file__).parent
    cta_base = script_dir / "ContextAgent"
    factory_dir = cta_base / "LLaMA-Factory"
    
    # If config_path not provided, use standard location
    if config_path is None:
        # Strip .yaml if already present (e.g., "deepseek_lora_sft.yaml" -> "deepseek_lora_sft")
        conf_name = experiment_conf.replace(".yaml", "") if experiment_conf.endswith(".yaml") else experiment_conf
        config_path = str(factory_dir / "experiments" / "configs" / "cab" / "train" / f"{conf_name}.yaml")
    elif not Path(config_path).is_absolute():
        config_path = str(cta_base / config_path)
    
    # Verify config path exists
    if not Path(config_path).exists():
        print(f"[ERROR] Config file not found: {config_path}")
        raise FileNotFoundError(f"Training config not found: {config_path}")
    
    train_cmd = [
        "python", "-m", "llamafactory.cli",
        "train",
        config_path
    ]
    
    print(f"[METRICS] Running training command: {' '.join(train_cmd)}")
    print(f"[METRICS] Config path: {config_path}")
    print(f"[METRICS] Working directory: {factory_dir}")
    
    # Run training with metrics
    training_start = collector.start_timing()
    
    try:
        # Start training as background process (don't wait for it to finish)
        print(f"[METRICS] Starting training subprocess...")
        proc = subprocess.Popen(
            train_cmd,
            cwd=str(factory_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        
        # Monitor GPU memory while training runs
        print(f"[METRICS] Monitoring GPU memory during training...")
        gpu_measurements = []
        poll_interval = 10  # Check every 10 seconds
        
        while proc.poll() is None:  # While training is still running
            try:
                gpu_info = collector.get_gpu_memory_from_nvidia_smi()
                if gpu_info and "gpu_memory_used_gb" in gpu_info:
                    gpu_measurements.append({
                        "timestamp": time.time(),
                        **gpu_info
                    })
                    # Update peak memory
                    current_used = gpu_info.get("gpu_memory_used_gb", 0)
                    collector.metrics["peak_gpu_memory_gb"] = max(
                        collector.metrics["peak_gpu_memory_gb"],
                        current_used
                    )
            except Exception as e:
                print(f"[DEBUG] Error monitoring GPU: {e}")
            
            time.sleep(poll_interval)
        
        # Get final return code and output
        returncode = proc.returncode
        stdout, stderr = proc.communicate()  # Get any remaining output
        
        training_duration = time.perf_counter() - training_start
        
        # Store GPU measurements
        if gpu_measurements:
            collector.metrics["gpu_measurements_during_training"] = gpu_measurements
            print(f"[METRICS] Captured {len(gpu_measurements)} GPU measurements during training")
        
        if returncode == 0:
            print(f"\n[METRICS] Training completed successfully")
            print(f"[METRICS] Training duration: {training_duration:.2f}s")
            
            # Find and load training artifacts
            artifact_dir = _find_sft_artifact_dir(factory_dir, base_model_value)
            trainer_state_path = artifact_dir / "trainer_state.json"
            all_results_path = artifact_dir / "all_results.json"
            train_results_path = artifact_dir / "train_results.json"
            trainer_log_path = artifact_dir / "trainer_log.jsonl"
            
            print(f"[METRICS] Looking for artifacts in: {artifact_dir}")
            
            trainer_state = _load_json_file(trainer_state_path)
            all_results = _load_json_file(all_results_path)
            train_results = _load_json_file(train_results_path)
            trainer_log_records = _load_jsonl_file(trainer_log_path)
            
            # Store artifact sources for reference
            collector.metrics["artifact_sources"] = {
                "artifact_dir": str(artifact_dir),
                "trainer_state": str(trainer_state_path) if trainer_state_path.exists() else None,
                "all_results": str(all_results_path) if all_results_path.exists() else None,
                "train_results": str(train_results_path) if train_results_path.exists() else None,
                "trainer_log": str(trainer_log_path) if trainer_log_path.exists() else None,
            }
            
            # Parse trainer_log.jsonl for per-step metrics
            if trainer_log_records:
                print(f"[METRICS] Found {len(trainer_log_records)} training log records")
                step_count = 0
                for record in trainer_log_records:
                    # trainer_log.jsonl uses "current_steps" and "lr" fields
                    step = record.get("current_steps") or record.get("global_step")
                    loss = record.get("loss")
                    learning_rate = record.get("lr") or record.get("learning_rate")
                    
                    if step is not None and loss is not None:
                        # Record this training step
                        collector.record_training_step(
                            step=int(step),
                            loss=float(loss),
                            learning_rate=float(learning_rate) if learning_rate else None,
                        )
                        step_count += 1
                print(f"[METRICS] Recorded {step_count} training steps from logs")
            
            # Store full trainer state for reference
            if trainer_state:
                collector.metrics["trainer_state"] = {
                    "global_step": trainer_state.get("global_step"),
                    "epoch": trainer_state.get("epoch"),
                    "eval_steps": trainer_state.get("eval_steps"),
                    "best_global_step": trainer_state.get("best_global_step"),
                    "best_metric": trainer_state.get("best_metric"),
                    "best_model_checkpoint": trainer_state.get("best_model_checkpoint"),
                }
                
                if trainer_state.get("log_history"):
                    collector.metrics["training_curve"] = trainer_state["log_history"]
            
            if trainer_log_records:
                collector.metrics["trainer_log_records_tail"] = trainer_log_records[-10:]
            
            # Extract summary metrics
            summary_source = all_results or train_results or {}
            
            # Get final metrics
            final_loss = summary_source.get("train_loss", 0.0)
            final_epoch = int(summary_source.get("epoch", 1) or 1)
            num_samples = int(summary_source.get("total_samples", 0) or 0)
            
            # Try to extract from trainer_state as fallback
            if trainer_state:
                final_epoch = trainer_state.get("epoch", final_epoch)
                final_loss = trainer_state.get("last_loss", final_loss)
            
            print(f"[METRICS] Final loss: {final_loss}")
            print(f"[METRICS] Final epoch: {final_epoch}")
            print(f"[METRICS] Total samples trained: {num_samples}")
            
            # Record epoch-level summary
            collector.record_training_epoch(
                epoch=final_epoch,
                avg_loss=float(final_loss),
                epoch_duration_seconds=training_duration,
            )
            
            # Update total runtime
            collector.metrics["total_runtime_seconds"] = training_duration
            
            # Print peak memory if available
            if collector.metrics.get("peak_gpu_memory_gb", 0) > 0:
                print(f"[METRICS] Peak GPU memory: {collector.metrics['peak_gpu_memory_gb']:.2f} GB")
            
        else:
            print(f"\n[ERROR] Training failed with return code: {returncode}")
            print(f"[ERROR] STDOUT:\n{stdout}")
            print(f"[ERROR] STDERR:\n{stderr}")
            raise RuntimeError(f"Training subprocess failed with code {returncode}")
    
    except Exception as e:
        print(f"\n[ERROR] Training execution failed: {e}")
        raise
    
    # Save metrics
    output_dir = str(script_dir / "performance/results")
    
    collector.print_summary()
    metrics_file = collector.save_results(output_dir=output_dir)
    
    print(f"\n[METRICS] Results saved to: {metrics_file}")
    return metrics_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SFT training with per-step metrics collection"
    )
    parser.add_argument("--experiment_conf", required=True,
                       help="Configuration name (e.g., baseline, pruned_5percent)")
    parser.add_argument("--base_model_value", required=True,
                       help="Base model name (e.g., qwen2.5, qwen7b)")
    parser.add_argument("--think", default="w_t",
                       help="Chain-of-thought condition (w_t or wo_t)")
    parser.add_argument("--personas", default="w_p",
                       help="Persona condition (w_p or wo_p)")
    parser.add_argument("--config_path", default=None,
                       help="Optional override for training config path")
    
    args = parser.parse_args()
    
    run_sft_training_with_metrics(
        experiment_conf=args.experiment_conf,
        base_model_value=args.base_model_value,
        think=args.think,
        personas=args.personas,
        config_path=args.config_path,
    )
