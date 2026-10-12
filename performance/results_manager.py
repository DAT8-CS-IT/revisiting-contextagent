"""
Results management and indexing for performance metrics.
Organizes results by type and date, maintains an index of all runs.
Handles granular per-step and per-sample metrics from new collectors.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


class ResultsIndex:
    """Manages an index of all performance result files."""
    
    def __init__(self, results_dir: Path = Path("performance/results")):
        """
        Initialize results index manager.
        
        Args:
            results_dir: Root results directory
        """
        self.results_dir = Path(results_dir)
        self.index_file = self.results_dir / "index.json"
    
    def build_index(self) -> Dict[str, Any]:
        """
        Scan all result files and build an index.
        Extracts summary statistics for quick comparison.
        
        Returns:
            Index dictionary mapping result types to runs
        """
        index = {
            "generated_at": datetime.now().isoformat(),
            "results_dir": str(self.results_dir),
            "by_type": {
                "icl": [],
                "sft_training": [],
                "sft_evaluation": [],
            },
            "by_date": {},
            "total_runs": 0,
        }
        
        # Scan all result files
        for result_file in self.results_dir.rglob("*.json"):
            # Skip index file itself
            if result_file.name == "index.json":
                continue
            
            try:
                with open(result_file, "r") as f:
                    data = json.load(f)
                
                # Determine type
                result_type = self._get_result_type(data)
                experiment = data.get("experiment", "unknown")
                timestamp = data.get("timestamp", "unknown")
                
                # Extract date from timestamp
                date = timestamp.split("T")[0] if "T" in timestamp else "unknown"
                
                # Extract summary metrics for efficient indexing
                summary = self._extract_summary(data, result_type)
                
                # Add to index
                entry = {
                    "file": str(result_file.relative_to(self.results_dir)),
                    "experiment": experiment,
                    "model": data.get("model", "unknown"),
                    "timestamp": timestamp,
                    "type": result_type,
                    "summary": summary,
                    "hardware": data.get("hardware", {}),
                }
                
                if result_type in index["by_type"]:
                    index["by_type"][result_type].append(entry)
                
                if date not in index["by_date"]:
                    index["by_date"][date] = []
                index["by_date"][date].append(entry)
                
                index["total_runs"] += 1
            
            except Exception as e:
                print(f"[WARNING] Failed to index {result_file}: {e}")
        
        return index
    
    def _get_result_type(self, data: Dict[str, Any]) -> str:
        """Determine result type from data."""
        phase = data.get("phase", "")
        experiment = data.get("experiment", "")
        
        if "training" in phase.lower():
            return "sft_training"
        elif "evaluation" in phase.lower():
            return "sft_evaluation"
        elif "icl" in experiment.lower():
            return "icl"
        else:
            return "unknown"
    
    def _extract_summary(self, data: Dict[str, Any], result_type: str) -> Dict[str, Any]:
        """
        Extract relevant summary metrics based on result type.
        This allows efficient search without loading full metric arrays.
        
        Args:
            data: Full result data
            result_type: Type of result
        
        Returns:
            Summary dictionary with key metrics
        """
        summary = {
            "total_runtime_seconds": data.get("total_runtime_seconds", 0),
            "peak_gpu_memory_gb": data.get("peak_gpu_memory_gb", 0),
            "total_samples": data.get("total_samples", 0),
        }
        
        # Add training-specific summaries
        if "training" in result_type:
            training_summary = data.get("training_summary", {})
            summary.update({
                "num_training_steps": training_summary.get("num_training_steps", 0),
                "final_loss": training_summary.get("final_loss"),
                "min_loss": training_summary.get("min_loss"),
                "avg_loss": training_summary.get("avg_loss"),
                "total_training_time": training_summary.get("total_training_time", 0),
            })
        
        # Add evaluation-specific summaries
        if "evaluation" in result_type:
            eval_summary = data.get("evaluation_summary", {})
            summary.update({
                "num_evaluations": eval_summary.get("num_evaluations", 0),
                "total_eval_time": eval_summary.get("total_eval_time", 0),
                "avg_eval_time": eval_summary.get("avg_eval_time", 0),
                "num_samples_evaluated": len(data.get("evaluation_metrics", [])),
            })
        
        # Add ICL-specific summaries (inference times, throughput)
        if "icl" in result_type:
            inference_times = data.get("inference_times", [])
            if inference_times:
                summary.update({
                    "num_samples_evaluated": len(inference_times),
                    "avg_inference_latency_ms": (sum(inference_times) / len(inference_times)) * 1000,
                    "throughput_samples_per_sec": len(inference_times) / data.get("total_runtime_seconds", 1),
                })
        
        return summary
    
    def save_index(self) -> Path:
        """Save index to disk."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        index = self.build_index()
        
        with open(self.index_file, "w") as f:
            json.dump(index, f, indent=2)
        
        print(f"[INFO] Index saved to {self.index_file}")
        print(f"[INFO] Total runs indexed: {index['total_runs']}")
        for result_type, entries in index["by_type"].items():
            if entries:
                print(f"       {result_type}: {len(entries)} runs")
        
        return self.index_file
    
    def load_index(self) -> Optional[Dict[str, Any]]:
        """Load index from disk if it exists."""
        if not self.index_file.exists():
            return None
        
        with open(self.index_file, "r") as f:
            return json.load(f)
    
    def get_runs_by_experiment(self, experiment_name: str) -> List[Dict[str, Any]]:
        """Get all runs matching an experiment name."""
        index = self.load_index()
        if not index:
            return []
        
        results = []
        for result_type, entries in index["by_type"].items():
            for entry in entries:
                if experiment_name.lower() in entry["experiment"].lower():
                    results.append(entry)
        
        return results
    
    def get_runs_by_date(self, date: str) -> List[Dict[str, Any]]:
        """Get all runs from a specific date (YYYY-MM-DD)."""
        index = self.load_index()
        if not index:
            return []
        
        return index["by_date"].get(date, [])


def organize_result_path(base_dir: str, result_type: str) -> str:
    """
    Generate organized result path: {base_dir}/{type}/{YYYY-MM-DD}/
    
    Args:
        base_dir: Base results directory
        result_type: Type of result (icl, sft_training, sft_evaluation)
    
    Returns:
        Path string for saving results
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    organized_path = Path(base_dir) / result_type / date_str
    organized_path.mkdir(parents=True, exist_ok=True)
    return str(organized_path)


def generate_index_report(index_file: str = "performance/results/index.json") -> str:
    """
    Generate a human-readable report from the results index.
    
    Args:
        index_file: Path to index.json
    
    Returns:
        Formatted report string
    """
    index_path = Path(index_file)
    if not index_path.exists():
        return "No index file found. Run ResultsIndex().save_index() first."
    
    with open(index_path, "r") as f:
        index = json.load(f)
    
    report = f"""
╔════════════════════════════════════════════════════════════════╗
║              PERFORMANCE METRICS RESULTS INDEX                 ║
╠════════════════════════════════════════════════════════════════╣
Generated: {index['generated_at']}
Results Directory: {index['results_dir']}
Total Runs: {index['total_runs']}

RESULTS BY TYPE:
"""
    
    for result_type, entries in index["by_type"].items():
        if entries:
            report += f"\n{result_type.upper()}: {len(entries)} runs\n"
            for entry in entries[-3:]:  # Show last 3 of each type
                summary = entry.get("summary", {})
                report += f"  • {entry['experiment']} ({entry['timestamp'][:10]})\n"
                
                if "final_loss" in summary and summary["final_loss"]:
                    report += f"    Loss: {summary['final_loss']:.4f}\n"
                if "num_samples_evaluated" in summary:
                    report += f"    Samples: {summary['num_samples_evaluated']}\n"
                if "throughput_samples_per_sec" in summary:
                    report += f"    Throughput: {summary['throughput_samples_per_sec']:.2f} samples/sec\n"
    
    report += "\n╚════════════════════════════════════════════════════════════════╝\n"
    return report

