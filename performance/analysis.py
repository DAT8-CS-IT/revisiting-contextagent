"""
Performance comparison and analysis script.
Aggregates metrics across multiple experiment runs and generates comparisons.

Usage:
  python performance/analysis.py --results_dir performance/results/
  python performance/analysis.py --results_dir performance/results/ --output_format csv
  python performance/analysis.py --results_dir performance/results/ --filter_experiment icl
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ExperimentResult:
    """Container for an experiment's results."""
    filename: str
    timestamp: str
    experiment: str
    model: str
    phase: Optional[str]
    total_samples: int
    total_runtime_s: float
    avg_runtime_s: float
    peak_gpu_memory_gb: float
    throughput_samples_per_sec: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "timestamp": self.timestamp,
            "experiment": self.experiment,
            "model": self.model,
            "phase": self.phase or "unknown",
            "total_samples": self.total_samples,
            "total_runtime_s": round(self.total_runtime_s, 2),
            "avg_runtime_s": round(self.avg_runtime_s, 4),
            "peak_gpu_memory_gb": round(self.peak_gpu_memory_gb, 2),
            "throughput_samples_per_sec": round(self.throughput_samples_per_sec, 2),
        }


def parse_result_file(filepath: Path) -> Optional[ExperimentResult]:
    """Parse a single results JSON file."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        
        # Extract summary statistics
        summary = data.get("summary", {})
        
        total_runtime = summary.get("total_runtime_seconds", 0)
        total_samples = data.get("total_samples", 0)
        avg_runtime = summary.get("avg_batch_time", 0)
        throughput = summary.get("avg_samples_per_sec", 0)
        peak_gpu = summary.get("peak_gpu_memory_gb", data.get("peak_gpu_memory_gb", 0))
        
        return ExperimentResult(
            filename=filepath.name,
            timestamp=data.get("timestamp", "unknown"),
            experiment=data.get("experiment", "unknown"),
            model=data.get("model", "unknown"),
            phase=data.get("phase"),
            total_samples=total_samples,
            total_runtime_s=total_runtime,
            avg_runtime_s=avg_runtime,
            peak_gpu_memory_gb=peak_gpu,
            throughput_samples_per_sec=throughput,
        )
    except Exception as e:
        print(f"[ERROR] Failed to parse {filepath}: {e}")
        return None


def load_results(results_dir: Path, experiment_filter: Optional[str] = None) -> List[ExperimentResult]:
    """Load all results from a directory."""
    results = []
    
    # Find all JSON files
    json_files = list(results_dir.glob("*.json"))
    print(f"[INFO] Found {len(json_files)} result files in {results_dir}")
    
    for filepath in sorted(json_files):
        result = parse_result_file(filepath)
        if result:
            # Apply filter if specified
            if experiment_filter and experiment_filter not in result.experiment:
                continue
            results.append(result)
    
    print(f"[INFO] Loaded {len(results)} results (after filtering)")
    return results


def print_table(results: List[ExperimentResult]):
    """Print results as formatted table."""
    if not results:
        print("[WARNING] No results to display")
        return
    
    # Header
    print("\n" + "=" * 130)
    print(f"{'Experiment':<30} {'Model':<20} {'Samples':<10} {'Time(s)':<12} "
          f"{'Avg(s)':<10} {'GPU(GB)':<10} {'Throughput':<12}")
    print("=" * 130)
    
    # Rows
    for r in results:
        print(f"{r.experiment:<30} {r.model:<20} {r.total_samples:<10} {r.total_runtime_s:<12.2f} "
              f"{r.avg_runtime_s:<10.4f} {r.peak_gpu_memory_gb:<10.2f} {r.throughput_samples_per_sec:<12.2f}")
    
    print("=" * 130 + "\n")


def print_csv(results: List[ExperimentResult]):
    """Print results as CSV."""
    if not results:
        print("[WARNING] No results to display")
        return
    
    # Header
    print(",".join([
        "timestamp", "experiment", "model", "total_samples", "total_runtime_s",
        "avg_runtime_s", "peak_gpu_memory_gb", "throughput_samples_per_sec"
    ]))
    
    # Rows
    for r in results:
        print(",".join([
            r.timestamp, r.experiment, r.model, str(r.total_samples),
            str(r.total_runtime_s), str(r.avg_runtime_s), str(r.peak_gpu_memory_gb),
            str(r.throughput_samples_per_sec)
        ]))


def group_by_experiment(results: List[ExperimentResult]) -> Dict[str, List[ExperimentResult]]:
    """Group results by experiment name."""
    groups = {}
    for r in results:
        if r.experiment not in groups:
            groups[r.experiment] = []
        groups[r.experiment].append(r)
    return groups


def compare_experiments(results: List[ExperimentResult]):
    """Print comparison analysis across experiments."""
    if not results:
        print("[INFO] No results for comparison")
        return
    
    groups = group_by_experiment(results)
    
    print("\n" + "=" * 80)
    print("COMPARISON BY EXPERIMENT")
    print("=" * 80)
    
    for exp_name, exp_results in sorted(groups.items()):
        exp_results = sorted(exp_results, key=lambda r: r.timestamp)
        
        print(f"\n📊 {exp_name}")
        print(f"   Runs: {len(exp_results)}")
        
        # Aggregated stats
        avg_throughput = sum(r.throughput_samples_per_sec for r in exp_results) / len(exp_results)
        avg_memory = sum(r.peak_gpu_memory_gb for r in exp_results) / len(exp_results)
        avg_time = sum(r.total_runtime_s for r in exp_results) / len(exp_results)
        
        print(f"   ├─ Avg throughput: {avg_throughput:.2f} samples/sec")
        print(f"   ├─ Avg GPU memory: {avg_memory:.2f} GB")
        print(f"   ├─ Avg total time: {avg_time:.2f} sec")
        
        # Latest run
        latest = exp_results[-1]
        print(f"   └─ Latest run ({latest.timestamp}):")
        print(f"       └─ Throughput: {latest.throughput_samples_per_sec:.2f} samples/sec")
    
    print("=" * 80 + "\n")


def save_analysis_report(results: List[ExperimentResult], output_file: Path):
    """Save detailed analysis report as JSON."""
    report = {
        "generated_at": datetime.now().isoformat(),
        "total_runs": len(results),
        "experiments": {},
        "all_results": [r.to_dict() for r in results],
    }
    
    # Group by experiment
    groups = group_by_experiment(results)
    for exp_name, exp_results in groups.items():
        throughputs = [r.throughput_samples_per_sec for r in exp_results]
        memories = [r.peak_gpu_memory_gb for r in exp_results]
        times = [r.total_runtime_s for r in exp_results]
        
        report["experiments"][exp_name] = {
            "num_runs": len(exp_results),
            "throughput": {
                "avg": sum(throughputs) / len(throughputs),
                "min": min(throughputs),
                "max": max(throughputs),
            },
            "gpu_memory_gb": {
                "avg": sum(memories) / len(memories),
                "min": min(memories),
                "max": max(memories),
            },
            "runtime_seconds": {
                "avg": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
            },
        }
    
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"[INFO] Report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Performance analysis tool")
    parser.add_argument("--results_dir", type=Path, default=Path("performance/results"),
                        help="Directory containing result JSON files")
    parser.add_argument("--output_format", choices=["table", "csv"], default="table",
                        help="Output format for results")
    parser.add_argument("--filter_experiment", type=str,
                        help="Only show results matching this experiment name")
    parser.add_argument("--compare", action="store_true",
                        help="Show comparison analysis across experiments")
    parser.add_argument("--save_report", type=Path,
                        help="Save detailed report to JSON file")
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.results_dir, args.filter_experiment)
    
    if not results:
        print("[WARNING] No results found!")
        return
    
    # Display results
    if args.output_format == "table":
        print_table(results)
    elif args.output_format == "csv":
        print_csv(results)
    
    # Show comparison if requested
    if args.compare:
        compare_experiments(results)
    
    # Save report if requested
    if args.save_report:
        save_analysis_report(results, args.save_report)


if __name__ == "__main__":
    main()
