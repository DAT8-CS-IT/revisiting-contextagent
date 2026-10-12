"""
Wrapper for ICL inference that adds performance metrics collection.
Call this instead of calling icl/inference.py directly.

Usage:
  python icl_inference_with_metrics.py --model_base qwen2.5:latest \\
    --zs false --think w_t --personas w_p --dataset cab
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "ContextAgent/src"))

from performance.icl_metrics import ICLMetricsCollector
from icl.inference import (
    apply_mode,
    load_json,
    load_prompts,
    build_system_prompt,
    run_single_sample,
    save_results_incremental,
    config,
)
from tqdm import tqdm


def run_icl_with_metrics():
    """
    Run ICL inference with metrics collection.
    This wraps the main ICL inference loop to collect performance data.
    """
    # Parse arguments with custom parser that includes ollama_host
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["live", "sandbox"], default="sandbox")
    parser.add_argument("--model_base", type=str, default="qwen2.5:latest")
    parser.add_argument("--dataset", type=str, default="cab", help="cab, cab_lite, cab_ood")
    parser.add_argument("--zs", type=str, default="false", help="false, true")
    parser.add_argument("--personas", type=str, default="w_p", help="w_p, wo_p")
    parser.add_argument("--think", type=str, default="w_t", help="w_t, wo_t")
    parser.add_argument("--n_fewshot", type=int, default=10)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Evaluate only the first N test samples (useful for a local smoke test).",
    )
    parser.add_argument("--ollama_host", type=str, default=None, 
                        help="Ollama host:port (e.g., 127.0.0.1:11434)")
    args = parser.parse_args()
    
    # Handle ollama_host parameter if provided
    if args.ollama_host:
        os.environ['OLLAMA_HOST'] = args.ollama_host
        print(f"[METRICS] Set OLLAMA_HOST={args.ollama_host}")
    
    apply_mode(args.mode)
    
    # Initialize metrics collector
    collector = ICLMetricsCollector(
        model_name=args.model_base,
        zs=args.zs,
        think=args.think,
        personas=args.personas,
        dataset=args.dataset,
    )
    
    print(f"[METRICS] Initialized collector for experiment: {collector.experiment_name}")
    print(f"[main] MODE = {config.MODE}")
    
    # Load prompts and data (same as original)
    base_prompt = load_prompts(args.zs, args.think)
    train_path = f"data/{args.dataset}/{args.dataset}_train.json"
    train_ds = load_json(train_path)
    sys_prompt, fewshot_keys = build_system_prompt(
        base_prompt, args.zs, train_ds, args.n_fewshot, args.think, args.personas
    )
    
    test_path = f"data/{args.dataset}/{args.dataset}_test.json"
    test_ds = load_json(test_path)
    
    # Main inference loop with metrics
    keys_list = list(test_ds.keys())
    if args.max_samples is not None:
        if args.max_samples < 1:
            parser.error("--max_samples must be at least 1")
        keys_list = keys_list[:args.max_samples]
    total = len(keys_list)
    sep = "=" * 80
    
    # Use overall timing for throughput calculation
    overall_start = time.perf_counter()
    
    for idx, k in enumerate(tqdm(keys_list, total=total, desc="Evaluating samples", unit="sample"), start=1):
        tqdm.write(f"\n{sep}\n[Sample {idx}/{total}] ID: {k}\n{sep}")
        
        # Time individual sample inference
        sample_start = collector.start_timing()
        
        try:
            preds = run_single_sample(k, test_ds[k], args, sys_prompt, args.dataset)
            test_ds[k]["predictions"] = preds
            
            # Record the inference time for this sample
            collector.record_inference(
                sample_start,
                response_text=str(preds.get("response", "")),
                num_output_tokens=None,  # Could be extracted if available
            )
            
        except Exception as e:
            tqdm.write(f"[ERROR] Failed to process sample {k}: {e}")
            collector.record_inference(sample_start, response_text="ERROR")
            raise
        
        # Save results incrementally
        save_path = save_results_incremental(test_ds, args)
        tqdm.write(f"[main] Saved: {save_path}")
        tqdm.write(f"{'-'*80}\n[Completed] {k} ({idx}/{total})\n{'-'*80}")
    
    overall_duration = time.perf_counter() - overall_start
    
    print("\nAll samples finished ✅")
    
    # Print summary and save metrics
    collector.print_summary()
    metrics_file = collector.save_results(output_dir="../performance/results")
    
    print(f"\n[METRICS] Results saved to: {metrics_file}")
    print(f"[METRICS] Total experiment time: {overall_duration:.2f}s")
    print(f"[METRICS] Throughput: {total / overall_duration:.2f} samples/sec")


if __name__ == "__main__":
    run_icl_with_metrics()
