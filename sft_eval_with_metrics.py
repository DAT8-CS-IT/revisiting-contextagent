"""
Wrapper for SFT evaluation that adds performance metrics collection.
Directly wraps sft/inference.py to collect per-sample performance metrics.
Includes setup/teardown of LLaMA-Factory API server.

Usage:
  python sft_eval_with_metrics.py --model_base "qwen7b" \\
    --dataset "cab" --think "w_t" --personas "w_p" --port 8009 \\
    --api_config "configs/cab/inference/qwen_lora_sft.yaml"
"""

import sys
import os
import argparse
import time
import subprocess
import socket
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "ContextAgent/src"))

from performance.sft_metrics import SFTMetricsCollector
from sft.inference import (
    apply_mode,
    load_json,
    load_prompt_base,
    get_sft_client,
    run_single_sample,
    save_results_incremental,
    config,
)
from tqdm import tqdm


def wait_for_port(port: int, timeout: int = 240) -> bool:
    """Wait for API service to be ready on port."""
    start = time.time()
    attempt = 0
    while time.time() - start < timeout:
        attempt += 1
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("127.0.0.1", port))
            sock.close()
            if result == 0:
                print(f"[METRICS] Port {port} is open ✓")
                return True
        except Exception as e:
            if attempt % 20 == 0:
                print(f"[METRICS] Waiting for port {port}... (attempt {attempt})")
        time.sleep(0.5)
    print(f"[ERROR] Port {port} did not open within {timeout} seconds")
    return False


def kill_port_process(port: int) -> None:
    """Kill process running on port."""
    try:
        subprocess.run(["fuser", "-k", f"{port}/tcp"], 
                      capture_output=True, timeout=5)
        time.sleep(1)
    except Exception as e:
        print(f"[WARNING] Failed to kill process on port {port}: {e}")


def run_sft_eval_with_metrics():
    """
    Run SFT inference with metrics collection.
    This wraps the main SFT inference loop to collect performance data per sample.
    """
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["live", "sandbox"], default="sandbox")
    parser.add_argument("--model_base", type=str, default="qwen7b",
                       help='base SFT model: qwen7b, llama8b, deepseek7b')
    parser.add_argument("--dataset", type=str, default="cab", help="cab, cab_lite, cab_ood")
    parser.add_argument("--think", type=str, default="w_t", help="w_t, wo_t")
    parser.add_argument("--personas", type=str, default="w_p", help="w_p, wo_p")
    parser.add_argument("--port", type=int, default=8009, help="port for local SFT model server")
    parser.add_argument("--api_config", type=str, default=None, 
                       help="path to LLaMA-Factory API config (starts API server if provided)")
    args = parser.parse_args()
    
    apply_mode(args.mode)
    
    # Initialize metrics collector
    collector = SFTMetricsCollector(
        model_name=args.model_base,
        experiment_conf="sft_inference",
        base_model_value=args.model_base,
        think=args.think,
        personas=args.personas,
    )
    
    print(f"[METRICS] Initialized collector for experiment: {collector.experiment_name}")
    print(f"[main] MODE = {config.MODE}")
    
    # Set API port from args
    if args.port:
        os.environ['API_PORT'] = str(args.port)
        print(f"[METRICS] Set API_PORT={args.port}")
    
    # Start LLaMA-Factory API server if config provided
    api_process = None
    api_log_file = None
    
    if args.api_config:
        print(f"\n[METRICS] Setting up LLaMA-Factory API server...")
        
        # Resolve API config path
        api_config_path = args.api_config
        if not Path(api_config_path).is_absolute():
            # Try common locations
            repo_dir = Path(__file__).parent
            factory_dir = repo_dir / "ContextAgent" / "LLaMA-Factory"
            api_config_path = factory_dir / api_config_path
        
        api_config_path = str(api_config_path)
        
        # If file doesn't exist, try adding .yaml extension
        if not Path(api_config_path).exists() and not api_config_path.endswith(".yaml"):
            api_config_path_with_ext = api_config_path + ".yaml"
            if Path(api_config_path_with_ext).exists():
                api_config_path = api_config_path_with_ext
                print(f"[METRICS] Auto-appended .yaml to config path")
        
        if not Path(api_config_path).exists():
            print(f"[ERROR] API config not found: {api_config_path}")
            raise FileNotFoundError(f"API config not found: {api_config_path}")
        
        print(f"[METRICS] API config: {api_config_path}")
        
        # Clean up any previous process
        print(f"[METRICS] Cleaning up previous processes on port {args.port}...")
        kill_port_process(args.port)
        time.sleep(1)
        
        # Create log directory
        repo_dir = Path(__file__).parent
        log_dir = repo_dir / "performance" / "results"
        log_dir.mkdir(parents=True, exist_ok=True)
        api_log_file = log_dir / f"api_port_{args.port}.log"
        
        print(f"[METRICS] Starting API server on port {args.port}...")
        print(f"[METRICS] API logs: {api_log_file}")
        
        # Start API server process
        factory_dir = repo_dir / "ContextAgent" / "LLaMA-Factory"
        with open(api_log_file, "a", buffering=1, encoding="utf-8") as log_handle:
            api_process = subprocess.Popen(
                ["llamafactory-cli", "api", api_config_path],
                cwd=str(factory_dir),
                env=os.environ.copy(),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        
        # Wait for API to be ready
        print(f"[METRICS] Waiting for API to be ready (max 240 seconds)...")
        if not wait_for_port(args.port, timeout=240):
            print(f"[ERROR] API server failed to start on port {args.port}")
            if api_process:
                api_process.terminate()
            raise RuntimeError(f"API startup timeout on port {args.port}")
        
        print(f"[METRICS] ✓ API server ready on port {args.port}\n")
    
    try:
        # Connect to SFT model
        print(f"[main] Connecting to SFT model on port {args.port}...")
        client = get_sft_client(args.port)
        
        # Test connection
        try:
            test_messages = [{"role": "system", "content": "test"}, {"role": "user", "content": "test"}]
            test_resp = client.chat.completions.create(messages=test_messages, model=args.model_base, max_tokens=1)
            print(f"[main] Connection test successful ✓\n")
        except Exception as e:
            print(f"[fatal] Connection failed: {e}")
            raise
        
        # Load prompt
        sys_prompt = load_prompt_base(args.think)
        
        # Load test dataset
        dataset_name = args.dataset
        script_dir = Path(__file__).parent
        test_path = script_dir / "ContextAgent" / "data" / dataset_name / f"{dataset_name}_test.json"
        dataset = load_json(str(test_path))
        print(f"[main] Loaded {len(dataset)} test samples from {dataset_name}")
        
        # Main inference loop with metrics
        keys_list = list(dataset.keys())
        total = len(keys_list)
        sep = "=" * 80
        
        # Use overall timing for throughput calculation
        overall_start = time.perf_counter()
        
        for idx, k in enumerate(tqdm(keys_list, total=total, desc="Evaluating samples", unit="sample"), start=1):
            tqdm.write(f"\n{sep}\n[Sample {idx}/{total}] ID: {k}\n{sep}")
            
            # Time individual sample inference
            sample_start = collector.start_timing()
            
            try:
                preds = run_single_sample(client, k, dataset[k], args, sys_prompt, dataset_name)
                dataset[k]["predictions"] = preds
                
                # Record the inference time for this sample
                collector.record_inference(
                    sample_start,
                    response_text=str(preds.get("response", "")),
                    num_output_tokens=None,
                )
                
            except Exception as e:
                tqdm.write(f"[ERROR] Failed to process sample {k}: {e}")
                collector.record_inference(sample_start, response_text="ERROR")
                raise
            
            # Save results incrementally
            save_path = save_results_incremental(dataset, args)
            tqdm.write(f"[main] Saved: {save_path}")
            tqdm.write(f"{'-'*80}\n[Completed] {k} ({idx}/{total})\n{'-'*80}")
        
        overall_duration = time.perf_counter() - overall_start
        
        print("\nAll samples finished ✅")
        
        # Print summary and save metrics
        collector.print_summary()
        metrics_file = collector.save_results(
            output_dir=str(Path(__file__).resolve().parent / "performance" / "results")
        )
        
        print(f"\n[METRICS] Results saved to: {metrics_file}")
        print(f"[METRICS] Total experiment time: {overall_duration:.2f}s")
        print(f"[METRICS] Throughput: {total / overall_duration:.2f} samples/sec")
    
    finally:
        # Clean up: shutdown API server if it was started
        if api_process:
            print(f"\n[METRICS] Shutting down API server on port {args.port}...")
            try:
                api_process.terminate()
                api_process.wait(timeout=10)
                print(f"[METRICS] ✓ API server terminated gracefully")
            except subprocess.TimeoutExpired:
                print(f"[WARNING] API server did not terminate, killing forcefully...")
                api_process.kill()
                api_process.wait()
            except Exception as e:
                print(f"[WARNING] Error terminating API process: {e}")
            
            # Also try to kill using fuser as backup
            kill_port_process(args.port)
            print(f"[METRICS] ✓ Port {args.port} cleaned up")


if __name__ == "__main__":
    run_sft_eval_with_metrics()
