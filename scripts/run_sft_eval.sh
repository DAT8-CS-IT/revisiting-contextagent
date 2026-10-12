#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Run one SFT evaluation locally.

Usage:
  run_sft_eval.sh CONFIG MODEL [THINK] [PERSONAS] [PORT] [MODE]

Example:
  run_sft_eval.sh \
    experiments/configs/cab/inference/qwen_lora_sft.yaml \
    qwen7b w_t w_p 8009 sandbox

The evaluation starts the local LLaMA-Factory API server and stops it when
evaluation finishes. The base model and adapter referenced by CONFIG must
already be available locally or downloadable by the configured backend.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

CONFIG="${1:?An inference configuration is required. Use --help for examples.}"
MODEL="${2:?A base-model identifier is required. Use --help for examples.}"
THINK="${3:-w_t}"
PERSONAS="${4:-w_p}"
PORT="${5:-8009}"
MODE="${6:-sandbox}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

command -v python3 >/dev/null || {
    echo "python3 is required." >&2
    exit 1
}

cd "$REPO_DIR"
python3 sft_eval_with_metrics.py \
    --model_base "$MODEL" \
    --dataset cab \
    --think "$THINK" \
    --personas "$PERSONAS" \
    --port "$PORT" \
    --mode "$MODE" \
    --api_config "$CONFIG"
