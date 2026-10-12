#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Run one ICL evaluation locally.

Usage:
  run_icl.sh MODEL [ZS] [THINK] [PERSONAS] [OLLAMA_PORT] [MAX_SAMPLES]

Example:
  run_icl.sh qwen2.5:0.5b false w_t w_p 11434 2

Ollama must be installed and running locally. Set START_OLLAMA=1 to start
an Ollama server in this script; the process is stopped when the script exits.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

MODEL="${1:-deepseek-r1:7b}"
ZS="${2:-false}"
THINK="${3:-w_t}"
PERSONAS="${4:-w_p}"
OLLAMA_PORT="${5:-11434}"
MAX_SAMPLES="${6:-}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"

command -v python3 >/dev/null || {
    echo "python3 is required." >&2
    exit 1
}
command -v ollama >/dev/null || {
    echo "ollama is required. Install it from https://ollama.com/." >&2
    exit 1
}

ollama_pid=""
cleanup() {
    if [[ -n "$ollama_pid" ]]; then
        kill "$ollama_pid" 2>/dev/null || true
        wait "$ollama_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

if [[ "${START_OLLAMA:-0}" == "1" ]]; then
    ollama serve &
    ollama_pid=$!
    for _ in {1..30}; do
        if ollama list >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done
fi

if ! ollama list >/dev/null 2>&1; then
    echo "Ollama is not reachable at ${OLLAMA_HOST}. Start it or set START_OLLAMA=1." >&2
    exit 1
fi

if ! ollama list | awk '{print $1}' | grep -Fxq "$MODEL"; then
    echo "Pulling Ollama model: $MODEL"
    ollama pull "$MODEL"
fi

cd "$REPO_DIR"
inference_args=(
    --model_base "$MODEL" \
    --mode sandbox \
    --zs "$ZS" \
    --think "$THINK" \
    --personas "$PERSONAS" \
    --dataset cab \
    --ollama_host "$OLLAMA_HOST"
)
if [[ -n "$MAX_SAMPLES" ]]; then
    inference_args+=(--max_samples "$MAX_SAMPLES")
fi
python3 icl_inference_with_metrics.py "${inference_args[@]}"

if [[ -z "$MAX_SAMPLES" ]]; then
    python3 ContextAgent/src/calculate_scores.py \
        --dataset cab \
        --methods icl \
        --model_base_icl "$MODEL" \
        --zs "$ZS" \
        --think "$THINK" \
        --personas "$PERSONAS"
else
    echo "[INFO] Smoke test complete; score calculation skipped for a partial dataset."
fi
