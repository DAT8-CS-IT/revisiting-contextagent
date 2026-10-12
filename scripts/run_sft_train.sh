#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Run one LoRA/SFT training job locally.

Usage:
  run_sft_train.sh CONFIG MODEL [THINK] [PERSONAS]

CONFIG is the training configuration name or path accepted by
sft_training_with_metrics.py, for example qwen_lora_sft.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

CONFIG="${1:?A training configuration is required. Use --help for examples.}"
MODEL="${2:?A base-model identifier is required. Use --help for examples.}"
THINK="${3:-w_t}"
PERSONAS="${4:-w_p}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

command -v python3 >/dev/null || {
    echo "python3 is required." >&2
    exit 1
}

cd "$REPO_DIR"
python3 sft_training_with_metrics.py \
    --experiment_conf "$CONFIG" \
    --base_model_value "$MODEL" \
    --think "$THINK" \
    --personas "$PERSONAS"
