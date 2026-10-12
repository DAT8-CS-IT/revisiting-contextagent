#!/bin/bash
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

echo "Running SFT training..."
cd "$BASE_DIR/LLaMA-Factory"
bash experiments/cab_lite_lora_train.sh
cd "$BASE_DIR"

echo "Running SFT evaluation..."
bash src/cab_lite/sft/eval_sft.sh