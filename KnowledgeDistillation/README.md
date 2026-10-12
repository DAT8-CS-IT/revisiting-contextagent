# Knowledge Distillation: Llama 3.1 8B → Llama 3.2 1B

This module implements response-based knowledge distillation to compress a Llama 3.1 8B teacher model into a smaller Llama 3.2 1B student model using KL divergence loss.

## Features

- **Response-based Distillation**: KL divergence between teacher and student logits
- **Flexible Teacher Loading**: Three backends to choose from:
  - `direct`: Load teacher directly via HuggingFace transformers
  - `ollama`: Query running Ollama instance
  - `vllm`: Query running vLLM instance
- **Configurable Training**: CLI arguments for all hyperparameters
- **Multiple Save Formats**: Save in both HuggingFace and PyTorch formats
- **Automatic Loss Graphing**: Loss curves with EMA smoothing
- **Sandbox Mode**: Test with dummy data without loading real models
- **ContextAgent-style Config**: Reusable argument pattern matching ContextAgent

## Directory Structure

```
KnowledgeDistillation/
├── kd.py                    # Main entry point
├── kd_config.py             # Global mode management
├── kd_args.py               # Argument definitions (dataclasses)
├── teacher_loaders.py       # Teacher loading strategies
├── student_model.py         # Student model initialization
├── distillation_trainer.py  # Training loop with KL loss
├── model_saver.py           # Save models in multiple formats
├── utils/
│   ├── __init__.py
│   └── plotting.py          # Loss visualization utilities
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── [outputs]/               # Saved models directory (auto-created)
```

## Installation

```bash
cd KnowledgeDistillation
pip install -r requirements.txt
```

## Usage

### Basic Training (Direct Teacher Loading)

```bash
python kd.py \
    --teacher_model_id meta-llama/Meta-Llama-3.1-8B \
    --student_model_id meta-llama/Llama-3.2-1B \
    --teacher_loading_method direct \
    --train_data_path /path/to/training_data.jsonl \
    --num_epochs 3 \
    --batch_size 8 \
    --temperature 3.0 \
    --kd_alpha 0.5
```

### Sandbox Mode (Testing with Dummy Data)

```bash
python kd.py \
    --mode sandbox \
    --teacher_loading_method direct \
    --num_epochs 1 \
    --batch_size 4
```

### Using Ollama Teacher

```bash
# Start Ollama in another terminal
ollama serve

# Run distillation
python kd.py \
    --teacher_model_id meta-llama/Meta-Llama-3.1-8B \
    --teacher_loading_method ollama \
    --train_data_path /path/to/data.jsonl \
    --teacher_ollama_port 11434
```

### Using vLLM Teacher

```bash
# Start vLLM in another terminal
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B \
    --port 8009

# Run distillation
python kd.py \
    --teacher_loading_method vllm \
    --train_data_path /path/to/data.jsonl \
    --teacher_vllm_port 8009
```

## Arguments Reference

### Distillation Arguments

- `--temperature`: Temperature for KL divergence softmax scaling (default: 3.0)
- `--kd_alpha`: Weight for KD loss in combined loss (default: 0.5)
  - Formula: `total_loss = alpha * CE_loss + (1 - alpha) * KD_loss`
- `--num_epochs`: Number of training epochs (default: 3)
- `--batch_size`: Training batch size (default: 8)
- `--eval_batch_size`: Evaluation batch size (default: 16)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--warmup_steps`: Warmup steps for learning rate scheduler (default: 500)
- `--weight_decay`: Weight decay for AdamW (default: 0.01)
- `--logging_steps`: Log metrics every N steps (default: 10)
- `--save_steps`: Save checkpoint every N steps (default: 500)

### Teacher Arguments

- `--teacher_model_id`: HuggingFace model ID (default: `meta-llama/Meta-Llama-3.1-8B`)
- `--teacher_loading_method`: How to load teacher: `direct`, `ollama`, or `vllm` (default: `direct`)
- `--teacher_device`: Device to load on: `cuda` or `cpu` (default: `cuda`)
- `--teacher_ollama_port`: Ollama service port (default: 11434)
- `--teacher_vllm_port`: vLLM service port (default: 8009)
- `--teacher_api_key`: API key for remote endpoints (default: `0`)

### Student Arguments

- `--student_model_id`: HuggingFace model ID (default: `meta-llama/Llama-3.2-1B`)
- `--student_device`: Device to load on: `cuda` or `cpu` (default: `cuda`)

### Data Arguments

- `--train_data_path`: Path to training data (JSONL format with `text` field)
- `--eval_data_path`: Path to evaluation data (optional)
- `--max_seq_length`: Maximum sequence length (default: 512)
- `--num_train_samples`: Limit number of training samples (optional)
- `--num_eval_samples`: Limit number of eval samples (optional)

### Output Arguments

- `--save_dir`: Base directory for saving models (default: `KnowledgeDistillation/`)
- `--save_hf_format`: Save in HuggingFace format (default: True)
- `--save_pytorch_format`: Save PyTorch checkpoint (default: True)
- `--save_trainer_state`: Save trainer state/loss history (default: True)

### Mode

- `--mode`: Run mode: `live` or `sandbox` (default: `live`)

## Data Format

Training data should be in JSONL format with a `text` field:

```jsonl
{"text": "First example text here..."}
{"text": "Second example text here..."}
{"text": "Third example text here..."}
...
```

## Output Structure

After training, models are saved in a timestamped directory:

```
KnowledgeDistillation/
└── Llama-3.2-1B_from_Meta-Llama-3.1-8B_2026-03-25_143022/
    ├── hf_model/                    # HuggingFace format
    │   ├── config.json
    │   ├── generation_config.json
    │   ├── model.safetensors
    │   ├── tokenizer.model
    │   └── tokenizer_config.json
    ├── pytorch_model.bin            # PyTorch checkpoint
    ├── config_snapshot.json         # Training config snapshot
    ├── trainer_state.json           # Loss history
    └── Llama-3.2-1B_from_Meta-Llama-3.1-8B_2026-03-25_143022_loss.png  # Loss plot
```

## Loss Plotting

The training script automatically generates a loss plot PNG showing:
- Raw loss curve
- EMA-smoothed loss curve
- Evaluation loss (if available)

To manually regenerate a plot from checkpoint:

```python
from utils.plotting import plot_from_checkpoint

plot_from_checkpoint(
    checkpoint_dir="KnowledgeDistillation/Llama-3.2-1B_from_Meta-Llama-3.1-8B_2026-03-25_143022/",
    output_filename="custom_loss_plot.png"
)
```

## Limitations & Future Enhancements

### Current Limitations:
- Only response-based distillation (no feature-based or relation-based)
- Ollama/vLLM backends need custom integration for full logit access
- No built-in quantization support for student model

### Completed Enhancements:
- ✅ Multi-GPU distributed training support (PyTorch DDP with torchrun)

### Future Enhancements:
- Feature-based distillation (intermediate layer matching)
- Relation-based distillation (attention/similarity matching)
- Optimizer state checkpointing for resumable training
- Adaptive temperature/alpha scheduling
- Integration with LoRA/QLoRA for smaller footprint
- Evaluation metrics (perplexity, BLEU, etc.)
- Multi-node distributed training (across multiple compute nodes)

## Troubleshooting

### CUDA Out of Memory (OOM)

Reduce batch size:
```bash
python kd.py ... --batch_size 4
```

### Ollama Connection Error

Ensure Ollama is running:
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Pull model if needed
ollama pull meta-llama/Meta-Llama-3.1-8B
```

### vLLM Connection Error

Ensure vLLM is running:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B \
    --port 8009
```

### No Training Data Found

Verify JSONL format:
```bash
head -n 1 /path/to/data.jsonl | python -m json.tool
```

Must have at least one valid JSON object per line with a `text` field.

## References

- **LLaMA Factory Plotting**: Inspired by plotting utilities from ContextAgent/LLaMA-Factory
- **KL Divergence Loss**: Standard knowledge distillation implementation
- **Temperature Scaling**: Following Hinton et al. (2015) "Distilling the Knowledge in a Neural Network"

## License

This code is part of the Reliable-Action-Recommendations project. See parent LICENSE file.

## Authors

Generated for knowledge distillation exploration in LLM-based adaptive systems.
