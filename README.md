# Towards More Efficient Proactive Agents

This repository contains the code and experiment configurations for our
reproducibility study of proactive LLM agents. It combines ContextAgent
inference, LoRA/SFT training and evaluation, performance instrumentation, and
knowledge-distillation utilities.

## Authors

Alexander Guldberg Redder, Andreas Svanberg Hummelmose, Beltrán J. Aceves Gil,
Eduard Brahas, Mahsa Hemmatpanah, Rasmus Præstegaard Pedersen, and Peter Dolog.

Peter Dolog is an Associate Professor at the Department of Computer Science,
Aalborg University, Denmark, and is the project supervisor.

## Repository layout

- `ContextAgent/`: agent code, data/prompt definitions, and LLaMA-Factory.
- `performance/`: metric collection and result-analysis utilities.
- `KnowledgeDistillation/`: teacher/student distillation code.
- `AiLab_scripts/`: portable local runners and result-formatting utilities.
- `icl_inference_with_metrics.py`: ICL evaluation with timing and GPU metrics.
- `sft_training_with_metrics.py`: LoRA/SFT training with training metrics.
- `sft_eval_with_metrics.py`: SFT evaluation with inference metrics.

Model weights, checkpoints, caches, credentials, and runtime logs are
intentionally not included in Git. Download the required base models from
their original model hosts and place trained adapters at the paths referenced
by the configuration files.

## Installation

The experiments require Python 3.10 or newer, PyTorch, Transformers, PEFT,
bitsandbytes for quantised runs, and the packages listed in the environment
files:

```bash
git clone git@github.com:DAT8-CS-IT/revisiting-contextagent.git
cd revisiting-contextagent

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r ContextAgent/requirements.txt
python -m pip install -e ContextAgent
python -m pip install -e ContextAgent/LLaMA-Factory
```

For the complete pinned environment, see
[`ContextAgent/environment.yml`](ContextAgent/environment.yml). A CUDA-capable
GPU is required for model training and local SFT evaluation. ICL evaluation
uses [Ollama](https://ollama.com/) and can run on a CPU if the selected model
fits.

Set credentials in the shell or in a local untracked `.env` file. Never add
real credentials to Git:

```bash
export HF_TOKEN="your-token"
```

## Running locally

These commands run on a normal workstation or server. They do not require a
cluster scheduler or a cluster-specific filesystem.

### ICL evaluation

Start Ollama separately, or set `START_OLLAMA=1` to let the wrapper start and
stop it:

```bash
ollama serve
AiLab_scripts/run_icl_local.sh deepseek-r1:7b false w_t w_p 11434
```

The script pulls the model if it is not already installed, evaluates the CAB
dataset, and calculates the scores.

### SFT/LoRA training

The base model and dataset referenced by the selected configuration must be
available locally or accessible through the configured model hub:

```bash
AiLab_scripts/run_sft_train_local.sh qwen_lora_sft qwen7b
```

### SFT evaluation

After training or downloading the corresponding adapter:

```bash
AiLab_scripts/run_sft_eval_local.sh \
  experiments/configs/cab/inference/qwen_lora_sft.yaml \
  qwen7b w_t w_p 8009 sandbox
```

Use `--help` with each local runner for its complete argument list. Evaluation
uses batch size 1, matching the reported experiments.

## Results

Generated results are written to the configured performance/results directories
and are ignored by Git. To generate a table from existing CSV results:

```bash
AiLab_scripts/generate_results_table.sh --format terminal --methods both
```

## Reproducibility and anonymisation

The public repository contains no model weights or private credentials. For an
anonymous review artifact, use a separate anonymous repository or archive:
remove author names, affiliations, repository ownership clues, personal paths,
job identifiers, and identifying logs from that copy. Do not rewrite this
public repository solely for anonymous review.

## License

See [`LICENSE`](LICENSE).
