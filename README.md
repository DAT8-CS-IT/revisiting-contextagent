# Revisiting ContextAgent

## A Reproducibility Study of Context-Aware Proactive LLM Agents under Resource Constraints

This repository contains the code, configurations, and evaluation scripts for
our reproducibility study of ContextAgent. It supports in-context learning
(ICL), LoRA/SFT training and evaluation, performance measurement, and knowledge
distillation.

## Authors

| Name | Role | Email |
| --- | --- | --- |
| Alexander Guldberg Redder | Author | [aredde22@student.aau.dk](mailto:aredde22@student.aau.dk) |
| Andreas Svanberg Hummelmose | Author | [ahumme22@student.aau.dk](mailto:ahumme22@student.aau.dk) |
| Beltrán J. Aceves Gil | Author | [baceve25@student.aau.dk](mailto:baceve25@student.aau.dk) |
| Eduard Brahas | Author | [ebraha25@student.aau.dk](mailto:ebraha25@student.aau.dk) |
| Mahsa Hemmatpanah | Author | [mhemma25@student.aau.dk](mailto:mhemma25@student.aau.dk) |
| Rasmus Præstegaard Pedersen | Author | [rpeder22@student.aau.dk](mailto:rpeder22@student.aau.dk) |
| Peter Dolog | Supervisor and Associate Professor | [dolog@cs.aau.dk](mailto:dolog@cs.aau.dk) |

Department of Computer Science, Aalborg University, Denmark.

## Repository contents

```text
ContextAgent/                 Agent code, data, prompts, and SFT framework
KnowledgeDistillation/        Knowledge-distillation implementation
performance/                  Metric collection and result analysis
scripts/                      Local experiment and result-formatting scripts
```

Model weights, checkpoints, caches, credentials, and runtime logs are not
included. Download the required base models from their original model hosts.

## 1. Install

The recommended setup uses Python 3.10 and a virtual environment:

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

For the pinned environment, use
[`ContextAgent/environment.yml`](ContextAgent/environment.yml) instead.
Training and SFT evaluation require a CUDA-capable GPU. ICL evaluation uses
[Ollama](https://ollama.com/) and can run on a CPU if the selected model fits.

If a model requires authentication, set the token only in your local shell:

```bash
export HF_TOKEN="your-hugging-face-token"
```

Never commit credentials.

## 2. Run a laptop smoke test

Before running the full evaluation, test the complete wiring on two samples.
The small Ollama model keeps this practical on a laptop:

```bash
ollama serve
ollama pull qwen2.5:0.5b

scripts/run_icl.sh qwen2.5:0.5b false w_t w_p 11434 2
```

The final argument limits the run to two samples. This checks Ollama
connectivity, imports, dataset loading, inference, incremental result saving,
and metric collection. Score calculation is intentionally skipped because a
two-sample result is not a meaningful evaluation score.

## 3. Run the full ICL evaluation

Install and start Ollama, then run one evaluation:

```bash
ollama serve
ollama pull deepseek-r1:7b

scripts/run_icl.sh deepseek-r1:7b false w_t w_p 11434
```

The script runs CAB inference, records performance metrics, and calculates
evaluation scores. Run `scripts/run_icl.sh --help` to see all arguments.

## 4. Run SFT training

Make sure the base model and training data referenced by the configuration are
available:

```bash
scripts/run_sft_train.sh qwen_lora_sft qwen7b
```

Run `scripts/run_sft_train.sh --help` for the argument description.

## 5. Run SFT evaluation

After training or downloading the corresponding adapter:

```bash
scripts/run_sft_eval.sh \
  experiments/configs/cab/inference/qwen_lora_sft.yaml \
  qwen7b w_t w_p 8009 sandbox
```

Run `scripts/run_sft_eval.sh --help` for all options. The evaluation uses
batch size 1, matching the reported experiments.

## 6. Format existing results

To format CSV metrics that have already been generated:

```bash
scripts/format_results.sh --format terminal --methods both
```

## Reproducibility notes

The scripts are designed to run directly on a workstation or server. They do
not require SLURM, Singularity, or the original cluster filesystem. Large
model files are intentionally excluded from Git; their model identifiers and
paths are supplied through the experiment configurations.

## License

See [`LICENSE`](LICENSE).
