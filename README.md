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

## 1. Install prerequisites

The repository requires Git, `curl`, and a virtual environment. The
lightweight ICL smoke test supports Python 3.10 or newer, including Python
3.13. The full training, SFT, pruning, and knowledge-distillation
environment **requires Python 3.10, 3.11, or 3.12**; it is not compatible
with Python 3.13 because of the pinned NumPy and training dependencies. Full
training also requires a CUDA-capable NVIDIA GPU. The smoke test and ICL
evaluation can run on a CPU if the selected Ollama model fits.

On Debian or Ubuntu, install the system prerequisites with:

```bash
sudo apt update
sudo apt install -y git curl python3 python3-venv python3-pip zstd
```

For the full environment, install a supported interpreter explicitly. For
example, on Ubuntu 22.04 or newer:

```bash
sudo apt install -y python3.12 python3.12-venv python3.12-dev
```

If your distribution does not provide that package, use the
[Python downloads](https://www.python.org/downloads/) or
[deadsnakes PPA](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa) for
Ubuntu. Verify the interpreter before creating the environment:

```bash
python3.12 --version
```

`zstd` is required by the Ollama installer. On Fedora/RHEL, use
`sudo dnf install git curl python3 python3-pip zstd`. If available in your
Fedora release, install a supported version with
`sudo dnf install python3.12 python3.12-devel`; otherwise use
[pyenv](https://github.com/pyenv/pyenv) or the official Python installer.
On Arch Linux, use `sudo pacman -S git curl python python-pip zstd` and
install a 3.12 interpreter through
[pyenv](https://github.com/pyenv/pyenv) if the system Python is newer.

On macOS, install Git and Python with
[Homebrew](https://brew.sh/), then install the Ollama desktop application:

```bash
brew install python@3.12 git
```

Use `python3.12` when creating the full environment.

On Windows, install Git, Python, and Ollama using their official installers:
[Git](https://git-scm.com/download/win),
[Python](https://www.python.org/downloads/windows/), and
[Ollama](https://ollama.com/download). Run the commands below in Git Bash or
PowerShell, and choose Python 3.12 (or 3.10/3.11) in the Python installer.
Enable **Add Python to PATH**.

Install Ollama separately. On Linux, run:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

On macOS or Windows, install the Ollama desktop application from
[ollama.com/download](https://ollama.com/download). Verify the installation:

```bash
ollama --version
```

If the command is not found after installation, open a new terminal so that
your `PATH` is refreshed.

## 2. Install the project

Clone the repository and create a Python virtual environment:

```bash
git clone git@github.com:DAT8-CS-IT/revisiting-contextagent.git
cd revisiting-contextagent

python3 -m venv .venv
source .venv/bin/activate
which python
python --version
python -m pip install --upgrade pip
python -m pip install -r ContextAgent/requirements-icl.txt
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1` instead of `source .venv/bin/activate`.

The `which python` command must point to
`.../revisiting-contextagent/.venv/bin/python`. If it points to
`/usr/bin/python` or another system location, stop and activate the
environment before installing packages. Do not continue if pip prints
`Defaulting to user installation`.

Verify the two imports that are needed during tool discovery:

```bash
python -c "import tqdm, dateparser; from gcsa.google_calendar import GoogleCalendar; print('Core tool dependencies are installed')"
```

The smoke-test runner imports the repository code directly, so it does not
need an editable installation of `ContextAgent`. This avoids installing the
full training package metadata and prevents misleading warnings about omitted
packages such as `accelerate`, `datasets`, and `peft`.

The smoke test uses `requirements-icl.txt`, which avoids the CUDA training
stack and supports Python 3.13. Do not run `pip check` after this lightweight
installation: the package metadata for the full training environment contains
dependencies that are intentionally not installed for the smoke test.
If you already installed the package editable while following an older
revision of this guide, remove that metadata-only installation with
`python -m pip uninstall -y contextagent` and continue; do not uninstall the
dependencies from `requirements-icl.txt`.

For the complete training and SFT environment, use Python 3.10--3.12.
The following example uses Python 3.12 explicitly; use `python3.10` or
`python3.11` if that is the supported interpreter you installed. Remove any
old environment created with Python 3.13 first:

```bash
deactivate 2>/dev/null || true
rm -rf .venv
python3.12 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip
python -m pip install -r ContextAgent/requirements.txt
python -m pip install -e ContextAgent
python -m pip install -e ContextAgent/LLaMA-Factory
python -m pip check
```

On Windows PowerShell, replace the environment creation and activation
commands with:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

The default installation intentionally excludes PyAV because it is only
needed for multimedia model inputs and may require system FFmpeg development
libraries. The text-only experiments in this repository do not need it. If
you need multimedia support, install it separately on a system with FFmpeg 7,
`pkg-config`, and a C compiler:

```bash
python -m pip install -r ContextAgent/requirements-audio.txt
```

For the pinned environment, use
[`ContextAgent/environment.yml`](ContextAgent/environment.yml) instead.

If a model requires authentication, set the token only in your local shell:

```bash
export HF_TOKEN="your-hugging-face-token"
```

Never commit credentials.

## 3. Run a smoke test

Before running a full evaluation, test the complete wiring on two samples.
The small Ollama model keeps this practical on a laptop. Start Ollama in one
terminal:

```bash
ollama serve
```

In a second terminal, from the repository root, download the small model and
run the smoke test:

```bash
ollama pull qwen2.5:0.5b

scripts/run_icl.sh qwen2.5:0.5b false w_t w_p 11434 2
```

Keep `ollama serve` running in a separate terminal. Alternatively, let the
runner start a temporary local server:

```bash
START_OLLAMA=1 scripts/run_icl.sh qwen2.5:0.5b false w_t w_p 11434 2
```

The final argument limits the run to two samples. This checks Ollama
connectivity, imports, dataset loading, inference, incremental result saving,
and metric collection. Score calculation is intentionally skipped because a
two-sample result is not a meaningful evaluation score.

## 4. Run the full ICL evaluation

Install and start Ollama, then run one evaluation:

```bash
ollama serve
ollama pull deepseek-r1:7b

scripts/run_icl.sh deepseek-r1:7b false w_t w_p 11434
```

The script runs CAB inference, records performance metrics, and calculates
evaluation scores. Run `scripts/run_icl.sh --help` to see all arguments.

## 5. Run SFT training

Make sure the base model and training data referenced by the configuration are
available:

```bash
scripts/run_sft_train.sh qwen_lora_sft qwen7b
```

Run `scripts/run_sft_train.sh --help` for the argument description.

## 6. Run SFT evaluation

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
