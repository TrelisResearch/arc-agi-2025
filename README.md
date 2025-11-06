# ARC-AGI 2025

This repository contains resources for working with the ARC-AGI (Abstraction and Reasoning Corpus for Artificial General Intelligence) datasets. 

> See [here](llm_python/experiment_notes.md) for experiment notes and [here](https://wandb.ai/trelis/arc-agi-2025) for Weights and Biases logs.

>[!TIP]
> Watch [this Youtube Video](https://youtube.com/live/ev2XuAktWpM?feature=share) to better understand how to use this repository.

## Setup / requirements

1. Install `uv`: https://docs.astral.sh/uv/getting-started/installation/
2. [Optional] set up `gcs` access (Linux only)
    a. **Linux**: Install `gcsfuse`: https://cloud.google.com/storage/docs/cloud-storage-fuse/quickstart-mount-bucket
    b. **Linux**: Configure the fuse mount with: `mkdir -p gcs && gcsfuse trelis-arc ./gcs`
    c. **macOS**: Use `gsutil` or `gcloud storage` commands directly (see examples below)

### Runpod Setup

You can start a GPU instance using [this Runpod template](https://console.runpod.io/deploy?template=bh0rvngapk&ref=jmfkcdio), which will install .toml dependencies AND unsloth, required for the fine-tuning notebook, via an on-start script.

To run the template, you'll need to set - in Runpod:
- `GITHUB_PAT`: a github personal access token with access to this arc-agi-2025 repo. YOU MUST GRANT CONTENTS: READ AND WRITE PERMISSION!
- `HUGGING_FACE_HUB_TOKEN`: a hugging face token with access the Trelis org [token](https://huggingface.co/settings/tokens)

To start via command line, you can try:
```bash
curl --request POST https://rest.runpod.io/v1/pods \
     --header "Authorization: Bearer $RUNPOD_API_KEY" \
     --header "Content-Type: application/json" \
     --data '{
       "templateId"     : "bh0rvngapk",
       "name"           : "arc-agi-h200",
       "cloudType"      : "SECURE",
       "gpuTypeIds"     : ["NVIDIA H200"],
       "gpuCount"       : 1,
       "volumeInGb"     : 150
     }'
```

### Quick Start - Evaluation

For evaluation, you can run models from a private endpoint OR from a Runpod instance that will automatically spin up (you need to set `RUNPOD_API_KEY` in a `.env` file for Runpod or set `OPENAI_API_KEY` to hit an openai style endpoint). To see evaluation options run:
```bash
uv run python3 -m llm_python.run_arc_tasks_soar -h
```
for example, to run 10 ARC PRIZE 2024 evaluation tasks using gpt-5-nano you would run:
```bash
uv run python3 -m llm_python.run_arc_tasks_soar --model gpt-5-nano --dataset arc-prize-2025 --subset evaluation --limit 10 --max_workers 64 --base-url https://openrouter.ai/api/v1 --unsafe-executor --max_attempts 2 --max-tokens 32000
```

You can also run a model from HuggingFace by automatically booting up a Runpod H100 and then evaluating that model:
```bash
PYTHONUNBUFFERED=1 nohup uv run runpod/create_pod_and_run_tasks.py arc-prize-2025 Trelis/Soar-qwen-14b-FP8-Dynamic --max-attempts 64 --subset training --max-workers 64 > julien31_soar_qwen_14b_all_100_training_64x.log 2>&1 &
```

### Quick Start - Training/Fine-tuning
Get started on a GPU (e.g. by booting up the runpod template above on an H100 SXM), and then navigate to `llm_python/fine-tuning/unsloth_arc_finetuning_soar.ipynb` in a jupyter terminal. Before running that notebook, adjust the values in the side-by-side config.yaml .

### UV Project Discovery

**Important**: When you run `uv` commands (like `uv venv`, `uv sync`, `uv run`) from subdirectories such as `llm_python/`, `uv` will automatically search upward through the directory hierarchy to find the root `pyproject.toml` file.

This means:
- Running `uv venv` from `llm_python/` will discover and use the root `pyproject.toml` configuration
- The virtual environment will respect the `requires-python = ">3.10,<3.12"` setting from the root
- All dependencies and project settings from the root will be applied

To create an isolated environment that ignores the root configuration, use:
```bash
uv venv --no-config
```

## Branches
-`o3-tools-images` a frozen version of the openai o3-tools branch that still maintains functionality for adding images to the prompts and feedback. This branch only supports openai models.
-`input-output-prog` contains a frozen version of the o3-tools branch with a prompt to generate input and output grids AS WELL as the transformation function. It uses openai responses api so is only compatible with openai models. This branch only supports openai models.

## Folders
- `data/` contains the ARC-AGI-1 and ARC-AGI-2 datasets.
- `llm_python` contains scripts for llm-generated python solutions to ARC tasks.

## Data

The `data/` folder contains:
- Complete ARC-AGI-1 dataset (800 tasks)
- Complete ARC-AGI-2 dataset (1,120 tasks)
- Predefined subsets for quick experimentation

For detailed information about the data structure, file formats, and how to work with the datasets, see the [data folder README](data/README.md).

## llm_python

The `llm_python/` folder contains a comprehensive testing framework for evaluating OpenAI models (o3, o4, GPT-4, Claude, custom Runpod endpoint, any openai style endpoint) on ARC-AGI tasks:

For complete documentation, usage examples, and detailed configuration options, see the [llm_python README](llm_python/README.md).

## Google Cloud Storage Operations

### Sync logs to GCS (Preferred Method - Parquet)

The preferred workflow is to first extract logs to parquet format, then upload to GCS for efficient processing in BigQuery:

#### Step 1: Extract logs to parquet
```bash
# Extract all logs to parquet format
uv run python -m llm_python.datasets.extract_from_logs --output your_output_file.parquet --logs-pattern="llm_python/logs/**/*.json" --batch-size 2000 --max-workers 4

# Example for capturing a specific day's run:
uv run python -m llm_python.datasets.extract_from_logs --output programs_20250811.parquet --logs-pattern="llm_python/logs/20250811*/*.json" --max-workers 4
```

#### Step 2: Upload parquet files to GCS
```bash
# Upload parquet file to GCS datasets folder
gsutil -m cp your_output_file.parquet gs://trelis-arc/datasets/your_output_file.parquet

# Example:
gsutil -m cp soar_log_programs_faking.parquet gs://trelis-arc/datasets/soar_log_programs_faking.parquet
```

Parquet files are stored in: https://console.cloud.google.com/storage/browser/trelis-arc/datasets

Once uploaded, these files can be quickly merged in BigQuery for analysis.

### Sync logs to GCS (Legacy Method - Direct sync)

```bash
# Using gsutil (works on all platforms, requires Python 3.8-3.12)
# Bidirectional sync (can both push and pull)
gsutil -m rsync -r llm_python/logs gs://trelis-arc/logs

# Push only (upload local files without downloading from GCS)
gsutil -m cp -r llm_python/logs/* gs://trelis-arc/logs/

# Using gcloud storage (modern alternative, recommended)
# Bidirectional sync (can both push and pull)
gcloud storage rsync llm_python/logs gs://trelis-arc/logs --recursive

# Push only (upload local files without downloading from GCS)
gcloud storage cp llm_python/logs gs://trelis-arc/logs --recursive
```

**Note:** 
- **rsync** commands perform **bidirectional synchronization** - they upload new/modified files and can also download files that exist in GCS but not locally
- **cp** commands are **push-only** - they only upload local files to GCS without downloading anything
- Both skip files that already exist and haven't changed, so you can safely run them repeatedly
- **Log optimization**: Prompts are now stored once per task instead of per attempt, reducing log file sizes by ~87% (7/8ths reduction)

### Other useful GCS commands (macOS/no fuse mount)

```bash
# List bucket contents
gcloud storage ls gs://trelis-arc/

# Download files
gcloud storage cp gs://trelis-arc/logs/* ./local-logs/ --recursive

# Upload files
gcloud storage cp ./local-files/* gs://trelis-arc/uploads/ --recursive
```