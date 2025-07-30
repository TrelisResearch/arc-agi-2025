# ARC-AGI 2025

This repository contains resources for working with the ARC-AGI (Abstraction and Reasoning Corpus for Artificial General Intelligence) datasets.

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

### UV Project Discovery

**Important**: When you run `uv` commands (like `uv venv`, `uv sync`, `uv run`) from subdirectories such as `llm-python/`, `uv` will automatically search upward through the directory hierarchy to find the root `pyproject.toml` file.

This means:
- Running `uv venv` from `llm-python/` will discover and use the root `pyproject.toml` configuration
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
- `llm-python` contains scripts for llm-generated python solutions to ARC tasks.

## Data

The `data/` folder contains:
- Complete ARC-AGI-1 dataset (800 tasks)
- Complete ARC-AGI-2 dataset (1,120 tasks)
- Predefined subsets for quick experimentation

For detailed information about the data structure, file formats, and how to work with the datasets, see the [data folder README](data/README.md).

## llm-python

The `llm-python/` folder contains a comprehensive testing framework for evaluating OpenAI models (o3, o4, GPT-4, Claude, custom Runpod endpoint, any openai style endpoint) on ARC-AGI tasks:

For complete documentation, usage examples, and detailed configuration options, see the [llm-python README](llm-python/README.md).

## Google Cloud Storage Operations

### Sync logs to GCS

```bash
# Using gsutil (works on all platforms, requires Python 3.8-3.12)
gsutil -m rsync -r llm-python/logs gs://trelis-arc/logs

# Using gcloud storage (modern alternative, recommended)
gcloud storage rsync llm-python/logs gs://trelis-arc/logs --recursive
```

**Note:** Both sync commands perform **incremental synchronization** - they only upload new, modified, or missing files. Files that already exist and haven't changed are automatically skipped, so you can safely run these commands repeatedly without duplicating uploads.

### Other useful GCS commands (macOS/no fuse mount)

```bash
# List bucket contents
gcloud storage ls gs://trelis-arc/

# Download files
gcloud storage cp gs://trelis-arc/logs/* ./local-logs/ --recursive

# Upload files
gcloud storage cp ./local-files/* gs://trelis-arc/uploads/ --recursive
```