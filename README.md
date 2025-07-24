# ARC-AGI 2025

This repository contains resources for working with the ARC-AGI (Abstraction and Reasoning Corpus for Artificial General Intelligence) datasets.

## Setup / requirements

1. Install `uv`: https://docs.astral.sh/uv/getting-started/installation/
2. [Optional] set up `gcs` fuse mount
    a. Install `gcsfuse`: https://cloud.google.com/storage/docs/cloud-storage-fuse/quickstart-mount-bucket
    b. Configure the fuse mount with: `mkdir -p gcs && gcsfuse trelis-arc ./gcs`

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

## Sync logs to GCS

```bash
gsutil -m rsync -r llm-python/logs gs://trelis-arc/logs
```