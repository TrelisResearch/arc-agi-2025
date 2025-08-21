# Fine-tuning

Contains fine-tuning notebooks and scripts for ARC-AGI task training.

## Notebooks

- **unsloth_arc_finetuning_soar.ipynb**: A fine-tuning notebook that uses the SOAR prompt. Recommended for now, unless results show another prompt to perform better.

Archived:
- **unsloth_arc_finetuning.ipynb**: A notebook with an incorrect v1 prompt that over-constrains the grid outputs.

## Notebook to Script Conversion

Convert Jupyter notebooks to executable Python scripts with YAML configuration support for easier experimentation and automation.

### Quick Start

1. **Convert a notebook to script:**
   ```bash
   uv run python notebook_to_script.py unsloth_arc_finetuning_soar.ipynb
   ```

2. **Run the converted script:**
   ```bash
   uv run python unsloth_arc_finetuning_soar_script.py
   ```

3. **Run with custom config:**
   ```bash
   uv run python unsloth_arc_finetuning_soar_script.py --config my_experiment.yaml
   ```

### Files

- **`notebook_to_script.py`** - Utility to convert notebooks to scripts
- **`config.yaml`** - Default configuration file with all tunable parameters
- **`unsloth_arc_finetuning_soar_script.py`** - Converted script from the main notebook

### Configuration

The `config.yaml` file contains all configuration options:

```yaml
# Test mode for quick runs
test_run: false

# Model settings
model:
  slug: "Qwen/Qwen3-4B"
  max_length: 32768
  lora_rank: 128

# Training settings
training:
  batch_size_global: 4
  dataset_slug: "Trelis/arc-agi-2-perfect-50"
  max_rows: null  # null for all data
  enable_thinking: false
```

### Benefits

- **Easy experiments**: Modify YAML instead of notebook cells
- **Version control**: Clean, readable configuration files
- **Reproducible**: Configuration is separate from code
- **Scriptable**: Run in automated pipelines
- **No code changes**: Existing notebook code works unchanged

### Advanced Usage

Convert any notebook with custom options:
```bash
# Convert with specific config and output
uv run python notebook_to_script.py my_notebook.ipynb \
  --config my_config.yaml \
  --output my_script.py

# Convert without config loader (keeps original config cell)
uv run python notebook_to_script.py my_notebook.ipynb --no-config-loader
```

Create experiment configs by copying and modifying `config.yaml`:
```bash
cp config.yaml experiment_1.yaml
# Edit experiment_1.yaml with different parameters
uv run python unsloth_arc_finetuning_soar_script.py --config experiment_1.yaml
```

## TensorBoard Logs
To view training logs:
```bash
uv venv
uv pip install tensorboard
tensorboard --logdir logs
```

Open browser to http://localhost:6006/ or via runpod proxy: https://<pod-id>-6006.proxy.runpod.net

## Runpod One-click Template

**Runpod One-click-template FOR FINE-TUNING**
You can start a GPU instance using [this Runpod template](https://console.runpod.io/deploy?template=bh0rvngapk&ref=jmfkcdio), which will install .toml dependencies AND unsloth, required for the fine-tuning notebook, via an on-start script.

To run the template, you'll need to set - in Runpod:
- `GITHUB_PAT`: a github personal access token with access to this arc-agi-2025 repo. YOU MUST GRANT CONTENTS: READ AND WRITE PERMISSION!
- `HUGGING_FACE_HUB_TOKEN`: a hugging face token with access the Trelis org [token](https://huggingface.co/settings/tokens)