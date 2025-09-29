# ARC Diffusion Training Pipeline

The `run_full_pipeline.py` script provides a convenient way to run the complete ARC diffusion training pipeline in sequence.

## Quick Start

```bash
# Run the full pipeline (training + size head + evaluation)
python experimental/diffusion/run_full_pipeline.py experimental/diffusion/configs/my_config.json

# Skip training and only run evaluation
python experimental/diffusion/run_full_pipeline.py experimental/diffusion/configs/my_config.json --skip-training --skip-size-head

# Run with limited evaluation tasks for testing
python experimental/diffusion/run_full_pipeline.py experimental/diffusion/configs/my_config.json --eval-limit 10
```

## Pipeline Steps

1. **Diffusion Backbone Training** - Trains the main diffusion model
2. **Size Head Training** - Trains the auxiliary size prediction head (optional)
3. **Model Evaluation** - Evaluates the model on ARC tasks with pass@2 scoring

## Options

- `--skip-training` - Skip diffusion backbone training
- `--skip-size-head` - Skip size head training
- `--skip-evaluation` - Skip evaluation
- `--eval-limit N` - Limit evaluation to N tasks (default: 5)

## Configuration

The script uses a single JSON config file that contains all settings for training, size head training, and evaluation. See `experimental/diffusion/configs/test_config.json` for an example.

Required config sections:
- `model` - Model architecture settings
- `training` - Training hyperparameters
- `size_head` - Size head settings (optional)
- `output` - Output directory configuration

## Output Files

The pipeline saves several key files in the configured output directory:

- `best_model.pt` - Best diffusion backbone model
- `best_size_head.pt` - Best size head model (if trained)
- `evaluation_*.json` - Evaluation results with pass@2 metrics
- `training_noise_visualization.png` - Visualization of training data and noise schedule

## Example Usage

```bash
# Create a config file
cat > my_config.json << EOF
{
  "model_version": "v4_diff_w_head",
  "model": {
    "vocab_size": 11,
    "d_model": 128,
    "nhead": 4,
    "num_layers": 8,
    "max_size": 20
  },
  "training": {
    "optimizer_steps": 10000,
    "num_timesteps": 128,
    "schedule_type": "cosine",
    "batch_size": 32,
    "learning_rate": 0.0005
  },
  "size_head": {
    "hidden_dim": 128,
    "epochs": 5,
    "learning_rate": 1e-3,
    "batch_size": 32
  },
  "output": {
    "output_dir": "experimental/diffusion/outputs/my_experiment",
    "use_wandb": false
  }
}
EOF

# Run the full pipeline
python experimental/diffusion/run_full_pipeline.py my_config.json
```

The pipeline will automatically handle error recovery and provides detailed progress reporting for each step.