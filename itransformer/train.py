#!/usr/bin/env python3
"""
ARC Iterative Refiner Training Script

Usage:
    uv run python itransformer/train.py --config itransformer/configs/test_config.json
    uv run python itransformer/train.py --config itransformer/configs/test_config.json --no-wandb
"""
import json
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from itransformer.src.training import train_arc_iterative


def load_and_flatten_config(config_path: str) -> dict:
    """Load configuration from JSON file and flatten nested structure."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Flatten nested config sections
    flat_config = {}

    # Copy top-level fields
    for key in ['model_version', 'tag', 'pretrained_model_path']:
        if key in config:
            flat_config[key] = config[key]

    # Flatten model, training, data, and output sections
    for section in ['model', 'training', 'data', 'output']:
        if section in config:
            flat_config.update(config[section])

    # Keep auxiliary_loss and lora as nested dicts
    if 'auxiliary_loss' in config:
        flat_config['auxiliary_loss'] = config['auxiliary_loss']
    if 'lora' in config:
        flat_config['lora'] = config['lora']

    return flat_config


def main():
    parser = argparse.ArgumentParser(
        description="Train ARC Iterative Refiner with JSON Config"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to JSON config file"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging (overrides config)"
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU training"
    )

    args = parser.parse_args()

    # Load and flatten config
    print(f"Loading config from: {args.config}")
    config = load_and_flatten_config(args.config)

    # Override wandb if specified
    if args.no_wandb:
        print("Disabling wandb logging (--no-wandb)")
        config['use_wandb'] = False

    # Override device if specified
    if args.force_cpu:
        print("Forcing CPU training (--force-cpu)")
        config['device'] = 'cpu'

    # Run training
    train_arc_iterative(config)


if __name__ == "__main__":
    main()
