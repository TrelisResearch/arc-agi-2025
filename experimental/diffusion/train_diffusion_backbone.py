#!/usr/bin/env python3
"""
ARC Diffusion Training with JSON Config Support

Usage:
    python experimental/diffusion/train_with_config.py --config configs/cpu_config.json
    python experimental/diffusion/train_with_config.py --config configs/gpu_config.json
"""
import json
import argparse
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experimental.diffusion.src.training import train_arc_diffusion


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Flatten the nested config for compatibility with existing training function
    flat_config = {}

    # Flatten model config
    flat_config.update(config['model'])

    # Flatten training config
    flat_config.update(config['training'])

    # Flatten data config
    flat_config.update(config['data'])

    # Flatten output config
    flat_config.update(config['output'])

    # Keep auxiliary_loss as nested dict (training.py expects config.get('auxiliary_loss', {}))
    flat_config['auxiliary_loss'] = config.get('auxiliary_loss', {})

    return flat_config


def main():
    parser = argparse.ArgumentParser(description="Train ARC Diffusion Model with JSON Config")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override wandb setting if requested
    if args.no_wandb:
        config['use_wandb'] = False

    print(f"Using config from: {args.config}")
    print(f"Config: {config}")

    # Train model
    model = train_arc_diffusion(config=config)

    print("Training completed successfully!")


if __name__ == "__main__":
    main()