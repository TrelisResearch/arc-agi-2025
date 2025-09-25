#!/usr/bin/env python3
"""
Main script to run manual task augmentation with black cell noise.

Usage:
    python run_augmentation.py --parquet-path /path/to/parquet --num-augmentations 10
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dataset_builder import AugmentedDatasetBuilder


def main():
    parser = argparse.ArgumentParser(description="Augment manual tasks with noise or geometric transformations")
    parser.add_argument(
        '--parquet-path',
        type=str,
        required=True,
        help="Path to parquet file with fully correct programs"
    )
    parser.add_argument(
        '--augmentation-type',
        type=str,
        choices=['noise', 'geometric'],
        default='noise',
        help="Type of augmentation to apply (default: noise)"
    )
    parser.add_argument(
        '--num-augmentations',
        type=int,
        default=10,
        help="Number of augmentations to attempt per task (default: 10)"
    )
    parser.add_argument(
        '--noise-percentage',
        type=float,
        default=0.1,
        help="Percentage of black cells to replace with noise (only for noise type, default: 0.1 = 10%%)"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help="Skip creating backup of original parquet file"
    )

    args = parser.parse_args()

    print("=== Manual Task Augmentation ===")
    print(f"Parquet file: {args.parquet_path}")
    print(f"Augmentation type: {args.augmentation_type}")
    print(f"Augmentations per task: {args.num_augmentations}")
    if args.augmentation_type == "noise":
        print(f"Noise percentage: {args.noise_percentage * 100:.1f}%")
    else:
        print("Transformations: horizontal/vertical flip, rotate (90°/180°/270°), recolor (+1 to +8)")
    print("Output directory: data/manual/ (arc-agi_augmented_*.json)")
    print(f"Random seed: {args.seed}")
    print()

    # Initialize builder
    builder = AugmentedDatasetBuilder(args.parquet_path, backup=not args.no_backup)

    # Generate augmentations
    print(f"Generating {args.augmentation_type} augmentations...")
    augmented_tasks, new_program_samples = builder.augment_tasks(
        num_augmentations=args.num_augmentations,
        augmentation_type=args.augmentation_type,
        noise_percentage=args.noise_percentage,
        base_seed=args.seed
    )

    if not augmented_tasks:
        print("No successful augmentations generated!")
        return

    # Save augmented tasks
    print(f"\nSaving {len(augmented_tasks)} augmented tasks...")
    builder.save_augmented_tasks(augmented_tasks, augmentation_type=args.augmentation_type)

    # Expand parquet file
    print(f"\nExpanding parquet with {len(new_program_samples)} new program samples...")
    expanded_parquet_path = builder.expand_parquet(new_program_samples, augmentation_type=args.augmentation_type)

    print(f"\n✅ {args.augmentation_type.title()} augmentation complete!")
    print(f"Expanded parquet saved to: {expanded_parquet_path}")


if __name__ == "__main__":
    main()