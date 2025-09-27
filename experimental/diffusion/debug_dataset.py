#!/usr/bin/env python3
"""
Debug script to inspect raw training data from the dataset.
"""
import sys
import numpy as np
from pathlib import Path

# Add the experimental directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from experimental.diffusion.src.dataset import ARCDataset, load_arc_data_paths

def debug_dataset():
    print("=== Debugging Dataset ===")

    # Load data paths (same as training)
    data_paths = load_arc_data_paths(
        data_dir='data/arc-prize-2024',
        datasets=['training_challenges', 'evaluation_challenges']
    )

    # Create dataset (same as training)
    dataset = ARCDataset(
        data_paths=data_paths['train'],
        max_size=30,
        augment=True,  # Include augmentation to match training
        n_augment=3,
        include_training_test_examples=True
    )

    print(f"Dataset loaded: {len(dataset)} examples from {len(dataset.task_id_to_idx)} tasks")

    # Show first 5 examples
    for i in range(min(5, len(dataset))):
        print(f"\n=== Example {i} ===")
        example = dataset[i]

        print(f"Task ID: {example['task_id']}")
        print(f"Task Index: {example['task_idx']}")
        print(f"Height: {example['height'].item()}")
        print(f"Width: {example['width'].item()}")

        # Get the raw grids (before padding)
        raw_example = dataset.examples[i]
        input_grid = raw_example['input_grid']
        output_grid = raw_example['output_grid']

        print(f"Raw input grid shape: {input_grid.shape}")
        print(f"Raw output grid shape: {output_grid.shape}")

        print("Raw input grid:")
        for row in input_grid:
            print(''.join(str(int(cell)) for cell in row))

        print("Raw output grid:")
        for row in output_grid:
            print(''.join(str(int(cell)) for cell in row))

        # Check padded versions
        padded_input = example['input_grid']
        padded_output = example['output_grid']
        print(f"Padded input shape: {padded_input.shape}")
        print(f"Padded output shape: {padded_output.shape}")

        # Show first few rows of padded output
        print("Padded output (first 5 rows, first 10 cols):")
        for row in padded_output[:5, :10].numpy():
            print(''.join(str(int(cell)) for cell in row))

        print("=" * 50)

    # Show task mapping
    print(f"\nTask mapping (first 10):")
    task_items = list(dataset.task_id_to_idx.items())[:10]
    for task_id, task_idx in task_items:
        print(f"  {task_id} -> {task_idx}")

if __name__ == "__main__":
    debug_dataset()