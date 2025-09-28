#!/usr/bin/env python3
"""
Debug script to show how much the grid actually changes during diffusion sampling.
"""
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experimental.diffusion.visualize_diffusion import DiffusionVisualizer
from llm_python.utils.task_loader import get_task_loader


def debug_progression_changes(model_path: str):
    """Check how much the grid actually changes during progression."""
    print(f"🔍 Debugging progression changes for: {model_path}")

    # Create visualizer
    visualizer = DiffusionVisualizer(
        model_path=model_path,
        device=None,  # Let it auto-detect
        num_visualization_steps=8
    )

    # Get a test task
    task_loader = get_task_loader()
    tasks = task_loader.get_dataset_subset("arc-prize-2024/evaluation", max_rows=1)
    if not tasks:
        tasks = task_loader.get_dataset_subset("arc-prize-2024/training", max_rows=1)

    task_id, task_data = tasks[0]
    test_example = task_data['test'][0]
    input_grid = np.array(test_example['input'])
    expected_output = np.array(test_example['output']) if 'output' in test_example else None

    print(f"🎯 Task: {task_id}")
    print(f"📐 Input shape: {input_grid.shape}")
    if expected_output is not None:
        print(f"📐 Expected output shape: {expected_output.shape}")

    # Run progression
    progression, timesteps = visualizer.sampler.sample_with_progression(
        input_grid=input_grid,
        task_id=task_id,
        num_steps=8
    )

    print(f"\\n📊 Progression Analysis:")
    print(f"Number of captured steps: {len(progression)}")
    print(f"Timesteps: {timesteps}")

    # Analyze changes between consecutive steps
    for i in range(1, len(progression)):
        prev_grid = progression[i-1]
        curr_grid = progression[i]

        # Count pixel differences
        if prev_grid.shape == curr_grid.shape:
            diff_pixels = np.sum(prev_grid != curr_grid)
            total_pixels = prev_grid.size
            change_percent = (diff_pixels / total_pixels) * 100

            print(f"\\nStep {i-1} → Step {i} (t={timesteps[i-1] if i-1 < len(timesteps) else 'noise'} → t={timesteps[i]}):")
            print(f"  Changed pixels: {diff_pixels}/{total_pixels} ({change_percent:.1f}%)")

            if diff_pixels > 0:
                # Show some examples of changes
                diff_positions = np.where(prev_grid != curr_grid)
                num_show = min(5, len(diff_positions[0]))
                print(f"  Example changes:")
                for j in range(num_show):
                    row, col = diff_positions[0][j], diff_positions[1][j]
                    old_val, new_val = prev_grid[row, col], curr_grid[row, col]
                    print(f"    ({row},{col}): {old_val} → {new_val}")
            else:
                print(f"  🔴 NO CHANGES detected!")

        else:
            print(f"\\nStep {i-1} → Step {i}: Different grid sizes, can't compare")

    # Check if final output makes sense
    final_grid = progression[-1]
    print(f"\\n🎯 Final Output Analysis:")
    print(f"Shape: {final_grid.shape}")
    print(f"Unique values: {np.unique(final_grid)}")
    print(f"Value counts:")
    unique_vals, counts = np.unique(final_grid, return_counts=True)
    for val, count in zip(unique_vals, counts):
        print(f"  {val}: {count} pixels")

    if expected_output is not None:
        if final_grid.shape == expected_output.shape:
            matches = np.sum(final_grid == expected_output)
            total = expected_output.size
            accuracy = (matches / total) * 100
            print(f"\\n✅ Accuracy vs Expected: {matches}/{total} ({accuracy:.1f}%)")
        else:
            print(f"\\n⚠️  Size mismatch: predicted {final_grid.shape} vs expected {expected_output.shape}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python debug_progression.py <model_path>")
        print("Example: python debug_progression.py experimental/diffusion/outputs/mps/best_model.pt")
        sys.exit(1)

    model_path = sys.argv[1]
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    debug_progression_changes(model_path)