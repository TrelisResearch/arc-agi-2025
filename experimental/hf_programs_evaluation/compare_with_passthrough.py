#!/usr/bin/env python3
"""
Compare the best SOAR program with a simple pass-through to see if they're similar.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_python.utils.task_loader import get_task_loader
import numpy as np


def calculate_pixel_match_percentage(predicted, expected):
    """Calculate pixel match percentage between predicted and expected grids."""
    if predicted is None:
        return 0.0

    pred_height = len(predicted)
    pred_width = len(predicted[0]) if pred_height > 0 else 0
    exp_height = len(expected)
    exp_width = len(expected[0]) if exp_height > 0 else 0

    if pred_height == 0 or pred_width == 0 or exp_height == 0 or exp_width == 0:
        return 0.0

    min_height = min(pred_height, exp_height)
    min_width = min(pred_width, exp_width)
    total_pixels = exp_height * exp_width
    matching_pixels = 0

    for i in range(min_height):
        for j in range(min_width):
            if predicted[i][j] == expected[i][j]:
                matching_pixels += 1

    return (matching_pixels / total_pixels) * 100.0


def passthrough_transform(grid_lst):
    """Simple pass-through - return input unchanged."""
    return grid_lst


def best_soar_transform(grid_lst):
    """Best SOAR program from parquet."""
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    output_grid = [row[:] for row in grid_lst]

    def is_surrounded_by_4s(i, j):
        if i > 0 and grid_lst[i - 1][j] != 4:
            return False
        if i < rows - 1 and grid_lst[i + 1][j] != 4:
            return False
        if j > 0 and grid_lst[i][j - 1] != 4:
            return False
        if j < cols - 1 and grid_lst[i][j + 1] != 4:
            return False
        return True

    for i in range(rows):
        for j in range(cols):
            if grid_lst[i][j] == 2:
                if is_surrounded_by_4s(i, j):
                    output_grid[i][j] = 8
            elif grid_lst[i][j] == 4:
                if is_surrounded_by_4s(i, j):
                    output_grid[i][j] = 8
    return output_grid


def load_task_1818057f():
    """Load task 1818057f from arc-prize-2025 evaluation."""
    task_loader = get_task_loader()
    eval_tasks = task_loader.get_subset_tasks("arc-prize-2025/evaluation")

    for tid, task_data in eval_tasks:
        if tid == "1818057f":
            return task_data

    raise ValueError("Task 1818057f not found in evaluation set")


def compare_programs():
    """Compare best SOAR program with pass-through."""
    print("Comparing best SOAR program vs simple pass-through on task 1818057f")

    task_data = load_task_1818057f()

    print(f"\nTask has {len(task_data['train'])} training examples and {len(task_data['test'])} test examples")

    # Test both programs
    programs = [
        ("Pass-through (no changes)", passthrough_transform),
        ("Best SOAR program", best_soar_transform)
    ]

    for program_name, transform_func in programs:
        print(f"\n{program_name}:")

        train_pixel_matches = []
        for i, example in enumerate(task_data["train"]):
            predicted = transform_func(example["input"])
            expected = example["output"]
            pixel_match = calculate_pixel_match_percentage(predicted, expected)
            train_pixel_matches.append(pixel_match)
            print(f"  Train {i+1}: {pixel_match:.1f}% pixel match")

        test_pixel_matches = []
        for i, example in enumerate(task_data["test"]):
            predicted = transform_func(example["input"])
            expected = example["output"]
            pixel_match = calculate_pixel_match_percentage(predicted, expected)
            test_pixel_matches.append(pixel_match)
            print(f"  Test {i+1}: {pixel_match:.1f}% pixel match")

        avg_train = np.mean(train_pixel_matches)
        avg_test = np.mean(test_pixel_matches)
        print(f"  Average train: {avg_train:.1f}%")
        print(f"  Average test: {avg_test:.1f}%")

    # Show first train example comparison
    print(f"\nFirst training example comparison:")
    first_input = task_data["train"][0]["input"]
    first_expected = task_data["train"][0]["output"]

    passthrough_output = passthrough_transform(first_input)
    soar_output = best_soar_transform(first_input)

    print(f"Input grid:")
    for row in first_input:
        print("  " + " ".join(str(x) for x in row))

    print(f"Expected output:")
    for row in first_expected:
        print("  " + " ".join(str(x) for x in row))

    print(f"Pass-through output:")
    for row in passthrough_output:
        print("  " + " ".join(str(x) for x in row))

    print(f"SOAR program output:")
    for row in soar_output:
        print("  " + " ".join(str(x) for x in row))


if __name__ == "__main__":
    compare_programs()