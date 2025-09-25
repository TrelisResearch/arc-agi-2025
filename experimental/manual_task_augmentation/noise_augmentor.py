#!/usr/bin/env python3
"""
Black cell noise augmentation for manual tasks.
Adds random colors (1-9) to 10% of black/0 cells in input grids.
"""

import random
from typing import List, Dict, Any, Tuple
from copy import deepcopy


def count_black_cells(grid: List[List[int]]) -> int:
    """Count the number of black (0) cells in a grid."""
    count = 0
    for row in grid:
        for cell in row:
            if cell == 0:
                count += 1
    return count


def add_noise_to_grid(grid: List[List[int]], noise_percentage: float = 0.1, seed: int = None) -> List[List[int]]:
    """
    Add noise to black cells in a grid by replacing them with random colors 1-9.

    Args:
        grid: Input grid as 2D list of integers
        noise_percentage: Fraction of black cells to replace (default 0.1 = 10%)
        seed: Random seed for reproducibility

    Returns:
        Noisy grid with some black cells replaced by random colors
    """
    if seed is not None:
        random.seed(seed)

    # Deep copy to avoid modifying original
    noisy_grid = deepcopy(grid)

    # Find all black cell positions
    black_positions = []
    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if grid[r][c] == 0:
                black_positions.append((r, c))

    # Calculate how many black cells to modify
    num_to_modify = int(len(black_positions) * noise_percentage)
    if num_to_modify == 0 and black_positions:
        num_to_modify = 1  # Always modify at least 1 if possible

    # Randomly select positions to modify
    positions_to_modify = random.sample(black_positions, min(num_to_modify, len(black_positions)))

    # Replace selected black cells with random colors 1-9
    for r, c in positions_to_modify:
        noisy_grid[r][c] = random.randint(1, 9)

    return noisy_grid


def augment_task(task_data: Dict[str, Any], augmentation_id: int, noise_percentage: float = 0.1, seed: int = None) -> Tuple[str, Dict[str, Any]]:
    """
    Create an augmented version of a task by adding noise to all input grids.

    Args:
        task_data: Original task data with 'train' and 'test' examples
        augmentation_id: Unique ID for this augmentation
        noise_percentage: Fraction of black cells to replace (default 0.1 = 10%)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (new_task_id, augmented_task_data)
    """
    original_task_id = task_data.get('task_id', 'unknown')
    new_task_id = f"{original_task_id}_aug_{augmentation_id:03d}"

    # Deep copy original task
    augmented_task = deepcopy(task_data)

    # Add noise to all training input grids
    for i, example in enumerate(augmented_task['train']):
        example_seed = seed + i if seed is not None else None
        example['input'] = add_noise_to_grid(example['input'], noise_percentage=noise_percentage, seed=example_seed)

    # Add noise to all test input grids
    for i, example in enumerate(augmented_task['test']):
        example_seed = seed + len(augmented_task['train']) + i if seed is not None else None
        example['input'] = add_noise_to_grid(example['input'], noise_percentage=noise_percentage, seed=example_seed)

    return new_task_id, augmented_task


def generate_augmentations(task_data: Dict[str, Any], num_augmentations: int, noise_percentage: float = 0.1, base_seed: int = None) -> Dict[str, Dict[str, Any]]:
    """
    Generate multiple augmented versions of a task.

    Args:
        task_data: Original task data
        num_augmentations: Number of augmented versions to create
        noise_percentage: Fraction of black cells to replace (default 0.1 = 10%)
        base_seed: Base seed for reproducibility

    Returns:
        Dictionary mapping new_task_id -> augmented_task_data
    """
    augmentations = {}

    for i in range(num_augmentations):
        aug_seed = base_seed + i if base_seed is not None else None
        new_task_id, aug_task = augment_task(task_data, i + 1, noise_percentage=noise_percentage, seed=aug_seed)
        augmentations[new_task_id] = aug_task

    return augmentations