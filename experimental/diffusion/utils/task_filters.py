"""
Utility functions for filtering ARC tasks based on various criteria.
"""
import numpy as np
from typing import Dict, List, Tuple, Any


def task_exceeds_max_size(task_data: Dict[str, Any], max_size: int) -> bool:
    """
    Check if any grid in the task exceeds max_size.

    Args:
        task_data: Task data with 'train' and 'test' splits
        max_size: Maximum allowed grid dimension

    Returns:
        True if any grid exceeds max_size, False otherwise
    """
    for split in ['train', 'test']:
        for example in task_data.get(split, []):
            # Handle both numpy arrays and lists
            input_grid = example['input']
            output_grid = example['output']

            if isinstance(input_grid, list):
                input_grid = np.array(input_grid)
            if isinstance(output_grid, list):
                output_grid = np.array(output_grid)

            # Check input grid size
            if input_grid.shape[0] > max_size or input_grid.shape[1] > max_size:
                return True

            # Check output grid size
            if output_grid.shape[0] > max_size or output_grid.shape[1] > max_size:
                return True

    return False


def filter_tasks_by_max_size(
    tasks: Dict[str, Dict[str, Any]],
    max_size: int,
    verbose: bool = True
) -> Tuple[Dict[str, Dict[str, Any]], int, int]:
    """
    Filter tasks by maximum grid size and return statistics.

    Args:
        tasks: Dictionary of task_id -> task_data
        max_size: Maximum allowed grid dimension
        verbose: Whether to print filtering statistics

    Returns:
        Tuple of (filtered_tasks, total_tasks, filtered_count)
    """
    filtered_tasks = {}
    total_tasks = len(tasks)
    filtered_count = 0

    for task_id, task_data in tasks.items():
        if task_exceeds_max_size(task_data, max_size):
            filtered_count += 1
        else:
            filtered_tasks[task_id] = task_data

    remaining_tasks = len(filtered_tasks)

    if verbose:
        print(f"📋 Task filtering (max_size={max_size}):")
        print(f"  Total tasks: {total_tasks}")
        print(f"  Tasks filtered out: {filtered_count}")
        print(f"  Tasks remaining: {remaining_tasks}")

    return filtered_tasks, total_tasks, filtered_count