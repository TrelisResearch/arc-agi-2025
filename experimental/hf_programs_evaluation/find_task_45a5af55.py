#!/usr/bin/env python3
"""
Find where task 45a5af55 is located.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_python.utils.task_loader import get_task_loader

def find_task_45a5af55():
    """Find task 45a5af55 in available subsets."""
    target_task_id = "45a5af55"
    task_loader = get_task_loader()

    # Try different subsets
    subsets_to_try = [
        "arc-prize-2024/evaluation",
        "arc-prize-2024/training",
        "arc-prize-2025/evaluation",
        "arc-prize-2025/training"
    ]

    for subset in subsets_to_try:
        print(f"Checking {subset}...")
        try:
            tasks = task_loader.get_subset_tasks(subset)
            task_ids = [tid for tid, _ in tasks]

            if target_task_id in task_ids:
                print(f"Found {target_task_id} in {subset}!")
                return subset
            else:
                print(f"  Not found in {subset} (has {len(task_ids)} tasks)")
                # Show a few example task IDs
                if task_ids:
                    print(f"  Sample task IDs: {task_ids[:5]}")
        except Exception as e:
            print(f"  Error accessing {subset}: {e}")

    print(f"Task {target_task_id} not found in any subset")
    return None

if __name__ == "__main__":
    find_task_45a5af55()