#!/usr/bin/env python3
"""
Debug data loading issues.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Import utilities from the main codebase
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_python.utils.task_loader import get_task_loader


def debug_data():
    print("Debugging data loading...")

    # Load evaluation tasks
    print("Loading evaluation tasks...")
    task_loader = get_task_loader()
    eval_tasks = task_loader.get_subset_tasks("arc-prize-2024/evaluation")
    eval_tasks_dict = dict(eval_tasks[:2])  # Just first 2 tasks
    print(f"Loaded {len(eval_tasks_dict)} tasks")

    for task_id, task_data in eval_tasks_dict.items():
        print(f"\nTask ID: {task_id}")
        print(f"Task data type: {type(task_data)}")
        print(f"Task data keys: {task_data.keys() if isinstance(task_data, dict) else 'Not a dict'}")

        if isinstance(task_data, dict):
            print(f"Train examples: {len(task_data.get('train', []))}")
            if 'train' in task_data and len(task_data['train']) > 0:
                print(f"First train example: {task_data['train'][0]}")
                print(f"First train example type: {type(task_data['train'][0])}")

        break  # Just look at first task for debugging


if __name__ == "__main__":
    debug_data()