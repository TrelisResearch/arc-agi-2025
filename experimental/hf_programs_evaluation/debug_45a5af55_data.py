#!/usr/bin/env python3
"""
Debug the actual data for task 45a5af55.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_python.utils.task_loader import get_task_loader

def debug_task_45a5af55():
    """Debug task 45a5af55 data."""
    task_loader = get_task_loader()
    eval_tasks = task_loader.get_subset_tasks("arc-prize-2025/evaluation")

    for tid, task_data in eval_tasks:
        if tid == "45a5af55":
            print(f"Task: {tid}")

            first_input = task_data["train"][0]["input"]
            first_expected = task_data["train"][0]["output"]

            print(f"Input dimensions: {len(first_input)}x{len(first_input[0])}")
            print(f"Expected dimensions: {len(first_expected)}x{len(first_expected[0])}")

            print("\nInput (first 5 rows):")
            for i, row in enumerate(first_input[:5]):
                print(f"  {row}")

            print(f"\nInput values: {sorted(set(val for row in first_input for val in row))}")

            print("\nExpected (first 5 rows):")
            for i, row in enumerate(first_expected[:5]):
                print(f"  {row}")

            print(f"\nExpected values: {sorted(set(val for row in first_expected for val in row))}")

            return

if __name__ == "__main__":
    debug_task_45a5af55()