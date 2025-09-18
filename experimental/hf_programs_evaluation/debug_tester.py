#!/usr/bin/env python3
"""
Debug ArcTester execution.
"""

import sys
from pathlib import Path
from datasets import load_dataset

# Import utilities from the main codebase
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_python.utils.arc_tester import ArcTester
from llm_python.utils.task_loader import get_task_loader


def debug_tester():
    print("Debugging ArcTester...")

    # Load one program and one task
    print("Loading data...")
    ds = load_dataset('Trelis/arc-agi-1-perfect-2')
    program_data = ds["train"][0]  # First program
    program_code = program_data['code']

    task_loader = get_task_loader()
    eval_tasks = task_loader.get_subset_tasks("arc-prize-2024/evaluation")
    task_id, task_data = eval_tasks[0]  # First task

    print(f"Program code: {program_code[:200]}...")
    print(f"Task ID: {task_id}")
    print(f"Task train examples: {len(task_data['train'])}")

    # Test execution manually first
    print("\nTesting manual execution...")
    test_input = task_data['train'][0]['input']
    expected_output = task_data['train'][0]['output']

    print(f"Test input: {test_input}")
    print(f"Expected output: {expected_output}")

    # Initialize tester
    print("\nInitializing ArcTester...")
    try:
        tester = ArcTester(timeout=10, executor_type="unrestricted")
        print("ArcTester initialized successfully")

        # Test program execution
        print("\nTesting program execution...")
        result = tester.test_program(program_code, task_data)
        print(f"Test result: {result}")
        print(f"Success: {result.success}")
        print(f"Train outputs: {result.train_outputs}")
        print(f"Train correct: {result.correct_train_input}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ArcTester.cleanup_executor()


if __name__ == "__main__":
    debug_tester()