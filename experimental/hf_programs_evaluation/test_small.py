#!/usr/bin/env python3
"""
Small test to verify the setup works correctly.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datasets import load_dataset

# Import utilities from the main codebase
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_python.utils.arc_tester import ArcTester, ProgramTestResult
from llm_python.utils.task_loader import get_task_loader


def test_small():
    print("Testing small setup...")

    # Load just a few programs
    print("Loading HuggingFace dataset...")
    ds = load_dataset('Trelis/arc-agi-1-perfect-2')
    hf_programs = [row for row in ds["train"][:5]]  # Just first 5
    print(f"Loaded {len(hf_programs)} programs")

    # Load evaluation tasks
    print("Loading evaluation tasks...")
    task_loader = get_task_loader()
    eval_tasks = task_loader.get_subset_tasks("arc-prize-2024/evaluation")
    eval_tasks_dict = dict(eval_tasks[:3])  # Just first 3 tasks
    print(f"Loaded {len(eval_tasks_dict)} tasks")

    # Test execution
    print("Testing execution...")
    tester = ArcTester(timeout=5, executor_type="unrestricted")

    for task_id, task_data in eval_tasks_dict.items():
        print(f"\nTesting task: {task_id}")
        for i, program_data in enumerate(hf_programs):
            print(f"  Program {i+1}: ", end="")
            try:
                result = tester.test_program(program_data['code'], task_data)
                train_correct = sum(result.correct_train_input)
                test_correct = sum(result.correct_test_input)
                print(f"train={train_correct}/{len(result.correct_train_input)}, test={test_correct}/{len(result.correct_test_input)}")
            except Exception as e:
                print(f"ERROR: {e}")

    ArcTester.cleanup_executor()
    print("\nTest complete!")


if __name__ == "__main__":
    test_small()