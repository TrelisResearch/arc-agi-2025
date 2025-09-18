#!/usr/bin/env python3
"""
Run a subset of programs from the HuggingFace dataset on a few ARC-AGI 2024 evaluation tasks.
This is a shorter test version.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, Counter
from datasets import load_dataset

# Import utilities from the main codebase
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_python.utils.arc_tester import ArcTester, ProgramTestResult
from llm_python.utils.task_loader import get_task_loader


def load_evaluation_tasks(max_tasks: int = 5) -> Dict[str, Any]:
    """Load ARC-AGI 2024 evaluation tasks."""
    task_loader = get_task_loader()
    eval_tasks = task_loader.get_subset_tasks("arc-prize-2024/evaluation")
    return dict(eval_tasks[:max_tasks])


def load_hf_programs(max_programs: int = 50) -> List[Dict[str, Any]]:
    """Load programs from the HuggingFace dataset."""
    print("Loading HuggingFace dataset...")
    ds = load_dataset('Trelis/arc-agi-1-perfect-2')
    return [ds["train"][i] for i in range(min(max_programs, len(ds["train"])))]


def analyze_task_performance(task_results: List[ProgramTestResult]) -> Dict[str, int]:
    """Analyze performance on a single task across all programs."""
    stats = {
        'total_programs': len(task_results),
        'all_correct': 0,  # Both train and test all correct
        'all_train_correct': 0,  # All training examples correct
        'at_least_one_train_correct': 0,  # At least one training example correct
        'no_execution': 0,  # Programs that failed to execute
        'no_output': 0,  # Programs that executed but produced no valid output
    }

    for result in task_results:
        # Check if program executed successfully
        if not result.success:
            stats['no_execution'] += 1
            continue

        # Check if any outputs were generated
        if not any(out is not None for out in result.train_outputs + result.test_outputs):
            stats['no_output'] += 1
            continue

        # Count correct train examples
        train_correct_count = sum(result.correct_train_input)
        test_correct_count = sum(result.correct_test_input)
        total_train = len(result.correct_train_input)
        total_test = len(result.correct_test_input)

        # All train correct
        if train_correct_count == total_train and total_train > 0:
            stats['all_train_correct'] += 1

        # All correct (both train and test)
        if (train_correct_count == total_train and test_correct_count == total_test and
            total_train > 0 and total_test > 0):
            stats['all_correct'] += 1

        # At least one train correct
        if train_correct_count > 0:
            stats['at_least_one_train_correct'] += 1

    return stats


def run_programs_on_evaluation():
    """Main function to run subset of programs on evaluation set."""
    print("Starting SHORT evaluation of HF programs on ARC-AGI 2024...")

    # Load data
    evaluation_tasks = load_evaluation_tasks(max_tasks=5)  # Only 5 tasks
    hf_programs = load_hf_programs(max_programs=50)  # Only first 50 programs

    print(f"Loaded {len(evaluation_tasks)} evaluation tasks")
    print(f"Loaded {len(hf_programs)} programs from HF dataset")

    # Initialize tester
    tester = ArcTester(timeout=10, executor_type="unrestricted")

    # Store results
    all_task_stats = {}
    program_performance = defaultdict(lambda: {'tasks_solved': 0, 'tasks_attempted': 0})

    # Process each evaluation task
    for task_idx, (task_id, task_data) in enumerate(evaluation_tasks.items()):
        print(f"\nProcessing task {task_idx + 1}/{len(evaluation_tasks)}: {task_id}")

        task_results = []

        # Run each program on this task
        for prog_idx, program_data in enumerate(hf_programs):
            if (prog_idx + 1) % 10 == 0:
                print(f"  Program {prog_idx + 1}/{len(hf_programs)}")

            program_code = program_data['code']

            try:
                result = tester.test_program(program_code, task_data)
                task_results.append(result)

                # Track program performance
                prog_id = program_data['row_id']
                program_performance[prog_id]['tasks_attempted'] += 1

                # Consider task solved if all train and test are correct
                train_correct = sum(result.correct_train_input) == len(result.correct_train_input)
                test_correct = sum(result.correct_test_input) == len(result.correct_test_input)
                if train_correct and test_correct and len(result.correct_train_input) > 0:
                    program_performance[prog_id]['tasks_solved'] += 1

            except Exception as e:
                print(f"    Error running program {prog_idx}: {e}")
                # Create a dummy result for failed execution
                task_results.append(ProgramTestResult(
                    train_outputs=[], test_outputs=[], train_inputs=[], test_inputs=[],
                    correct_train_input=[], correct_test_input=[], success=False
                ))

        # Analyze results for this task
        task_stats = analyze_task_performance(task_results)
        all_task_stats[task_id] = task_stats

        print(f"  Task {task_id} results:")
        print(f"    All correct (train + test): {task_stats['all_correct']}")
        print(f"    All train correct: {task_stats['all_train_correct']}")
        print(f"    At least one train correct: {task_stats['at_least_one_train_correct']}")

    # Generate summary report
    print("\n" + "="*60)
    print("SHORT EVALUATION SUMMARY REPORT")
    print("="*60)

    # Overall statistics
    total_all_correct = sum(stats['all_correct'] for stats in all_task_stats.values())
    total_all_train_correct = sum(stats['all_train_correct'] for stats in all_task_stats.values())
    total_at_least_one_train = sum(stats['at_least_one_train_correct'] for stats in all_task_stats.values())

    print(f"\nAcross all {len(evaluation_tasks)} evaluation tasks:")
    print(f"Total program-task pairs with all correct (train + test): {total_all_correct}")
    print(f"Total program-task pairs with all train correct: {total_all_train_correct}")
    print(f"Total program-task pairs with at least one train correct: {total_at_least_one_train}")
    print(f"Total program-task pairs tested: {len(evaluation_tasks) * len(hf_programs)}")

    # Tasks that had at least one solution
    tasks_with_all_correct = len([task for task, stats in all_task_stats.items() if stats['all_correct'] > 0])
    tasks_with_train_correct = len([task for task, stats in all_task_stats.items() if stats['all_train_correct'] > 0])
    tasks_with_some_train = len([task for task, stats in all_task_stats.items() if stats['at_least_one_train_correct'] > 0])

    print(f"\nTasks with at least one perfect solution (train + test): {tasks_with_all_correct}/{len(evaluation_tasks)}")
    print(f"Tasks with at least one all-train solution: {tasks_with_train_correct}/{len(evaluation_tasks)}")
    print(f"Tasks with at least one partial train solution: {tasks_with_some_train}/{len(evaluation_tasks)}")

    # Top performing programs
    print(f"\nTop 10 programs by tasks solved:")
    sorted_programs = sorted(program_performance.items(), key=lambda x: x[1]['tasks_solved'], reverse=True)
    for i, (prog_id, stats) in enumerate(sorted_programs[:10]):
        print(f"  {i+1}. {prog_id}: {stats['tasks_solved']}/{stats['tasks_attempted']} tasks")

    # Save detailed results
    output_file = Path(__file__).parent / "short_evaluation_results.json"
    results_data = {
        'summary': {
            'total_evaluation_tasks': len(evaluation_tasks),
            'total_programs': len(hf_programs),
            'total_all_correct': total_all_correct,
            'total_all_train_correct': total_all_train_correct,
            'total_at_least_one_train': total_at_least_one_train,
            'tasks_with_all_correct': tasks_with_all_correct,
            'tasks_with_train_correct': tasks_with_train_correct,
            'tasks_with_some_train': tasks_with_some_train,
        },
        'per_task_stats': all_task_stats,
        'program_performance': dict(program_performance)
    }

    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")
    print("\nShort evaluation complete!")

    # Cleanup executor
    ArcTester.cleanup_executor()


if __name__ == "__main__":
    run_programs_on_evaluation()