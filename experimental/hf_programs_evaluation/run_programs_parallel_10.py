#!/usr/bin/env python3
"""
Parallelized evaluation of all programs from the HuggingFace dataset on 10 ARC-AGI 2024 evaluation tasks.
Uses 4 cores and includes detailed pixel match metrics.

This script provides:
1. Standard correctness metrics (all-correct, all-train-correct, partial-correct)
2. Pixel match percentages for partial solutions
3. Median and max pixel match ratios per task
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from collections import defaultdict, Counter
from datasets import load_dataset
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

# Import utilities from the main codebase
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_python.utils.arc_tester import ArcTester, ProgramTestResult
from llm_python.utils.task_loader import get_task_loader


class PixelMatchResult(NamedTuple):
    """Result including pixel match percentages for partial solutions."""
    program_result: ProgramTestResult
    train_pixel_matches: List[float]  # Pixel match % for each train example
    test_pixel_matches: List[float]   # Pixel match % for each test example
    program_id: str
    task_id: str


def calculate_pixel_match_percentage(predicted: Optional[List[List[int]]],
                                   expected: List[List[int]]) -> float:
    """Calculate pixel match percentage between predicted and expected grids."""
    if predicted is None:
        return 0.0

    # Handle different grid sizes by comparing overlapping region
    pred_height = len(predicted)
    pred_width = len(predicted[0]) if pred_height > 0 else 0
    exp_height = len(expected)
    exp_width = len(expected[0]) if exp_height > 0 else 0

    if pred_height == 0 or pred_width == 0 or exp_height == 0 or exp_width == 0:
        return 0.0

    # Compare overlapping region
    min_height = min(pred_height, exp_height)
    min_width = min(pred_width, exp_width)

    total_pixels = exp_height * exp_width  # Total pixels in expected output
    matching_pixels = 0

    # Count matches in overlapping region
    for i in range(min_height):
        for j in range(min_width):
            if predicted[i][j] == expected[i][j]:
                matching_pixels += 1

    return (matching_pixels / total_pixels) * 100.0


def evaluate_single_program_task(args: Tuple[str, Dict[str, Any], str, Any]) -> PixelMatchResult:
    """Evaluate a single program on a single task. Designed for multiprocessing."""
    program_code, task_data, program_id, task_id = args

    # Initialize tester for this process
    tester = ArcTester(timeout=10, executor_type="unrestricted")

    try:
        # Test the program
        result = tester.test_program(program_code, task_data)

        # Calculate pixel matches for train examples
        train_pixel_matches = []
        for i, (predicted, expected) in enumerate(zip(result.train_outputs, [ex["output"] for ex in task_data["train"]])):
            pixel_match = calculate_pixel_match_percentage(predicted, expected)
            train_pixel_matches.append(pixel_match)

        # Calculate pixel matches for test examples
        test_pixel_matches = []
        for i, (predicted, expected) in enumerate(zip(result.test_outputs, [ex["output"] for ex in task_data["test"]])):
            pixel_match = calculate_pixel_match_percentage(predicted, expected)
            test_pixel_matches.append(pixel_match)

        return PixelMatchResult(
            program_result=result,
            train_pixel_matches=train_pixel_matches,
            test_pixel_matches=test_pixel_matches,
            program_id=program_id,
            task_id=task_id
        )

    except Exception as e:
        # Return dummy result for failed execution
        dummy_result = ProgramTestResult(
            train_outputs=[], test_outputs=[], train_inputs=[], test_inputs=[],
            correct_train_input=[], correct_test_input=[], success=False
        )
        return PixelMatchResult(
            program_result=dummy_result,
            train_pixel_matches=[],
            test_pixel_matches=[],
            program_id=program_id,
            task_id=task_id
        )
    finally:
        ArcTester.cleanup_executor()


def load_evaluation_tasks(max_tasks: int = 10) -> Dict[str, Any]:
    """Load ARC-AGI 2024 evaluation tasks."""
    task_loader = get_task_loader()
    eval_tasks = task_loader.get_subset_tasks("arc-prize-2024/evaluation")
    return dict(eval_tasks[:max_tasks])


def load_hf_programs() -> List[Dict[str, Any]]:
    """Load programs from the HuggingFace dataset."""
    print("Loading HuggingFace dataset...")
    ds = load_dataset('Trelis/arc-agi-1-perfect-2')
    return [ds["train"][i] for i in range(len(ds["train"]))]


def analyze_task_performance_with_pixels(task_results: List[PixelMatchResult]) -> Dict[str, Any]:
    """Analyze performance on a single task across all programs, including pixel metrics."""
    stats = {
        'total_programs': len(task_results),

        # Standard correctness metrics
        'all_correct': 0,  # Both train and test all correct
        'all_train_correct': 0,  # All training examples correct
        'at_least_one_train_correct': 0,  # At least one training example correct
        'no_execution': 0,  # Programs that failed to execute
        'no_output': 0,  # Programs that executed but produced no valid output

        # Pixel match metrics
        'median_train_pixel_match': 0.0,
        'max_train_pixel_match': 0.0,
        'median_test_pixel_match': 0.0,
        'max_test_pixel_match': 0.0,
        'median_max_ratio': 0.0,  # median/max ratio for this task
    }

    all_train_pixels = []
    all_test_pixels = []

    for pixel_result in task_results:
        result = pixel_result.program_result

        # Collect pixel matches
        all_train_pixels.extend(pixel_result.train_pixel_matches)
        all_test_pixels.extend(pixel_result.test_pixel_matches)

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

    # Calculate pixel match statistics
    if all_train_pixels:
        stats['median_train_pixel_match'] = float(np.median(all_train_pixels))
        stats['max_train_pixel_match'] = float(np.max(all_train_pixels))

    if all_test_pixels:
        stats['median_test_pixel_match'] = float(np.median(all_test_pixels))
        stats['max_test_pixel_match'] = float(np.max(all_test_pixels))

    # Calculate median/max ratio
    if stats['max_train_pixel_match'] > 0:
        stats['median_max_ratio'] = stats['median_train_pixel_match'] / stats['max_train_pixel_match']

    return stats


def run_programs_on_evaluation_parallel_10():
    """Main function to run all programs on 10 evaluation tasks using parallel processing."""
    print("Starting PARALLEL evaluation of HF programs on 10 ARC-AGI 2024 tasks...")
    start_time = time.time()

    # Load data
    print("Loading data...")
    evaluation_tasks = load_evaluation_tasks(max_tasks=10)  # Only 10 tasks
    hf_programs = load_hf_programs()

    print(f"Loaded {len(evaluation_tasks)} evaluation tasks")
    print(f"Loaded {len(hf_programs)} programs from HF dataset")
    print(f"Total program-task pairs to evaluate: {len(evaluation_tasks) * len(hf_programs)}")

    # Prepare all work items
    work_items = []
    for task_id, task_data in evaluation_tasks.items():
        for program_data in hf_programs:
            work_items.append((
                program_data['code'],  # program_code
                task_data,            # task_data
                program_data['row_id'],  # program_id
                task_id               # task_id
            ))

    print(f"Using 4 cores to process {len(work_items)} evaluations...")

    # Process with parallel workers
    all_results = []
    completed_count = 0

    with ProcessPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        future_to_work = {executor.submit(evaluate_single_program_task, item): item for item in work_items}

        # Process completed tasks
        for future in as_completed(future_to_work):
            completed_count += 1
            if completed_count % 500 == 0:  # More frequent updates for smaller run
                elapsed = time.time() - start_time
                rate = completed_count / elapsed
                remaining = len(work_items) - completed_count
                eta = remaining / rate if rate > 0 else 0
                print(f"Completed {completed_count}/{len(work_items)} evaluations ({completed_count/len(work_items)*100:.1f}%). "
                      f"Rate: {rate:.1f}/sec. ETA: {eta/60:.1f} minutes")

            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"Error in evaluation: {e}")

    print(f"Completed all evaluations in {(time.time() - start_time)/60:.1f} minutes")

    # Group results by task
    task_results = defaultdict(list)
    program_performance = defaultdict(lambda: {'tasks_solved': 0, 'tasks_attempted': 0})

    for pixel_result in all_results:
        task_results[pixel_result.task_id].append(pixel_result)

        # Track program performance
        program_performance[pixel_result.program_id]['tasks_attempted'] += 1

        result = pixel_result.program_result
        if result.success:
            train_correct = sum(result.correct_train_input) == len(result.correct_train_input)
            test_correct = sum(result.correct_test_input) == len(result.correct_test_input)
            if train_correct and test_correct and len(result.correct_train_input) > 0:
                program_performance[pixel_result.program_id]['tasks_solved'] += 1

    # Analyze results per task
    print("\nAnalyzing results...")
    all_task_stats = {}
    for task_id, results in task_results.items():
        task_stats = analyze_task_performance_with_pixels(results)
        all_task_stats[task_id] = task_stats

    # Generate summary report
    print("\n" + "="*80)
    print("PARALLEL EVALUATION SUMMARY REPORT (10 TASKS) WITH PIXEL METRICS")
    print("="*80)

    # Overall statistics
    total_all_correct = sum(stats['all_correct'] for stats in all_task_stats.values())
    total_all_train_correct = sum(stats['all_train_correct'] for stats in all_task_stats.values())
    total_at_least_one_train = sum(stats['at_least_one_train_correct'] for stats in all_task_stats.values())

    print(f"\nSTANDARD CORRECTNESS METRICS:")
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

    # Pixel match statistics
    all_median_ratios = [stats['median_max_ratio'] for stats in all_task_stats.values() if stats['median_max_ratio'] > 0]
    overall_median_ratio = np.median(all_median_ratios) if all_median_ratios else 0.0

    print(f"\nPIXEL MATCH METRICS:")
    print(f"Overall median/max pixel match ratio across all tasks: {overall_median_ratio:.3f}")
    print(f"Tasks with pixel match data: {len(all_median_ratios)}/{len(evaluation_tasks)}")

    # Detailed per-task breakdown
    print(f"\nDETAILED PER-TASK BREAKDOWN:")
    sorted_tasks = sorted(all_task_stats.items(), key=lambda x: x[1]['median_train_pixel_match'], reverse=True)

    for i, (task_id, stats) in enumerate(sorted_tasks):
        print(f"\n{i+1}. Task {task_id}:")
        print(f"   Correctness: all-correct={stats['all_correct']}, all-train={stats['all_train_correct']}, partial={stats['at_least_one_train_correct']}")
        print(f"   Pixel match: median={stats['median_train_pixel_match']:.1f}%, max={stats['max_train_pixel_match']:.1f}%, ratio={stats['median_max_ratio']:.3f}")
        print(f"   Execution:   successful={stats['total_programs'] - stats['no_execution']}/{stats['total_programs']}")

    # Top performing programs
    print(f"\nTOP 10 PROGRAMS BY TASKS SOLVED:")
    sorted_programs = sorted(program_performance.items(), key=lambda x: x[1]['tasks_solved'], reverse=True)
    for i, (prog_id, stats) in enumerate(sorted_programs[:10]):
        print(f"  {i+1}. {prog_id}: {stats['tasks_solved']}/{stats['tasks_attempted']} tasks")

    # Save detailed results
    output_file = Path(__file__).parent / "parallel_10_evaluation_results.json"
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
            'overall_median_max_ratio': float(overall_median_ratio),
            'evaluation_time_minutes': (time.time() - start_time) / 60,
        },
        'per_task_stats': all_task_stats,
        'program_performance': dict(program_performance)
    }

    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")
    print(f"Total evaluation time: {(time.time() - start_time)/60:.1f} minutes")
    print("\nParallel evaluation complete!")


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    mp.set_start_method('spawn', force=True)
    run_programs_on_evaluation_parallel_10()