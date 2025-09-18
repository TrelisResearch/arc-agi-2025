#!/usr/bin/env python3
"""
Optimized evaluation on 25 tasks with FILTERED pixel metrics and EARLY TERMINATION.
Uses 2 cores with early termination: if program fails on first training example, skip the rest.

This script provides:
1. Standard correctness metrics (all-correct, all-train-correct, partial-correct)
2. Pixel match percentages for partial solutions
3. Median and max pixel match ratios per task
4. EARLY TERMINATION: Skip remaining examples if first training example fails
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
    early_terminated: bool  # True if evaluation was terminated early


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


def evaluate_single_program_task_fast(args: Tuple[str, Dict[str, Any], str, Any]) -> PixelMatchResult:
    """
    Evaluate a single program on a single task with EARLY TERMINATION.
    If the program fails on the first training example, skip the rest.
    """
    program_code, task_data, program_id, task_id = args

    # Initialize tester for this process - 3 second timeout
    tester = ArcTester(timeout=3, executor_type="unrestricted")

    try:
        # Check if we have training examples
        if not task_data["train"]:
            # No training examples, run full evaluation
            result = tester.test_program(program_code, task_data)
            early_terminated = False
        else:
            # EARLY TERMINATION: Test first training example
            first_train_input = task_data["train"][0]["input"]
            first_output, error_message, timed_out = tester.execute_program_with_timeout(
                program_code, first_train_input
            )

            # Validate the first output
            first_validated = tester._validate_grid(first_output)

            # If first example failed (execution error or invalid output), terminate early
            if first_validated is None or error_message or timed_out:
                # Create dummy result indicating early termination
                dummy_result = ProgramTestResult(
                    train_outputs=[None] * len(task_data["train"]),
                    test_outputs=[None] * len(task_data["test"]),
                    train_inputs=[ex["input"] for ex in task_data["train"]],
                    test_inputs=[ex["input"] for ex in task_data["test"]],
                    correct_train_input=[False] * len(task_data["train"]),
                    correct_test_input=[False] * len(task_data["test"]),
                    success=False
                )

                train_pixel_matches = [0.0] * len(task_data["train"])
                test_pixel_matches = [0.0] * len(task_data["test"])
                early_terminated = True

                return PixelMatchResult(
                    program_result=dummy_result,
                    train_pixel_matches=train_pixel_matches,
                    test_pixel_matches=test_pixel_matches,
                    program_id=program_id,
                    task_id=task_id,
                    early_terminated=early_terminated
                )

            # First example succeeded, run full evaluation
            result = tester.test_program(program_code, task_data)
            early_terminated = False

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
            task_id=task_id,
            early_terminated=early_terminated
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
            task_id=task_id,
            early_terminated=True  # Exception counts as early termination
        )
    finally:
        ArcTester.cleanup_executor()


def load_evaluation_tasks(max_tasks: int = 25) -> Dict[str, Any]:
    """Load ARC-AGI 2024 evaluation tasks."""
    task_loader = get_task_loader()
    eval_tasks = task_loader.get_subset_tasks("arc-prize-2024/evaluation")
    return dict(eval_tasks[:max_tasks])


def load_hf_programs() -> List[Dict[str, Any]]:
    """Load programs from the HuggingFace dataset."""
    print("Loading HuggingFace dataset...")
    ds = load_dataset('Trelis/arc-agi-1-perfect-2')
    return [ds["train"][i] for i in range(len(ds["train"]))]


def analyze_task_performance_with_filtered_pixels(task_results: List[PixelMatchResult]) -> Dict[str, Any]:
    """Analyze performance on a single task across all programs, with FILTERED pixel metrics."""
    stats = {
        'total_programs': len(task_results),

        # Standard correctness metrics
        'all_correct': 0,  # Both train and test all correct
        'all_train_correct': 0,  # All training examples correct
        'at_least_one_train_correct': 0,  # At least one training example correct
        'no_execution': 0,  # Programs that failed to execute
        'no_output': 0,  # Programs that executed but produced no valid output

        # Early termination tracking
        'early_terminated': 0,  # Programs that were terminated early

        # Standard (unfiltered) pixel match metrics
        'median_train_pixel_match': 0.0,
        'max_train_pixel_match': 0.0,
        'median_test_pixel_match': 0.0,
        'max_test_pixel_match': 0.0,
        'median_max_ratio': 0.0,

        # FILTERED pixel match metrics (only programs with ALL train examples wrong)
        'all_wrong_programs': 0,  # Count of programs that got ALL training examples wrong
        'filtered_median_train_pixel_match': 0.0,
        'filtered_max_train_pixel_match': 0.0,
        'filtered_median_test_pixel_match': 0.0,
        'filtered_max_test_pixel_match': 0.0,
        'filtered_median_max_ratio': 0.0,
    }

    # Collect all pixel matches (unfiltered)
    all_train_pixels = []
    all_test_pixels = []

    # Collect filtered pixel matches (only from programs that got ALL train wrong)
    all_wrong_train_pixels = []
    all_wrong_test_pixels = []

    for pixel_result in task_results:
        result = pixel_result.program_result

        # Track early termination
        if pixel_result.early_terminated:
            stats['early_terminated'] += 1

        # Collect pixel matches (unfiltered)
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

        # FILTER: Only programs that got ALL training examples wrong
        if train_correct_count == 0 and total_train > 0:
            stats['all_wrong_programs'] += 1
            all_wrong_train_pixels.extend(pixel_result.train_pixel_matches)
            all_wrong_test_pixels.extend(pixel_result.test_pixel_matches)

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

    # Calculate unfiltered pixel match statistics
    if all_train_pixels:
        stats['median_train_pixel_match'] = float(np.median(all_train_pixels))
        stats['max_train_pixel_match'] = float(np.max(all_train_pixels))

    if all_test_pixels:
        stats['median_test_pixel_match'] = float(np.median(all_test_pixels))
        stats['max_test_pixel_match'] = float(np.max(all_test_pixels))

    # Calculate unfiltered median/max ratio
    if stats['max_train_pixel_match'] > 0:
        stats['median_max_ratio'] = stats['median_train_pixel_match'] / stats['max_train_pixel_match']

    # Calculate FILTERED pixel match statistics (only all-wrong programs)
    if all_wrong_train_pixels:
        stats['filtered_median_train_pixel_match'] = float(np.median(all_wrong_train_pixels))
        stats['filtered_max_train_pixel_match'] = float(np.max(all_wrong_train_pixels))

    if all_wrong_test_pixels:
        stats['filtered_median_test_pixel_match'] = float(np.median(all_wrong_test_pixels))
        stats['filtered_max_test_pixel_match'] = float(np.max(all_wrong_test_pixels))

    # Calculate filtered median/max ratio
    if stats['filtered_max_train_pixel_match'] > 0:
        stats['filtered_median_max_ratio'] = (stats['filtered_median_train_pixel_match'] /
                                            stats['filtered_max_train_pixel_match'])

    return stats


def run_programs_on_evaluation_25_filtered_fast():
    """Main function to run 25-task evaluation with filtered pixel metrics and early termination."""
    print("Starting FAST evaluation on 25 tasks with FILTERED pixel metrics and EARLY TERMINATION...")
    start_time = time.time()

    # Load data
    print("Loading data...")
    evaluation_tasks = load_evaluation_tasks(max_tasks=25)  # 25 tasks
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

    print(f"Using 2 cores with 3-second timeout and EARLY TERMINATION to process {len(work_items)} evaluations...")

    # Process with parallel workers (2 cores)
    all_results = []
    completed_count = 0

    with ProcessPoolExecutor(max_workers=2) as executor:
        # Submit all tasks
        future_to_work = {executor.submit(evaluate_single_program_task_fast, item): item for item in work_items}

        # Process completed tasks
        for future in as_completed(future_to_work):
            completed_count += 1
            if completed_count % 500 == 0:  # Updates every 500
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
    early_termination_stats = {'total_early_terminated': 0, 'total_evaluations': len(all_results)}

    for pixel_result in all_results:
        task_results[pixel_result.task_id].append(pixel_result)

        # Track early termination stats
        if pixel_result.early_terminated:
            early_termination_stats['total_early_terminated'] += 1

        # Track program performance
        program_performance[pixel_result.program_id]['tasks_attempted'] += 1

        result = pixel_result.program_result
        if result.success:
            train_correct = sum(result.correct_train_input) == len(result.correct_train_input)
            test_correct = sum(result.correct_test_input) == len(result.correct_test_input)
            if train_correct and test_correct and len(result.correct_train_input) > 0:
                program_performance[pixel_result.program_id]['tasks_solved'] += 1

    # Analyze results per task
    print("\\nAnalyzing results...")
    all_task_stats = {}
    for task_id, results in task_results.items():
        task_stats = analyze_task_performance_with_filtered_pixels(results)
        all_task_stats[task_id] = task_stats

    # Generate summary report
    print("\\n" + "="*80)
    print("FAST EVALUATION SUMMARY REPORT (25 TASKS) WITH FILTERED PIXEL METRICS & EARLY TERMINATION")
    print("="*80)

    # Overall statistics
    total_all_correct = sum(stats['all_correct'] for stats in all_task_stats.values())
    total_all_train_correct = sum(stats['all_train_correct'] for stats in all_task_stats.values())
    total_at_least_one_train = sum(stats['at_least_one_train_correct'] for stats in all_task_stats.values())
    total_all_wrong_programs = sum(stats['all_wrong_programs'] for stats in all_task_stats.values())

    print(f"\\nEARLY TERMINATION STATISTICS:")
    print(f"Total evaluations terminated early: {early_termination_stats['total_early_terminated']}")
    print(f"Early termination rate: {early_termination_stats['total_early_terminated']/early_termination_stats['total_evaluations']*100:.1f}%")

    print(f"\\nSTANDARD CORRECTNESS METRICS:")
    print(f"Total program-task pairs with all correct (train + test): {total_all_correct}")
    print(f"Total program-task pairs with all train correct: {total_all_train_correct}")
    print(f"Total program-task pairs with at least one train correct: {total_at_least_one_train}")
    print(f"Total program-task pairs tested: {len(evaluation_tasks) * len(hf_programs)}")

    # Tasks that had at least one solution
    tasks_with_all_correct = len([task for task, stats in all_task_stats.items() if stats['all_correct'] > 0])
    tasks_with_train_correct = len([task for task, stats in all_task_stats.items() if stats['all_train_correct'] > 0])
    tasks_with_some_train = len([task for task, stats in all_task_stats.items() if stats['at_least_one_train_correct'] > 0])

    print(f"\\nTASK-LEVEL SUCCESS RATES:")
    print(f"Tasks with at least one perfect solution (train + test): {tasks_with_all_correct}/{len(evaluation_tasks)}")
    print(f"Tasks with at least one all-train solution: {tasks_with_train_correct}/{len(evaluation_tasks)}")
    print(f"Tasks with at least one partial train solution: {tasks_with_some_train}/{len(evaluation_tasks)}")

    # FILTERED pixel match statistics
    print(f"\\nFILTERED PIXEL MATCH METRICS (ALL-WRONG PROGRAMS ONLY):")
    print(f"Total programs that got ALL training examples wrong: {total_all_wrong_programs}")

    # Calculate overall filtered ratios
    all_filtered_ratios = [stats['filtered_median_max_ratio'] for stats in all_task_stats.values()
                          if stats['filtered_median_max_ratio'] > 0]
    overall_filtered_ratio = np.median(all_filtered_ratios) if all_filtered_ratios else 0.0

    print(f"Overall filtered median/max pixel match ratio: {overall_filtered_ratio:.3f}")
    print(f"Tasks with filtered pixel match data: {len(all_filtered_ratios)}/{len(evaluation_tasks)}")

    # Comparison with unfiltered metrics
    all_unfiltered_ratios = [stats['median_max_ratio'] for stats in all_task_stats.values()
                            if stats['median_max_ratio'] > 0]
    overall_unfiltered_ratio = np.median(all_unfiltered_ratios) if all_unfiltered_ratios else 0.0

    print(f"\\nCOMPARISON - UNFILTERED vs FILTERED:")
    print(f"Unfiltered median/max ratio: {overall_unfiltered_ratio:.3f}")
    print(f"Filtered median/max ratio: {overall_filtered_ratio:.3f}")
    if overall_unfiltered_ratio > 0:
        print(f"Filtering signal improvement: {(overall_filtered_ratio/overall_unfiltered_ratio - 1)*100:+.1f}%")

    # Top tasks by filtered pixel match performance
    print(f"\\nTOP 10 TASKS BY FILTERED MEDIAN TRAIN PIXEL MATCH:")
    sorted_tasks_by_filtered = sorted(
        all_task_stats.items(),
        key=lambda x: x[1]['filtered_median_train_pixel_match'],
        reverse=True
    )
    for i, (task_id, stats) in enumerate(sorted_tasks_by_filtered[:10]):
        if stats['filtered_median_train_pixel_match'] > 0:
            print(f"  {i+1}. {task_id}: median={stats['filtered_median_train_pixel_match']:.1f}%, "
                  f"max={stats['filtered_max_train_pixel_match']:.1f}%, "
                  f"ratio={stats['filtered_median_max_ratio']:.3f}, "
                  f"all-wrong-programs={stats['all_wrong_programs']}")

    # Performance metrics
    total_time_minutes = (time.time() - start_time) / 60
    rate_per_second = len(work_items) / (time.time() - start_time)
    print(f"\\nPERFORMANCE METRICS:")
    print(f"Total evaluation time: {total_time_minutes:.1f} minutes")
    print(f"Evaluation rate: {rate_per_second:.1f} evaluations/second")

    # Save detailed results
    output_file = Path(__file__).parent / "fast_25_filtered_evaluation_results.json"
    results_data = {
        'summary': {
            'total_evaluation_tasks': len(evaluation_tasks),
            'total_programs': len(hf_programs),
            'total_all_correct': total_all_correct,
            'total_all_train_correct': total_all_train_correct,
            'total_at_least_one_train': total_at_least_one_train,
            'total_all_wrong_programs': total_all_wrong_programs,
            'tasks_with_all_correct': tasks_with_all_correct,
            'tasks_with_train_correct': tasks_with_train_correct,
            'tasks_with_some_train': tasks_with_some_train,
            'overall_unfiltered_ratio': float(overall_unfiltered_ratio),
            'overall_filtered_ratio': float(overall_filtered_ratio),
            'evaluation_time_minutes': total_time_minutes,
            'evaluation_rate_per_second': rate_per_second,
            'early_termination_stats': early_termination_stats,
        },
        'per_task_stats': all_task_stats,
        'program_performance': dict(program_performance)
    }

    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\\nDetailed results saved to: {output_file}")
    print("\\nFast filtered evaluation with early termination complete!")


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    mp.set_start_method('spawn', force=True)
    run_programs_on_evaluation_25_filtered_fast()