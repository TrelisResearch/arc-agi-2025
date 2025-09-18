#!/usr/bin/env python3
"""
Small test for the parallelized version with pixel metrics.
Tests 20 programs on 3 tasks.
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
    tester = ArcTester(timeout=5, executor_type="unrestricted")

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


def test_parallel_small():
    """Test the parallelized version with a small subset."""
    print("Testing PARALLEL evaluation with pixel metrics...")
    start_time = time.time()

    # Load small subset of data
    print("Loading data...")
    task_loader = get_task_loader()
    eval_tasks = task_loader.get_subset_tasks("arc-prize-2024/evaluation")
    evaluation_tasks = dict(eval_tasks[:3])  # Just 3 tasks

    ds = load_dataset('Trelis/arc-agi-1-perfect-2')
    hf_programs = [ds["train"][i] for i in range(20)]  # Just 20 programs

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
            if completed_count % 10 == 0:
                elapsed = time.time() - start_time
                rate = completed_count / elapsed
                remaining = len(work_items) - completed_count
                eta = remaining / rate if rate > 0 else 0
                print(f"Completed {completed_count}/{len(work_items)} evaluations ({completed_count/len(work_items)*100:.1f}%). "
                      f"Rate: {rate:.1f}/sec. ETA: {eta:.1f} seconds")

            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                print(f"Error in evaluation: {e}")

    print(f"Completed all evaluations in {(time.time() - start_time):.1f} seconds")

    # Group results by task and analyze
    task_results = defaultdict(list)
    for pixel_result in all_results:
        task_results[pixel_result.task_id].append(pixel_result)

    # Analyze each task
    print("\nTask Analysis:")
    for task_id, results in task_results.items():
        print(f"\nTask {task_id}:")

        # Collect all pixel matches
        all_train_pixels = []
        all_test_pixels = []
        successful_programs = 0

        for pixel_result in results:
            all_train_pixels.extend(pixel_result.train_pixel_matches)
            all_test_pixels.extend(pixel_result.test_pixel_matches)
            if pixel_result.program_result.success:
                successful_programs += 1

        print(f"  Programs that executed successfully: {successful_programs}/{len(results)}")

        if all_train_pixels:
            median_train = np.median(all_train_pixels)
            max_train = np.max(all_train_pixels)
            ratio = median_train / max_train if max_train > 0 else 0
            print(f"  Train pixel matches: median={median_train:.1f}%, max={max_train:.1f}%, ratio={ratio:.3f}")

        if all_test_pixels:
            median_test = np.median(all_test_pixels)
            max_test = np.max(all_test_pixels)
            print(f"  Test pixel matches:  median={median_test:.1f}%, max={max_test:.1f}%")

        # Show a few sample pixel match results
        print(f"  Sample train pixel matches: {all_train_pixels[:10]}")

    print(f"\nTest completed in {(time.time() - start_time):.1f} seconds")


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    mp.set_start_method('spawn', force=True)
    test_parallel_small()