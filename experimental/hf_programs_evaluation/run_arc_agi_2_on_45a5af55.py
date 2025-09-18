#!/usr/bin/env python3
"""
Run all ARC-AGI-2 dataset programs on task 45a5af55, excluding programs from that same task.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_python.utils.arc_tester import ArcTester
from llm_python.utils.task_loader import get_task_loader
from datasets import load_dataset
import numpy as np
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import NamedTuple, List, Dict, Any, Optional, Tuple


class ProgramResult(NamedTuple):
    """Result for a single program on the target task."""
    program_id: str
    task_id_origin: str
    program_code: str
    success: bool
    train_correct: List[bool]
    test_correct: List[bool]
    train_pixel_matches: List[float]
    test_pixel_matches: List[float]
    avg_train_pixel_match: float
    avg_test_pixel_match: float
    error: Optional[str] = None


def calculate_pixel_match_percentage(predicted, expected):
    """Calculate pixel match percentage between predicted and expected grids."""
    if predicted is None:
        return 0.0

    pred_height = len(predicted)
    pred_width = len(predicted[0]) if pred_height > 0 else 0
    exp_height = len(expected)
    exp_width = len(expected[0]) if exp_height > 0 else 0

    if pred_height == 0 or pred_width == 0 or exp_height == 0 or exp_width == 0:
        return 0.0

    min_height = min(pred_height, exp_height)
    min_width = min(pred_width, exp_width)
    total_pixels = exp_height * exp_width
    matching_pixels = 0

    for i in range(min_height):
        for j in range(min_width):
            if predicted[i][j] == expected[i][j]:
                matching_pixels += 1

    return (matching_pixels / total_pixels) * 100.0


def evaluate_single_program(args: Tuple[str, str, str, Dict[str, Any]]) -> ProgramResult:
    """Evaluate a single program on the target task."""
    program_code, program_id, task_id_origin, task_data = args

    # Initialize tester for this process
    tester = ArcTester(timeout=5, executor_type="unrestricted")

    try:
        # Test the program
        result = tester.test_program(program_code, task_data)

        # Calculate pixel matches for train examples
        train_pixel_matches = []
        for predicted, expected in zip(result.train_outputs, [ex["output"] for ex in task_data["train"]]):
            pixel_match = calculate_pixel_match_percentage(predicted, expected)
            train_pixel_matches.append(pixel_match)

        # Calculate pixel matches for test examples
        test_pixel_matches = []
        for predicted, expected in zip(result.test_outputs, [ex["output"] for ex in task_data["test"]]):
            pixel_match = calculate_pixel_match_percentage(predicted, expected)
            test_pixel_matches.append(pixel_match)

        return ProgramResult(
            program_id=program_id,
            task_id_origin=task_id_origin,
            program_code=program_code,
            success=result.success,
            train_correct=result.correct_train_input,
            test_correct=result.correct_test_input,
            train_pixel_matches=train_pixel_matches,
            test_pixel_matches=test_pixel_matches,
            avg_train_pixel_match=np.mean(train_pixel_matches) if train_pixel_matches else 0.0,
            avg_test_pixel_match=np.mean(test_pixel_matches) if test_pixel_matches else 0.0
        )

    except Exception as e:
        return ProgramResult(
            program_id=program_id,
            task_id_origin=task_id_origin,
            program_code=program_code,
            success=False,
            train_correct=[],
            test_correct=[],
            train_pixel_matches=[],
            test_pixel_matches=[],
            avg_train_pixel_match=0.0,
            avg_test_pixel_match=0.0,
            error=str(e)
        )
    finally:
        ArcTester.cleanup_executor()


def load_task_45a5af55():
    """Load task 45a5af55 from arc-prize-2025 evaluation."""
    task_loader = get_task_loader()
    eval_tasks = task_loader.get_subset_tasks("arc-prize-2025/evaluation")

    for tid, task_data in eval_tasks:
        if tid == "45a5af55":
            return task_data

    raise ValueError("Task 45a5af55 not found in evaluation set")


def load_arc_agi_2_programs():
    """Load programs from the Trelis ARC-AGI dataset."""
    print("Loading Trelis ARC-AGI dataset...")
    ds = load_dataset('Trelis/arc-agi-1-perfect-2')
    return [ds["train"][i] for i in range(len(ds["train"]))]


def run_arc_agi_2_on_45a5af55():
    """Main function to run ARC-AGI-2 programs on task 45a5af55."""
    target_task_id = "45a5af55"
    print(f"Starting evaluation of ARC-AGI-2 programs on task {target_task_id}...")
    start_time = time.time()

    # Load data
    print("Loading data...")
    task_data = load_task_45a5af55()
    arc_programs = load_arc_agi_2_programs()

    print(f"Loaded task: {target_task_id}")
    print(f"Task has {len(task_data['train'])} training examples and {len(task_data['test'])} test examples")
    print(f"Loaded {len(arc_programs)} programs from ARC-AGI-2 dataset")

    # Filter out programs from the same task
    filtered_programs = []
    excluded_count = 0
    for program in arc_programs:
        if program.get('task_id', '') == target_task_id:
            excluded_count += 1
        else:
            filtered_programs.append(program)

    print(f"Excluded {excluded_count} programs that originated from task {target_task_id}")
    print(f"Evaluating {len(filtered_programs)} programs")

    # Prepare work items
    work_items = []
    for program_data in filtered_programs:
        work_items.append((
            program_data['code'],
            program_data.get('id', program_data.get('row_id', 'unknown')),
            program_data.get('task_id', 'unknown'),
            task_data
        ))

    print(f"\nUsing 4 cores with 5-second timeout...")

    # Process with parallel workers
    all_results = []
    completed_count = 0

    with ProcessPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        future_to_work = {executor.submit(evaluate_single_program, item): item for item in work_items}

        # Process completed tasks
        for future in as_completed(future_to_work):
            completed_count += 1
            if completed_count % 100 == 0:
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

    print(f"\nCompleted all evaluations in {(time.time() - start_time)/60:.1f} minutes")

    # Analyze results
    print("\nAnalyzing results...")

    # Categorize results
    all_correct = []       # All train + test correct
    all_train_correct = [] # All train correct
    partial_correct = []   # Some train correct
    all_wrong = []         # No train correct
    execution_failed = []  # Failed to execute

    for result in all_results:
        if result.error:
            execution_failed.append(result)
        elif not result.success:
            execution_failed.append(result)
        else:
            train_correct_count = sum(result.train_correct)
            test_correct_count = sum(result.test_correct)
            total_train = len(result.train_correct)
            total_test = len(result.test_correct)

            if train_correct_count == total_train and test_correct_count == total_test and total_train > 0:
                all_correct.append(result)
            elif train_correct_count == total_train and total_train > 0:
                all_train_correct.append(result)
            elif train_correct_count > 0:
                partial_correct.append(result)
            else:
                all_wrong.append(result)

    # Generate report
    print(f"\n" + "="*80)
    print(f"ARC-AGI-2 PROGRAMS ON TASK {target_task_id}")
    print("="*80)

    print(f"\nEVALUATION SUMMARY:")
    print(f"Total programs evaluated: {len(all_results)}")
    print(f"Programs excluded (same task): {excluded_count}")
    print(f"All correct (train + test): {len(all_correct)}")
    print(f"All training correct: {len(all_train_correct)}")
    print(f"Partial training correct: {len(partial_correct)}")
    print(f"All training wrong: {len(all_wrong)}")
    print(f"Execution failed: {len(execution_failed)}")

    # Show all correct programs
    if all_correct:
        print(f"\nALL CORRECT PROGRAMS ({len(all_correct)}):")
        for i, result in enumerate(all_correct):
            print(f"  {i+1}. {result.program_id} (from {result.task_id_origin})")
            if i < 3:  # Show code for first 3
                print(f"     Code preview: {result.program_code[:100].replace(chr(10), ' ')}...")

    # Show all train correct programs
    if all_train_correct:
        print(f"\nALL TRAIN CORRECT PROGRAMS ({len(all_train_correct)}):")
        for i, result in enumerate(all_train_correct):
            print(f"  {i+1}. {result.program_id} (from {result.task_id_origin})")

    # Top performers by pixel match (from all_wrong programs)
    if all_wrong:
        print(f"\nTOP 10 PROGRAMS BY AVERAGE TRAIN PIXEL MATCH (ALL-WRONG ONLY, N={len(all_wrong)}):")
        all_wrong_sorted = sorted(all_wrong, key=lambda x: x.avg_train_pixel_match, reverse=True)

        for i, result in enumerate(all_wrong_sorted[:10]):
            print(f"  {i+1}. {result.program_id} (from {result.task_id_origin}): {result.avg_train_pixel_match:.1f}% train, {result.avg_test_pixel_match:.1f}% test")

    # Show top 3 programs with code
    if all_wrong:
        print(f"\nTOP 3 PROGRAMS BY PIXEL MATCH WITH CODE:")
        for i, result in enumerate(all_wrong_sorted[:3]):
            print(f"\n{i+1}. Program {result.program_id} (from {result.task_id_origin})")
            print(f"   Train pixel match: {result.avg_train_pixel_match:.1f}%")
            print(f"   Test pixel match: {result.avg_test_pixel_match:.1f}%")
            print(f"   Code:")
            print("   " + "\n   ".join(result.program_code.split("\n")[:10]))  # First 10 lines
            if len(result.program_code.split("\n")) > 10:
                print("   ...")

    # Performance metrics
    total_time_minutes = (time.time() - start_time) / 60
    rate_per_second = len(work_items) / (time.time() - start_time)
    print(f"\nPERFORMANCE METRICS:")
    print(f"Total evaluation time: {total_time_minutes:.2f} minutes")
    print(f"Evaluation rate: {rate_per_second:.1f} programs/second")

    print(f"\nEvaluation complete!")

    return all_results


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    results = run_arc_agi_2_on_45a5af55()