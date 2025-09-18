#!/usr/bin/env python3
"""
Run all HF programs on a single specific task with detailed analysis.
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


class SingleTaskResult(NamedTuple):
    """Result for a single program on the target task."""
    program_result: ProgramTestResult
    train_pixel_matches: List[float]  # Pixel match % for each train example
    test_pixel_matches: List[float]   # Pixel match % for each test example
    program_id: str
    program_code: str
    early_terminated: bool


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


def evaluate_single_program_on_task(args: Tuple[str, Dict[str, Any], str, str]) -> SingleTaskResult:
    """
    Evaluate a single program on the target task with early termination.
    """
    program_code, task_data, program_id, full_program_code = args

    # Initialize tester for this process
    tester = ArcTester(timeout=5, executor_type="unrestricted")

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

                return SingleTaskResult(
                    program_result=dummy_result,
                    train_pixel_matches=train_pixel_matches,
                    test_pixel_matches=test_pixel_matches,
                    program_id=program_id,
                    program_code=full_program_code,
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

        return SingleTaskResult(
            program_result=result,
            train_pixel_matches=train_pixel_matches,
            test_pixel_matches=test_pixel_matches,
            program_id=program_id,
            program_code=full_program_code,
            early_terminated=early_terminated
        )

    except Exception as e:
        # Return dummy result for failed execution
        dummy_result = ProgramTestResult(
            train_outputs=[], test_outputs=[], train_inputs=[], test_inputs=[],
            correct_train_input=[], correct_test_input=[], success=False
        )
        return SingleTaskResult(
            program_result=dummy_result,
            train_pixel_matches=[],
            test_pixel_matches=[],
            program_id=program_id,
            program_code=full_program_code,
            early_terminated=True  # Exception counts as early termination
        )
    finally:
        ArcTester.cleanup_executor()


def load_single_task(task_id: str) -> Dict[str, Any]:
    """Load a specific ARC-AGI 2025 evaluation task."""
    task_loader = get_task_loader()
    eval_tasks = task_loader.get_subset_tasks("arc-prize-2025/evaluation")

    for tid, task_data in eval_tasks:
        if tid == task_id:
            return {task_id: task_data}

    raise ValueError(f"Task {task_id} not found in evaluation set")


def load_hf_programs() -> List[Dict[str, Any]]:
    """Load programs from the HuggingFace dataset."""
    print("Loading HuggingFace dataset...")
    ds = load_dataset('Trelis/arc-agi-1-perfect-2')
    return [ds["train"][i] for i in range(len(ds["train"]))]


def run_programs_on_single_task(target_task_id: str = "1818057f"):
    """Main function to run all programs on a single task."""
    print(f"Starting evaluation of all HF programs on task {target_task_id}...")
    start_time = time.time()

    # Load data
    print("Loading data...")
    try:
        evaluation_tasks = load_single_task(target_task_id)
    except ValueError as e:
        print(f"Error: {e}")
        return

    hf_programs = load_hf_programs()

    print(f"Loaded task: {target_task_id}")
    print(f"Loaded {len(hf_programs)} programs from HF dataset")
    print(f"Total evaluations to run: {len(hf_programs)}")

    # Show task details
    task_data = evaluation_tasks[target_task_id]
    print(f"\\nTask Details:")
    print(f"- Training examples: {len(task_data['train'])}")
    print(f"- Test examples: {len(task_data['test'])}")

    # Show first training example dimensions
    if task_data["train"]:
        first_train = task_data["train"][0]
        input_dims = f"{len(first_train['input'])}x{len(first_train['input'][0])}"
        output_dims = f"{len(first_train['output'])}x{len(first_train['output'][0])}"
        print(f"- First training example: {input_dims} -> {output_dims}")

    # Prepare work items
    work_items = []
    for program_data in hf_programs:
        work_items.append((
            program_data['code'],       # program_code
            task_data,                  # task_data
            program_data['row_id'],     # program_id
            program_data['code']        # full_program_code for analysis
        ))

    print(f"\\nUsing 4 cores with 5-second timeout and early termination...")

    # Process with parallel workers
    all_results = []
    completed_count = 0

    with ProcessPoolExecutor(max_workers=4) as executor:
        # Submit all tasks
        future_to_work = {executor.submit(evaluate_single_program_on_task, item): item for item in work_items}

        # Process completed tasks
        for future in as_completed(future_to_work):
            completed_count += 1
            if completed_count % 50 == 0:  # More frequent updates for single task
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

    print(f"\\nCompleted all evaluations in {(time.time() - start_time)/60:.1f} minutes")

    # Analyze results
    print("\\nAnalyzing results...")

    # Categorize results
    perfect_solutions = []  # All correct (train + test)
    train_correct = []      # All train correct
    partial_correct = []    # Some train correct
    all_wrong = []          # No train correct
    execution_failed = []   # Failed to execute
    early_terminated = []   # Early termination

    for result in all_results:
        if result.early_terminated:
            early_terminated.append(result)
        elif not result.program_result.success:
            execution_failed.append(result)
        else:
            train_correct_count = sum(result.program_result.correct_train_input)
            test_correct_count = sum(result.program_result.correct_test_input)
            total_train = len(result.program_result.correct_train_input)
            total_test = len(result.program_result.correct_test_input)

            if train_correct_count == total_train and test_correct_count == total_test and total_train > 0:
                perfect_solutions.append(result)
            elif train_correct_count == total_train and total_train > 0:
                train_correct.append(result)
            elif train_correct_count > 0:
                partial_correct.append(result)
            else:
                all_wrong.append(result)

    # Generate detailed report
    print("\\n" + "="*80)
    print(f"SINGLE TASK EVALUATION REPORT: {target_task_id}")
    print("="*80)

    print(f"\\nEVALUATION SUMMARY:")
    print(f"Total programs evaluated: {len(all_results)}")
    print(f"Perfect solutions (train + test): {len(perfect_solutions)}")
    print(f"All training correct: {len(train_correct)}")
    print(f"Partial training correct: {len(partial_correct)}")
    print(f"All training wrong: {len(all_wrong)}")
    print(f"Execution failed: {len(execution_failed)}")
    print(f"Early terminated: {len(early_terminated)}")

    # Perfect solutions analysis
    if perfect_solutions:
        print(f"\\nPERFECT SOLUTIONS ({len(perfect_solutions)}):")
        for i, result in enumerate(perfect_solutions[:5]):  # Show first 5
            print(f"  {i+1}. Program {result.program_id}")
            if i == 0:  # Show code for first perfect solution
                print(f"     Code preview: {result.program_code[:100]}...")

    # Pixel match analysis for all-wrong programs
    if all_wrong:
        print(f"\\nPIXEL MATCH ANALYSIS (ALL-WRONG PROGRAMS, N={len(all_wrong)}):")

        train_pixel_matches = []
        test_pixel_matches = []

        for result in all_wrong:
            train_pixel_matches.extend(result.train_pixel_matches)
            test_pixel_matches.extend(result.test_pixel_matches)

        if train_pixel_matches:
            median_train = np.median(train_pixel_matches)
            max_train = np.max(train_pixel_matches)
            mean_train = np.mean(train_pixel_matches)

            print(f"Training pixel matches:")
            print(f"  Median: {median_train:.1f}%")
            print(f"  Max: {max_train:.1f}%")
            print(f"  Mean: {mean_train:.1f}%")
            print(f"  Median/Max ratio: {median_train/max_train:.3f}")

        if test_pixel_matches:
            median_test = np.median(test_pixel_matches)
            max_test = np.max(test_pixel_matches)
            mean_test = np.mean(test_pixel_matches)

            print(f"Test pixel matches:")
            print(f"  Median: {median_test:.1f}%")
            print(f"  Max: {max_test:.1f}%")
            print(f"  Mean: {mean_test:.1f}%")

        # Top performing all-wrong programs
        print(f"\\nTOP 10 ALL-WRONG PROGRAMS BY AVERAGE PIXEL MATCH:")
        scored_programs = []
        for result in all_wrong:
            if result.train_pixel_matches:
                avg_train_pixel = np.mean(result.train_pixel_matches)
                scored_programs.append((result, avg_train_pixel))

        scored_programs.sort(key=lambda x: x[1], reverse=True)
        for i, (result, score) in enumerate(scored_programs[:10]):
            print(f"  {i+1}. Program {result.program_id}: {score:.1f}% avg pixel match")
            if i < 3:  # Show code for top 3
                print(f"     Code preview: {result.program_code[:100].replace(chr(10), ' ')}...")

    # Performance metrics
    total_time_minutes = (time.time() - start_time) / 60
    rate_per_second = len(work_items) / (time.time() - start_time)
    print(f"\\nPERFORMANCE METRICS:")
    print(f"Total evaluation time: {total_time_minutes:.2f} minutes")
    print(f"Evaluation rate: {rate_per_second:.1f} programs/second")
    print(f"Early termination rate: {len(early_terminated)/len(all_results)*100:.1f}%")

    # Save detailed results
    output_file = Path(__file__).parent / f"single_task_{target_task_id}_results.json"
    results_data = {
        'task_id': target_task_id,
        'task_data': task_data,
        'summary': {
            'total_programs': len(all_results),
            'perfect_solutions': len(perfect_solutions),
            'train_correct': len(train_correct),
            'partial_correct': len(partial_correct),
            'all_wrong': len(all_wrong),
            'execution_failed': len(execution_failed),
            'early_terminated': len(early_terminated),
            'evaluation_time_minutes': total_time_minutes,
            'evaluation_rate_per_second': rate_per_second,
        },
        'perfect_solution_ids': [r.program_id for r in perfect_solutions],
        'top_partial_programs': [(r.program_id, np.mean(r.train_pixel_matches) if r.train_pixel_matches else 0)
                                for r in scored_programs[:20]] if all_wrong else [],
    }

    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\\nDetailed results saved to: {output_file}")
    print(f"\\nSingle task evaluation complete!")


if __name__ == "__main__":
    # Set multiprocessing start method for compatibility
    mp.set_start_method('spawn', force=True)

    # Run on the specified task
    run_programs_on_single_task("1818057f")