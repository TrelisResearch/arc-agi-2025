#!/usr/bin/env python3
"""
Load programs from a parquet file and run them on task 1818057f to calculate pixel match scores.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_python.datasets.io import read_soar_parquet
from llm_python.utils.arc_tester import ArcTester
from llm_python.utils.task_loader import get_task_loader
import numpy as np


def calculate_pixel_match_percentage(predicted, expected):
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


def load_task_1818057f():
    """Load task 1818057f from arc-prize-2025 evaluation."""
    task_loader = get_task_loader()
    eval_tasks = task_loader.get_subset_tasks("arc-prize-2025/evaluation")

    for tid, task_data in eval_tasks:
        if tid == "1818057f":
            return task_data

    raise ValueError("Task 1818057f not found in evaluation set")


def run_parquet_programs_on_task(parquet_path: str):
    """Main function to run parquet programs on task 1818057f."""
    print(f"Loading programs from parquet file: {parquet_path}")

    # Load programs from parquet
    df = read_soar_parquet(parquet_path)
    print(f"Loaded {len(df)} programs from parquet file")

    # Show program info
    print("\nProgram IDs and correctness:")
    for idx, row in df.iterrows():
        train_correct = all(row['correct_train_input']) if row['correct_train_input'] else False
        test_correct = all(row['correct_test_input']) if row['correct_test_input'] else False
        print(f"  {row['row_id']}: Train={train_correct}, Test={test_correct}")

    # Load task 1818057f
    print(f"\nLoading task 1818057f...")
    task_data = load_task_1818057f()
    print(f"Task has {len(task_data['train'])} training examples and {len(task_data['test'])} test examples")

    # Initialize tester
    tester = ArcTester(timeout=5, executor_type="unrestricted")

    results = []

    print(f"\nRunning {len(df)} programs on task 1818057f...")

    for idx, row in df.iterrows():
        program_id = row['row_id']
        program_code = row['code']

        print(f"\nProgram {idx+1}/{len(df)}: {program_id}")

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

            # Store result
            program_result = {
                'program_id': program_id,
                'program_code': program_code,
                'success': result.success,
                'train_correct': result.correct_train_input,
                'test_correct': result.correct_test_input,
                'train_pixel_matches': train_pixel_matches,
                'test_pixel_matches': test_pixel_matches,
                'avg_train_pixel_match': np.mean(train_pixel_matches) if train_pixel_matches else 0.0,
                'avg_test_pixel_match': np.mean(test_pixel_matches) if test_pixel_matches else 0.0
            }

            results.append(program_result)

            print(f"  Success: {result.success}")
            print(f"  Train correct: {sum(result.correct_train_input)}/{len(result.correct_train_input)}")
            print(f"  Test correct: {sum(result.correct_test_input)}/{len(result.correct_test_input)}")
            print(f"  Avg train pixel match: {program_result['avg_train_pixel_match']:.1f}%")
            print(f"  Avg test pixel match: {program_result['avg_test_pixel_match']:.1f}%")

        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'program_id': program_id,
                'program_code': program_code,
                'success': False,
                'error': str(e),
                'avg_train_pixel_match': 0.0,
                'avg_test_pixel_match': 0.0
            })

    # Summary
    print(f"\n" + "="*60)
    print("SUMMARY RESULTS")
    print("="*60)

    successful_programs = [r for r in results if r.get('success', False)]
    print(f"Successfully executed: {len(successful_programs)}/{len(results)} programs")

    # Sort by average train pixel match
    results_sorted = sorted(results, key=lambda x: x.get('avg_train_pixel_match', 0), reverse=True)

    print(f"\nTop programs by average train pixel match:")
    for i, result in enumerate(results_sorted[:len(results)]):
        if 'error' not in result:
            train_correct_count = sum(result['train_correct']) if result.get('train_correct') else 0
            test_correct_count = sum(result['test_correct']) if result.get('test_correct') else 0
            print(f"  {i+1}. {result['program_id']}: {result['avg_train_pixel_match']:.1f}% train, {result['avg_test_pixel_match']:.1f}% test")
            print(f"      Train: {train_correct_count}/{len(result.get('train_correct', []))} correct, Test: {test_correct_count}/{len(result.get('test_correct', []))} correct")
        else:
            print(f"  {i+1}. {result['program_id']}: ERROR - {result.get('error', 'Unknown error')}")

    # Cleanup
    tester.cleanup_executor()

    return results


if __name__ == "__main__":
    parquet_file = "/Users/ronanmcgovern/TR/arc-agi-2025/llm_python/datasets/inference/20250918_151237_julien31_Soar-qwen-14b_arc-prize-2025_evaluation.parquet"
    results = run_parquet_programs_on_task(parquet_file)