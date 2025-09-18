#!/usr/bin/env python3
"""
Calculate pixel match scores for programs already executed and stored in parquet file.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_python.datasets.io import read_soar_parquet
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


def calculate_parquet_pixel_matches(parquet_path: str):
    """Calculate pixel matches for programs in parquet file."""
    print(f"Loading programs from parquet file: {parquet_path}")

    # Load programs from parquet
    df = read_soar_parquet(parquet_path)
    print(f"Loaded {len(df)} programs from parquet file")

    # Show program info
    print("\nProgram IDs and correctness:")
    for idx, row in df.iterrows():
        train_array = row['correct_train_input']
        test_array = row['correct_test_input']
        train_correct = train_array.all() if hasattr(train_array, 'all') else all(train_array) if train_array else False
        test_correct = test_array.all() if hasattr(test_array, 'all') else all(test_array) if test_array else False
        print(f"  {row['row_id']}: Train={train_correct}, Test={test_correct}")

    # Load task 1818057f to get expected outputs
    print(f"\nLoading task 1818057f for expected outputs...")
    task_data = load_task_1818057f()
    print(f"Task has {len(task_data['train'])} training examples and {len(task_data['test'])} test examples")

    results = []

    print(f"\nCalculating pixel matches for {len(df)} programs...")

    for idx, row in df.iterrows():
        program_id = row['row_id']

        print(f"\nProgram {idx+1}/{len(df)}: {program_id}")

        try:
            # Get predicted outputs from parquet (already computed)
            predicted_train_outputs = row['predicted_train_output']
            predicted_test_outputs = row['predicted_test_output']

            # Convert numpy arrays to lists for compatibility
            predicted_train_outputs = [arr.tolist() if hasattr(arr, 'tolist') else arr for arr in predicted_train_outputs]
            predicted_test_outputs = [arr.tolist() if hasattr(arr, 'tolist') else arr for arr in predicted_test_outputs]

            # Calculate pixel matches for train examples
            train_pixel_matches = []
            for i, (predicted, expected) in enumerate(zip(predicted_train_outputs, [ex["output"] for ex in task_data["train"]])):
                pixel_match = calculate_pixel_match_percentage(predicted, expected)
                train_pixel_matches.append(pixel_match)
                print(f"    Train {i+1}: {pixel_match:.1f}% pixel match")

            # Calculate pixel matches for test examples
            test_pixel_matches = []
            for i, (predicted, expected) in enumerate(zip(predicted_test_outputs, [ex["output"] for ex in task_data["test"]])):
                pixel_match = calculate_pixel_match_percentage(predicted, expected)
                test_pixel_matches.append(pixel_match)
                print(f"    Test {i+1}: {pixel_match:.1f}% pixel match")

            # Store result
            program_result = {
                'program_id': program_id,
                'program_code': row['code'],
                'train_correct': row['correct_train_input'],
                'test_correct': row['correct_test_input'],
                'train_pixel_matches': train_pixel_matches,
                'test_pixel_matches': test_pixel_matches,
                'avg_train_pixel_match': np.mean(train_pixel_matches) if train_pixel_matches else 0.0,
                'avg_test_pixel_match': np.mean(test_pixel_matches) if test_pixel_matches else 0.0
            }

            results.append(program_result)

            print(f"  Train correct: {sum(row['correct_train_input'])}/{len(row['correct_train_input'])}")
            print(f"  Test correct: {sum(row['correct_test_input'])}/{len(row['correct_test_input'])}")
            print(f"  Avg train pixel match: {program_result['avg_train_pixel_match']:.1f}%")
            print(f"  Avg test pixel match: {program_result['avg_test_pixel_match']:.1f}%")

        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'program_id': program_id,
                'program_code': row['code'],
                'error': str(e),
                'avg_train_pixel_match': 0.0,
                'avg_test_pixel_match': 0.0
            })

    # Summary
    print(f"\n" + "="*60)
    print("SUMMARY RESULTS")
    print("="*60)

    successful_programs = [r for r in results if 'error' not in r]
    print(f"Successfully processed: {len(successful_programs)}/{len(results)} programs")

    # Sort by average train pixel match
    results_sorted = sorted(results, key=lambda x: x.get('avg_train_pixel_match', 0), reverse=True)

    print(f"\nPrograms ranked by average train pixel match:")
    for i, result in enumerate(results_sorted):
        if 'error' not in result:
            train_array = result.get('train_correct', [])
            test_array = result.get('test_correct', [])
            train_correct_count = train_array.sum() if hasattr(train_array, 'sum') else sum(train_array) if train_array else 0
            test_correct_count = test_array.sum() if hasattr(test_array, 'sum') else sum(test_array) if test_array else 0
            print(f"  {i+1}. {result['program_id']}: {result['avg_train_pixel_match']:.1f}% train, {result['avg_test_pixel_match']:.1f}% test")
            print(f"      Train: {train_correct_count}/{len(result.get('train_correct', []))} correct, Test: {test_correct_count}/{len(result.get('test_correct', []))} correct")
        else:
            print(f"  {i+1}. {result['program_id']}: ERROR - {result.get('error', 'Unknown error')}")

    # Show best program code
    if successful_programs:
        best_program = results_sorted[0]
        print(f"\nBest program code ({best_program['program_id']}):")
        print("="*60)
        print(best_program['program_code'])
        print("="*60)

    return results


if __name__ == "__main__":
    parquet_file = "/Users/ronanmcgovern/TR/arc-agi-2025/llm_python/datasets/inference/20250918_151237_julien31_Soar-qwen-14b_arc-prize-2025_evaluation.parquet"
    results = calculate_parquet_pixel_matches(parquet_file)