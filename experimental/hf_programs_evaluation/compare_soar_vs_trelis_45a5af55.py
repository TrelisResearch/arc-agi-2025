#!/usr/bin/env python3
"""
Compare SOAR programs from parquet file with Trelis programs on task 45a5af55.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_python.datasets.io import read_soar_parquet
from llm_python.utils.task_loader import get_task_loader
from llm_python.utils.arc_tester import ArcTester
import numpy as np


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


def load_task_45a5af55():
    """Load task 45a5af55 from arc-prize-2025 evaluation."""
    task_loader = get_task_loader()
    eval_tasks = task_loader.get_subset_tasks("arc-prize-2025/evaluation")

    for tid, task_data in eval_tasks:
        if tid == "45a5af55":
            return task_data

    raise ValueError("Task 45a5af55 not found in evaluation set")


def test_soar_programs_on_45a5af55(parquet_path: str):
    """Test SOAR programs from parquet on task 45a5af55."""
    print(f"Loading SOAR programs from: {parquet_path}")

    # Load programs from parquet
    df = read_soar_parquet(parquet_path)
    print(f"Loaded {len(df)} SOAR programs from parquet file")

    # Load task
    task_data = load_task_45a5af55()
    print(f"Task 45a5af55 has {len(task_data['train'])} training examples and {len(task_data['test'])} test examples")

    # Check what task these programs were designed for
    if 'task_id' in df.columns:
        orig_tasks = df['task_id'].unique()
        print(f"These programs were originally designed for task(s): {list(orig_tasks)}")

    # If the parquet already contains results for 45a5af55, use those
    if len(df) > 0 and 'task_id' in df.columns and '45a5af55' in df['task_id'].values:
        print("Found pre-computed results for 45a5af55 in parquet file")

        results = []
        for idx, row in df.iterrows():
            if row['task_id'] == '45a5af55':
                # Use pre-computed outputs
                predicted_train_outputs = row['predicted_train_output']
                predicted_test_outputs = row['predicted_test_output']

                # Convert numpy arrays to lists
                predicted_train_outputs = [arr.tolist() if hasattr(arr, 'tolist') else arr for arr in predicted_train_outputs]
                predicted_test_outputs = [arr.tolist() if hasattr(arr, 'tolist') else arr for arr in predicted_test_outputs]

                # Calculate pixel matches
                train_pixel_matches = []
                for predicted, expected in zip(predicted_train_outputs, [ex["output"] for ex in task_data["train"]]):
                    pixel_match = calculate_pixel_match_percentage(predicted, expected)
                    train_pixel_matches.append(pixel_match)

                test_pixel_matches = []
                for predicted, expected in zip(predicted_test_outputs, [ex["output"] for ex in task_data["test"]]):
                    pixel_match = calculate_pixel_match_percentage(predicted, expected)
                    test_pixel_matches.append(pixel_match)

                results.append({
                    'program_id': row['row_id'],
                    'avg_train_pixel_match': np.mean(train_pixel_matches),
                    'avg_test_pixel_match': np.mean(test_pixel_matches),
                    'train_pixel_matches': train_pixel_matches,
                    'test_pixel_matches': test_pixel_matches,
                    'program_code': row['code']
                })

        return results

    else:
        print("Running SOAR programs on task 45a5af55...")

        # Initialize tester
        tester = ArcTester(timeout=5, executor_type="unrestricted")

        results = []

        for idx, row in df.iterrows():
            program_id = row['row_id']
            program_code = row['code']

            print(f"Testing program {idx+1}/{len(df)}: {program_id}")

            try:
                # Test the program
                result = tester.test_program(program_code, task_data)

                # Calculate pixel matches
                train_pixel_matches = []
                for predicted, expected in zip(result.train_outputs, [ex["output"] for ex in task_data["train"]]):
                    pixel_match = calculate_pixel_match_percentage(predicted, expected)
                    train_pixel_matches.append(pixel_match)

                test_pixel_matches = []
                for predicted, expected in zip(result.test_outputs, [ex["output"] for ex in task_data["test"]]):
                    pixel_match = calculate_pixel_match_percentage(predicted, expected)
                    test_pixel_matches.append(pixel_match)

                results.append({
                    'program_id': program_id,
                    'avg_train_pixel_match': np.mean(train_pixel_matches),
                    'avg_test_pixel_match': np.mean(test_pixel_matches),
                    'train_pixel_matches': train_pixel_matches,
                    'test_pixel_matches': test_pixel_matches,
                    'program_code': program_code,
                    'train_correct': sum(result.correct_train_input),
                    'test_correct': sum(result.correct_test_input)
                })

                print(f"  Train pixel match: {np.mean(train_pixel_matches):.1f}%")
                print(f"  Test pixel match: {np.mean(test_pixel_matches):.1f}%")

            except Exception as e:
                print(f"  Error: {e}")
                results.append({
                    'program_id': program_id,
                    'avg_train_pixel_match': 0.0,
                    'avg_test_pixel_match': 0.0,
                    'error': str(e)
                })

        tester.cleanup_executor()
        return results


def compare_soar_vs_trelis():
    """Compare SOAR programs vs Trelis baseline on task 45a5af55."""

    # Test SOAR programs
    soar_parquet = "/Users/ronanmcgovern/TR/arc-agi-2025/llm_python/datasets/inference/20250918_152152_julien31_Soar-qwen-14b_arc-prize-2025_evaluation.parquet"
    soar_results = test_soar_programs_on_45a5af55(soar_parquet)

    print(f"\n" + "="*80)
    print("COMPARISON: SOAR vs TRELIS ON TASK 45a5af55")
    print("="*80)

    # Trelis baseline (from previous run)
    trelis_best_pixel_match = 56.8
    print(f"\nTRELIS BASELINE:")
    print(f"Best average train pixel match: {trelis_best_pixel_match}%")
    print(f"Programs evaluated: 718 (excluded 2 from same task)")
    print(f"All correct: 0, All train correct: 0, Partial correct: 0")

    # SOAR results
    valid_soar_results = [r for r in soar_results if 'error' not in r]
    if valid_soar_results:
        soar_results_sorted = sorted(valid_soar_results, key=lambda x: x['avg_train_pixel_match'], reverse=True)
        best_soar = soar_results_sorted[0]

        print(f"\nSOAR PROGRAMS:")
        print(f"Programs evaluated: {len(soar_results)}")
        print(f"Successfully executed: {len(valid_soar_results)}")
        print(f"Best average train pixel match: {best_soar['avg_train_pixel_match']:.1f}%")

        print(f"\nSOAR PROGRAM RESULTS:")
        for i, result in enumerate(soar_results_sorted):
            train_correct = result.get('train_correct', 'N/A')
            test_correct = result.get('test_correct', 'N/A')
            print(f"  {i+1}. {result['program_id']}: {result['avg_train_pixel_match']:.1f}% train, {result['avg_test_pixel_match']:.1f}% test")
            if train_correct != 'N/A':
                print(f"      Train correct: {train_correct}/2, Test correct: {test_correct}/1")

        # Comparison
        improvement = best_soar['avg_train_pixel_match'] - trelis_best_pixel_match
        print(f"\nCOMPARISON:")
        print(f"SOAR best: {best_soar['avg_train_pixel_match']:.1f}%")
        print(f"Trelis best: {trelis_best_pixel_match}%")
        print(f"Improvement: {improvement:+.1f} percentage points")

        if improvement > 0:
            print("✅ SOAR programs outperform Trelis baseline")
        elif improvement < 0:
            print("❌ SOAR programs underperform Trelis baseline")
        else:
            print("🔄 SOAR programs match Trelis baseline")

        # Show best SOAR program code
        print(f"\nBEST SOAR PROGRAM CODE ({best_soar['program_id']}):")
        print("="*60)
        print(best_soar['program_code'][:500] + ("..." if len(best_soar['program_code']) > 500 else ""))
        print("="*60)

    else:
        print(f"\nSOAR PROGRAMS:")
        print(f"All {len(soar_results)} programs failed to execute")
        print("❌ SOAR programs failed completely")


if __name__ == "__main__":
    compare_soar_vs_trelis()