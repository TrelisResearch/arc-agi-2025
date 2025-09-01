#!/usr/bin/env python3
"""
Analyze correctness distribution for non-transductive programs in task 135a2760.
Shows how many programs have 1-correct, 2-correct, 3-correct, etc.
"""

import sys
import os
# Add the project root to the path so we can import from llm_python
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from llm_python.datasets.io import read_soar_parquet
import pandas as pd

def analyze_correctness_distribution(parquet_path: str, task_id: str = "135a2760"):
    """
    Analyze correctness distribution for non-transductive programs.
    
    Args:
        parquet_path: Path to the parquet file
        task_id: Task ID to filter for (default: "135a2760")
    
    Returns:
        tuple: (total_non_trans_programs, correctness_distribution, programs_with_any_correct)
    """
    print(f"\nAnalyzing: {parquet_path}")
    
    # Read the parquet file
    df = read_soar_parquet(parquet_path)
    
    # Filter for the specific task_id
    task_df = df[df['task_id'] == task_id]
    
    if len(task_df) == 0:
        print(f"No programs found for task ID {task_id}")
        return 0, {}, 0
    
    print(f"Found {len(task_df)} total programs for task {task_id}")
    
    # Filter out transductive programs
    non_trans_df = task_df[task_df['is_transductive'] == False]
    print(f"Found {len(non_trans_df)} non-transductive programs")
    
    if len(non_trans_df) == 0:
        print("No non-transductive programs found!")
        return 0, {}, 0
    
    # First, figure out the maximum possible correct solutions
    sample_row = non_trans_df.iloc[0]
    train_values_sample = list(sample_row['correct_train_input']) if hasattr(sample_row['correct_train_input'], '__iter__') else [sample_row['correct_train_input']]
    test_values_sample = list(sample_row['correct_test_input']) if hasattr(sample_row['correct_test_input'], '__iter__') else [sample_row['correct_test_input']]
    max_possible_correct = len(train_values_sample) + len(test_values_sample)
    
    print(f"Maximum possible correct solutions per program: {max_possible_correct}")
    
    # Count correctness distribution
    correctness_counts = {}  # {num_correct: count}
    programs_with_any_correct = 0
    
    for idx, row in non_trans_df.iterrows():
        # Convert numpy arrays to Python lists if needed, then count total correct
        train_values = list(row['correct_train_input']) if hasattr(row['correct_train_input'], '__iter__') else [row['correct_train_input']]
        test_values = list(row['correct_test_input']) if hasattr(row['correct_test_input'], '__iter__') else [row['correct_test_input']]
        
        total_correct = sum(train_values) + sum(test_values)
        
        if total_correct > 0:
            programs_with_any_correct += 1
            
        if total_correct not in correctness_counts:
            correctness_counts[total_correct] = 0
        correctness_counts[total_correct] += 1
    
    print(f"\nCorrectness Distribution (Non-Transductive Programs):")
    print(f"Total non-transductive programs: {len(non_trans_df)}")
    print(f"Programs with at least one correct: {programs_with_any_correct}")
    
    # Show complete breakdown from 0 to max_possible_correct
    for num_correct in range(max_possible_correct + 1):
        count = correctness_counts.get(num_correct, 0)
        percentage_of_total = count/len(non_trans_df)*100
        
        if programs_with_any_correct > 0 and num_correct > 0:
            percentage_of_correct = count/programs_with_any_correct*100
            print(f"  {num_correct} correct: {count} programs ({percentage_of_total:.1f}% of total, {percentage_of_correct:.1f}% of programs with ≥1 correct)")
        else:
            print(f"  {num_correct} correct: {count} programs ({percentage_of_total:.1f}% of total)")
    
    return len(non_trans_df), correctness_counts, programs_with_any_correct

def main():
    # Define the parquet file paths
    parquet1 = "/Users/ronanmcgovern/TR/arc-agi-2025/llm_python/datasets/inference/20250830_082909_Trelis_Qwen3-4B_ds-arc-agi-2-partial-100-c2806_arc-prize-2025_evaluation.parquet"
    parquet2 = "/Users/ronanmcgovern/TR/arc-agi-2025/llm_python/datasets/inference/20250830_114154__workspace_arc-agi-2025_llm_python_fine-tuning_Qwen3-4B_ds-arc-agi-2-partial-100-c2806_ds-inference-final_arc-prize-2025_evaluation.parquet"
    
    task_id = "4c7dc4dd"
    
    print(f"Analyzing task ID: {task_id} (Non-Transductive Programs Only)")
    print("="*80)
    
    # Analyze first parquet file
    total1, dist1, correct1 = analyze_correctness_distribution(parquet1, task_id)
    
    # Analyze second parquet file  
    total2, dist2, correct2 = analyze_correctness_distribution(parquet2, task_id)
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    print(f"Task ID: {task_id}")
    print(f"\nFile 1 (Trelis_Qwen3-4B):")
    print(f"  Non-transductive programs: {total1}")
    print(f"  Programs with ≥1 correct: {correct1} ({correct1/total1*100:.1f}%)")
    
    print(f"\nFile 2 (fine-tuning):")
    print(f"  Non-transductive programs: {total2}")
    print(f"  Programs with ≥1 correct: {correct2} ({correct2/total2*100:.1f}%)")

if __name__ == "__main__":
    main()