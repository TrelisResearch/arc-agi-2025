#!/usr/bin/env python3
"""
Check how many programs have at least one correct solution for task ID 135a2760
in the specified parquet files.
"""

import sys
import os
# Add the project root to the path so we can import from llm_python
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from llm_python.datasets.io import read_soar_parquet
import pandas as pd

def check_correct_programs(parquet_path: str, task_id: str = "135a2760"):
    """
    Check how many programs have at least one correct solution for the given task_id.
    
    Args:
        parquet_path: Path to the parquet file
        task_id: Task ID to filter for (default: "135a2760")
    
    Returns:
        tuple: (total_programs, programs_with_correct)
    """
    print(f"\nAnalyzing: {parquet_path}")
    
    # Read the parquet file
    df = read_soar_parquet(parquet_path)
    
    # Filter for the specific task_id
    task_df = df[df['task_id'] == task_id]
    
    if len(task_df) == 0:
        print(f"No programs found for task ID {task_id}")
        return 0, 0
    
    print(f"Found {len(task_df)} total programs for task {task_id}")
    
    # Check which programs have at least one correct solution
    print("Columns in dataset:", list(df.columns))
    
    # Let's first examine the structure of the correctness columns
    print("\nExamining correctness column structure...")
    sample_row = task_df.iloc[0]
    print(f"Type of correct_train_input: {type(sample_row['correct_train_input'])}")
    print(f"Sample correct_train_input: {sample_row['correct_train_input']}")
    print(f"Type of correct_test_input: {type(sample_row['correct_test_input'])}")
    print(f"Sample correct_test_input: {sample_row['correct_test_input']}")
    
    # Based on the schema, we have 'correct_train_input' and 'correct_test_input' columns
    # These appear to be lists of booleans, so we need to check if any element is True
    if 'correct_train_input' in task_df.columns and 'correct_test_input' in task_df.columns:
        programs_with_correct = 0
        train_correct = 0
        test_correct = 0
        both_correct = 0
        
        for idx, row in task_df.iterrows():
            # Convert numpy arrays to Python lists if needed, then check for any True values
            train_values = list(row['correct_train_input']) if hasattr(row['correct_train_input'], '__iter__') else [row['correct_train_input']]
            test_values = list(row['correct_test_input']) if hasattr(row['correct_test_input'], '__iter__') else [row['correct_test_input']]
            
            train_has_correct = any(train_values) if train_values else False
            test_has_correct = any(test_values) if test_values else False
            
            
            if train_has_correct:
                train_correct += 1
            if test_has_correct:
                test_correct += 1
            if train_has_correct and test_has_correct:
                both_correct += 1
            if train_has_correct or test_has_correct:
                programs_with_correct += 1
        
        print(f"Programs with at least one correct solution: {programs_with_correct}")
        print(f"  - Train input correct: {train_correct}")
        print(f"  - Test input correct: {test_correct}")
        print(f"  - Both train and test correct: {both_correct}")
        
        return len(task_df), programs_with_correct
    else:
        # Look for other columns that might indicate correctness
        correctness_cols = [col for col in task_df.columns if 'correct' in col.lower() or 'score' in col.lower()]
        print(f"Potential correctness columns: {correctness_cols}")
        
        # If we can't determine correctness, just return program count
        print(f"Total programs found: {len(task_df)}")
        return len(task_df), "Unknown (no clear correctness indicator)"

def main():
    # Define the parquet file paths
    parquet1 = "/Users/ronanmcgovern/TR/arc-agi-2025/llm_python/datasets/inference/20250830_082909_Trelis_Qwen3-4B_ds-arc-agi-2-partial-100-c2806_arc-prize-2025_evaluation.parquet"
    parquet2 = "/Users/ronanmcgovern/TR/arc-agi-2025/llm_python/datasets/inference/20250830_114154__workspace_arc-agi-2025_llm_python_fine-tuning_Qwen3-4B_ds-arc-agi-2-partial-100-c2806_ds-inference-final_arc-prize-2025_evaluation.parquet"
    
    task_id = "135a2760"
    
    print(f"Checking task ID: {task_id}")
    print("="*50)
    
    # Check first parquet file
    total1, correct1 = check_correct_programs(parquet1, task_id)
    
    # Check second parquet file  
    total2, correct2 = check_correct_programs(parquet2, task_id)
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Task ID: {task_id}")
    print(f"File 1 (Trelis_Qwen3-4B): {correct1} programs with correct solutions (out of {total1} total)")
    print(f"File 2 (fine-tuning): {correct2} programs with correct solutions (out of {total2} total)")

if __name__ == "__main__":
    main()