"""
Check task 045e512c for programs with partial train success.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_python.datasets.io import read_soar_parquet
import pandas as pd
import numpy as np

def main():
    # Read the parquet file
    parquet_path = Path("/Users/ronanmcgovern/TR/arc-agi-2025/llm_python/datasets/inference/20250902_122615_openai_gpt-oss-120b_arc-prize-2025_training-hard.parquet")
    
    print(f"Reading parquet file: {parquet_path}")
    df = read_soar_parquet(parquet_path)
    
    # Filter for task 045e512c
    task_df = df[df['task_id'] == '045e512c'].copy()
    
    print(f"\nTask 045e512c:")
    print(f"  Total programs: {len(task_df)}")
    
    if len(task_df) == 0:
        print("  No programs found for this task")
        return
    
    # Analyze train correctness
    partial_success_programs = []
    all_success_programs = []
    no_success_programs = []
    
    print(f"\nDebugging first few rows:")
    for i, (idx, row) in enumerate(task_df.iterrows()):
        correct_train = row['correct_train_input']
        
        if i < 3:  # Debug first 3 rows
            print(f"  Row {i}: correct_train = {correct_train} (type: {type(correct_train)})")
        
        # Handle both lists and numpy arrays
        if isinstance(correct_train, (list, np.ndarray)) or hasattr(correct_train, '__len__'):
            try:
                num_correct = sum(correct_train)
                total_examples = len(correct_train)
                
                if num_correct == total_examples:
                    all_success_programs.append((idx, correct_train, row))
                elif num_correct > 0:
                    partial_success_programs.append((idx, correct_train, row))
                else:
                    no_success_programs.append((idx, correct_train, row))
            except Exception as e:
                print(f"  Error processing correct_train: {e}")
                print(f"  correct_train = {correct_train} (type: {type(correct_train)})")
        else:
            print(f"  Warning: correct_train is not a list/array: {correct_train} (type: {type(correct_train)})")
    
    print(f"  Programs with all train correct: {len(all_success_programs)}")
    print(f"  Programs with partial train correct: {len(partial_success_programs)}")
    print(f"  Programs with no train correct: {len(no_success_programs)}")
    
    # Show details for partial success programs
    if partial_success_programs:
        print(f"\n=== Programs with partial train success ===")
        for i, (idx, correct_train, row) in enumerate(partial_success_programs):
            num_correct = sum(correct_train)
            total = len(correct_train)
            is_transductive = row['is_transductive']
            refined = row['refined_from_id'] if pd.notna(row['refined_from_id']) else None
            
            print(f"  Program {i+1}: {num_correct}/{total} train correct")
            print(f"    Row index: {idx}")
            print(f"    Train correctness: {correct_train}")
            print(f"    Is transductive: {is_transductive}")
            print(f"    Refined from: {refined}")
            print(f"    Test correctness: {row['correct_test_input']}")
            print()
    
    # Show details for all success programs if any
    if all_success_programs:
        print(f"\n=== Programs with all train correct ===")
        for i, (idx, correct_train, row) in enumerate(all_success_programs[:3]):  # Show first 3
            is_transductive = row['is_transductive']
            refined = row['refined_from_id'] if pd.notna(row['refined_from_id']) else None
            
            print(f"  Program {i+1}: {sum(correct_train)}/{len(correct_train)} train correct")
            print(f"    Row index: {idx}")
            print(f"    Is transductive: {is_transductive}")
            print(f"    Refined from: {refined}")
            print(f"    Test correctness: {row['correct_test_input']}")
            print()

if __name__ == "__main__":
    main()