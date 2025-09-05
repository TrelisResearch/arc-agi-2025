"""
Inspect the parquet file to understand the data structure better.
"""

import sys
from pathlib import Path

# Add parent directories to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_python.datasets.io import read_soar_parquet
import pandas as pd

def main():
    # Read the parquet file
    parquet_path = Path("/Users/ronanmcgovern/TR/arc-agi-2025/llm_python/datasets/inference/20250901_214154_gpt-5-nano_arc-prize-2025_training.parquet")
    
    print(f"Reading parquet file: {parquet_path}")
    df = read_soar_parquet(parquet_path)
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Look at first few rows
    print("\nFirst row details:")
    for col in df.columns:
        val = df.iloc[0][col]
        print(f"  {col}: {type(val).__name__} = {val if not isinstance(val, (list, str)) or len(str(val)) < 100 else str(val)[:100] + '...'}")
    
    # Check unique values for some columns
    print(f"\nUnique task_ids: {df['task_id'].nunique()}")
    print(f"Unique models: {df['model'].unique()}")
    
    # Look at correct_train_input column more closely
    print("\n--- Analyzing correct_train_input column ---")
    sample_values = df['correct_train_input'].head(10)
    for i, val in enumerate(sample_values):
        print(f"Row {i}: type={type(val).__name__}, value={val}")
    
    # Check how many rows have all True in correct_train_input
    def count_all_true(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return False
        if isinstance(val, list):
            # Debug: print first few
            if len(df) < 20:
                print(f"  Checking list: {val}, all={all(val)}")
            return all(val) if val else False
        return False
    
    all_true_count = df['correct_train_input'].apply(count_all_true).sum()
    print(f"\nRows with all True in correct_train_input: {all_true_count}/{len(df)} ({100*all_true_count/len(df):.1f}%)")
    
    # Group by task and see how many tasks have at least one all-true
    df['all_train_correct'] = df['correct_train_input'].apply(count_all_true)
    tasks_with_success = df.groupby('task_id')['all_train_correct'].max().sum()
    total_tasks = df['task_id'].nunique()
    print(f"Tasks with at least one all-true: {tasks_with_success}/{total_tasks} ({100*tasks_with_success/total_tasks:.1f}%)")
    
    # Look at test correctness too
    print("\n--- Analyzing correct_test_input column ---")
    test_sample = df['correct_test_input'].head(10)
    for i, val in enumerate(test_sample):
        print(f"Row {i}: type={type(val).__name__}, value={val}")

if __name__ == "__main__":
    main()