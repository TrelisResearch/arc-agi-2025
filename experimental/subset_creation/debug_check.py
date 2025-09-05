"""
Debug the all() function on the data.
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
    
    df = read_soar_parquet(parquet_path)
    
    print("Testing all() function on first 10 rows:")
    for i in range(min(10, len(df))):
        val = df.iloc[i]['correct_train_input']
        if isinstance(val, list):
            result = all(val)
            print(f"Row {i}: {val} -> all() = {result}")
    
    # Now count properly
    print("\nCounting rows with all train correct:")
    count = 0
    for idx in range(len(df)):
        val = df.iloc[idx]['correct_train_input']
        if isinstance(val, list) and val and all(val):
            count += 1
            if count <= 3:  # Print first few matches
                print(f"  Found match at row {idx}: task_id={df.iloc[idx]['task_id']}")
    
    print(f"Total rows with all train correct: {count}/{len(df)} ({100*count/len(df):.1f}%)")
    
    # Group by task - using a more robust approach
    all_train_correct_list = []
    for idx in range(len(df)):
        val = df.iloc[idx]['correct_train_input']
        result = isinstance(val, list) and bool(val) and all(val)
        all_train_correct_list.append(result)
    
    df['all_train_correct'] = all_train_correct_list
    
    print(f"\nDouble-check: {df['all_train_correct'].sum()} rows have all_train_correct=True")
    
    tasks_with_success = df.groupby('task_id')['all_train_correct'].max()
    num_success = tasks_with_success.sum()
    total = len(tasks_with_success)
    
    print(f"Tasks with at least one all train correct: {num_success}/{total} ({100*num_success/total:.1f}%)")

if __name__ == "__main__":
    main()