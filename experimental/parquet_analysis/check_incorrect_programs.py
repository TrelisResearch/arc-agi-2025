#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
import sys

# Add llm_python to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_python.datasets.io import read_soar_parquet

def main():
    # Path to the specific parquet file
    parquet_path = Path(__file__).parent.parent.parent / "llm_python/datasets/inference/20250902_142348_openai_gpt-oss-120b_arc-prize-2025_training-hard.parquet"
    
    if not parquet_path.exists():
        print(f"❌ File not found: {parquet_path}")
        return
    
    print(f"📊 Analyzing incorrect programs in: {parquet_path.name}")
    
    # Load the parquet file
    df = read_soar_parquet(parquet_path)
    print(f"Total programs: {len(df)}")
    
    # Filter non-transductive
    non_trans_df = df[df['is_transductive'] == False].copy()
    print(f"Non-transductive programs: {len(non_trans_df)}")
    
    if len(non_trans_df) == 0:
        print("❌ No non-transductive programs found!")
        return
    
    # Helper functions to check correctness with PyArrow arrays
    def check_all_correct(x):
        if x is None or (hasattr(x, '__len__') and len(x) == 0):
            return False
        # Convert to Python list if it's a PyArrow array
        if hasattr(x, 'to_pylist'):
            x = x.to_pylist()
        return all(x)
    
    def check_any_correct(x):
        if x is None or (hasattr(x, '__len__') and len(x) == 0):
            return False
        if hasattr(x, 'to_pylist'):
            x = x.to_pylist()
        return any(x)
    
    # Add analysis columns
    non_trans_df['all_train_correct'] = non_trans_df['correct_train_input'].apply(check_all_correct)
    non_trans_df['all_test_correct'] = non_trans_df['correct_test_input'].apply(check_all_correct)
    non_trans_df['all_correct'] = non_trans_df['all_train_correct'] & non_trans_df['all_test_correct']
    
    non_trans_df['any_train_correct'] = non_trans_df['correct_train_input'].apply(check_any_correct)
    non_trans_df['any_test_correct'] = non_trans_df['correct_test_input'].apply(check_any_correct)
    
    # Define "incorrect" programs - those that get NO train examples correct AND NO test examples correct
    non_trans_df['completely_incorrect'] = (~non_trans_df['any_train_correct']) & (~non_trans_df['any_test_correct'])
    
    # Task-level analysis
    task_stats = non_trans_df.groupby('task_id').agg({
        'all_correct': 'sum',
        'any_train_correct': 'sum', 
        'any_test_correct': 'sum',
        'completely_incorrect': 'sum',
        'row_id': 'count'
    }).rename(columns={'row_id': 'total_programs'})
    
    # Count tasks with completely incorrect programs
    tasks_with_incorrect = (task_stats['completely_incorrect'] > 0).sum()
    tasks_with_only_incorrect = (task_stats['completely_incorrect'] == task_stats['total_programs']).sum()
    
    print(f"\n❌ Incorrect Program Analysis:")
    print(f"  - Total completely incorrect programs: {non_trans_df['completely_incorrect'].sum()}")
    print(f"  - Tasks with ≥1 completely incorrect program: {tasks_with_incorrect}/{len(task_stats)} ({tasks_with_incorrect/len(task_stats)*100:.1f}%)")
    print(f"  - Tasks with ONLY completely incorrect programs: {tasks_with_only_incorrect}/{len(task_stats)} ({tasks_with_only_incorrect/len(task_stats)*100:.1f}%)")
    
    # Show breakdown by correctness categories
    print(f"\n📊 Program Categories (non-transductive only):")
    all_correct_count = non_trans_df['all_correct'].sum()
    any_train_count = non_trans_df['any_train_correct'].sum()
    any_test_count = non_trans_df['any_test_correct'].sum()
    completely_incorrect_count = non_trans_df['completely_incorrect'].sum()
    
    print(f"  - All correct (train AND test): {all_correct_count} ({all_correct_count/len(non_trans_df)*100:.1f}%)")
    print(f"  - Any train correct: {any_train_count} ({any_train_count/len(non_trans_df)*100:.1f}%)")
    print(f"  - Any test correct: {any_test_count} ({any_test_count/len(non_trans_df)*100:.1f}%)")
    print(f"  - Completely incorrect: {completely_incorrect_count} ({completely_incorrect_count/len(non_trans_df)*100:.1f}%)")
    
    # Show worst performing tasks (most incorrect programs)
    print(f"\n🔴 Tasks with most completely incorrect programs:")
    worst_tasks = task_stats.nlargest(10, 'completely_incorrect')
    for task_id, row in worst_tasks.iterrows():
        if row['completely_incorrect'] > 0:
            print(f"  - {task_id}: {int(row['completely_incorrect'])}/{int(row['total_programs'])} completely incorrect ({row['completely_incorrect']/row['total_programs']*100:.1f}%)")

if __name__ == "__main__":
    main()