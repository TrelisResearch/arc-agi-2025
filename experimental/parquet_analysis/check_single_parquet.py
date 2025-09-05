#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
import sys

# Add llm_python to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_python.datasets.io import read_soar_parquet

def main():
    # Path to the specific parquet file
    parquet_path = Path(__file__).parent.parent.parent / "llm_python/datasets/inference/20250903_085020_openai_gpt-oss-120b_arc-prize-2025_training-hard.parquet"
    
    if not parquet_path.exists():
        print(f"❌ File not found: {parquet_path}")
        return
    
    print(f"📊 Analyzing: {parquet_path.name}")
    
    # Load the parquet file
    df = read_soar_parquet(parquet_path)
    print(f"Total programs: {len(df)}")
    
    # Filter non-transductive
    non_trans_df = df[df['is_transductive'] == False].copy()
    print(f"Non-transductive programs: {len(non_trans_df)}")
    print(f"Transductive programs: {len(df) - len(non_trans_df)}")
    
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
    
    # Count results
    all_correct_count = non_trans_df['all_correct'].sum()
    all_train_correct = non_trans_df['all_train_correct'].sum()
    all_test_correct = non_trans_df['all_test_correct'].sum()
    any_train_correct = non_trans_df['any_train_correct'].sum()
    any_test_correct = non_trans_df['any_test_correct'].sum()
    
    print(f"\n🎯 Results for non-transductive programs:")
    print(f"  - All correct (train AND test): {all_correct_count} ({all_correct_count/len(non_trans_df)*100:.2f}%)")
    print(f"  - All train correct: {all_train_correct} ({all_train_correct/len(non_trans_df)*100:.2f}%)")
    print(f"  - All test correct: {all_test_correct} ({all_test_correct/len(non_trans_df)*100:.2f}%)")
    print(f"  - Any train correct: {any_train_correct} ({any_train_correct/len(non_trans_df)*100:.2f}%)")
    print(f"  - Any test correct: {any_test_correct} ({any_test_correct/len(non_trans_df)*100:.2f}%)")
    
    # Task-level analysis
    task_stats = non_trans_df.groupby('task_id').agg({
        'all_correct': 'sum',
        'all_train_correct': 'sum',
        'any_train_correct': 'sum',
        'row_id': 'count'
    }).rename(columns={'row_id': 'total_programs'})
    
    tasks_with_all_correct = (task_stats['all_correct'] > 0).sum()
    tasks_with_all_train = (task_stats['all_train_correct'] > 0).sum()
    tasks_with_any_train = (task_stats['any_train_correct'] > 0).sum()
    
    print(f"\n📈 Task-level analysis:")
    print(f"  - Unique tasks: {len(task_stats)}")
    print(f"  - Tasks with ≥1 all-correct program: {tasks_with_all_correct}/{len(task_stats)} ({tasks_with_all_correct/len(task_stats)*100:.1f}%)")
    print(f"  - Tasks with ≥1 all-train-correct program: {tasks_with_all_train}/{len(task_stats)} ({tasks_with_all_train/len(task_stats)*100:.1f}%)")
    print(f"  - Tasks with ≥1 any-train-correct program: {tasks_with_any_train}/{len(task_stats)} ({tasks_with_any_train/len(task_stats)*100:.1f}%)")
    
    # Show top performing tasks if any
    if all_correct_count > 0:
        print(f"\n🏆 Top tasks with all-correct programs:")
        top_tasks = task_stats[task_stats['all_correct'] > 0].nlargest(10, 'all_correct')
        for task_id, row in top_tasks.iterrows():
            print(f"  - {task_id}: {int(row['all_correct'])} all-correct (out of {int(row['total_programs'])} total)")
    
    # Model info
    models = non_trans_df['model'].unique()
    print(f"\n🔧 Models in this file: {models}")

if __name__ == "__main__":
    main()