#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
import sys

# Add llm_python to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_python.datasets.io import read_soar_parquet
from llm_python.utils.task_loader import get_task_loader

def load_training_hard_tasks():
    """Load the list of tasks in training-hard subset"""
    loader = get_task_loader()
    try:
        tasks = loader.get_subset_tasks("arc-prize-2025/training-hard")
        return set(task[0] for task in tasks)
    except Exception as e:
        print(f"Error loading training-hard tasks: {e}")
        tasks = loader.get_dataset_subset("arc-prize-2025/training-hard")
        return set(task[0] for task in tasks)

def main():
    # Path to the new gpt-5-mini parquet file
    parquet_path = Path(__file__).parent.parent.parent / "llm_python/datasets/inference/20250903_120046_gpt-5-mini_arc-prize-2025_training-hard.parquet"
    
    if not parquet_path.exists():
        print(f"❌ File not found: {parquet_path}")
        return
    
    print(f"📊 Analyzing: {parquet_path.name}")
    
    # Load training-hard task list
    training_hard_tasks = load_training_hard_tasks()
    print(f"Total training-hard tasks: {len(training_hard_tasks)}")
    
    # Load the parquet file
    df = read_soar_parquet(parquet_path)
    print(f"Total programs in file: {len(df)}")
    
    # Check coverage of training-hard tasks
    unique_tasks_in_file = set(df['task_id'].unique())
    tasks_in_training_hard = unique_tasks_in_file.intersection(training_hard_tasks)
    
    print(f"Unique tasks in file: {len(unique_tasks_in_file)}")
    print(f"Tasks that are in training-hard: {len(tasks_in_training_hard)}")
    print(f"Training-hard coverage: {len(tasks_in_training_hard)}/{len(training_hard_tasks)} ({len(tasks_in_training_hard)/len(training_hard_tasks)*100:.1f}%)")
    
    # Filter for training-hard tasks only
    df_filtered = df[df['task_id'].isin(training_hard_tasks)].copy()
    print(f"Programs in training-hard tasks: {len(df_filtered)}")
    
    # Filter non-transductive
    non_trans_df = df_filtered[df_filtered['is_transductive'] == False].copy()
    print(f"Non-transductive programs: {len(non_trans_df)}")
    print(f"Transductive programs: {len(df_filtered) - len(non_trans_df)}")
    
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
    
    def check_partial_correct(x):
        if x is None or (hasattr(x, '__len__') and len(x) == 0):
            return False
        if hasattr(x, 'to_pylist'):
            x = x.to_pylist()
        return any(x) and not all(x)
    
    # Add analysis columns
    non_trans_df['all_train_correct'] = non_trans_df['correct_train_input'].apply(check_all_correct)
    non_trans_df['all_test_correct'] = non_trans_df['correct_test_input'].apply(check_all_correct)
    non_trans_df['all_correct'] = non_trans_df['all_train_correct'] & non_trans_df['all_test_correct']
    
    non_trans_df['any_train_correct'] = non_trans_df['correct_train_input'].apply(check_any_correct)
    non_trans_df['any_test_correct'] = non_trans_df['correct_test_input'].apply(check_any_correct)
    
    non_trans_df['partial_train_correct'] = non_trans_df['correct_train_input'].apply(check_partial_correct)
    non_trans_df['partial_test_correct'] = non_trans_df['correct_test_input'].apply(check_partial_correct)
    
    # Define "completely incorrect" programs
    non_trans_df['completely_incorrect'] = (~non_trans_df['any_train_correct']) & (~non_trans_df['any_test_correct'])
    
    # Count results
    all_correct_count = non_trans_df['all_correct'].sum()
    all_train_correct = non_trans_df['all_train_correct'].sum()
    all_test_correct = non_trans_df['all_test_correct'].sum()
    any_train_correct = non_trans_df['any_train_correct'].sum()
    any_test_correct = non_trans_df['any_test_correct'].sum()
    partial_train_correct = non_trans_df['partial_train_correct'].sum()
    partial_test_correct = non_trans_df['partial_test_correct'].sum()
    completely_incorrect = non_trans_df['completely_incorrect'].sum()
    
    print(f"\n🎯 Results for non-transductive programs:")
    print(f"  - All correct (train AND test): {all_correct_count} ({all_correct_count/len(non_trans_df)*100:.2f}%)")
    print(f"  - All train correct: {all_train_correct} ({all_train_correct/len(non_trans_df)*100:.2f}%)")
    print(f"  - All test correct: {all_test_correct} ({all_test_correct/len(non_trans_df)*100:.2f}%)")
    print(f"  - Any train correct: {any_train_correct} ({any_train_correct/len(non_trans_df)*100:.2f}%)")
    print(f"  - Any test correct: {any_test_correct} ({any_test_correct/len(non_trans_df)*100:.2f}%)")
    print(f"  - Partial train correct: {partial_train_correct} ({partial_train_correct/len(non_trans_df)*100:.2f}%)")
    print(f"  - Partial test correct: {partial_test_correct} ({partial_test_correct/len(non_trans_df)*100:.2f}%)")
    print(f"  - Completely incorrect: {completely_incorrect} ({completely_incorrect/len(non_trans_df)*100:.2f}%)")
    
    # Task-level analysis
    task_stats = non_trans_df.groupby('task_id').agg({
        'all_correct': 'sum',
        'all_train_correct': 'sum',
        'any_train_correct': 'sum',
        'partial_train_correct': 'sum',
        'any_test_correct': 'sum',
        'partial_test_correct': 'sum',
        'completely_incorrect': 'sum',
        'row_id': 'count'
    }).rename(columns={'row_id': 'total_programs'})
    
    tasks_with_all_correct = (task_stats['all_correct'] > 0).sum()
    tasks_with_all_train = (task_stats['all_train_correct'] > 0).sum()
    tasks_with_any_train = (task_stats['any_train_correct'] > 0).sum()
    tasks_with_partial_train = (task_stats['partial_train_correct'] > 0).sum()
    tasks_with_any_test = (task_stats['any_test_correct'] > 0).sum()
    tasks_with_only_incorrect = (task_stats['completely_incorrect'] == task_stats['total_programs']).sum()
    
    print(f"\n📈 Task-level analysis (out of {len(tasks_in_training_hard)} training-hard tasks covered):")
    print(f"  - Tasks with ≥1 all-correct program: {tasks_with_all_correct} ({tasks_with_all_correct/len(tasks_in_training_hard)*100:.1f}%)")
    print(f"  - Tasks with ≥1 all-train-correct: {tasks_with_all_train} ({tasks_with_all_train/len(tasks_in_training_hard)*100:.1f}%)")
    print(f"  - Tasks with ≥1 any-train-correct: {tasks_with_any_train} ({tasks_with_any_train/len(tasks_in_training_hard)*100:.1f}%)")
    print(f"  - Tasks with ≥1 partial-train-correct: {tasks_with_partial_train} ({tasks_with_partial_train/len(tasks_in_training_hard)*100:.1f}%)")
    print(f"  - Tasks with ≥1 any-test-correct: {tasks_with_any_test} ({tasks_with_any_test/len(tasks_in_training_hard)*100:.1f}%)")
    print(f"  - Tasks with ONLY incorrect programs: {tasks_with_only_incorrect} ({tasks_with_only_incorrect/len(tasks_in_training_hard)*100:.1f}%)")
    
    # Show top performing tasks if any
    if all_correct_count > 0:
        print(f"\n🏆 Top tasks with all-correct programs:")
        top_tasks = task_stats[task_stats['all_correct'] > 0].nlargest(10, 'all_correct')
        for task_id, row in top_tasks.iterrows():
            print(f"  - {task_id}: {int(row['all_correct'])} all-correct (out of {int(row['total_programs'])} total)")
    
    # Model info
    models = non_trans_df['model'].unique()
    print(f"\n🔧 Model: {models[0] if len(models) == 1 else models}")
    
    print(f"\n📊 Summary:")
    print(f"  - Training-hard coverage: {len(tasks_in_training_hard)}/137 tasks ({len(tasks_in_training_hard)/137*100:.1f}%)")
    print(f"  - All-correct programs: {all_correct_count} ({all_correct_count/len(non_trans_df)*100:.2f}% of programs)")
    print(f"  - Tasks with success: {tasks_with_all_correct}/137 ({tasks_with_all_correct/137*100:.1f}%)")

if __name__ == "__main__":
    main()