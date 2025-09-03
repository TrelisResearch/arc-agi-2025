#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
import sys

# Add llm_python to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_python.datasets.io import read_soar_parquet

def main():
    # Load the combined filtered data we just created
    data_path = Path(__file__).parent / "combined_non_transductive_training_hard.parquet"
    
    print("Loading non-transductive training-hard data...")
    df = read_soar_parquet(data_path)
    print(f"Total non-transductive programs: {len(df)}")
    print(f"Total unique tasks: {df['task_id'].nunique()}")
    
    # Helper function to check if any training example is correct
    def has_any_train_correct(x):
        if x is None or (hasattr(x, '__len__') and len(x) == 0):
            return False
        # Convert to Python list if it's a PyArrow array
        if hasattr(x, 'to_pylist'):
            x = x.to_pylist()
        return any(x)
    
    # Add column for programs with at least one training example correct
    df['any_train_correct'] = df['correct_train_input'].apply(has_any_train_correct)
    
    # Count programs with at least one training example correct
    programs_with_any_train = df['any_train_correct'].sum()
    print(f"\n📊 Programs with at least 1 training example correct: {programs_with_any_train}/{len(df)} ({programs_with_any_train/len(df)*100:.1f}%)")
    
    # Count tasks that have at least one program with any training correct
    tasks_with_any_train = df[df['any_train_correct']].groupby('task_id').size()
    num_tasks_with_any_train = len(tasks_with_any_train)
    
    print(f"\n📈 Tasks with at least 1 program that gets ≥1 training example correct:")
    print(f"  - {num_tasks_with_any_train}/{df['task_id'].nunique()} tasks ({num_tasks_with_any_train/df['task_id'].nunique()*100:.1f}%)")
    
    # Show distribution of how many programs per task get at least one training correct
    print(f"\n📊 Distribution of programs with ≥1 training correct per task:")
    print(f"  - Mean: {tasks_with_any_train.mean():.1f} programs per task (for tasks with any)")
    print(f"  - Max: {tasks_with_any_train.max()} programs (task: {tasks_with_any_train.idxmax()})")
    print(f"  - Min: {tasks_with_any_train.min()} programs")
    
    # List tasks that have ZERO programs with any training correct
    all_task_ids = set(df['task_id'].unique())
    tasks_with_programs = set(tasks_with_any_train.index)
    tasks_with_zero = all_task_ids - tasks_with_programs
    
    print(f"\n❌ Tasks with ZERO programs that get any training example correct: {len(tasks_with_zero)}")
    if len(tasks_with_zero) <= 20:
        print(f"  Task IDs: {sorted(list(tasks_with_zero))}")
    else:
        print(f"  First 20 task IDs: {sorted(list(tasks_with_zero))[:20]}")
    
    # Also check for all training correct (for comparison)
    def has_all_train_correct(x):
        if x is None or (hasattr(x, '__len__') and len(x) == 0):
            return False
        if hasattr(x, 'to_pylist'):
            x = x.to_pylist()
        return all(x)
    
    df['all_train_correct'] = df['correct_train_input'].apply(has_all_train_correct)
    programs_with_all_train = df['all_train_correct'].sum()
    tasks_with_all_train = df[df['all_train_correct']].groupby('task_id').size()
    
    print(f"\n📊 For comparison - Programs with ALL training examples correct:")
    print(f"  - Programs: {programs_with_all_train}/{len(df)} ({programs_with_all_train/len(df)*100:.1f}%)")
    print(f"  - Tasks: {len(tasks_with_all_train)}/{df['task_id'].nunique()} ({len(tasks_with_all_train)/df['task_id'].nunique()*100:.1f}%)")

if __name__ == "__main__":
    main()