#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# Add llm_python to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_python.datasets.io import read_soar_parquet, write_soar_parquet
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

def check_all_correct(x):
    """Helper to check if all examples are correct"""
    if x is None or (hasattr(x, '__len__') and len(x) == 0):
        return False
    if hasattr(x, 'to_pylist'):
        x = x.to_pylist()
    return all(x)

def check_any_correct(x):
    """Helper to check if any examples are correct"""
    if x is None or (hasattr(x, '__len__') and len(x) == 0):
        return False
    if hasattr(x, 'to_pylist'):
        x = x.to_pylist()
    return any(x)

def check_partial_correct(x):
    """Helper to check if some but not all examples are correct"""
    if x is None or (hasattr(x, '__len__') and len(x) == 0):
        return False
    if hasattr(x, 'to_pylist'):
        x = x.to_pylist()
    return any(x) and not all(x)

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*60)
    print(f"📊 COMPREHENSIVE PARQUET ANALYSIS - {timestamp}")
    print("="*60)
    
    # Step 1: Load training-hard task list
    print("\nStep 1: Loading training-hard task list...")
    training_hard_tasks = load_training_hard_tasks()
    print(f"✅ Found {len(training_hard_tasks)} tasks in training-hard subset")
    
    # Step 2: Find and load all parquet files
    parquet_dir = Path(__file__).parent.parent.parent / "llm_python/datasets/inference"
    parquet_files = list(parquet_dir.glob("*.parquet"))
    print(f"\nStep 2: Found {len(parquet_files)} parquet files total")
    
    # Load all parquet files
    all_dfs = []
    file_stats = []
    
    print("\nStep 3: Loading and analyzing individual files...")
    print("-"*50)
    
    for pf in sorted(parquet_files):
        try:
            df = read_soar_parquet(pf)
            total_rows = len(df)
            non_trans = len(df[df['is_transductive'] == False])
            trans = total_rows - non_trans
            
            # Filter for training-hard tasks
            df_filtered = df[df['task_id'].isin(training_hard_tasks)]
            in_training_hard = len(df_filtered)
            
            file_stats.append({
                'file': pf.name,
                'total': total_rows,
                'non_trans': non_trans,
                'trans': trans,
                'in_training_hard': in_training_hard
            })
            
            print(f"📄 {pf.name[:40]:40} | Total: {total_rows:5} | NonTrans: {non_trans:5} | InTH: {in_training_hard:5}")
            all_dfs.append(df)
        except Exception as e:
            print(f"  ❌ Error loading {pf.name}: {e}")
    
    print("-"*50)
    
    # Step 4: Combine all dataframes
    print("\nStep 4: Combining all dataframes...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"✅ Combined data: {len(combined_df)} total rows")
    
    # Step 5: Filter for non-transductive and training-hard
    print("\nStep 5: Applying filters...")
    filtered_df = combined_df[
        (combined_df['is_transductive'] == False) & 
        (combined_df['task_id'].isin(training_hard_tasks))
    ].copy()
    
    print(f"  - Before filters: {len(combined_df)} rows")
    print(f"  - Non-transductive only: {len(combined_df[combined_df['is_transductive'] == False])} rows")
    print(f"  - Non-trans + training-hard: {len(filtered_df)} rows")
    print(f"  - Filtered out: {len(combined_df) - len(filtered_df)} rows")
    
    # Step 6: Add analysis columns
    print("\nStep 6: Analyzing correctness...")
    filtered_df['all_train_correct'] = filtered_df['correct_train_input'].apply(check_all_correct)
    filtered_df['all_test_correct'] = filtered_df['correct_test_input'].apply(check_all_correct)
    filtered_df['any_train_correct'] = filtered_df['correct_train_input'].apply(check_any_correct)
    filtered_df['any_test_correct'] = filtered_df['correct_test_input'].apply(check_any_correct)
    filtered_df['partial_train_correct'] = filtered_df['correct_train_input'].apply(check_partial_correct)
    filtered_df['partial_test_correct'] = filtered_df['correct_test_input'].apply(check_partial_correct)
    
    filtered_df['all_correct'] = filtered_df['all_train_correct'] & filtered_df['all_test_correct']
    filtered_df['none_correct'] = (~filtered_df['any_train_correct']) & (~filtered_df['any_test_correct'])
    
    # Step 7: Calculate statistics
    print("\n" + "="*60)
    print("📈 OVERALL STATISTICS")
    print("="*60)
    
    print(f"\n📊 Dataset Overview:")
    print(f"  - Total non-transductive training-hard programs: {len(filtered_df)}")
    print(f"  - Unique tasks: {filtered_df['task_id'].nunique()}")
    print(f"  - Unique models: {filtered_df['model'].nunique()}")
    if filtered_df['model'].nunique() <= 5:
        models = filtered_df['model'].unique()
        for model in models:
            count = len(filtered_df[filtered_df['model'] == model])
            print(f"    • {model}: {count} programs")
    
    print(f"\n✅ All-Correct Programs:")
    all_correct_count = filtered_df['all_correct'].sum()
    print(f"  - Count: {all_correct_count}")
    print(f"  - Percentage: {all_correct_count/len(filtered_df)*100:.2f}%")
    
    print(f"\n📊 Training Correctness:")
    all_train = filtered_df['all_train_correct'].sum()
    partial_train = filtered_df['partial_train_correct'].sum()
    any_train = filtered_df['any_train_correct'].sum()
    none_train = (~filtered_df['any_train_correct']).sum()
    
    print(f"  - All train correct: {all_train} ({all_train/len(filtered_df)*100:.2f}%)")
    print(f"  - Partial train correct: {partial_train} ({partial_train/len(filtered_df)*100:.2f}%)")
    print(f"  - Any train correct: {any_train} ({any_train/len(filtered_df)*100:.2f}%)")
    print(f"  - No train correct: {none_train} ({none_train/len(filtered_df)*100:.2f}%)")
    
    print(f"\n📊 Test Correctness:")
    all_test = filtered_df['all_test_correct'].sum()
    partial_test = filtered_df['partial_test_correct'].sum()
    any_test = filtered_df['any_test_correct'].sum()
    none_test = (~filtered_df['any_test_correct']).sum()
    
    print(f"  - All test correct: {all_test} ({all_test/len(filtered_df)*100:.2f}%)")
    print(f"  - Partial test correct: {partial_test} ({partial_test/len(filtered_df)*100:.2f}%)")
    print(f"  - Any test correct: {any_test} ({any_test/len(filtered_df)*100:.2f}%)")
    print(f"  - No test correct: {none_test} ({none_test/len(filtered_df)*100:.2f}%)")
    
    print(f"\n❌ Completely Incorrect Programs:")
    none_correct = filtered_df['none_correct'].sum()
    print(f"  - Count: {none_correct}")
    print(f"  - Percentage: {none_correct/len(filtered_df)*100:.2f}%")
    
    # Task-level analysis
    print("\n" + "="*60)
    print("📊 TASK-LEVEL ANALYSIS")
    print("="*60)
    
    task_stats = filtered_df.groupby('task_id').agg({
        'all_correct': 'sum',
        'all_train_correct': 'sum',
        'partial_train_correct': 'sum',
        'any_train_correct': 'sum',
        'all_test_correct': 'sum',
        'partial_test_correct': 'sum',
        'any_test_correct': 'sum',
        'none_correct': 'sum',
        'row_id': 'count'
    }).rename(columns={'row_id': 'total_programs'})
    
    tasks_with_all_correct = (task_stats['all_correct'] > 0).sum()
    tasks_with_all_train = (task_stats['all_train_correct'] > 0).sum()
    tasks_with_any_train = (task_stats['any_train_correct'] > 0).sum()
    tasks_with_none_correct = task_stats[task_stats['none_correct'] == task_stats['total_programs']]
    
    print(f"\n📈 Task Success Rates:")
    print(f"  - Tasks with ≥1 all-correct program: {tasks_with_all_correct}/{len(task_stats)} ({tasks_with_all_correct/len(task_stats)*100:.1f}%)")
    print(f"  - Tasks with ≥1 all-train-correct: {tasks_with_all_train}/{len(task_stats)} ({tasks_with_all_train/len(task_stats)*100:.1f}%)")
    print(f"  - Tasks with ≥1 any-train-correct: {tasks_with_any_train}/{len(task_stats)} ({tasks_with_any_train/len(task_stats)*100:.1f}%)")
    print(f"  - Tasks with ONLY incorrect programs: {len(tasks_with_none_correct)}/{len(task_stats)} ({len(tasks_with_none_correct)/len(task_stats)*100:.1f}%)")
    
    # Top performing tasks
    print(f"\n🏆 Top 10 Tasks by All-Correct Programs:")
    top_tasks = task_stats.nlargest(10, 'all_correct')
    for task_id, row in top_tasks.iterrows():
        if row['all_correct'] > 0:
            success_rate = row['all_correct']/row['total_programs']*100
            print(f"  - {task_id}: {int(row['all_correct'])}/{int(row['total_programs'])} all-correct ({success_rate:.1f}%)")
    
    # Worst performing tasks
    print(f"\n🔴 Tasks with Zero Successful Programs:")
    zero_success = task_stats[task_stats['any_train_correct'] == 0]
    print(f"  - Count: {len(zero_success)} tasks")
    if len(zero_success) <= 10:
        for task_id in zero_success.index:
            programs = int(task_stats.loc[task_id, 'total_programs'])
            print(f"    • {task_id}: 0/{programs} success")
    
    # Save the combined filtered data
    output_path = Path(__file__).parent / f"combined_nontrans_traininghard_{timestamp}.parquet"
    print(f"\n💾 Saving combined filtered data...")
    write_soar_parquet(filtered_df, output_path)
    print(f"✅ Saved to: {output_path.name}")
    print(f"   - Total rows: {len(filtered_df)}")
    print(f"   - File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Save summary statistics
    stats_path = Path(__file__).parent / f"task_statistics_{timestamp}.csv"
    task_stats.to_csv(stats_path)
    print(f"📊 Task statistics saved to: {stats_path.name}")
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()