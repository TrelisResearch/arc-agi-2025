#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
import sys

# Add llm_python to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_python.datasets.io import read_soar_parquet

def main():
    # Path to the new Qwen evaluation file
    parquet_path = Path(__file__).parent.parent.parent / "llm_python/datasets/inference/20250903_173058_Trelis_Qwen3-4B_ds-arc-agi-2-partial-100-c2806_ds-arc-agi-2-training-hard-curriculum-c262_arc-prize-2025_evaluation.parquet"
    
    if not parquet_path.exists():
        print(f"❌ File not found: {parquet_path}")
        return
    
    print(f"📊 Analyzing: {parquet_path.name[:80]}...")
    print(f"File size: {parquet_path.stat().st_size / 1024:.1f} KB")
    
    # Load the parquet file
    df = read_soar_parquet(parquet_path)
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Total programs: {len(df)}")
    print(f"Unique tasks: {df['task_id'].nunique()}")
    
    # Filter non-transductive
    non_trans_df = df[df['is_transductive'] == False].copy()
    print(f"Non-transductive programs: {len(non_trans_df)}")
    print(f"Transductive programs: {len(df) - len(non_trans_df)}")
    
    if len(non_trans_df) == 0:
        print("❌ No non-transductive programs found!")
        return
    
    # Helper function to check if all examples are correct
    def check_all_correct(x):
        if x is None or (hasattr(x, '__len__') and len(x) == 0):
            return False
        if hasattr(x, 'to_pylist'):
            x = x.to_pylist()
        return all(x)
    
    def check_any_correct(x):
        if x is None or (hasattr(x, '__len__') and len(x) == 0):
            return False
        if hasattr(x, 'to_pylist'):
            x = x.to_pylist()
        return any(x)
    
    # Compute all-correct columns
    non_trans_df['all_train_correct'] = non_trans_df['correct_train_input'].apply(check_all_correct)
    non_trans_df['all_test_correct'] = non_trans_df['correct_test_input'].apply(check_all_correct)
    non_trans_df['all_correct'] = non_trans_df['all_train_correct'] & non_trans_df['all_test_correct']
    
    non_trans_df['any_train_correct'] = non_trans_df['correct_train_input'].apply(check_any_correct)
    non_trans_df['any_test_correct'] = non_trans_df['correct_test_input'].apply(check_any_correct)
    
    # Count all-correct programs
    all_correct_count = non_trans_df['all_correct'].sum()
    all_train_correct = non_trans_df['all_train_correct'].sum()
    all_test_correct = non_trans_df['all_test_correct'].sum()
    any_train_correct = non_trans_df['any_train_correct'].sum()
    any_test_correct = non_trans_df['any_test_correct'].sum()
    
    print(f"\n🎯 Program-level results (non-transductive):")
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
    total_tasks = len(task_stats)
    
    print(f"\n📈 TASK-LEVEL RESULTS:")
    print(f"  - Tasks with ≥1 all-correct program: **{tasks_with_all_correct}/{total_tasks}** ({tasks_with_all_correct/total_tasks*100:.1f}%)")
    print(f"  - Tasks with ≥1 all-train-correct: {tasks_with_all_train}/{total_tasks} ({tasks_with_all_train/total_tasks*100:.1f}%)")
    print(f"  - Tasks with ≥1 any-train-correct: {tasks_with_any_train}/{total_tasks} ({tasks_with_any_train/total_tasks*100:.1f}%)")
    
    # Show which tasks have all-correct programs
    if tasks_with_all_correct > 0:
        print(f"\n✅ Tasks with all-correct programs:")
        successful_tasks = task_stats[task_stats['all_correct'] > 0]['all_correct'].sort_values(ascending=False)
        for task_id, count in successful_tasks.items():
            print(f"  - {task_id}: {int(count)} all-correct programs")
    
    # Model info
    models = non_trans_df['model'].unique()
    print(f"\n🔧 Model: {models}")

if __name__ == "__main__":
    main()