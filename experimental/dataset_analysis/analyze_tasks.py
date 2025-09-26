#!/usr/bin/env python3

import pandas as pd
import numpy as np
from collections import defaultdict

def analyze_task_correctness_patterns():
    """
    Analyze tasks to find ones with mixed correctness patterns:
    - Some rows with all train+test correct
    - Some rows with partial correctness
    - Some rows with no correctness
    """

    print("📊 Loading dataset...")
    df = pd.read_parquet('experimental/dataset_analysis/dataset.parquet')

    print(f"Dataset shape: {df.shape}")
    print(f"Unique tasks: {df['task_id'].nunique()}")

    # Group by task_id to analyze patterns
    task_stats = {}

    for task_id, group in df.groupby('task_id'):
        rows_data = []

        for _, row in group.iterrows():
            # Calculate correctness for each row
            train_correct = row['correct_train_input'].sum() if hasattr(row['correct_train_input'], 'sum') else 0
            train_total = len(row['correct_train_input']) if hasattr(row['correct_train_input'], '__len__') else 1

            test_correct = row['correct_test_input'].sum() if hasattr(row['correct_test_input'], 'sum') else 0
            test_total = len(row['correct_test_input']) if hasattr(row['correct_test_input'], '__len__') else 1

            total_correct = train_correct + test_correct
            total_examples = train_total + test_total

            row_correctness = total_correct / total_examples if total_examples > 0 else 0.0

            rows_data.append({
                'row_id': row['row_id'],
                'train_correct': train_correct,
                'train_total': train_total,
                'test_correct': test_correct,
                'test_total': test_total,
                'total_correct': total_correct,
                'total_examples': total_examples,
                'correctness_pct': row_correctness
            })

        # Categorize rows by correctness
        all_correct = [r for r in rows_data if r['correctness_pct'] == 1.0]
        partially_correct = [r for r in rows_data if 0.0 < r['correctness_pct'] < 1.0]
        none_correct = [r for r in rows_data if r['correctness_pct'] == 0.0]

        task_stats[task_id] = {
            'total_rows': len(rows_data),
            'all_correct': len(all_correct),
            'partially_correct': len(partially_correct),
            'none_correct': len(none_correct),
            'rows_data': rows_data,
            'has_mixed_patterns': len(all_correct) > 0 and len(partially_correct) > 0 and len(none_correct) > 0
        }

    # Find tasks with mixed patterns
    mixed_tasks = {k: v for k, v in task_stats.items() if v['has_mixed_patterns']}

    print(f"\n🎯 Found {len(mixed_tasks)} tasks with mixed correctness patterns:")

    # Sort by diversity (tasks with more balanced distribution)
    sorted_mixed = sorted(mixed_tasks.items(),
                         key=lambda x: min(x[1]['all_correct'], x[1]['partially_correct'], x[1]['none_correct']),
                         reverse=True)

    # Show top candidates
    print("\n📈 Top candidate tasks (most balanced mix):")
    for i, (task_id, stats) in enumerate(sorted_mixed[:10]):
        print(f"{i+1:2d}. {task_id}: {stats['all_correct']:2d} all-correct, "
              f"{stats['partially_correct']:2d} partial, {stats['none_correct']:2d} none "
              f"(total: {stats['total_rows']} rows)")

    # Select the best candidate
    if sorted_mixed:
        selected_task_id, selected_stats = sorted_mixed[0]
        print(f"\n🎯 SELECTED TASK: {selected_task_id}")
        print(f"   Rows: {selected_stats['total_rows']}")
        print(f"   All correct: {selected_stats['all_correct']}")
        print(f"   Partially correct: {selected_stats['partially_correct']}")
        print(f"   None correct: {selected_stats['none_correct']}")

        # Show correctness distribution for selected task
        correctness_values = [r['correctness_pct'] for r in selected_stats['rows_data']]
        print(f"   Correctness range: {min(correctness_values):.2f} - {max(correctness_values):.2f}")
        print(f"   Average correctness: {np.mean(correctness_values):.2f}")

        # Save task data for analysis
        task_df = df[df['task_id'] == selected_task_id].copy()
        task_df.to_parquet('experimental/dataset_analysis/selected_task.parquet', index=False)
        print(f"   Saved task data to: experimental/dataset_analysis/selected_task.parquet")

        return selected_task_id, task_df
    else:
        print("❌ No tasks with mixed patterns found!")
        return None, None

if __name__ == "__main__":
    selected_task_id, task_df = analyze_task_correctness_patterns()
    if selected_task_id:
        print(f"\n✅ Analysis complete! Selected task: {selected_task_id}")
    else:
        print("\n❌ No suitable task found for analysis.")