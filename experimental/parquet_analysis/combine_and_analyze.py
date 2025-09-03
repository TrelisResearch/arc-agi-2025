#!/usr/bin/env python3

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Add llm_python to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_python.datasets.io import read_soar_parquet, write_soar_parquet
from llm_python.utils.task_loader import get_task_loader

def load_training_hard_tasks():
    """Load the list of tasks in training-hard subset"""
    loader = get_task_loader()
    try:
        # Try to load training-hard subset
        tasks = loader.get_subset_tasks("arc-prize-2025/training-hard")
        return set(task[0] for task in tasks)  # Extract task IDs
    except Exception as e:
        print(f"Error loading training-hard tasks: {e}")
        # Fallback: try different approach
        tasks = loader.get_dataset_subset("arc-prize-2025/training-hard")
        return set(task[0] for task in tasks)

def main():
    print("Step 1: Loading training-hard task list...")
    training_hard_tasks = load_training_hard_tasks()
    print(f"Found {len(training_hard_tasks)} tasks in training-hard subset")
    
    # Path to parquet files - use absolute path from repo root
    parquet_dir = Path(__file__).parent.parent.parent / "llm_python/datasets/inference"
    
    # Find all training-hard parquet files
    parquet_files = list(parquet_dir.glob("*training-hard*.parquet"))
    print(f"\nStep 2: Found {len(parquet_files)} training-hard parquet files")
    
    # Load and combine all parquet files
    print("\nStep 3: Loading and combining parquet files...")
    all_dfs = []
    
    for pf in parquet_files:
        try:
            df = read_soar_parquet(pf)
            print(f"  - Loaded {pf.name}: {len(df)} rows")
            all_dfs.append(df)
        except Exception as e:
            print(f"  - Error loading {pf.name}: {e}")
    
    # Combine all dataframes
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nCombined data: {len(combined_df)} total rows")
    
    # Step 4: Filter for non-transductive programs only
    print("\nStep 4: Filtering non-transductive programs...")
    non_transductive_df = combined_df[combined_df['is_transductive'] == False].copy()
    print(f"  - Non-transductive programs: {len(non_transductive_df)} rows")
    print(f"  - Transductive programs filtered out: {len(combined_df) - len(non_transductive_df)} rows")
    
    # Step 5: Filter for tasks that are in training-hard subset
    print("\nStep 5: Filtering for training-hard tasks only...")
    filtered_df = non_transductive_df[non_transductive_df['task_id'].isin(training_hard_tasks)].copy()
    print(f"  - Final filtered data: {len(filtered_df)} rows")
    
    # Print some statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"  - Unique tasks: {filtered_df['task_id'].nunique()}")
    print(f"  - Unique models: {filtered_df['model'].nunique()}")
    print(f"  - Models: {filtered_df['model'].unique()}")
    
    # Step 6: Analyze all-correct programs
    print("\nStep 6: Analyzing all-correct programs...")
    
    # Add columns for all-correct analysis - handle PyArrow arrays
    def check_all_correct(x):
        if x is None or (hasattr(x, '__len__') and len(x) == 0):
            return False
        # Convert to Python list if it's a PyArrow array
        if hasattr(x, 'to_pylist'):
            x = x.to_pylist()
        return all(x)
    
    filtered_df['all_train_correct'] = filtered_df['correct_train_input'].apply(check_all_correct)
    filtered_df['all_test_correct'] = filtered_df['correct_test_input'].apply(check_all_correct)
    filtered_df['all_correct'] = filtered_df['all_train_correct'] & filtered_df['all_test_correct']
    
    # Count all-correct programs per task
    task_stats = filtered_df.groupby('task_id').agg({
        'all_correct': 'sum',  # Count of all-correct programs per task
        'row_id': 'count'  # Total programs per task
    }).rename(columns={'row_id': 'total_programs'})
    
    # Sort by task_id to have consistent ordering
    task_stats = task_stats.sort_index()
    
    print(f"\n📈 All-Correct Analysis:")
    print(f"  - Tasks with at least 1 all-correct program: {(task_stats['all_correct'] > 0).sum()}")
    print(f"  - Total all-correct programs: {task_stats['all_correct'].sum()}")
    print(f"  - Average all-correct programs per task: {task_stats['all_correct'].mean():.2f}")
    
    # Save the filtered combined data
    output_path = Path(__file__).parent / "combined_non_transductive_training_hard.parquet"
    print(f"\nStep 7: Saving combined filtered data to {output_path.name}...")
    write_soar_parquet(filtered_df, output_path)
    print(f"  - Saved {len(filtered_df)} rows")
    
    # Step 8: Create visualization
    print("\nStep 8: Creating visualization...")
    
    # Prepare data for visualization
    task_list = task_stats.index.tolist()
    all_correct_counts = task_stats['all_correct'].tolist()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create bar chart
    x_positions = np.arange(len(task_list))
    bars = ax.bar(x_positions, all_correct_counts, color='steelblue', alpha=0.8, edgecolor='navy', linewidth=0.5)
    
    # Customize x-axis - show labels every 25 tasks
    tick_positions = np.arange(0, len(task_list), 25)
    tick_labels = [f"{i}" for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    
    # Labels and title
    ax.set_xlabel('Task Index', fontsize=12)
    ax.set_ylabel('Number of All-Correct Programs', fontsize=12)
    ax.set_title(f'All-Correct Programs per Task in Training-Hard Subset\n(Non-Transductive Programs Only)', 
                 fontsize=14, fontweight='bold')
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Add summary statistics as text
    stats_text = (f"Total Tasks: {len(task_list)}\n"
                  f"Tasks with All-Correct: {(task_stats['all_correct'] > 0).sum()}\n"
                  f"Total All-Correct Programs: {task_stats['all_correct'].sum()}\n"
                  f"Max per Task: {task_stats['all_correct'].max()}")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Highlight tasks with high counts
    max_count = task_stats['all_correct'].max()
    for i, count in enumerate(all_correct_counts):
        if count == max_count and count > 0:
            bars[i].set_color('darkgreen')
        elif count > 10:
            bars[i].set_color('green')
    
    plt.tight_layout()
    
    # Save figure
    output_fig_path = Path(__file__).parent / "all_correct_programs_chart.png"
    plt.savefig(output_fig_path, dpi=150, bbox_inches='tight')
    print(f"  - Chart saved to {output_fig_path.name}")
    
    # Also save task statistics to CSV for reference
    stats_output_path = Path(__file__).parent / "task_statistics.csv"
    task_stats.to_csv(stats_output_path)
    print(f"  - Task statistics saved to {stats_output_path.name}")
    
    # Print top performing tasks
    print("\n🏆 Top 10 Tasks with Most All-Correct Programs:")
    top_tasks = task_stats.nlargest(10, 'all_correct')
    for task_id, row in top_tasks.iterrows():
        print(f"  - {task_id}: {int(row['all_correct'])} all-correct programs (out of {int(row['total_programs'])} total)")
    
    plt.show()

if __name__ == "__main__":
    main()