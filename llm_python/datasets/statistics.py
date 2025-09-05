"""
Dataset statistics analysis functions.
Provides analysis and visualization for SOAR format datasets.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Optional, Dict, Any


def analyze_dataset_statistics(
    df: pd.DataFrame, 
    dataset_name: str = "unknown",
    subset_mapping: Optional[Dict[str, str]] = None
) -> None:
    """
    Analyze and visualize dataset statistics for SOAR format data.
    
    Args:
        df: DataFrame in SOAR format with columns:
            - task_id
            - correct_train_input (list of bools)
            - correct_test_input (list of bools)
        dataset_name: Name of the dataset for display
        subset_mapping: Optional dict mapping task_id to subset name
    """
    print("=" * 80)
    print(f"DATASET STATISTICS ANALYSIS: {dataset_name}")
    print("=" * 80)
    
    # 1. Basic statistics
    basic_stats = _compute_basic_statistics(df)
    _print_basic_statistics(basic_stats)
    
    # 2. Subset breakdown if available
    if subset_mapping:
        subset_stats = _compute_subset_statistics(df, subset_mapping)
        _print_subset_statistics(subset_stats)
    
    # 3. Per-task statistics
    per_task_stats = _compute_per_task_statistics(df)
    _print_per_task_statistics(per_task_stats)
    
    # 4. Quantile analysis
    quantiles = _compute_quantiles(per_task_stats)
    _print_quantile_statistics(quantiles)
    
    # 5. Generate visualizations
    _create_visualizations(per_task_stats, quantiles, dataset_name)
    
    print("\n✓ Analysis complete! Generated visualizations showing:")
    print("   • Program distribution patterns across tasks")
    print("   • Quantile distribution of programs per task")
    print("   • Relationship between total and correct programs per task")


def _compute_basic_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute basic dataset statistics."""
    # Calculate correctness for each row
    all_train_correct = df['correct_train_input'].apply(lambda x: all(x))
    all_test_correct = df['correct_test_input'].apply(lambda x: all(x))
    
    total_correct = df['correct_train_input'].apply(lambda x: sum(x)) + \
                   df['correct_test_input'].apply(lambda x: sum(x))
    total_possible = df['correct_train_input'].apply(len) + \
                    df['correct_test_input'].apply(len)
    
    fully_correct = all_train_correct & all_test_correct
    partially_correct = (total_correct > 0) & ~fully_correct
    completely_incorrect = (total_correct == 0)
    
    return {
        'unique_tasks': df['task_id'].nunique(),
        'total_programs': len(df),
        'fully_correct_programs': fully_correct.sum(),
        'partially_correct_programs': partially_correct.sum(),
        'completely_incorrect_programs': completely_incorrect.sum(),
        'avg_correctness_rate': (total_correct / total_possible).mean()
    }


def _print_basic_statistics(stats: Dict[str, Any]) -> None:
    """Print basic statistics."""
    print("1. Computing basic statistics...")
    print("📊 Basic Statistics:")
    print(f"   • Unique tasks: {stats['unique_tasks']:,}")
    print(f"   • Total programs: {stats['total_programs']:,}")
    print(f"   • Fully correct programs: {stats['fully_correct_programs']:,} ({100*stats['fully_correct_programs']/stats['total_programs']:.1f}%)")
    print(f"   • Partially correct programs: {stats['partially_correct_programs']:,} ({100*stats['partially_correct_programs']/stats['total_programs']:.1f}%)")
    print(f"   • Completely incorrect programs: {stats['completely_incorrect_programs']:,} ({100*stats['completely_incorrect_programs']/stats['total_programs']:.1f}%)")
    print(f"   • Average correctness rate: {stats['avg_correctness_rate']:.3f}")


def _compute_subset_statistics(df: pd.DataFrame, subset_mapping: Dict[str, str]) -> pd.DataFrame:
    """Compute statistics by ARC subset."""
    # Add subset information to DataFrame
    df_with_subsets = df.copy()
    df_with_subsets['subset'] = df_with_subsets['task_id'].map(subset_mapping)
    
    # Group by subset and count unique tasks
    subset_stats = df_with_subsets.dropna(subset=['subset']).groupby('subset')['task_id'].nunique().reset_index()
    subset_stats = subset_stats.rename(columns={'task_id': 'unique_tasks_in_subset'})
    
    return subset_stats.sort_values('subset')


def _print_subset_statistics(subset_stats: pd.DataFrame) -> None:
    """Print subset statistics."""
    print("\n📋 ARC Subsets represented in dataset:")
    for _, row in subset_stats.iterrows():
        print(f"   • {row['subset']}: {row['unique_tasks_in_subset']} tasks")
    
    total_subset_tasks = subset_stats['unique_tasks_in_subset'].sum()
    print(f"   • Note: Total adds to {total_subset_tasks} because some tasks appear in multiple subsets")


def _compute_per_task_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-task statistics."""
    # Calculate correctness for each row
    all_train_correct = df['correct_train_input'].apply(lambda x: all(x))
    all_test_correct = df['correct_test_input'].apply(lambda x: all(x))
    
    total_correct = df['correct_train_input'].apply(lambda x: sum(x)) + \
                   df['correct_test_input'].apply(lambda x: sum(x))
    
    fully_correct = all_train_correct & all_test_correct
    partially_correct = (total_correct > 0) & ~fully_correct
    
    # Group by task
    task_stats = df.groupby('task_id').agg({
        'task_id': 'count'  # Count programs per task
    }).rename(columns={'task_id': 'total_programs_per_task'})
    
    # Add correctness counts per task
    df_with_correctness = df.copy()
    df_with_correctness['fully_correct'] = fully_correct
    df_with_correctness['partially_correct'] = partially_correct
    
    correctness_stats = df_with_correctness.groupby('task_id').agg({
        'fully_correct': 'sum',
        'partially_correct': 'sum'
    }).rename(columns={
        'fully_correct': 'fully_correct_per_task',
        'partially_correct': 'partially_correct_per_task'
    })
    
    # Merge statistics
    per_task_stats = task_stats.join(correctness_stats)
    per_task_stats['incorrect_per_task'] = (
        per_task_stats['total_programs_per_task'] - 
        per_task_stats['fully_correct_per_task'] - 
        per_task_stats['partially_correct_per_task']
    )
    
    return per_task_stats.reset_index().sort_values(['total_programs_per_task', 'task_id'], ascending=[False, True])


def _print_per_task_statistics(per_task_stats: pd.DataFrame) -> None:
    """Print per-task statistics."""
    print("\n2. Computing per-task distributions...")
    print("📈 Per-task Statistics:")
    print(f"   • Average programs per task: {per_task_stats['total_programs_per_task'].mean():.1f}")
    print(f"   • Median programs per task: {per_task_stats['total_programs_per_task'].median():.1f}")
    print(f"   • Min programs per task: {per_task_stats['total_programs_per_task'].min()}")
    print(f"   • Max programs per task: {per_task_stats['total_programs_per_task'].max()}")
    
    max_programs = per_task_stats['total_programs_per_task'].max()
    print(f"   • Tasks with {max_programs} programs (max): {(per_task_stats['total_programs_per_task'] == max_programs).sum()}")
    print(f"   • Tasks with fully correct programs: {(per_task_stats['fully_correct_per_task'] > 0).sum()}")
    print(f"   • Tasks with no correct programs: {(per_task_stats['fully_correct_per_task'] == 0).sum()}")


def _compute_quantiles(per_task_stats: pd.DataFrame) -> pd.Series:
    """Compute quantiles for programs per task."""
    programs_per_task = per_task_stats['total_programs_per_task']
    
    # Compute quantiles (0-100 percentiles)
    quantiles = pd.Series([
        programs_per_task.quantile(p/100) for p in range(101)
    ], index=range(101))
    
    return quantiles


def _print_quantile_statistics(quantiles: pd.Series) -> None:
    """Print quantile statistics."""
    print("\n3. Computing quantile distribution of programs per task...")
    print("📊 Programs per Task - Quantile Distribution:")
    print(f"   • 0th percentile (min): {quantiles[0]}")
    print(f"   • 25th percentile: {quantiles[25]}")
    print(f"   • 50th percentile (median): {quantiles[50]}")
    print(f"   • 75th percentile: {quantiles[75]}")
    print(f"   • 90th percentile: {quantiles[90]}")
    print(f"   • 95th percentile: {quantiles[95]}")
    print(f"   • 99th percentile: {quantiles[99]}")
    print(f"   • 100th percentile (max): {quantiles[100]}")


def _create_visualizations(per_task_stats: pd.DataFrame, quantiles: pd.Series, dataset_name: str) -> None:
    """Create and display visualizations."""
    print("\n4. Creating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Dataset Analysis: {dataset_name}', fontsize=16, fontweight='bold')
    
    # 1. Distribution of total programs per task
    axes[0, 0].hist(per_task_stats['total_programs_per_task'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Total Programs per Task')
    axes[0, 0].set_xlabel('Number of Programs')
    axes[0, 0].set_ylabel('Number of Tasks')
    axes[0, 0].axvline(per_task_stats['total_programs_per_task'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {per_task_stats["total_programs_per_task"].mean():.1f}')
    axes[0, 0].axvline(per_task_stats['total_programs_per_task'].median(), color='orange', linestyle='--', 
                       label=f'Median: {per_task_stats["total_programs_per_task"].median():.1f}')
    axes[0, 0].legend()
    
    # 2. Distribution of fully correct programs per task
    axes[0, 1].hist(per_task_stats['fully_correct_per_task'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Distribution of Fully Correct Programs per Task')
    axes[0, 1].set_xlabel('Number of Fully Correct Programs')
    axes[0, 1].set_ylabel('Number of Tasks')
    axes[0, 1].axvline(per_task_stats['fully_correct_per_task'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {per_task_stats["fully_correct_per_task"].mean():.1f}')
    axes[0, 1].legend()
    
    # 3. Quantile distribution plot
    percentiles = list(range(0, 101, 5))  # Every 5th percentile
    quantile_values = [quantiles[p] for p in percentiles]
    
    axes[1, 0].plot(percentiles, quantile_values, 'b-o', markersize=4)
    axes[1, 0].set_title('Programs per Task: Quantile Distribution')
    axes[1, 0].set_xlabel('Percentile')
    axes[1, 0].set_ylabel('Number of Programs per Task')
    axes[1, 0].grid(True, alpha=0.3)
    # Add some key percentile annotations
    for p in [25, 50, 75, 90, 95]:
        axes[1, 0].annotate(f'{p}th: {quantiles[p]:.0f}', 
                           xy=(p, quantiles[p]), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.8)
    
    # 4. Scatter plot: total programs vs correct programs per task
    scatter = axes[1, 1].scatter(per_task_stats['total_programs_per_task'], 
                                per_task_stats['fully_correct_per_task'],
                                alpha=0.6, c=per_task_stats['partially_correct_per_task'], 
                                cmap='viridis', s=50)
    axes[1, 1].set_title('Total vs Fully Correct Programs per Task')
    axes[1, 1].set_xlabel('Total Programs per Task')
    axes[1, 1].set_ylabel('Fully Correct Programs per Task')
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('Partially Correct Programs')
    
    # Add diagonal line for reference
    max_val = max(per_task_stats['total_programs_per_task'].max(), per_task_stats['fully_correct_per_task'].max())
    axes[1, 1].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect Correctness')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()