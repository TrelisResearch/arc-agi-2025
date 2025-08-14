#!/usr/bin/env python3
"""
Quick script to analyze and visualize the distribution of programs per task in the SOAR dataset.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from collections import Counter

def analyze_task_distribution():
    """Analyze and visualize the distribution of programs per task."""
    print("📊 Analyzing SOAR Dataset Task Distribution")
    print("=" * 50)
    
    # Load the SOAR dataset
    print("📁 Loading SOAR dataset...")
    dataset = load_dataset("Trelis/soar-program-samples")
    df = dataset['train'].to_pandas()
    
    print(f"✅ Total programs: {len(df)}")
    print(f"✅ Unique tasks: {df['task_id'].nunique()}")
    
    # Count programs per task
    task_counts = df['task_id'].value_counts().sort_values(ascending=False)
    
    print(f"\n📋 Task Distribution Summary:")
    print(f"   Min programs per task: {task_counts.min()}")
    print(f"   Max programs per task: {task_counts.max()}")
    print(f"   Mean programs per task: {task_counts.mean():.1f}")
    print(f"   Median programs per task: {task_counts.median():.1f}")
    
    print(f"\n🔝 Top 10 Tasks by Program Count:")
    for i, (task_id, count) in enumerate(task_counts.head(10).items(), 1):
        print(f"   {i:2d}. {task_id}: {count} programs")
    
    print(f"\n🔻 Bottom 10 Tasks by Program Count:")
    for i, (task_id, count) in enumerate(task_counts.tail(10).items(), 1):
        print(f"   {i:2d}. {task_id}: {count} programs")
    
    # Create visualizations
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram of programs per task
    ax1.hist(task_counts.values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Number of Programs per Task')
    ax1.set_ylabel('Number of Tasks')
    ax1.set_title('Distribution of Programs per Task')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"""Statistics:
    Total Tasks: {len(task_counts)}
    Min: {task_counts.min()}
    Max: {task_counts.max()}
    Mean: {task_counts.mean():.1f}
    Median: {task_counts.median():.1f}"""
    ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Bar plot of top tasks
    top_10 = task_counts.head(10)
    bars = ax2.bar(range(len(top_10)), top_10.values, color='lightcoral', alpha=0.7)
    ax2.set_xlabel('Task Rank')
    ax2.set_ylabel('Number of Programs')
    ax2.set_title('Top 10 Tasks by Program Count')
    ax2.set_xticks(range(len(top_10)))
    ax2.set_xticklabels([f'{i+1}' for i in range(len(top_10))])
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = "classification/data/task_distribution_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n📊 Saved visualization: {output_file}")
    
    # Save detailed task information
    task_details = {
        'total_programs': len(df),
        'total_tasks': len(task_counts),
        'task_counts': task_counts.to_dict(),
        'statistics': {
            'min': int(task_counts.min()),
            'max': int(task_counts.max()),
            'mean': float(task_counts.mean()),
            'median': float(task_counts.median()),
            'std': float(task_counts.std())
        },
        'top_10_tasks': top_10.to_dict(),
        'bottom_10_tasks': task_counts.tail(10).to_dict()
    }
    
    output_json = "classification/data/task_distribution_details.json"
    with open(output_json, 'w') as f:
        json.dump(task_details, f, indent=2)
    print(f"💾 Saved detailed analysis: {output_json}")
    
    plt.show()
    
    return task_counts

if __name__ == "__main__":
    task_distribution = analyze_task_distribution()