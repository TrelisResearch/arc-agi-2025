#!/usr/bin/env python3
"""
Download and analyze Trelis/arc-agi-2-partial-100 dataset.
Filter to tasks with max 10 programs that are all-correct on train AND test.
Plot column chart showing stacked correctness counts by task.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple

# Add parent directories to path to import llm_python utilities
current_dir = Path(__file__).parent
repo_root = current_dir.parent.parent
sys.path.insert(0, str(repo_root))

from llm_python.datasets.io import read_soar_parquet, write_soar_parquet
from llm_python.utils.deduplication import deduplicate_by_task


def download_dataset(cache_dir: str = "./data") -> pd.DataFrame:
    """Download Trelis/arc-agi-2-partial-100 dataset."""
    print("📥 Downloading Trelis/arc-agi-2-partial-100 dataset...")
    
    # Create cache directory
    Path(cache_dir).mkdir(exist_ok=True)
    
    # Download the dataset
    dataset = load_dataset("Trelis/arc-agi-2-partial-100", cache_dir=cache_dir)
    
    # Convert to DataFrame
    if 'train' in dataset:
        df = dataset['train'].to_pandas()
    else:
        # If there's no explicit train split, use the first available split
        split_name = list(dataset.keys())[0]
        df = dataset[split_name].to_pandas()
        print(f"📊 Using split: {split_name}")
    
    print(f"✅ Downloaded {len(df)} rows")
    return df


def check_dataset_structure(df: pd.DataFrame) -> None:
    """Examine the dataset structure to understand its format."""
    print("\n🔍 Dataset Structure Analysis:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    print("\nColumn dtypes:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    
    print("\nFirst few rows preview:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df.head(2))
    
    # Check for correctness columns
    correctness_cols = [col for col in df.columns if 'correct' in col.lower()]
    print(f"\nCorrectness columns found: {correctness_cols}")
    
    if correctness_cols:
        for col in correctness_cols:
            sample_values = df[col].dropna().head(5)
            print(f"\nSample values for {col}:")
            for i, val in enumerate(sample_values):
                print(f"  [{i}]: {val} (type: {type(val)})")


def filter_tasks_with_all_correct_programs(df: pd.DataFrame, max_programs: int = 10) -> pd.DataFrame:
    """
    Filter to tasks that have a maximum of max_programs that are 100% correct (percentage-based).
    """
    print(f"\n🔍 Filtering tasks with max {max_programs} programs that achieve 100% correctness...")
    
    # First, let's understand what makes a program "100% correct"
    # Look for columns that indicate correctness
    train_correct_col = None
    test_correct_col = None
    
    for col in df.columns:
        if 'correct_train' in col.lower():
            train_correct_col = col
        elif 'correct_test' in col.lower():
            test_correct_col = col
    
    if not train_correct_col or not test_correct_col:
        print("❌ Could not find train/test correctness columns")
        print(f"Available columns: {list(df.columns)}")
        return df
    
    print(f"✅ Using train correctness: {train_correct_col}")
    print(f"✅ Using test correctness: {test_correct_col}")
    
    def count_correct_and_total(correctness_data):
        """Count correct and total test cases."""
        if correctness_data is None:
            return 0, 0
        # Check for numpy array first
        if hasattr(correctness_data, 'tolist'):  # numpy array
            if correctness_data.size == 0:
                return 0, 0
            return int(correctness_data.sum()), int(correctness_data.size)
        elif isinstance(correctness_data, list):
            return sum(correctness_data), len(correctness_data)
        else:
            try:
                return int(bool(correctness_data)), 1
            except ValueError:
                return 0, 1
    
    # Calculate percentage correctness for each program and mark 100% programs
    df_with_percentages = []
    for _, row in df.iterrows():
        # Count correct on train and test
        train_correct, train_total = count_correct_and_total(row[train_correct_col])
        test_correct, test_total = count_correct_and_total(row[test_correct_col])
        
        # Calculate percentage correctness
        total_correct = train_correct + test_correct
        total_cases = train_total + test_total
        
        if total_cases == 0:
            percentage = 0.0
        else:
            percentage = (total_correct / total_cases) * 100
        
        # Mark as 100% correct
        is_100_percent = (percentage == 100.0)
        
        row_dict = row.to_dict()
        row_dict['percentage_correct'] = percentage
        row_dict['is_100_percent'] = is_100_percent
        df_with_percentages.append(row_dict)
    
    df_with_percentages = pd.DataFrame(df_with_percentages)
    
    # Count 100% programs per task
    task_100_percent_counts = df_with_percentages[df_with_percentages['is_100_percent']].groupby('task_id').size()
    
    print(f"📊 Programs with 100% correctness: {df_with_percentages['is_100_percent'].sum()}")
    
    print(f"\n📈 Distribution of 100% correct programs per task:")
    count_distribution = task_100_percent_counts.value_counts().sort_index()
    for count, num_tasks in count_distribution.items():
        print(f"  {count} programs with 100% correctness: {num_tasks} tasks")
    
    # Filter to tasks with <= max_programs that have 100% correctness
    eligible_tasks = task_100_percent_counts[task_100_percent_counts <= max_programs].index.tolist()
    filtered_df = df_with_percentages[df_with_percentages['task_id'].isin(eligible_tasks)].copy()
    
    print(f"\n✅ Found {len(eligible_tasks)} tasks with ≤{max_programs} programs achieving 100% correctness")
    print(f"✅ Filtered dataset has {len(filtered_df)} total programs")
    
    return filtered_df


def analyze_correctness_distribution(df: pd.DataFrame, train_correct_col: str, test_correct_col: str) -> Tuple[Dict[str, List[int]], pd.DataFrame]:
    """
    Analyze correctness distribution for each task using percentage correctness.
    Filter to highest-performing programs per task, with tie-breaker by code length.
    Returns dict mapping task_id to bucketed counts: [<25%, 25-50%, 50-75%, >75% but <100%, 100%].
    """
    print(f"\n📊 Analyzing correctness distribution with percentage bucketing...")
    
    def count_correct_and_total(correctness_data):
        """Count correct and total test cases."""
        if correctness_data is None:
            return 0, 0
        # Check for numpy array first
        if hasattr(correctness_data, 'tolist'):  # numpy array
            if correctness_data.size == 0:
                return 0, 0
            return int(correctness_data.sum()), int(correctness_data.size)
        elif isinstance(correctness_data, list):
            return sum(correctness_data), len(correctness_data)
        else:
            try:
                return int(bool(correctness_data)), 1
            except ValueError:
                return 0, 1
    
    # Calculate percentage correctness for each program and add analysis columns
    df_with_analysis = df.copy()
    percentages = []
    code_lengths = []
    
    for _, row in df_with_analysis.iterrows():
        # Count correct on train and test
        train_correct, train_total = count_correct_and_total(row[train_correct_col])
        test_correct, test_total = count_correct_and_total(row[test_correct_col])
        
        # Calculate percentage correctness
        total_correct = train_correct + test_correct
        total_cases = train_total + test_total
        
        if total_cases == 0:
            percentage = 0.0
        else:
            percentage = (total_correct / total_cases) * 100
        
        # Get code length for tie-breaking
        code_length = len(row.get('code', '') or '')
        
        percentages.append(percentage)
        code_lengths.append(code_length)
    
    df_with_analysis['percentage'] = percentages
    df_with_analysis['code_length'] = code_lengths
    
    # Group by task and find highest-performing programs per task
    filtered_rows = []
    for task_id in df_with_analysis['task_id'].unique():
        task_rows = df_with_analysis[df_with_analysis['task_id'] == task_id].copy()
        
        # Sort by percentage (desc) then by code length (asc) for tie-breaking
        task_rows = task_rows.sort_values(['percentage', 'code_length'], ascending=[False, True])
        
        # Keep top 10 programs per task (or all if fewer than 10)
        top_rows = task_rows.head(10)
        filtered_rows.append(top_rows)
    
    # Combine all filtered rows
    final_filtered_df = pd.concat(filtered_rows, ignore_index=True)
    
    # Create program data list for bucketing analysis
    filtered_programs = []
    for _, row in final_filtered_df.iterrows():
        filtered_programs.append({
            'task_id': row['task_id'],
            'percentage': row['percentage'],
            'code_length': row['code_length'],
            'total_correct': 0,  # Not needed for bucketing
            'total_cases': 0     # Not needed for bucketing
        })
    
    print(f"📊 Filtered to {len(filtered_programs)} highest-performing programs across {len(df_with_analysis['task_id'].unique())} tasks")
    
    # Bucket the filtered programs by percentage
    def get_percentage_bucket(percentage):
        """Convert percentage to bucket index: [<25%, 25-50%, 50-75%, >75% but <100%, 100%]"""
        if percentage == 100.0:
            return 4  # 100%
        elif percentage > 75.0:
            return 3  # >75% but <100%
        elif percentage >= 50.0:
            return 2  # 50-75%
        elif percentage >= 25.0:
            return 1  # 25-50%
        else:
            return 0  # <25%
    
    task_correctness = defaultdict(lambda: [0, 0, 0, 0, 0])  # 5 buckets
    
    for prog in filtered_programs:
        task_id = prog['task_id']
        bucket = get_percentage_bucket(prog['percentage'])
        task_correctness[task_id][bucket] += 1
    
    # Show bucket statistics
    bucket_labels = ['<25%', '25-50%', '50-75%', '>75% but <100%', '100%']
    bucket_totals = [0, 0, 0, 0, 0]
    
    for counts in task_correctness.values():
        for i, count in enumerate(counts):
            bucket_totals[i] += count
    
    total_programs = sum(bucket_totals)
    print(f"\n📈 Percentage correctness distribution:")
    for i, (label, count) in enumerate(zip(bucket_labels, bucket_totals)):
        percentage = (count / total_programs) * 100 if total_programs > 0 else 0
        print(f"  {label}: {count} programs ({percentage:.1f}%)")
    
    return dict(task_correctness), final_filtered_df


def plot_correctness_distribution(task_correctness: Dict[str, List[int]], output_path: str = "correctness_distribution.png"):
    """
    Plot column chart showing stacked percentage correctness buckets by task.
    X-axis shows task enumeration in increments of 25.
    Shows 5 buckets: <25%, 25-50%, 50-75%, >75% but <100%, 100%.
    """
    print(f"\n📊 Creating percentage correctness distribution plot...")
    
    # Simple sorting: by total programs, then by 100%, then by >75%, etc.
    def sort_key(task_id):
        counts = task_correctness[task_id]
        total_programs = sum(counts)
        # Sort by: (total_programs desc, 100% desc, >75% desc, 50-75% desc, 25-50% desc, <25% desc)
        return (-total_programs, -counts[4], -counts[3], -counts[2], -counts[1], -counts[0])
    
    sorted_tasks = sorted(task_correctness.keys(), key=sort_key)
    
    if not sorted_tasks:
        print("❌ No tasks to plot")
        return
    
    # Prepare data for stacking (5 percentage buckets, highest percentage at bottom)
    num_buckets = 5
    stacked_data = np.zeros((num_buckets, len(sorted_tasks)))
    
    for task_idx, task_id in enumerate(sorted_tasks):
        bucket_counts = task_correctness[task_id]
        # Reverse the order so highest percentage (100%) is at bottom (index 0)
        # Original order: [<25%, 25-50%, 50-75%, >75% but <100%, 100%]
        # Stacking order: [100%, >75% but <100%, 50-75%, 25-50%, <25%]
        for bucket_idx in range(len(bucket_counts)):
            reversed_idx = len(bucket_counts) - 1 - bucket_idx
            stacked_data[reversed_idx, task_idx] = bucket_counts[bucket_idx]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Define colors and labels for stacking order (100% at bottom, <25% at top)
    # Stacking order: [100%, >75% but <100%, 50-75%, 25-50%, <25%]
    stacking_colors = ['#2e7d32', '#90a4ae', '#ffd54f', '#ffab40', '#ff6b6b']  # Dark Green -> Blue-Gray -> Yellow -> Orange -> Red
    stacking_labels = ['100%', '>75% but <100%', '50-75%', '25-50%', '<25%']
    
    # Create stacked bars
    bottom = np.zeros(len(sorted_tasks))
    bars = []
    
    for stack_idx in range(num_buckets):
        if np.any(stacked_data[stack_idx] > 0):  # Only plot if there's data
            color = stacking_colors[stack_idx]
            label = stacking_labels[stack_idx]
            bar = ax.bar(range(len(sorted_tasks)), stacked_data[stack_idx], 
                        bottom=bottom, label=label, 
                        color=color, alpha=0.8)
            bars.append(bar)
            bottom += stacked_data[stack_idx]
    
    # Customize the plot
    ax.set_xlabel('Task Index')
    ax.set_ylabel('Number of Programs')
    ax.set_title('Top 10 Program Correctness Distribution by Task\n(Sorted by total programs, then by 100% programs; 100% at bottom)', pad=20)
    
    # Set x-axis ticks in increments of 25
    total_tasks = len(sorted_tasks)
    tick_positions = list(range(0, total_tasks, 25))
    if total_tasks - 1 not in tick_positions:
        tick_positions.append(total_tasks - 1)
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(pos) for pos in tick_positions])
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_path}")
    
    # Show some statistics
    total_programs = sum(sum(counts) for counts in task_correctness.values())
    avg_programs_per_task = total_programs / len(sorted_tasks) if sorted_tasks else 0
    
    print(f"\n📈 Statistics:")
    print(f"  Total tasks: {len(sorted_tasks)}")
    print(f"  Total programs (top 10 per task): {total_programs}")
    print(f"  Average programs per task: {avg_programs_per_task:.1f}")


def upload_dataset_to_huggingface(df: pd.DataFrame, repo_name: str = "Trelis/arc-agi-2-partial-100-tricky-10") -> bool:
    """
    Upload the filtered dataset to Hugging Face Hub.
    
    Args:
        df: Filtered dataset DataFrame
        repo_name: Repository name on Hugging Face Hub
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from datasets import Dataset
        
        print(f"📤 Uploading dataset to {repo_name}...")
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_pandas(df)
        
        # Upload to Hub
        dataset.push_to_hub(repo_name)
        
        print(f"✅ Successfully uploaded dataset to {repo_name}")
        return True
        
    except ImportError:
        print("❌ Could not import datasets library for upload")
        return False
    except Exception as e:
        print(f"❌ Error uploading dataset: {e}")
        return False


def main():
    """Main execution function."""
    print("🚀 Starting Trelis/arc-agi-2-partial-100 analysis...")
    
    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Step 1: Download dataset
        df = download_dataset(cache_dir=str(output_dir / "hf_cache"))
        
        # Step 2: Check dataset structure
        check_dataset_structure(df)
        
        # Step 3: Filter tasks with max 10 programs with 100% correctness
        filtered_df = filter_tasks_with_all_correct_programs(df, max_programs=10)
        
        if len(filtered_df) == 0:
            print("❌ No tasks found matching criteria")
            return 1
        
        # Step 4: Apply deduplication AFTER filtering tasks
        print(f"\n🔧 Applying deduplication...")
        deduplicated_df, dedup_stats = deduplicate_by_task(filtered_df, task_column='task_id', code_column='code')
        
        # Find correctness columns for analysis
        train_correct_col = None
        test_correct_col = None
        
        for col in deduplicated_df.columns:
            if 'correct_train' in col.lower():
                train_correct_col = col
            elif 'correct_test' in col.lower():
                test_correct_col = col
        
        if not train_correct_col or not test_correct_col:
            print("❌ Could not find train/test correctness columns")
            return 1
        
        # Step 5: Analyze correctness distribution (this also filters to top 10 per task)
        task_correctness, final_filtered_df = analyze_correctness_distribution(deduplicated_df, train_correct_col, test_correct_col)
        
        # Step 6: Create plot
        plot_path = output_dir / "correctness_distribution.png"
        plot_correctness_distribution(task_correctness, str(plot_path))
        
        # Step 7: Save final filtered dataset locally
        filtered_csv_path = output_dir / "filtered_tricky_tasks.csv"
        final_filtered_df.to_csv(filtered_csv_path, index=False)
        print(f"✅ Final filtered dataset saved to: {filtered_csv_path}")
        
        # Step 8: Upload to Hugging Face if --upload flag is provided
        if len(sys.argv) > 1 and '--upload' in sys.argv:
            # Remove analysis columns before upload (keep original dataset format)
            upload_df = final_filtered_df.drop(columns=['percentage', 'code_length', 'percentage_correct', 'is_100_percent'], errors='ignore')
            upload_success = upload_dataset_to_huggingface(upload_df)
            if not upload_success:
                print("⚠️ Dataset upload failed, but local analysis completed successfully")
        else:
            print("⏭️ To upload dataset, run with --upload flag")
        
        print(f"\n🎉 Analysis complete! Check the output directory: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)