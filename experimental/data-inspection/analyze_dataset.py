#!/usr/bin/env python3
"""
Script to download and analyze the arc-programs-50-full-200-partial dataset from HuggingFace.
Generates plots of task difficulty vs program length metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import ast

def download_dataset():
    """Download the dataset from HuggingFace."""
    print("Downloading arc-programs-50-full-200-partial dataset...")
    dataset = load_dataset("Trelis/arc-programs-50-full-200-partial", split="train")
    df = pd.DataFrame(dataset)
    print(f"Dataset downloaded successfully. Shape: {df.shape}")
    return df

def calculate_percentage_correct(df):
    """Calculate percentage correct for each row."""
    print("Calculating percentage correct for each row...")
    
    def count_true_values(lst):
        """Count True values in a list."""
        if isinstance(lst, list):
            return sum(1 for x in lst if x is True)
        return 0
    
    # Calculate correct counts for train and test
    df['correct_train_count'] = df['correct_train_input'].apply(count_true_values)
    df['correct_test_count'] = df['correct_test_input'].apply(count_true_values)
    
    # Calculate total correct and total possible
    df['total_correct'] = df['correct_train_count'] + df['correct_test_count']
    df['total_train_len'] = df['correct_train_input'].apply(len)
    df['total_test_len'] = df['correct_test_input'].apply(len)
    df['total_possible'] = df['total_train_len'] + df['total_test_len']
    
    # Calculate percentage correct
    df['percentage_correct'] = (df['total_correct'] / df['total_possible']) * 100
    
    return df

def calculate_program_length(df):
    """Calculate program length for each row."""
    print("Calculating program length for each row...")
    
    def get_code_length(code):
        """Get length of code string."""
        if isinstance(code, str):
            return len(code)
        return 0
    
    df['program_length'] = df['code'].apply(get_code_length)
    return df

def get_min_length_from_best_correctness(group):
    """Get minimum program length from the highest correctness programs in a task."""
    # Sort by percentage_correct descending, then by program_length ascending
    sorted_group = group.sort_values(['percentage_correct', 'program_length'], 
                                   ascending=[False, True])
    
    # Start with the highest correctness and work down
    max_correctness = sorted_group['percentage_correct'].max()
    
    # Find programs with the highest correctness
    best_programs = sorted_group[sorted_group['percentage_correct'] == max_correctness]
    
    if len(best_programs) > 0:
        return best_programs['program_length'].min()
    else:
        # Fallback to overall minimum (shouldn't happen)
        return group['program_length'].min()

def calculate_task_metrics(df):
    """Calculate task-level metrics."""
    print("Calculating task-level metrics...")
    
    # Group by task_id and calculate basic metrics
    task_metrics = df.groupby('task_id').agg({
        'percentage_correct': 'mean',  # Task difficulty (average % correct)
        'program_length': ['mean', 'max', 'count']
    }).round(2)
    
    # Flatten column names
    task_metrics.columns = ['task_difficulty', 'avg_program_length', 
                           'max_program_length', 'num_programs']
    
    # Calculate min program length from highest correctness programs
    print("Calculating minimum program length from highest correctness programs...")
    min_lengths = df.groupby('task_id').apply(get_min_length_from_best_correctness)
    task_metrics['min_program_length'] = min_lengths.values
    
    # Calculate ratio of mean to reference min program length
    task_metrics['length_ratio'] = (task_metrics['avg_program_length'] / 
                                   task_metrics['min_program_length']).replace([np.inf, -np.inf], np.nan)
    
    # Reset index to make task_id a column
    task_metrics = task_metrics.reset_index()
    
    print(f"Calculated metrics for {len(task_metrics)} tasks")
    return task_metrics

def generate_plots(task_metrics):
    """Generate the requested plots."""
    print("Generating plots...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Task difficulty vs Average program length
    scatter1 = ax1.scatter(task_metrics['task_difficulty'], task_metrics['avg_program_length'], 
                          alpha=0.6, s=60)
    ax1.set_xlabel('Task Difficulty (Average % Correct)')
    ax1.set_ylabel('Average Program Length')
    ax1.set_title('Task Difficulty vs Average Program Length')
    ax1.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    corr1 = task_metrics['task_difficulty'].corr(task_metrics['avg_program_length'])
    ax1.text(0.05, 0.95, f'Correlation: {corr1:.3f}', transform=ax1.transAxes, 
             bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    
    # Plot 2: Task difficulty vs Ratio of mean to reference min program length
    # Filter out NaN and infinite values
    valid_mask = np.isfinite(task_metrics['length_ratio'])
    valid_data = task_metrics[valid_mask]
    
    scatter2 = ax2.scatter(valid_data['task_difficulty'], valid_data['length_ratio'], 
                          alpha=0.6, s=60, color='orange')
    ax2.set_xlabel('Task Difficulty (Average % Correct)')
    ax2.set_ylabel('Ratio of Mean to Reference Min Program Length')
    ax2.set_title('Task Difficulty vs Mean/Min Program Length Ratio')
    ax2.grid(True, alpha=0.3)
    
    # Add correlation coefficient for valid data
    corr2 = valid_data['task_difficulty'].corr(valid_data['length_ratio'])
    ax2.text(0.05, 0.95, f'Correlation: {corr2:.3f}', transform=ax2.transAxes, 
             bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    # Save the plots
    plt.savefig('task_analysis_plots.png', dpi=300, bbox_inches='tight')
    print("Plots saved to 'task_analysis_plots.png'")
    
    # Don't show plots in headless environment, just save
    # plt.show()

def print_summary_stats(df, task_metrics):
    """Print summary statistics."""
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    print(f"\nDataset Overview:")
    print(f"  Total rows: {len(df)}")
    print(f"  Unique tasks: {df['task_id'].nunique()}")
    print(f"  Average programs per task: {len(df) / df['task_id'].nunique():.1f}")
    
    print(f"\nPercentage Correct Statistics:")
    print(f"  Mean: {df['percentage_correct'].mean():.2f}%")
    print(f"  Median: {df['percentage_correct'].median():.2f}%")
    print(f"  Std: {df['percentage_correct'].std():.2f}%")
    print(f"  Min: {df['percentage_correct'].min():.2f}%")
    print(f"  Max: {df['percentage_correct'].max():.2f}%")
    
    print(f"\nProgram Length Statistics:")
    print(f"  Mean: {df['program_length'].mean():.0f} characters")
    print(f"  Median: {df['program_length'].median():.0f} characters")
    print(f"  Std: {df['program_length'].std():.0f} characters")
    print(f"  Min: {df['program_length'].min()}")
    print(f"  Max: {df['program_length'].max()}")
    
    print(f"\nTask Difficulty Statistics:")
    print(f"  Mean: {task_metrics['task_difficulty'].mean():.2f}%")
    print(f"  Median: {task_metrics['task_difficulty'].median():.2f}%")
    print(f"  Std: {task_metrics['task_difficulty'].std():.2f}%")
    print(f"  Min: {task_metrics['task_difficulty'].min():.2f}%")
    print(f"  Max: {task_metrics['task_difficulty'].max():.2f}%")
    
    valid_ratios = task_metrics['length_ratio'].dropna()
    if len(valid_ratios) > 0:
        print(f"\nMean/Min Length Ratio Statistics:")
        print(f"  Mean: {valid_ratios.mean():.2f}")
        print(f"  Median: {valid_ratios.median():.2f}")
        print(f"  Std: {valid_ratios.std():.2f}")
        print(f"  Min: {valid_ratios.min():.2f}")
        print(f"  Max: {valid_ratios.max():.2f}")

def main():
    """Main analysis pipeline."""
    try:
        # Download dataset
        df = download_dataset()
        
        # Calculate metrics
        df = calculate_percentage_correct(df)
        df = calculate_program_length(df)
        task_metrics = calculate_task_metrics(df)
        
        # Generate plots
        generate_plots(task_metrics)
        
        # Print summary statistics
        print_summary_stats(df, task_metrics)
        
        # Save processed data
        df.to_csv('processed_data.csv', index=False)
        task_metrics.to_csv('task_metrics.csv', index=False)
        print(f"\nProcessed data saved to 'processed_data.csv'")
        print(f"Task metrics saved to 'task_metrics.csv'")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()