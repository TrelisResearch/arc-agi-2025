#!/usr/bin/env python3

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add the parent directory to path to import our utils
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_python.utils.refinement_utils import _calculate_pixel_match_percentage
from llm_python.utils.program_compression import calculate_normalized_gzip_ratio


def load_ground_truth_task(task_id: str):
    """Load the ground truth ARC task data."""
    with open('data/arc-prize-2025/arc-agi_training_challenges.json', 'r') as f:
        training_challenges = json.load(f)

    with open('data/arc-prize-2025/arc-agi_training_solutions.json', 'r') as f:
        training_solutions = json.load(f)

    if task_id not in training_challenges:
        raise ValueError(f"Task {task_id} not found in training challenges")

    task_data = training_challenges[task_id]

    # Add solutions to test examples
    if task_id in training_solutions:
        test_solutions = training_solutions[task_id]
        for i, solution in enumerate(test_solutions):
            if i < len(task_data['test']):
                task_data['test'][i]['output'] = solution

    return task_data


def compute_metrics_for_task():
    """Compute pixel match and gzip metrics for all rows of the selected task."""

    print("📊 Loading task data...")

    # Load our selected task predictions
    df = pd.read_parquet('experimental/dataset_analysis/selected_task.parquet')
    task_id = df['task_id'].iloc[0]

    print(f"Task ID: {task_id}")
    print(f"Total rows: {len(df)}")

    # Load ground truth
    ground_truth = load_ground_truth_task(task_id)

    print(f"Ground truth - Train examples: {len(ground_truth['train'])}, Test examples: {len(ground_truth['test'])}")

    # Prepare results
    results = []

    for idx, row in df.iterrows():
        print(f"Processing row {idx+1}/{len(df)}: {row['row_id']}")

        row_results = {
            'row_id': row['row_id'],
            'model': row['model'],
            'is_transductive': row['is_transductive']
        }

        # Recompute correctness based on pixel-perfect matching (ignore dataset labels)
        train_correct_count = 0
        test_correct_count = 0
        train_total = len(ground_truth['train'])
        test_total = len(ground_truth['test'])

        # Calculate pixel matches for individual examples
        pixel_matches = []

        # Collect training inputs/outputs for normalized gzip calculation (training only)
        train_inputs = []
        train_ground_truth_outputs = []
        train_predicted_outputs = []

        # Process training examples
        predicted_train = row['predicted_train_output']
        for i, gt_example in enumerate(ground_truth['train']):
            if i < len(predicted_train):
                predicted = predicted_train[i]
                ground_truth_output = gt_example['output']

                # Convert predicted grid format if needed
                if hasattr(predicted, 'tolist'):
                    predicted = predicted.tolist()

                # Calculate pixel match (0 for wrong size grids)
                pixel_match = _calculate_pixel_match_percentage(predicted, ground_truth_output)
                pixel_matches.append(pixel_match)

                # Check if this example is pixel-perfect correct (1.0 pixel match)
                if pixel_match == 1.0:
                    train_correct_count += 1

                # Collect for normalized gzip calculation (training examples only)
                train_inputs.append(gt_example['input'])
                train_ground_truth_outputs.append(ground_truth_output)
                train_predicted_outputs.append(predicted)

        # Process test examples
        predicted_test = row['predicted_test_output']
        for i, gt_example in enumerate(ground_truth['test']):
            if i < len(predicted_test):
                predicted = predicted_test[i]
                ground_truth_output = gt_example['output']

                # Convert predicted grid format if needed
                if hasattr(predicted, 'tolist'):
                    predicted = predicted.tolist()

                # Calculate pixel match (0 for wrong size grids)
                pixel_match = _calculate_pixel_match_percentage(predicted, ground_truth_output)
                pixel_matches.append(pixel_match)

                # Check if this example is pixel-perfect correct (1.0 pixel match)
                if pixel_match == 1.0:
                    test_correct_count += 1

                # Note: Test examples not included in gzip calculation

        # Calculate normalized gzip ratio using training examples only
        normalized_gzip_ratio = calculate_normalized_gzip_ratio(
            train_inputs, train_ground_truth_outputs, train_predicted_outputs
        )

        # Calculate overall correctness based on pixel-perfect matching
        total_correct = train_correct_count + test_correct_count
        total_examples = train_total + test_total
        overall_correctness = total_correct / total_examples if total_examples > 0 else 0.0

        # Store results
        row_results['overall_correctness'] = overall_correctness
        row_results['train_correct_count'] = train_correct_count
        row_results['train_total_count'] = train_total
        row_results['test_correct_count'] = test_correct_count
        row_results['test_total_count'] = test_total
        row_results['avg_pixel_match'] = np.mean(pixel_matches) if pixel_matches else 0.0
        row_results['normalized_gzip_ratio'] = normalized_gzip_ratio
        row_results['num_examples_processed'] = len(pixel_matches)

        results.append(row_results)

    return pd.DataFrame(results)


def create_plots(results_df: pd.DataFrame):
    """Create the requested plots."""

    print("📈 Creating plots...")

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Analysis of Task {results_df.iloc[0]["row_id"].split("_")[0] if "_" in results_df.iloc[0]["row_id"] else "Unknown"} (n={len(results_df)})', fontsize=16)

    # 1. Pixel Match vs Normalized Gzip Ratio
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(results_df['normalized_gzip_ratio'], results_df['avg_pixel_match'],
                          c=results_df['overall_correctness'], cmap='RdYlBu_r',
                          s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Normalized Gzip Ratio (Pred/GT)')
    ax1.set_ylabel('Average Pixel Match Score')
    ax1.set_title('Pixel Match vs Normalized Gzip Ratio')
    ax1.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Equal compressibility')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Overall Correctness')

    # 2. Pixel Match vs Correctness
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(results_df['overall_correctness'], results_df['avg_pixel_match'],
                          c=results_df['normalized_gzip_ratio'], cmap='viridis',
                          s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax2.set_xlabel('Overall Correctness')
    ax2.set_ylabel('Average Pixel Match Score')
    ax2.set_title('Pixel Match vs Correctness')
    ax2.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Normalized Gzip Ratio')

    # 3. Normalized Gzip Ratio vs Correctness
    ax3 = axes[1, 0]
    scatter3 = ax3.scatter(results_df['overall_correctness'], results_df['normalized_gzip_ratio'],
                          c=results_df['avg_pixel_match'], cmap='plasma',
                          s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax3.set_xlabel('Overall Correctness')
    ax3.set_ylabel('Normalized Gzip Ratio (Pred/GT)')
    ax3.set_title('Normalized Gzip Ratio vs Correctness')
    ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Equal compressibility')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('Avg Pixel Match')

    # 4. Distribution plot
    ax4 = axes[1, 1]
    ax4.hist(results_df['overall_correctness'], bins=20, alpha=0.6, color='skyblue', edgecolor='black')
    ax4.axvline(results_df['overall_correctness'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {results_df["overall_correctness"].mean():.2f}')
    ax4.set_xlabel('Overall Correctness')
    ax4.set_ylabel('Count')
    ax4.set_title('Distribution of Overall Correctness')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = 'experimental/dataset_analysis/analysis_plots.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"📊 Plots saved to: {output_path}")
    plt.show()

    # Print summary statistics
    print("\n📋 Summary Statistics:")
    print(f"Overall Correctness: {results_df['overall_correctness'].mean():.3f} ± {results_df['overall_correctness'].std():.3f}")
    print(f"Avg Pixel Match: {results_df['avg_pixel_match'].mean():.3f} ± {results_df['avg_pixel_match'].std():.3f}")
    print(f"Normalized Gzip Ratio: {results_df['normalized_gzip_ratio'].mean():.3f} ± {results_df['normalized_gzip_ratio'].std():.3f}")

    # Correlations
    print("\n📊 Correlations:")
    corr_pixel_correctness = results_df['avg_pixel_match'].corr(results_df['overall_correctness'])
    corr_gzip_correctness = results_df['normalized_gzip_ratio'].corr(results_df['overall_correctness'])
    corr_pixel_gzip = results_df['avg_pixel_match'].corr(results_df['normalized_gzip_ratio'])

    print(f"Pixel Match ↔ Correctness: {corr_pixel_correctness:.3f}")
    print(f"Gzip Ratio ↔ Correctness: {corr_gzip_correctness:.3f}")
    print(f"Pixel Match ↔ Gzip Ratio: {corr_pixel_gzip:.3f}")


if __name__ == "__main__":
    try:
        # Compute metrics
        results_df = compute_metrics_for_task()

        # Save results
        results_path = 'experimental/dataset_analysis/computed_metrics.parquet'
        results_df.to_parquet(results_path, index=False)
        print(f"💾 Results saved to: {results_path}")

        # Create plots
        create_plots(results_df)

        print("\n✅ Analysis complete!")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()