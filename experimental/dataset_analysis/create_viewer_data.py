#!/usr/bin/env python3

import json
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_python.utils.refinement_utils import _calculate_pixel_match_percentage
from llm_python.utils.compression_utils import calculate_combined_gzip_ratio

def convert_to_lists(obj):
    """Recursively convert numpy arrays to lists for JSON serialization."""
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_lists(item) for item in obj]
    else:
        return obj

def create_viewer_data():
    """Create JSON data file for the web viewer."""

    print("📊 Loading data for web viewer...")

    # Load our analysis results
    results_df = pd.read_parquet('experimental/dataset_analysis/computed_metrics.parquet')
    task_df = pd.read_parquet('experimental/dataset_analysis/selected_task.parquet')

    # Load ground truth
    with open('data/arc-prize-2025/arc-agi_training_challenges.json', 'r') as f:
        challenges = json.load(f)
    with open('data/arc-prize-2025/arc-agi_training_solutions.json', 'r') as f:
        solutions = json.load(f)

    task_id = 'b2bc3ffd'
    ground_truth = challenges[task_id]

    # Add test solutions
    if task_id in solutions:
        for i, solution in enumerate(solutions[task_id]):
            if i < len(ground_truth['test']):
                ground_truth['test'][i]['output'] = solution

    print(f"Ground truth: {len(ground_truth['train'])} train, {len(ground_truth['test'])} test examples")

    # Create viewer data
    viewer_data = {
        'task_id': task_id,
        'task_info': {
            'train_examples': len(ground_truth['train']),
            'test_examples': len(ground_truth['test']),
            'total_rows': len(results_df)
        },
        'ground_truth': ground_truth,
        'rows': []
    }

    # Process each row
    for idx, (_, result_row) in enumerate(results_df.iterrows()):
        row_id = result_row['row_id']

        # Find corresponding task row
        task_row = task_df[task_df['row_id'] == row_id].iloc[0]

        print(f"Processing row {idx+1}/{len(results_df)}: {row_id[:8]}...")

        # Calculate detailed pixel matches for each example
        example_details = []

        # Process training examples
        predicted_train = task_row['predicted_train_output']
        for i, gt_example in enumerate(ground_truth['train']):
            if i < len(predicted_train):
                predicted = predicted_train[i]
                if hasattr(predicted, 'tolist'):
                    predicted = predicted.tolist()

                ground_truth_output = gt_example['output']

                # Ensure all nested numpy arrays are converted to lists
                predicted = convert_to_lists(predicted)
                pixel_match = _calculate_pixel_match_percentage(predicted, ground_truth_output)

                example_details.append({
                    'type': 'train',
                    'example_id': i,
                    'input_grid': gt_example['input'],
                    'expected_output': ground_truth_output,
                    'predicted_output': predicted,
                    'pixel_match': pixel_match,
                    'is_correct': pixel_match == 1.0
                })

        # Process test examples
        predicted_test = task_row['predicted_test_output']
        for i, gt_example in enumerate(ground_truth['test']):
            if i < len(predicted_test):
                predicted = predicted_test[i]
                if hasattr(predicted, 'tolist'):
                    predicted = predicted.tolist()

                ground_truth_output = gt_example['output']

                # Ensure all nested numpy arrays are converted to lists
                predicted = convert_to_lists(predicted)
                pixel_match = _calculate_pixel_match_percentage(predicted, ground_truth_output)

                example_details.append({
                    'type': 'test',
                    'example_id': i,
                    'input_grid': gt_example['input'],
                    'expected_output': ground_truth_output,
                    'predicted_output': predicted,
                    'pixel_match': pixel_match,
                    'is_correct': pixel_match == 1.0
                })

        # Add row data
        row_data = {
            'row_id': row_id,
            'model': task_row['model'],
            'is_transductive': bool(task_row['is_transductive']),
            'overall_correctness': float(result_row['overall_correctness']),
            'avg_pixel_match': float(result_row['avg_pixel_match']),
            'normalized_gzip_ratio': float(result_row['normalized_gzip_ratio']),
            'train_correct_count': int(result_row['train_correct_count']),
            'train_total_count': int(result_row['train_total_count']),
            'test_correct_count': int(result_row['test_correct_count']),
            'test_total_count': int(result_row['test_total_count']),
            'examples': example_details,
            'code': task_row['code'] if 'code' in task_row else None
        }

        viewer_data['rows'].append(row_data)

    # Save viewer data
    output_path = 'experimental/dataset_analysis/viewer_data.json'
    with open(output_path, 'w') as f:
        json.dump(viewer_data, f, indent=2)

    print(f"✅ Viewer data saved to: {output_path}")
    print(f"📊 Created data for {len(viewer_data['rows'])} rows")

    return output_path

if __name__ == "__main__":
    create_viewer_data()