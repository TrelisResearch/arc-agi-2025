#!/usr/bin/env python3

import json
import os
from collections import defaultdict

def analyze_tasks(dataset_name, training_path, evaluation_path):
    """Analyze task structure and find tasks with multiple test outputs."""
    print(f"\nAnalyzing {dataset_name}...")
    
    stats = {
        'total_tasks': 0,
        'tasks_with_multiple_tests': [],
        'train_examples_distribution': defaultdict(int),
        'test_examples_distribution': defaultdict(int),
        'max_train_examples': 0,
        'max_test_examples': 0,
        'grid_size_stats': {
            'min_width': float('inf'),
            'max_width': 0,
            'min_height': float('inf'),
            'max_height': 0
        }
    }
    
    # Function to analyze a single directory
    def analyze_directory(directory_path, split_name):
        for filename in os.listdir(directory_path):
            if filename.endswith('.json'):
                task_id = filename[:-5]  # Remove .json extension
                filepath = os.path.join(directory_path, filename)
                
                try:
                    with open(filepath, 'r') as f:
                        task_data = json.load(f)
                        
                        stats['total_tasks'] += 1
                        
                        # Analyze train examples
                        train_count = len(task_data.get('train', []))
                        stats['train_examples_distribution'][train_count] += 1
                        stats['max_train_examples'] = max(stats['max_train_examples'], train_count)
                        
                        # Analyze test examples
                        test_count = len(task_data.get('test', []))
                        stats['test_examples_distribution'][test_count] += 1
                        stats['max_test_examples'] = max(stats['max_test_examples'], test_count)
                        
                        # Track tasks with multiple test outputs
                        if test_count > 1:
                            stats['tasks_with_multiple_tests'].append({
                                'id': task_id,
                                'test_count': test_count,
                                'split': split_name
                            })
                        
                        # Analyze grid sizes
                        for example_type in ['train', 'test']:
                            for example in task_data.get(example_type, []):
                                for grid_type in ['input', 'output']:
                                    grid = example.get(grid_type, [])
                                    if grid:
                                        height = len(grid)
                                        width = len(grid[0]) if grid[0] else 0
                                        
                                        stats['grid_size_stats']['min_height'] = min(stats['grid_size_stats']['min_height'], height)
                                        stats['grid_size_stats']['max_height'] = max(stats['grid_size_stats']['max_height'], height)
                                        stats['grid_size_stats']['min_width'] = min(stats['grid_size_stats']['min_width'], width)
                                        stats['grid_size_stats']['max_width'] = max(stats['grid_size_stats']['max_width'], width)
                
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    
    # Analyze both training and evaluation directories
    analyze_directory(training_path, 'training')
    analyze_directory(evaluation_path, 'evaluation')
    
    # Sort tasks with multiple tests by test count
    stats['tasks_with_multiple_tests'].sort(key=lambda x: x['test_count'], reverse=True)
    
    return stats

def main():
    # Analyze both datasets
    arc1_stats = analyze_tasks(
        "ARC-AGI-1",
        "data/arc-agi-1/training",
        "data/arc-agi-1/evaluation"
    )
    
    arc2_stats = analyze_tasks(
        "ARC-AGI-2",
        "data/arc-agi-2/training",
        "data/arc-agi-2/evaluation"
    )
    
    # Print results
    for dataset, stats in [("ARC-AGI-1", arc1_stats), ("ARC-AGI-2", arc2_stats)]:
        print(f"\n{'='*50}")
        print(f"{dataset} Analysis")
        print(f"{'='*50}")
        print(f"Total tasks: {stats['total_tasks']}")
        print(f"\nTrain examples distribution:")
        for count in sorted(stats['train_examples_distribution'].keys()):
            print(f"  {count} examples: {stats['train_examples_distribution'][count]} tasks")
        
        print(f"\nTest examples distribution:")
        for count in sorted(stats['test_examples_distribution'].keys()):
            print(f"  {count} examples: {stats['test_examples_distribution'][count]} tasks")
        
        print(f"\nGrid size ranges:")
        print(f"  Width: {stats['grid_size_stats']['min_width']} - {stats['grid_size_stats']['max_width']}")
        print(f"  Height: {stats['grid_size_stats']['min_height']} - {stats['grid_size_stats']['max_height']}")
        
        print(f"\nTasks with multiple test examples: {len(stats['tasks_with_multiple_tests'])}")
        if stats['tasks_with_multiple_tests']:
            print("Task IDs with multiple test examples:")
            for task in stats['tasks_with_multiple_tests'][:10]:  # Show first 10
                print(f"  {task['id']} ({task['split']}): {task['test_count']} test examples")
            if len(stats['tasks_with_multiple_tests']) > 10:
                print(f"  ... and {len(stats['tasks_with_multiple_tests']) - 10} more")
        
        # Save full list of tasks with multiple tests
        output_file = f"data/subsets/{dataset.lower()}/tasks_with_multiple_tests.json"
        with open(output_file, 'w') as f:
            json.dump(stats['tasks_with_multiple_tests'], f, indent=2)
        print(f"\nFull list saved to: {output_file}")

if __name__ == "__main__":
    main()