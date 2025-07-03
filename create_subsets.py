#!/usr/bin/env python3

import json
import os
from pathlib import Path

def calculate_task_size(task_data):
    """Calculate the total size of a task based on grid dimensions."""
    total_size = 0
    
    # Calculate size for training examples
    for example in task_data.get('train', []):
        input_grid = example.get('input', [])
        output_grid = example.get('output', [])
        total_size += len(input_grid) * len(input_grid[0]) if input_grid else 0
        total_size += len(output_grid) * len(output_grid[0]) if output_grid else 0
    
    # Calculate size for test examples
    for example in task_data.get('test', []):
        input_grid = example.get('input', [])
        output_grid = example.get('output', [])
        total_size += len(input_grid) * len(input_grid[0]) if input_grid else 0
        total_size += len(output_grid) * len(output_grid[0]) if output_grid else 0
    
    return total_size

def analyze_dataset(dataset_name, training_path, evaluation_path):
    """Analyze a dataset and create subset files."""
    print(f"\nAnalyzing {dataset_name}...")
    
    all_tasks = []
    
    # Analyze training tasks
    for filename in os.listdir(training_path):
        if filename.endswith('.json'):
            task_id = filename[:-5]  # Remove .json extension
            filepath = os.path.join(training_path, filename)
            
            try:
                with open(filepath, 'r') as f:
                    task_data = json.load(f)
                    size = calculate_task_size(task_data)
                    all_tasks.append({
                        'id': task_id,
                        'size': size,
                        'split': 'training',
                        'filename': filename
                    })
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Analyze evaluation tasks
    for filename in os.listdir(evaluation_path):
        if filename.endswith('.json'):
            task_id = filename[:-5]  # Remove .json extension
            filepath = os.path.join(evaluation_path, filename)
            
            try:
                with open(filepath, 'r') as f:
                    task_data = json.load(f)
                    size = calculate_task_size(task_data)
                    all_tasks.append({
                        'id': task_id,
                        'size': size,
                        'split': 'evaluation',
                        'filename': filename
                    })
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Sort tasks by size
    all_tasks.sort(key=lambda x: x['size'])
    
    print(f"Total tasks analyzed: {len(all_tasks)}")
    print(f"Smallest task size: {all_tasks[0]['size']} (ID: {all_tasks[0]['id']})")
    print(f"Largest task size: {all_tasks[-1]['size']} (ID: {all_tasks[-1]['id']})")
    
    # Create subset files
    subset_dir = f"data/subsets/{dataset_name}"
    
    # Top 1 shortest
    with open(f"{subset_dir}/shortest_1.txt", 'w') as f:
        for task in all_tasks[:1]:
            f.write(f"{task['id']}\n")
    
    # Top 10 shortest
    with open(f"{subset_dir}/shortest_10.txt", 'w') as f:
        for task in all_tasks[:10]:
            f.write(f"{task['id']}\n")
    
    # Top 100 shortest
    with open(f"{subset_dir}/shortest_100.txt", 'w') as f:
        for task in all_tasks[:100]:
            f.write(f"{task['id']}\n")
    
    # Also create JSON files with more details
    with open(f"{subset_dir}/shortest_1_details.json", 'w') as f:
        json.dump(all_tasks[:1], f, indent=2)
    
    with open(f"{subset_dir}/shortest_10_details.json", 'w') as f:
        json.dump(all_tasks[:10], f, indent=2)
    
    with open(f"{subset_dir}/shortest_100_details.json", 'w') as f:
        json.dump(all_tasks[:100], f, indent=2)
    
    print(f"Created subset files in {subset_dir}")
    
    return all_tasks

# Main execution
if __name__ == "__main__":
    # Analyze ARC-AGI-1
    arc1_tasks = analyze_dataset(
        "arc-agi-1",
        "data/arc-agi-1/training",
        "data/arc-agi-1/evaluation"
    )
    
    # Analyze ARC-AGI-2
    arc2_tasks = analyze_dataset(
        "arc-agi-2",
        "data/arc-agi-2/training",
        "data/arc-agi-2/evaluation"
    )
    
    print("\nSubset files created successfully!")
    print("\nFiles created:")
    print("- data/subsets/arc-agi-1/shortest_1.txt")
    print("- data/subsets/arc-agi-1/shortest_10.txt")
    print("- data/subsets/arc-agi-1/shortest_100.txt")
    print("- data/subsets/arc-agi-2/shortest_1.txt")
    print("- data/subsets/arc-agi-2/shortest_10.txt")
    print("- data/subsets/arc-agi-2/shortest_100.txt")
    print("\nAdditional JSON files with task details also created.")