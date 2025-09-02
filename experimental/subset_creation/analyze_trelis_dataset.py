"""
Analyze the Trelis/arc-agi-2-partial-100 dataset to find which training-hard tasks
have at least one program that gets all train AND test examples correct.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets import load_dataset
import json

def main():
    # Load the training-hard tasks
    hard_tasks_path = Path("data/subsets/arc-prize-2025/training-hard.txt")
    with open(hard_tasks_path, 'r') as f:
        hard_tasks = set(line.strip() for line in f)
    
    print(f"Loaded {len(hard_tasks)} training-hard tasks")
    
    # Load the Trelis dataset
    print("\nLoading Trelis/arc-agi-2-partial-100 dataset...")
    dataset = load_dataset("Trelis/arc-agi-2-partial-100", split="train")
    
    print(f"Dataset has {len(dataset)} rows")
    print(f"Columns: {dataset.column_names}")
    
    # Analyze which tasks have perfect solutions
    task_perfect_solutions = {}
    
    for idx, row in enumerate(dataset):
        task_id = row['task_id']
        
        # Only analyze if it's in our hard tasks
        if task_id not in hard_tasks:
            continue
        
        # Check if both train and test are correct
        correct_train = row['correct_train_input']
        correct_test = row['correct_test_input']
        
        # Check if all train examples are correct
        all_train_correct = all(correct_train) if isinstance(correct_train, list) else correct_train
        
        # Check if all test examples are correct  
        all_test_correct = all(correct_test) if isinstance(correct_test, list) else correct_test
        
        # Track perfect solutions (all train AND all test correct)
        if all_train_correct and all_test_correct:
            if task_id not in task_perfect_solutions:
                task_perfect_solutions[task_id] = 0
            task_perfect_solutions[task_id] += 1
    
    # Sort tasks by number of perfect solutions
    sorted_tasks = sorted(task_perfect_solutions.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n=== Results ===")
    print(f"Total training-hard tasks: {len(hard_tasks)}")
    print(f"Tasks with at least one perfect solution (train AND test): {len(task_perfect_solutions)}")
    print(f"Tasks with NO perfect solution: {len(hard_tasks - set(task_perfect_solutions.keys()))}")
    
    if sorted_tasks:
        print(f"\n=== Tasks with perfect solutions ===")
        for task_id, count in sorted_tasks[:20]:  # Show top 20
            print(f"  {task_id}: {count} perfect solution(s)")
        
        if len(sorted_tasks) > 20:
            print(f"  ... and {len(sorted_tasks) - 20} more tasks")
    
    # Create a new subset excluding tasks with perfect solutions
    ultra_hard_tasks = hard_tasks - set(task_perfect_solutions.keys())
    ultra_hard_tasks = sorted(list(ultra_hard_tasks))
    
    print(f"\n=== Creating ultra-hard subset ===")
    print(f"Tasks with NO perfect solution in Trelis dataset: {len(ultra_hard_tasks)}")
    
    # Save the ultra-hard subset
    output_dir = Path("experimental/subset_creation")
    output_file = output_dir / "training-ultra-hard.txt"
    with open(output_file, 'w') as f:
        for task_id in ultra_hard_tasks:
            f.write(f"{task_id}\n")
    print(f"Saved to: {output_file}")
    
    # Also save details
    details = {
        "source": "training-hard subset filtered by Trelis/arc-agi-2-partial-100",
        "description": "Tasks from training-hard that have NO perfect solutions (all train AND test correct) in Trelis dataset",
        "original_hard_tasks": len(hard_tasks),
        "tasks_with_perfect_solutions": len(task_perfect_solutions),
        "ultra_hard_tasks": len(ultra_hard_tasks),
        "task_ids": ultra_hard_tasks
    }
    
    details_file = output_dir / "training-ultra-hard_details.json"
    with open(details_file, 'w') as f:
        json.dump(details, f, indent=2)
    print(f"Saved details to: {details_file}")
    
    # Also save in standard location
    standard_dir = Path("data/subsets/arc-prize-2025")
    std_file = standard_dir / "training-ultra-hard.txt"
    with open(std_file, 'w') as f:
        for task_id in ultra_hard_tasks:
            f.write(f"{task_id}\n")
    print(f"Also saved to: {std_file}")
    
    std_details_file = standard_dir / "training-ultra-hard_details.json"
    with open(std_details_file, 'w') as f:
        json.dump(details, f, indent=2)
    print(f"Saved details to: {std_details_file}")

if __name__ == "__main__":
    main()