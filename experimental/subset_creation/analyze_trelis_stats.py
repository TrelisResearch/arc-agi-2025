"""
Analyze the Trelis/arc-agi-2-partial-100 dataset to get statistics on training-hard tasks.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets import load_dataset

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
    
    # Track statistics
    tasks_with_perfect_solution = set()  # All train AND test correct
    tasks_with_any_train_correct = set()  # At least one train example correct
    tasks_with_all_train_correct = set()  # All train examples correct
    
    for row in dataset:
        task_id = row['task_id']
        
        # Only analyze if it's in our hard tasks
        if task_id not in hard_tasks:
            continue
        
        correct_train = row['correct_train_input']
        correct_test = row['correct_test_input']
        
        # Check train correctness
        all_train_correct = all(correct_train) if isinstance(correct_train, list) else correct_train
        any_train_correct = any(correct_train) if isinstance(correct_train, list) else correct_train
        
        # Check test correctness
        all_test_correct = all(correct_test) if isinstance(correct_test, list) else correct_test
        
        # Track statistics
        if any_train_correct:
            tasks_with_any_train_correct.add(task_id)
        
        if all_train_correct:
            tasks_with_all_train_correct.add(task_id)
            
            # Perfect solution: all train AND all test correct
            if all_test_correct:
                tasks_with_perfect_solution.add(task_id)
    
    print(f"\n=== Statistics for training-hard tasks in Trelis dataset ===")
    print(f"Total training-hard tasks: {len(hard_tasks)}")
    print(f"\nTasks with at least ONE train example correct: {len(tasks_with_any_train_correct)}/{len(hard_tasks)}")
    print(f"Tasks with ALL train examples correct: {len(tasks_with_all_train_correct)}/{len(hard_tasks)}")
    print(f"Tasks with perfect solution (all train AND all test correct): {len(tasks_with_perfect_solution)}/{len(hard_tasks)}")
    
    print(f"\nTasks with NO train examples correct: {len(hard_tasks - tasks_with_any_train_correct)}/{len(hard_tasks)}")
    print(f"Tasks without all train correct: {len(hard_tasks - tasks_with_all_train_correct)}/{len(hard_tasks)}")
    print(f"Tasks without perfect solution: {len(hard_tasks - tasks_with_perfect_solution)}/{len(hard_tasks)}")

if __name__ == "__main__":
    main()