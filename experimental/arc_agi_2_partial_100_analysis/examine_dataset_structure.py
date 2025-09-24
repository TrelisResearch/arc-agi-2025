#!/usr/bin/env python3

from datasets import load_from_disk
from collections import Counter
import pandas as pd

def examine_dataset_structure():
    """Examine the actual structure of the dataset"""
    print("Loading dataset...")
    dataset = load_from_disk("./arc_agi_2_partial_100_data")
    train_data = dataset['train']

    print(f"Total rows: {len(train_data)}")
    print(f"Columns: {train_data.column_names}")

    # Look at first few examples
    print("\n=== First 5 Examples ===")
    for i in range(min(5, len(train_data))):
        example = train_data[i]
        print(f"\nExample {i+1}:")
        print(f"  task_id: {example['task_id']}")
        print(f"  code length: {len(example['code']) if example['code'] else 0}")
        print(f"  reasoning length: {len(example['reasoning']) if example['reasoning'] else 0}")
        if example['code']:
            # Show first few lines of code
            code_lines = example['code'].split('\n')[:3]
            print(f"  code preview: {code_lines}")

    # Count task_ids to see how many programs per task
    task_ids = [example['task_id'] for example in train_data]
    task_id_counts = Counter(task_ids)

    print(f"\n=== Task ID Analysis ===")
    print(f"Unique task IDs: {len(task_id_counts)}")
    print(f"Total entries: {len(task_ids)}")
    print(f"Average programs per task: {len(task_ids) / len(task_id_counts):.2f}")

    # Show distribution of programs per task
    count_distribution = Counter(task_id_counts.values())
    print(f"\nPrograms per task distribution:")
    for num_programs, num_tasks in sorted(count_distribution.items()):
        print(f"  {num_programs} programs: {num_tasks} tasks")

    # Show some examples of tasks with multiple programs
    print(f"\n=== Examples of Tasks with Multiple Programs ===")
    multi_program_tasks = [(task_id, count) for task_id, count in task_id_counts.items() if count > 1]

    for i, (task_id, count) in enumerate(multi_program_tasks[:3]):
        print(f"\nTask {task_id} has {count} programs:")
        task_examples = [ex for ex in train_data if ex['task_id'] == task_id]
        for j, example in enumerate(task_examples[:3]):  # Show first 3 programs for this task
            code_preview = example['code'][:100] + "..." if len(example['code']) > 100 else example['code']
            print(f"  Program {j+1}: {code_preview}")

if __name__ == "__main__":
    examine_dataset_structure()