"""
Create a new subset from gpt-5-nano test results, excluding tasks that got all train correct.
"""

import sys
import json
from pathlib import Path

# Add parent directories to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_python.datasets.io import read_soar_parquet
import pandas as pd

def main():
    # Read the parquet file
    parquet_path = Path("/Users/ronanmcgovern/TR/arc-agi-2025/llm_python/datasets/inference/20250901_214154_gpt-5-nano_arc-prize-2025_training.parquet")
    
    print(f"Reading parquet file: {parquet_path}")
    df = read_soar_parquet(parquet_path)
    
    print(f"Total rows in parquet: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check if 'correct_train_input' contains at least one True value for each row
    # This indicates at least one train example was correctly predicted
    any_train_correct_list = []
    for idx in range(len(df)):
        val = df.iloc[idx]['correct_train_input']
        result = isinstance(val, list) and bool(val) and any(val)
        any_train_correct_list.append(result)
    
    df['any_train_correct'] = any_train_correct_list
    
    # Group by task_id and check if any program got any_train_correct
    task_stats = df.groupby('task_id').agg({
        'any_train_correct': 'max',  # True if any program got at least one train correct
        'row_id': 'count'  # Number of programs per task
    }).rename(columns={'row_id': 'num_programs'})
    
    print(f"\nTotal unique tasks: {len(task_stats)}")
    
    # Find tasks where NO program got any_train_correct (completely failed tasks)
    hard_tasks = task_stats[task_stats['any_train_correct'] == False].index.tolist()
    
    print(f"Tasks where NO program got any train correct: {len(hard_tasks)}")
    print(f"Tasks where at least one program got min 1 train correct: {len(task_stats) - len(hard_tasks)}")
    
    # Sort task IDs for consistent ordering
    hard_tasks.sort()
    
    # Create subset files in the experimental folder
    output_dir = Path("experimental/subset_creation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create text file with just task IDs (one per line)
    txt_file = output_dir / "training-gpt-5-nano-hard.txt"
    with open(txt_file, 'w') as f:
        for task_id in hard_tasks:
            f.write(f"{task_id}\n")
    print(f"\nCreated: {txt_file}")
    print(f"  Contains {len(hard_tasks)} task IDs")
    
    # Create details JSON file with metadata
    details = {
        "source": "20250901_214154_gpt-5-nano_arc-prize-2025_training.parquet",
        "description": "Tasks from arc-prize-2025 training set where gpt-5-nano got ZERO train examples correct in all attempts",
        "total_tasks": len(hard_tasks),
        "task_ids": hard_tasks
    }
    
    json_file = output_dir / "training-gpt-5-nano-hard_details.json"
    with open(json_file, 'w') as f:
        json.dump(details, f, indent=2)
    print(f"Created: {json_file}")
    
    # Also create subset files in the standard location
    standard_dir = Path("data/subsets/arc-prize-2025")
    
    # Create text file
    std_txt_file = standard_dir / "training-gpt-5-nano-hard.txt"
    with open(std_txt_file, 'w') as f:
        for task_id in hard_tasks:
            f.write(f"{task_id}\n")
    print(f"\nAlso created in standard location: {std_txt_file}")
    
    # Create details JSON file
    std_json_file = standard_dir / "training-gpt-5-nano-hard_details.json"
    with open(std_json_file, 'w') as f:
        json.dump(details, f, indent=2)
    print(f"Created: {std_json_file}")
    
    # Print some statistics about the hard tasks
    print("\n=== Statistics for hard tasks ===")
    hard_df = df[df['task_id'].isin(hard_tasks)]
    
    print(f"Total programs for hard tasks: {len(hard_df)}")
    print(f"Average programs per hard task: {len(hard_df) / len(hard_tasks):.2f}")
    
    # Check performance metrics for hard tasks
    print(f"Any train correct rate for hard tasks: {hard_df['any_train_correct'].mean():.2%}")
    print("(Should be 0.0% since these are tasks with zero train examples correct)")
    
    print("\n✅ Done! New subset 'training-gpt-5-nano-hard' created with {} tasks".format(len(hard_tasks)))

if __name__ == "__main__":
    main()