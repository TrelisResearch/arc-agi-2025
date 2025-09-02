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
    # Read both parquet files
    nano_path = Path("/Users/ronanmcgovern/TR/arc-agi-2025/llm_python/datasets/inference/20250901_214154_gpt-5-nano_arc-prize-2025_training.parquet")
    oss_path = Path("/Users/ronanmcgovern/TR/arc-agi-2025/llm_python/datasets/inference/20250902_094825_openai_gpt-oss-120b_arc-prize-2025_training-gpt-5-nano-hard.parquet")
    
    print(f"Reading gpt-5-nano parquet: {nano_path}")
    df_nano = read_soar_parquet(nano_path)
    
    print(f"Reading gpt-oss-120b parquet: {oss_path}")
    df_oss = read_soar_parquet(oss_path)
    
    # Analyze gpt-5-nano performance
    print(f"\n=== gpt-5-nano analysis ===")
    print(f"Total rows: {len(df_nano)}")
    print(f"Columns: {df_nano.columns.tolist()}")
    
    # Check gpt-5-nano performance
    any_train_correct_nano = []
    for idx in range(len(df_nano)):
        val = df_nano.iloc[idx]['correct_train_input']
        result = isinstance(val, list) and bool(val) and any(val)
        any_train_correct_nano.append(result)
    
    df_nano['any_train_correct'] = any_train_correct_nano
    nano_task_stats = df_nano.groupby('task_id')['any_train_correct'].max()
    nano_failed_tasks = set(nano_task_stats[nano_task_stats == False].index)
    
    print(f"gpt-5-nano tasks with zero train correct: {len(nano_failed_tasks)}")
    
    # Analyze gpt-oss-120b performance  
    print(f"\n=== gpt-oss-120b analysis ===")
    print(f"Total rows: {len(df_oss)}")
    
    # Check gpt-oss-120b performance
    any_train_correct_oss = []
    for idx in range(len(df_oss)):
        val = df_oss.iloc[idx]['correct_train_input']
        result = isinstance(val, list) and bool(val) and any(val)
        any_train_correct_oss.append(result)
    
    df_oss['any_train_correct'] = any_train_correct_oss
    oss_task_stats = df_oss.groupby('task_id')['any_train_correct'].max()
    oss_failed_tasks = set(oss_task_stats[oss_task_stats == False].index)
    
    print(f"gpt-oss-120b tasks with zero train correct: {len(oss_failed_tasks)}")
    
    # Find intersection - tasks both models completely failed
    hard_tasks = sorted(list(nano_failed_tasks.intersection(oss_failed_tasks)))
    
    print(f"\n=== Combined analysis ===")
    print(f"Tasks failed by BOTH models: {len(hard_tasks)}")
    print(f"Tasks failed by gpt-5-nano only: {len(nano_failed_tasks - oss_failed_tasks)}")
    print(f"Tasks failed by gpt-oss-120b only: {len(oss_failed_tasks - nano_failed_tasks)}")
    
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
        "source_files": [
            "20250901_214154_gpt-5-nano_arc-prize-2025_training.parquet",
            "20250902_094825_openai_gpt-oss-120b_arc-prize-2025_training-gpt-5-nano-hard.parquet"
        ],
        "description": "Tasks from arc-prize-2025 training set where BOTH gpt-5-nano AND gpt-oss-120b got ZERO train examples correct in all attempts",
        "models_tested": ["gpt-5-nano", "gpt-oss-120b"],
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
    
    print(f"\n✅ Done! Updated subset 'training-gpt-5-nano-hard' with {len(hard_tasks)} tasks")
    print("These are tasks that completely failed for BOTH gpt-5-nano AND gpt-oss-120b")

if __name__ == "__main__":
    main()