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
    # Read all four parquet files
    nano_path = Path("/Users/ronanmcgovern/TR/arc-agi-2025/llm_python/datasets/inference/20250901_214154_gpt-5-nano_arc-prize-2025_training.parquet")
    oss_hard_path = Path("/Users/ronanmcgovern/TR/arc-agi-2025/llm_python/datasets/inference/20250902_094825_openai_gpt-oss-120b_arc-prize-2025_training-gpt-5-nano-hard.parquet")
    oss_run2_path = Path("/Users/ronanmcgovern/TR/arc-agi-2025/llm_python/datasets/inference/20250902_105628_openai_gpt-oss-120b_arc-prize-2025_training-hard.parquet")
    oss_run3_path = Path("/Users/ronanmcgovern/TR/arc-agi-2025/llm_python/datasets/inference/20250902_113816_openai_gpt-oss-120b_arc-prize-2025_training-hard.parquet")
    
    print(f"Reading gpt-5-nano parquet: {nano_path}")
    df_nano = read_soar_parquet(nano_path)
    
    print(f"Reading gpt-oss-120b (run 1) parquet: {oss_hard_path}")
    df_oss_1 = read_soar_parquet(oss_hard_path)
    
    print(f"Reading gpt-oss-120b (run 2) parquet: {oss_run2_path}")
    df_oss_2 = read_soar_parquet(oss_run2_path)
    
    print(f"Reading gpt-oss-120b (run 3) parquet: {oss_run3_path}")
    df_oss_3 = read_soar_parquet(oss_run3_path)
    
    def analyze_model_performance(df, model_name):
        """Analyze model performance, considering only non-transductive programs."""
        print(f"\n=== {model_name} analysis ===")
        print(f"Total rows: {len(df)}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Filter to non-transductive programs only
        if 'is_transductive' in df.columns:
            non_transductive_df = df[df['is_transductive'] == False].copy()
            print(f"Non-transductive rows: {len(non_transductive_df)} (filtered from {len(df)})")
        else:
            non_transductive_df = df.copy()
            print(f"No transductive column found, using all rows: {len(non_transductive_df)}")
        
        if len(non_transductive_df) == 0:
            print(f"No non-transductive programs found for {model_name}")
            return set(), set()
        
        # Check performance on non-transductive programs only
        any_train_correct_list = []
        for idx in range(len(non_transductive_df)):
            val = non_transductive_df.iloc[idx]['correct_train_input']
            result = isinstance(val, list) and bool(val) and any(val)
            any_train_correct_list.append(result)
        
        non_transductive_df['any_train_correct'] = any_train_correct_list
        task_stats = non_transductive_df.groupby('task_id')['any_train_correct'].max()
        
        successful_tasks = set(task_stats[task_stats == True].index)
        failed_tasks = set(task_stats[task_stats == False].index)
        
        print(f"Tasks with at least one non-transductive program getting min 1 train correct: {len(successful_tasks)}")
        print(f"Tasks with zero train correct (non-transductive only): {len(failed_tasks)}")
        
        return successful_tasks, failed_tasks
    
    # Analyze all models
    nano_success, nano_failed = analyze_model_performance(df_nano, "gpt-5-nano")
    oss1_success, oss1_failed = analyze_model_performance(df_oss_1, "gpt-oss-120b (run 1)")
    oss2_success, oss2_failed = analyze_model_performance(df_oss_2, "gpt-oss-120b (run 2)")
    oss3_success, oss3_failed = analyze_model_performance(df_oss_3, "gpt-oss-120b (run 3)")
    
    # Combine all successful tasks from any model/run
    all_successful_tasks = nano_success.union(oss1_success).union(oss2_success).union(oss3_success)
    
    # Find intersection - tasks ALL models completely failed on (non-transductive)
    hard_tasks = sorted(list(nano_failed.intersection(oss1_failed).intersection(oss2_failed).intersection(oss3_failed)))
    
    print(f"\n=== Combined analysis (non-transductive programs only) ===")
    print(f"Total tasks with any success across all models/runs: {len(all_successful_tasks)}")
    print(f"Tasks failed by ALL models/runs: {len(hard_tasks)}")
    print(f"Tasks failed by gpt-5-nano only: {len(nano_failed - oss1_success - oss2_success - oss3_success)}")
    print(f"Tasks failed by gpt-oss-120b runs only: {len(oss1_failed.intersection(oss2_failed).intersection(oss3_failed) - nano_success)}")
    
    # Sort task IDs for consistent ordering
    hard_tasks.sort()
    
    # Create subset files in the experimental folder
    output_dir = Path("experimental/subset_creation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create text file with just task IDs (one per line)
    txt_file = output_dir / "training-hard.txt"
    with open(txt_file, 'w') as f:
        for task_id in hard_tasks:
            f.write(f"{task_id}\n")
    print(f"\nCreated: {txt_file}")
    print(f"  Contains {len(hard_tasks)} task IDs")
    
    # Create details JSON file with metadata
    details = {
        "source_files": [
            "20250901_214154_gpt-5-nano_arc-prize-2025_training.parquet",
            "20250902_094825_openai_gpt-oss-120b_arc-prize-2025_training-gpt-5-nano-hard.parquet",
            "20250902_105628_openai_gpt-oss-120b_arc-prize-2025_training-hard.parquet",
            "20250902_113816_openai_gpt-oss-120b_arc-prize-2025_training-hard.parquet"
        ],
        "description": "Tasks from arc-prize-2025 training set where ALL models got ZERO train examples correct in non-transductive programs only",
        "filtering_criteria": "Only non-transductive programs considered. Transductive programs ignored.",
        "models_tested": ["gpt-5-nano", "gpt-oss-120b (3 runs)"],
        "total_tasks": len(hard_tasks),
        "task_ids": hard_tasks
    }
    
    json_file = output_dir / "training-hard_details.json"
    with open(json_file, 'w') as f:
        json.dump(details, f, indent=2)
    print(f"Created: {json_file}")
    
    # Also create subset files in the standard location
    standard_dir = Path("data/subsets/arc-prize-2025")
    
    # Create text file
    std_txt_file = standard_dir / "training-hard.txt"
    with open(std_txt_file, 'w') as f:
        for task_id in hard_tasks:
            f.write(f"{task_id}\n")
    print(f"\nAlso created in standard location: {std_txt_file}")
    
    # Create details JSON file
    std_json_file = standard_dir / "training-hard_details.json"
    with open(std_json_file, 'w') as f:
        json.dump(details, f, indent=2)
    print(f"Created: {std_json_file}")
    
    print(f"\n✅ Done! Created subset 'training-hard' with {len(hard_tasks)} tasks")
    print("These are tasks where ALL models got ZERO train examples correct (non-transductive programs only)")

if __name__ == "__main__":
    main()