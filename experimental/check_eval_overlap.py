#!/usr/bin/env python3
"""
Specifically check which ARC-AGI-2 evaluation tasks overlap with ARC-AGI-1
"""

import os
from pathlib import Path

def get_task_ids_from_directory(directory_path):
    """Get all task IDs from a directory of JSON files."""
    task_ids = set()
    if os.path.exists(directory_path):
        for filename in os.listdir(directory_path):
            if filename.endswith('.json'):
                task_id = filename[:-5]  # Remove .json extension
                task_ids.add(task_id)
    return task_ids

def main():
    base_path = Path("/Users/ronanmcgovern/TR/arc-agi-2025/data")
    
    # Get task IDs from ARC-AGI-1
    arc1_training_ids = get_task_ids_from_directory(base_path / "arc-agi-1" / "training")
    arc1_evaluation_ids = get_task_ids_from_directory(base_path / "arc-agi-1" / "evaluation")
    
    # Get task IDs from ARC-AGI-2 evaluation
    arc2_evaluation_ids = get_task_ids_from_directory(base_path / "arc-agi-2" / "evaluation")
    
    print(f"ARC-AGI-2 evaluation tasks: {len(arc2_evaluation_ids)}")
    print(f"ARC-AGI-1 training tasks: {len(arc1_training_ids)}")
    print(f"ARC-AGI-1 evaluation tasks: {len(arc1_evaluation_ids)}")
    
    # Check overlaps
    arc2_eval_in_arc1_train = arc2_evaluation_ids.intersection(arc1_training_ids)
    arc2_eval_in_arc1_eval = arc2_evaluation_ids.intersection(arc1_evaluation_ids)
    
    print(f"\n=== OVERLAP ANALYSIS ===")
    print(f"ARC-AGI-2 evaluation tasks that are in ARC-AGI-1 training: {len(arc2_eval_in_arc1_train)}")
    if arc2_eval_in_arc1_train:
        print("Tasks:")
        for task_id in sorted(arc2_eval_in_arc1_train):
            print(f"  {task_id}")
    
    print(f"\nARC-AGI-2 evaluation tasks that are in ARC-AGI-1 evaluation: {len(arc2_eval_in_arc1_eval)}")
    if arc2_eval_in_arc1_eval:
        print("Tasks:")
        for task_id in sorted(arc2_eval_in_arc1_eval):
            print(f"  {task_id}")
    
    total_overlap = len(arc2_eval_in_arc1_train) + len(arc2_eval_in_arc1_eval)
    print(f"\nTotal ARC-AGI-2 evaluation tasks overlapping with ARC-AGI-1: {total_overlap}")
    
    # Let's also show a few sample ARC-AGI-2 evaluation task IDs for reference
    print(f"\nSample ARC-AGI-2 evaluation task IDs:")
    for task_id in sorted(list(arc2_evaluation_ids))[:10]:
        print(f"  {task_id}")

if __name__ == "__main__":
    main()