#!/usr/bin/env python3
"""
Analyze overlap between ARC-AGI-1 and ARC-AGI-2 datasets and create a subset
of ARC-AGI-2 training tasks that are not present in ARC-AGI-1.
"""

import os
import json
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
    
    # Get task IDs from ARC-AGI-1 (both training and evaluation)
    print("Analyzing ARC-AGI-1 datasets...")
    arc1_training_ids = get_task_ids_from_directory(base_path / "arc-agi-1" / "training")
    arc1_evaluation_ids = get_task_ids_from_directory(base_path / "arc-agi-1" / "evaluation")
    arc1_all_ids = arc1_training_ids.union(arc1_evaluation_ids)
    
    print(f"ARC-AGI-1 Training tasks: {len(arc1_training_ids)}")
    print(f"ARC-AGI-1 Evaluation tasks: {len(arc1_evaluation_ids)}")
    print(f"ARC-AGI-1 Total unique tasks: {len(arc1_all_ids)}")
    
    # Get task IDs from ARC-AGI-2
    print("\nAnalyzing ARC-AGI-2 datasets...")
    arc2_training_ids = get_task_ids_from_directory(base_path / "arc-agi-2" / "training")
    arc2_evaluation_ids = get_task_ids_from_directory(base_path / "arc-agi-2" / "evaluation")
    arc2_all_ids = arc2_training_ids.union(arc2_evaluation_ids)
    
    print(f"ARC-AGI-2 Training tasks: {len(arc2_training_ids)}")
    print(f"ARC-AGI-2 Evaluation tasks: {len(arc2_evaluation_ids)}")
    print(f"ARC-AGI-2 Total unique tasks: {len(arc2_all_ids)}")
    
    # Find overlaps
    print("\nAnalyzing overlaps...")
    
    # ARC-AGI-2 training tasks that are in ARC-AGI-1 (training or evaluation)
    arc2_train_overlap_with_arc1 = arc2_training_ids.intersection(arc1_all_ids)
    
    # Specifically check overlap with ARC-AGI-1 training and evaluation separately
    arc2_train_overlap_with_arc1_train = arc2_training_ids.intersection(arc1_training_ids)
    arc2_train_overlap_with_arc1_eval = arc2_training_ids.intersection(arc1_evaluation_ids)
    
    # ARC-AGI-2 training tasks that are NOT in ARC-AGI-1
    arc2_train_unique = arc2_training_ids - arc1_all_ids
    
    print(f"ARC-AGI-2 training tasks that overlap with ARC-AGI-1 (total): {len(arc2_train_overlap_with_arc1)}")
    print(f"  - Overlap with ARC-AGI-1 training: {len(arc2_train_overlap_with_arc1_train)}")
    print(f"  - Overlap with ARC-AGI-1 evaluation: {len(arc2_train_overlap_with_arc1_eval)}")
    print(f"ARC-AGI-2 training tasks that are unique (not in ARC-AGI-1): {len(arc2_train_unique)}")
    
    # Additional analysis for ARC-AGI-2 evaluation
    arc2_eval_overlap_with_arc1 = arc2_evaluation_ids.intersection(arc1_all_ids)
    arc2_eval_unique = arc2_evaluation_ids - arc1_all_ids
    
    print(f"\nARC-AGI-2 evaluation tasks that overlap with ARC-AGI-1: {len(arc2_eval_overlap_with_arc1)}")
    print(f"ARC-AGI-2 evaluation tasks that are unique (not in ARC-AGI-1): {len(arc2_eval_unique)}")
    
    # Create the subset file of unique ARC-AGI-2 training tasks
    output_file = base_path / "subsets" / "arc-agi-2" / "unique_training_tasks.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Sort the unique task IDs for consistency
    sorted_unique_tasks = sorted(list(arc2_train_unique))
    
    with open(output_file, 'w') as f:
        for task_id in sorted_unique_tasks:
            f.write(f"{task_id}\n")
    
    print(f"\nCreated subset file: {output_file}")
    print(f"Contains {len(sorted_unique_tasks)} unique ARC-AGI-2 training tasks")
    
    # Create a detailed JSON report
    report_file = base_path / "subsets" / "arc-agi-2" / "unique_training_tasks_details.json"
    
    report_data = {
        "analysis_summary": {
            "arc_agi_1_training_tasks": len(arc1_training_ids),
            "arc_agi_1_evaluation_tasks": len(arc1_evaluation_ids),
            "arc_agi_1_total_tasks": len(arc1_all_ids),
            "arc_agi_2_training_tasks": len(arc2_training_ids),
            "arc_agi_2_evaluation_tasks": len(arc2_evaluation_ids),
            "arc_agi_2_total_tasks": len(arc2_all_ids),
            "arc_agi_2_training_overlap_with_arc_agi_1_total": len(arc2_train_overlap_with_arc1),
            "arc_agi_2_training_overlap_with_arc_agi_1_training": len(arc2_train_overlap_with_arc1_train),
            "arc_agi_2_training_overlap_with_arc_agi_1_evaluation": len(arc2_train_overlap_with_arc1_eval),
            "arc_agi_2_training_unique_tasks": len(arc2_train_unique),
            "arc_agi_2_evaluation_overlap_with_arc_agi_1": len(arc2_eval_overlap_with_arc1),
            "arc_agi_2_evaluation_unique_tasks": len(arc2_eval_unique)
        },
        "overlapping_tasks": {
            "arc_agi_2_training_in_arc_agi_1_training": sorted(list(arc2_train_overlap_with_arc1_train)),
            "arc_agi_2_training_in_arc_agi_1_evaluation": sorted(list(arc2_train_overlap_with_arc1_eval))
        },
        "unique_arc_agi_2_training_tasks": sorted_unique_tasks
    }
    
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"Created detailed report: {report_file}")
    
    # Print some sample overlapping tasks
    if arc2_train_overlap_with_arc1_train:
        print(f"\nSample ARC-AGI-2 training tasks that are in ARC-AGI-1 training:")
        for task_id in sorted(list(arc2_train_overlap_with_arc1_train))[:5]:
            print(f"  {task_id}")
    
    if arc2_train_overlap_with_arc1_eval:
        print(f"\nSample ARC-AGI-2 training tasks that are in ARC-AGI-1 evaluation:")
        for task_id in sorted(list(arc2_train_overlap_with_arc1_eval))[:5]:
            print(f"  {task_id}")
    
    if sorted_unique_tasks:
        print(f"\nSample unique ARC-AGI-2 training tasks (not in ARC-AGI-1):")
        for task_id in sorted_unique_tasks[:5]:
            print(f"  {task_id}")

if __name__ == "__main__":
    main()