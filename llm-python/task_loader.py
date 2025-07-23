#!/usr/bin/env python3

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict

# Type definitions for ARC-AGI data structures
Grid = List[List[int]]

class TaskExample(TypedDict):
    """A single training or test example"""
    input: Grid
    output: Grid

class TaskData(TypedDict):
    """Complete task data structure"""
    train: List[TaskExample]
    test: List[TaskExample]

class TaskLoader:
    """Loads ARC-AGI tasks from the data directory.
    
    Subset naming convention (2025+):
      - shortest_1_training, shortest_10_training, shortest_30_training
      - shortest_1_evaluation, ...
      - middle_1_training, ...
      - longest_1_evaluation, ...
    Each subset contains only task IDs from the relevant split.
    Legacy mixed subsets are in archive/.
    """
    
    def __init__(self, data_root: str = "../data"):
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise ValueError(f"Data root directory not found: {data_root}")
    
    def load_all_tasks(self, dataset: str = "arc-agi-1") -> Dict[str, TaskData]:
        """Load all tasks from the specified dataset"""
        tasks = {}
        for split in ["training", "evaluation"]:
            split_path = self.data_root / dataset / split
            if not split_path.exists():
                continue
            
            for task_file in split_path.glob("*.json"):
                with open(task_file, 'r') as f:
                    task_data = json.load(f)
                    task_id = task_file.stem
                    tasks[task_id] = task_data
        return tasks

    def load_task(self, task_id: str, dataset: str = "arc-agi-1") -> TaskData:
        """Load a single task by ID from the specified dataset"""
        # Check training directory first, then evaluation
        for split in ["training", "evaluation"]:
            task_path = self.data_root / dataset / split / f"{task_id}.json"
            if task_path.exists():
                with open(task_path, 'r') as f:
                    return json.load(f)
        
        raise FileNotFoundError(f"Task {task_id} not found in {dataset}")
    
    def load_subset(self, subset_name: str, dataset: str = "arc-agi-1") -> List[str]:
        """Load task IDs from a subset file"""
        subset_path = self.data_root / "subsets" / dataset / f"{subset_name}.txt"
        if not subset_path.exists():
            raise FileNotFoundError(f"Subset file not found: {subset_path}")
        
        with open(subset_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    def load_tasks_from_subset(self, subset_name: str, dataset: str = "arc-agi-1") -> List[Tuple[str, TaskData]]:
        """Load all tasks from a subset, returning (task_id, task_data) tuples"""
        task_ids = self.load_subset(subset_name, dataset)
        tasks = []
        
        for task_id in task_ids:
            try:
                task_data = self.load_task(task_id, dataset)
                tasks.append((task_id, task_data))
            except FileNotFoundError:
                print(f"Warning: Task {task_id} not found, skipping")
                
        return tasks
    
    def get_available_subsets(self, dataset: str = "arc-agi-1") -> List[str]:
        """List available subset files for a dataset (ignores archive/ and details)."""
        subset_dir = self.data_root / "subsets" / dataset
        if not subset_dir.exists():
            return []
        subsets = []
        for file in subset_dir.glob("*.txt"):
            if (
                not file.name.endswith("_details.json")
                and "archive" not in str(file.parent)
            ):
                subsets.append(file.stem)
        return sorted(subsets)
    
    def format_task_for_prompt(self, task_data: TaskData, include_test: bool = False) -> str:
        """Format task data into a string suitable for prompting"""
        lines = []
        
        # Format training examples
        lines.append("Training Examples:")
        for i, example in enumerate(task_data.get('train', [])):
            lines.append(f"\nExample {i+1}:")
            lines.append("Input:")
            lines.append(self._format_grid(example['input']))
            lines.append("Output:")
            lines.append(self._format_grid(example['output']))
        
        # Only include test if explicitly requested
        if include_test and task_data.get('test'):
            lines.append("\nTest Input:")
            lines.append(self._format_grid(task_data['test'][0]['input']))
        
        return '\n'.join(lines)
    
    def _format_grid(self, grid: Grid) -> str:
        """Format a grid as a string"""
        return '\n'.join(' '.join(str(cell) for cell in row) for row in grid)
    
    def get_test_outputs(self, task_data: TaskData) -> List[Grid]:
        """Extract all test outputs from a task"""
        return [test_case['output'] for test_case in task_data.get('test', [])]


# Example usage
if __name__ == "__main__":
    loader = TaskLoader()
    
    # Show available subsets
    print("Available subsets for arc-agi-1:")
    for subset_name in loader.get_available_subsets("arc-agi-1"):
        print(f"  - {subset_name}")
    
    # Load and display a single task
    print("\nLoading shortest task from arc-agi-1...")
    task_ids = loader.load_subset("shortest_training_1", "arc-agi-1")
    if task_ids:
        task_id = task_ids[0]
        task_data = loader.load_task(task_id, "arc-agi-1")
        
        print(f"\nTask ID: {task_id}")
        print(f"Number of training examples: {len(task_data.get('train', []))}")
        print(f"Number of test examples: {len(task_data.get('test', []))}")
        
        # Show formatted task
        print("\nFormatted task for prompt:")
        print("-" * 50)
        print(loader.format_task_for_prompt(task_data))
        
    # Show available subsets for arc-agi-2
    print(f"\nAvailable subsets for arc-agi-2:")
    for subset_name in loader.get_available_subsets("arc-agi-2"):
        print(f"  - {subset_name}")