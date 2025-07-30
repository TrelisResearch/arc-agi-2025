#!/usr/bin/env python3

import json
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

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

    def __init__(self, data_root: Optional[str] = None):
        if data_root is None:
            # Find the data directory by searching up the directory tree
            current_path = Path(__file__).resolve()
            while current_path.parent != current_path:  # Stop at filesystem root
                data_path = current_path / "data"
                if data_path.exists() and data_path.is_dir():
                    data_root = data_path.as_posix()
                    break
                current_path = current_path.parent
            else:
                # Fallback to the original calculation
                data_root = (Path(__file__).parent.parent.parent / "data").resolve().as_posix()
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise ValueError(f"Data root directory not found: {data_root}")

    def load_task(self, task_id: str, dataset: Optional[str] = None) -> TaskData:
        """Load a single task by ID from the specified dataset(s).
        
        Args:
            task_id: The task ID to load
            dataset: Specific dataset to search in. If None, searches both arc-agi-1 and arc-agi-2
            
        Returns:
            The task data
            
        Raises:
            FileNotFoundError: If task is not found
        """
        # If no dataset specified, search across both datasets
        if dataset is None:
            datasets_to_search = ["arc-agi-1", "arc-agi-2"]
            for ds in datasets_to_search:
                try:
                    return self._load_task_from_dataset(task_id, ds)
                except FileNotFoundError:
                    continue
            raise FileNotFoundError(f"Task {task_id} not found in any dataset")
        else:
            # Search in specific dataset
            return self._load_task_from_dataset(task_id, dataset)
    
    def _load_task_from_dataset(self, task_id: str, dataset: str) -> TaskData:
        """Helper method to load a task from a specific dataset"""
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