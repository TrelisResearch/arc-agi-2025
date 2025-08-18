#!/usr/bin/env python3

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict

# Type definitions for ARC-AGI data structures
Grid = List[List[int]]


class TaskExample(TypedDict):
    input: Grid
    output: Optional[Grid]  # Optional for test challenges without solutions


class TaskData(TypedDict):
    """Complete task data structure"""

    train: List[TaskExample]
    test: List[TaskExample]


class TaskLoader:
    """Loads ARC-AGI tasks from competition format data sources into memory.

    On initialization, loads all tasks from:
    - Competition format: arc-prize-2024, arc-prize-2025 (training/evaluation/test)
    - All existing subset definitions from data/subsets/

    Subset naming:
    - Competition: "arc-prize-2024/training", "arc-prize-2025/evaluation", "arc-prize-2024/test"
    - Legacy subsets: "arc-agi-1/shortest_training_1", "arc-agi-2/all_evaluation", etc.
    """

    def __init__(self, data_root: Optional[str] = None):
        if data_root is None:
            # Check environment variable first
            data_root = os.getenv('ARC_DATA_ROOT')
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
                    data_root = (
                        (Path(__file__).parent.parent.parent / "data").resolve().as_posix()
                    )
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise ValueError(f"Data root directory not found: {data_root}")

        # Cache for loaded data
        self.tasks: Dict[str, TaskData] = {}
        self.subsets: Dict[str, List[str]] = {}
        self._loaded_subsets: set = set()  # Track what we've already loaded

        # Load subset definitions only
        self._load_all_subsets()

    def _ensure_subset_loaded(self, subset_name: str):
        """Load a specific subset on demand if not already loaded"""
        if subset_name in self._loaded_subsets:
            return
            
        print(f"Loading subset {subset_name}...")
        
        # Parse subset name to determine dataset and split
        if "/" in subset_name:
            dataset, split = subset_name.split("/", 1)
            if dataset in ["arc-prize-2024", "arc-prize-2025"]:
                # Load competition format data for this specific subset
                subset_tasks = self._load_competition_split(dataset, split)
                self.tasks.update(subset_tasks)
                if subset_tasks:
                    self.subsets[subset_name] = list(subset_tasks.keys())
                    print(f"  Loaded {len(subset_tasks)} tasks from {subset_name}")
        
        self._loaded_subsets.add(subset_name)

    def _load_competition_data(self):
        """Load all competition format data from arc-prize-2024 and arc-prize-2025"""
        for dataset in ["arc-prize-2024", "arc-prize-2025"]:
            dataset_path = self.data_root / dataset
            if not dataset_path.exists():
                continue

            print(f"Loading {dataset}...")

            # Load training split
            training_tasks = self._load_competition_split(dataset, "training")
            self.tasks.update(training_tasks)
            if training_tasks:
                self.subsets[f"{dataset}/training"] = list(training_tasks.keys())
                print(f"  Training: {len(training_tasks)} tasks")

            # Load evaluation split
            eval_tasks = self._load_competition_split(dataset, "evaluation")
            self.tasks.update(eval_tasks)
            if eval_tasks:
                self.subsets[f"{dataset}/evaluation"] = list(eval_tasks.keys())
                print(f"  Evaluation: {len(eval_tasks)} tasks")

            # Load test split (challenges only, no solutions expected)
            test_tasks = self._load_competition_split(dataset, "test")
            self.tasks.update(test_tasks)
            if test_tasks:
                self.subsets[f"{dataset}/test"] = list(test_tasks.keys())
                print(f"  Test: {len(test_tasks)} tasks")

    def _load_competition_split(self, dataset: str, split: str) -> Dict[str, TaskData]:
        """Load a specific split (training/evaluation) from competition format"""
        dataset_path = self.data_root / dataset
        challenges_path = dataset_path / f"arc-agi_{split}_challenges.json"
        solutions_path = dataset_path / f"arc-agi_{split}_solutions.json"

        if not challenges_path.exists():
            return {}

        # Load challenges
        with open(challenges_path, "r") as f:
            challenges = json.load(f)

        # Load solutions if they exist (they might not for test splits)
        solutions = {}
        if solutions_path.exists():
            with open(solutions_path, "r") as f:
                solutions = json.load(f)

        # Combine challenges and solutions
        tasks = {}
        for task_id, challenge_data in challenges.items():
            task_data = TaskData(train=challenge_data["train"], test=[])

            # Add test data, with solutions if available
            test_outputs = solutions.get(task_id, [])
            for i, test_input in enumerate(challenge_data["test"]):
                test_example = TaskExample(
                    input=test_input["input"],
                    output=test_outputs[i] if i < len(test_outputs) else None,
                )
                task_data["test"].append(test_example)

            tasks[task_id] = task_data

        return tasks

    def _load_all_subsets(self):
        """Load subset definitions from data/subsets and discover competition splits"""
        # Load legacy subset definitions from data/subsets
        subsets_dir = self.data_root / "subsets"
        if subsets_dir.exists():
            for dataset_dir in subsets_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue

                dataset_name = dataset_dir.name
                for subset_file in dataset_dir.glob("*.txt"):
                    if "_details.json" in subset_file.name or "archive" in str(subset_file):
                        continue

                    subset_name = subset_file.stem
                    subset_key = f"{dataset_name}/{subset_name}"

                    try:
                        with open(subset_file, "r") as f:
                            task_ids = [line.strip() for line in f if line.strip()]
                        self.subsets[subset_key] = task_ids
                    except Exception as e:
                        print(f"Warning: Could not load subset {subset_file}: {e}")

        # Discover competition format subsets (but don't load task data yet)
        for dataset in ["arc-prize-2024", "arc-prize-2025"]:
            dataset_path = self.data_root / dataset
            if not dataset_path.exists():
                continue

            for split in ["training", "evaluation", "test"]:
                challenges_path = dataset_path / f"arc-agi_{split}_challenges.json"
                if challenges_path.exists():
                    subset_key = f"{dataset}/{split}"
                    # Just mark that this subset exists - we'll load task IDs on demand
                    if subset_key not in self.subsets:
                        self.subsets[subset_key] = []  # Empty list means "exists but not loaded yet"

    def get_task(self, task_id: str) -> TaskData:
        """Get a task by ID"""
        if task_id not in self.tasks:
            raise FileNotFoundError(f"Task {task_id} not found")
        return self.tasks[task_id]

    def get_subset_tasks(self, subset_name: str) -> List[Tuple[str, TaskData]]:
        """Get all tasks from a subset, returning (task_id, task_data) tuples"""
        # Load the subset on demand if not already loaded
        self._ensure_subset_loaded(subset_name)
        
        if subset_name not in self.subsets:
            raise ValueError(
                f"Subset {subset_name} not found. Available: {list(self.subsets.keys())}"
            )

        tasks = []
        missing_tasks = []

        for task_id in self.subsets[subset_name]:
            if task_id in self.tasks:
                tasks.append((task_id, self.tasks[task_id]))
            else:
                missing_tasks.append(task_id)

        if missing_tasks:
            print(
                f"Warning: {len(missing_tasks)} tasks from subset {subset_name} not found: {missing_tasks[:5]}{'...' if len(missing_tasks) > 5 else ''}"
            )

        return tasks

    def get_available_subsets(self) -> List[str]:
        """Get list of all available subset names"""
        return sorted(self.subsets.keys())

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about loaded data"""
        stats = {"total_tasks": len(self.tasks), "total_subsets": len(self.subsets)}

        # Count tasks by competition dataset subset
        for subset_name in self.subsets:
            if subset_name.startswith("arc-prize-") and "/" in subset_name:
                stats[f"{subset_name.replace('/', '_')}_tasks"] = len(
                    self.subsets[subset_name]
                )

        return stats

    # Backward compatibility methods
    def load_task(self, task_id: str, dataset: Optional[str] = None) -> TaskData:
        """Backward compatibility: Load a single task by ID

        Args:
            task_id: The task ID to load
            dataset: Ignored (kept for compatibility)
        """
        return self.get_task(task_id)

    def load_tasks_from_subset(
        self, subset_name: str, dataset: str = "arc-agi-1"
    ) -> List[Tuple[str, TaskData]]:
        """Backward compatibility: Load all tasks from a subset

        Args:
            subset_name: Name of the subset
            dataset: Dataset name (used to construct full subset path)
        """
        full_subset_name = f"{dataset}/{subset_name}"
        return self.get_subset_tasks(full_subset_name)
