import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict

import threading


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

    When duplicate task IDs are found with conflicting outputs, the canonical_dataset
    parameter determines which version to prefer.

    Args:
        data_root: Path to the data directory. If None, searches for 'data' directory.
        canonical_dataset: Dataset to prefer when resolving conflicts (default: "arc-prize-2025")

    Subset naming:
    - Competition: "arc-prize-2024/training", "arc-prize-2025/evaluation", "arc-prize-2024/test"
    - Legacy subsets: "arc-agi-1/shortest_training_1", "arc-agi-2/all_evaluation", etc.
    """

    def __init__(self, data_root: str, canonical_dataset: str = "arc-prize-2025"):
        self.canonical_dataset = canonical_dataset
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise ValueError(f"Data root directory not found: {data_root}")

        # Cache for loaded data
        self.tasks: Dict[str, TaskData] = {}
        self.subsets: Dict[str, List[str]] = {}
        self.task_sources: Dict[
            str, str
        ] = {}  # Track which dataset each task came from

        # Load everything upfront
        self._load_all_data()

    def _load_all_data(self):
        """Load all subset definitions and task data upfront"""
        # First load subset definitions
        self._load_all_subsets()

        # Then load all competition format data
        self._load_all_competition_data()

    def _has_outputs(self, task_data: TaskData) -> bool:
        """Check if a task has any test outputs defined"""
        return any(
            test_example.get("output") is not None for test_example in task_data["test"]
        )

    def _outputs_are_equal(self, task1: TaskData, task2: TaskData) -> bool:
        """Check if two tasks have identical test outputs"""
        test1, test2 = task1["test"], task2["test"]
        if len(test1) != len(test2):
            return False

        for t1, t2 in zip(test1, test2):
            output1, output2 = t1.get("output"), t2.get("output")
            # Both None is equal, otherwise compare the actual outputs
            if output1 is None and output2 is None:
                continue
            if output1 is None or output2 is None:
                return False
            if output1 != output2:
                return False
        return True

    def _merge_task_data(
        self,
        existing_task: TaskData,
        new_task: TaskData,
        task_id: str,
        new_source_dataset: str,
        existing_source_dataset: str,
    ) -> TaskData:
        """Merge two task data entries, preserving outputs when available"""
        # If existing task has outputs and new task doesn't, keep existing
        if self._has_outputs(existing_task) and not self._has_outputs(new_task):
            return existing_task

        # If new task has outputs and existing doesn't, use new task
        if self._has_outputs(new_task) and not self._has_outputs(existing_task):
            return new_task

        # If both have outputs, check if they're the same
        if self._has_outputs(existing_task) and self._has_outputs(new_task):
            if not self._outputs_are_equal(existing_task, new_task):
                # Use canonical dataset to resolve conflicts
                if new_source_dataset == self.canonical_dataset:
                    return new_task
                elif existing_source_dataset == self.canonical_dataset:
                    return existing_task
                else:
                    # Neither is canonical, prefer the new one
                    return new_task

        # If both have same outputs or neither has outputs, use the new one
        return new_task

    def _add_tasks_safely(
        self, new_tasks: Dict[str, TaskData], source_dataset: str
    ) -> None:
        """Add tasks to the cache, handling duplicates by preserving outputs when possible"""
        for task_id, task_data in new_tasks.items():
            if task_id in self.tasks:
                existing_source = self.task_sources.get(task_id, "unknown")
                merged_task = self._merge_task_data(
                    self.tasks[task_id],
                    task_data,
                    task_id,
                    source_dataset,
                    existing_source,
                )
                self.tasks[task_id] = merged_task

                # Update source based on which task we ended up using
                if merged_task is task_data:  # Using new task
                    self.task_sources[task_id] = source_dataset
                # If using existing task, keep the existing source
            else:
                self.tasks[task_id] = task_data
                self.task_sources[task_id] = source_dataset

    def _load_all_competition_data(self):
        """Load all competition format data from arc-prize-2024 and arc-prize-2025"""
        for dataset in ["arc-prize-2024", "arc-prize-2025"]:
            dataset_path = self.data_root / dataset
            if not dataset_path.exists():
                continue

            # Load training split
            training_tasks = self._load_competition_split(dataset, "training")
            if training_tasks:
                self._add_tasks_safely(training_tasks, dataset)
                self.subsets[f"{dataset}/training"] = list(training_tasks.keys())
                # print(f"  {dataset}/training: {len(training_tasks)} tasks")  # Debug info

            # Load evaluation split
            eval_tasks = self._load_competition_split(dataset, "evaluation")
            if eval_tasks:
                self._add_tasks_safely(eval_tasks, dataset)
                self.subsets[f"{dataset}/evaluation"] = list(eval_tasks.keys())

            # Load test split (challenges only, no solutions expected)
            test_tasks = self._load_competition_split(dataset, "test")
            if test_tasks:
                self._add_tasks_safely(test_tasks, dataset)
                self.subsets[f"{dataset}/test"] = list(test_tasks.keys())

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
        """Load subset definitions from data/subsets"""
        # Load legacy subset definitions from data/subsets
        subsets_dir = self.data_root / "subsets"
        if subsets_dir.exists():
            for dataset_dir in subsets_dir.iterdir():
                if not dataset_dir.is_dir():
                    continue

                dataset_name = dataset_dir.name
                for subset_file in dataset_dir.glob("*.txt"):
                    if "_details.json" in subset_file.name or "archive" in str(
                        subset_file
                    ):
                        continue

                    subset_name = subset_file.stem
                    subset_key = f"{dataset_name}/{subset_name}"

                    try:
                        with open(subset_file, "r") as f:
                            task_ids = [line.strip() for line in f if line.strip()]
                        self.subsets[subset_key] = task_ids
                    except Exception as e:
                        print(f"Warning: Could not load subset {subset_file}: {e}")

    def get_task(self, task_id: str) -> TaskData:
        """Get a task by ID"""
        if task_id not in self.tasks:
            raise FileNotFoundError(f"Task {task_id} not found")
        return self.tasks[task_id]

    def get_subset_tasks(self, subset_name: str) -> List[Tuple[str, TaskData]]:
        """Get all tasks from a subset, returning (task_id, task_data) tuples"""
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


def get_default_data_root() -> str:
    # Compute data_root if not provided
    data_root = os.getenv("ARC_DATA_ROOT")
    if data_root is None:
        current_path = Path(__file__).resolve()
        while current_path.parent != current_path:
            data_path = current_path / "data"
            if data_path.exists() and data_path.is_dir():
                data_root = data_path.as_posix()
                break
            current_path = current_path.parent
        else:
            data_root = (
                (Path(__file__).parent.parent.parent / "data").resolve().as_posix()
            )
    return data_root


_default_task_loader_instance = None
_default_task_loader_lock = threading.Lock()


def get_task_loader() -> TaskLoader:
    """
    Lazily create and return a singleton TaskLoader instance in a thread-safe way.
    Computes data_root if not provided, using environment or search logic.
    """
    global _default_task_loader_instance
    if _default_task_loader_instance is not None:
        return _default_task_loader_instance
    with _default_task_loader_lock:
        if _default_task_loader_instance is None:
            _default_task_loader_instance = TaskLoader(
                data_root=get_default_data_root(), canonical_dataset="arc-prize-2025"
            )
    return _default_task_loader_instance
