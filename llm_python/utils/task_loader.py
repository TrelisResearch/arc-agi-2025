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

    def inject_subset(self, dataset_name: str, subset_name: str, tasks: Dict[str, TaskData]) -> None:
        """
        Inject a new subset into the TaskLoader from external data.

        Args:
            subset_name: Name for the new subset.
            tasks: Dict mapping task_id to TaskData.
            source_dataset: Source label for the injected tasks (default: "injected").
        """
        self._add_tasks_safely(tasks, dataset_name)
        self.subsets[subset_name] = list(tasks.keys())

    def _detect_dataset_type(self, identifier: str) -> str:
        """Detect if identifier is parquet file/directory, HuggingFace dataset, or traditional subset"""
        from pathlib import Path
        
        path = Path(identifier)
        
        # Check if it's a parquet file or directory containing parquet files
        if path.exists():
            if path.is_file() and path.suffix == '.parquet':
                return "parquet"
            elif path.is_dir() and any(path.glob("*.parquet")):
                return "parquet"
        
        # Check if it's a path that should be parquet but doesn't exist
        if identifier.endswith('.parquet') or 'parquet' in identifier.lower():
            return "parquet"
        
        # Check if it's a known traditional subset first
        if identifier in self.subsets:
            return "traditional"
        
        # Check if it looks like a HuggingFace dataset (contains slash and not a known subset)
        if "/" in identifier and identifier not in self.subsets:
            return "huggingface"
        
        # Otherwise assume it's a traditional subset
        return "traditional"

    def get_dataset_subset(self, identifier: str, max_rows: Optional[int] = None) -> List[Tuple[str, TaskData]]:
        """Get tasks from HuggingFace dataset, parquet file, or traditional subset.
        
        For HF/parquet datasets, extracts unique task_ids and looks up the actual
        task data from the cached tasks. For traditional subsets, uses existing logic.
        
        Args:
            identifier: Dataset identifier (HF dataset slug, parquet path, or traditional subset name)
            max_rows: Maximum number of rows to load from dataset (for HF/parquet only)
            
        Returns:
            List of (task_id, task_data) tuples matching existing interface
            
        Raises:
            ValueError: If dataset type not supported or tasks not found
        """
        dataset_type = self._detect_dataset_type(identifier)
        
        if dataset_type == "traditional":
            # Use existing logic for traditional subsets
            tasks = self.get_subset_tasks(identifier)
            if max_rows and len(tasks) > max_rows:
                tasks = tasks[:max_rows]
            return tasks
        
        elif dataset_type == "parquet":
            # Load parquet and extract unique task_ids
            from llm_python.datasets.io import read_soar_parquet
            
            try:
                df = read_soar_parquet(identifier)
                if max_rows and len(df) > max_rows:
                    df = df.head(max_rows)
                task_ids = df['task_id'].unique().tolist()
                print(f"📊 Extracted {len(task_ids)} unique task IDs from parquet dataset")
            except Exception as e:
                raise ValueError(f"Failed to load parquet dataset '{identifier}': {e}")
        
        elif dataset_type == "huggingface":
            # Load HF dataset and extract unique task_ids
            try:
                from datasets import load_dataset
                ds = load_dataset(identifier, split="train")
                if max_rows and max_rows < len(ds):
                    ds = ds.select(range(max_rows))
                
                # Handle datasets with row_id column
                if 'task_id' in ds.column_names:
                    task_ids = list(set(ds['task_id']))
                elif 'row_id' in ds.column_names:
                    # Use row_id as task_id if task_id not present
                    task_ids = list(set(ds['row_id']))
                else:
                    raise ValueError("Dataset must contain either 'task_id' or 'row_id' column")
                
                print(f"📊 Extracted {len(task_ids)} unique task IDs from HuggingFace dataset")
            except Exception as e:
                raise ValueError(f"Failed to load HuggingFace dataset '{identifier}': {e}")
        
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        # Look up actual task data from cached tasks
        tasks = []
        missing_tasks = []
        
        for task_id in task_ids:
            if task_id in self.tasks:
                tasks.append((task_id, self.tasks[task_id]))
            else:
                missing_tasks.append(task_id)
        
        if missing_tasks:
            print(
                f"Warning: {len(missing_tasks)} task IDs from dataset not found in cached tasks: {missing_tasks[:5]}{'...' if len(missing_tasks) > 5 else ''}"
            )
        
        if not tasks:
            raise ValueError(f"No valid tasks found for dataset '{identifier}'")
        
        print(f"✅ Successfully loaded {len(tasks)} tasks from {dataset_type} dataset")
        return tasks

    def get_program_data_for_refinement(self, identifier: str, max_rows: Optional[int] = None) -> Dict:
        """Get program data from HF/parquet datasets for refinement mode.
        
        This method is specifically for refinement mode and returns the raw program data
        along with metadata like correctness information, rather than just task IDs.
        
        Args:
            identifier: Dataset identifier (HF dataset slug or parquet path)
            max_rows: Maximum number of rows to load from dataset
            
        Returns:
            Dictionary mapping task_id to program information including:
            - programs: List of program dictionaries with 'code', 'correct_train_input', etc.
            - task_data: The actual TaskData for the task
            
        Raises:
            ValueError: If dataset type not supported, no programs found, or only traditional subset provided
        """
        dataset_type = self._detect_dataset_type(identifier)
        
        if dataset_type == "traditional":
            raise ValueError("Refinement mode requires HuggingFace dataset or parquet file. Traditional subsets not supported.")
        
        programs_by_task = {}
        
        if dataset_type == "parquet":
            # Load parquet and get full program data
            from llm_python.datasets.io import read_soar_parquet
            
            try:
                df = read_soar_parquet(identifier)
                if max_rows and len(df) > max_rows:
                    df = df.head(max_rows)
                
                # Filter programs for refinement using new strategy:
                # 1. Exclude transductive programs
                # 2. Exclude programs that solve ALL training examples (100% correct - nothing to improve)
                # 3. Include ALL other programs (0% correct might have useful partial logic)
                from llm_python.utils.refinement_utils import is_program_valid_for_refinement
                
                valid_df = df[df.apply(is_program_valid_for_refinement, axis=1)].copy()
                
                if len(valid_df) == 0:
                    raise ValueError("No valid programs found for refinement (need non-transductive, non-perfect programs)")
                
                print(f"📊 Found {len(valid_df)} refineable programs out of {len(df)} total programs (non-transductive, non-perfect)")
                
                # Group by task_id and collect program data
                for _, row in valid_df.iterrows():
                    task_id = row['task_id']
                    
                    program_info = {
                        'row_id': row.get('row_id', ''),  # Capture row_id for refinement tracking
                        'code': row.get('code', ''),
                        'correct_train_input': row['correct_train_input'],
                        'correct_test_input': row.get('correct_test_input', []),
                        'model': row.get('model', 'unknown'),
                        'reasoning': row.get('reasoning', ''),
                        'is_transductive': row.get('is_transductive', False)
                    }
                    
                    if task_id not in programs_by_task:
                        programs_by_task[task_id] = {
                            'programs': [],
                            'task_data': self.tasks.get(task_id)
                        }
                    
                    programs_by_task[task_id]['programs'].append(program_info)
                    
            except Exception as e:
                raise ValueError(f"Failed to load parquet dataset '{identifier}': {e}")
                
        elif dataset_type == "huggingface":
            # Load HF dataset and get full program data
            try:
                from datasets import load_dataset, Dataset
                ds = load_dataset(identifier, split="train")
                if max_rows and max_rows < len(ds):
                    ds = ds.select(range(max_rows))
                
                # Convert to pandas for easier filtering
                import pandas as pd
                df = ds.to_pandas()
                
                # Determine task_id column
                task_id_col = 'task_id' if 'task_id' in df.columns else 'row_id'
                if task_id_col not in df.columns:
                    raise ValueError("Dataset must contain either 'task_id' or 'row_id' column")
                
                # Filter programs for refinement using new strategy (same logic as parquet)
                from llm_python.utils.refinement_utils import is_program_valid_for_refinement
                
                if 'correct_train_input' not in df.columns:
                    raise ValueError("Dataset must contain 'correct_train_input' column for refinement mode")
                    
                valid_df = df[df.apply(is_program_valid_for_refinement, axis=1)].copy()
                
                if len(valid_df) == 0:
                    raise ValueError("No valid programs found for refinement (need non-transductive, non-perfect programs)")
                
                print(f"📊 Found {len(valid_df)} programs that are non-transductive and non-perfect for refinement out of {len(df)} total programs")
                
                # Group by task_id and collect program data
                for _, row in valid_df.iterrows():
                    task_id = row[task_id_col]
                    
                    program_info = {
                        'row_id': row.get('row_id', ''),  # Capture row_id for refinement tracking
                        'code': row.get('code', ''),
                        'correct_train_input': row['correct_train_input'],
                        'correct_test_input': row.get('correct_test_input', []),
                        'model': row.get('model', 'unknown'),
                        'reasoning': row.get('reasoning', ''),
                        'is_transductive': row.get('is_transductive', False)
                    }
                    
                    if task_id not in programs_by_task:
                        programs_by_task[task_id] = {
                            'programs': [],
                            'task_data': self.tasks.get(task_id)
                        }
                    
                    programs_by_task[task_id]['programs'].append(program_info)
                    
            except Exception as e:
                raise ValueError(f"Failed to load HuggingFace dataset '{identifier}': {e}")
        
        else:
            raise ValueError(f"Unsupported dataset type for refinement: {dataset_type}")
        
        # Remove tasks that don't have corresponding task data in our cache
        valid_programs_by_task = {}
        missing_tasks = []
        
        for task_id, data in programs_by_task.items():
            if data['task_data'] is not None:
                valid_programs_by_task[task_id] = data
            else:
                missing_tasks.append(task_id)
        
        if missing_tasks:
            print(f"Warning: {len(missing_tasks)} tasks from dataset not found in cached task data: {missing_tasks[:5]}{'...' if len(missing_tasks) > 5 else ''}")
        
        if not valid_programs_by_task:
            raise ValueError(f"No valid tasks with program data found for refinement from dataset '{identifier}'")
        
        print(f"✅ Successfully loaded program data for {len(valid_programs_by_task)} tasks for refinement")
        return valid_programs_by_task
    
    def get_all_programs_for_early_stopping(self, identifier: str) -> Dict[str, int]:
        """Get count of all non-transductive all-train-correct programs by task_id for early stopping.
        
        This loads ALL programs from the dataset (not filtered for refinement) and counts
        only the non-transductive all-train-correct ones.
        
        Args:
            identifier: Dataset identifier (HF dataset slug, parquet file, or traditional subset name)
            
        Returns:
            Dict mapping task_id -> count of non-transductive all-train-correct programs
        """
        dataset_type = self._detect_dataset_type(identifier)
        
        if dataset_type == "huggingface":
            # Load the full HuggingFace dataset without filtering
            try:
                import datasets
                dataset = datasets.load_dataset(identifier, split="train")
                df = dataset.to_pandas()
            except ImportError:
                raise ImportError("datasets library not installed. Install with: pip install datasets")
                
        elif dataset_type == "parquet":
            # Load full parquet without filtering
            import pandas as pd
            df = pd.read_parquet(identifier)
            
        else:
            raise ValueError(f"Early stopping count only supported for HuggingFace and Parquet datasets, got: {dataset_type}")
        
        if 'task_id' not in df.columns or 'correct_train_input' not in df.columns or 'is_transductive' not in df.columns:
            raise ValueError("Dataset must contain 'task_id', 'correct_train_input', and 'is_transductive' columns")
        
        # Count all-train-correct non-transductive programs per task
        task_counts = {}
        for _, row in df.iterrows():
            task_id = row['task_id']
            
            # Count only non-transductive all-train-correct programs
            if not row.get('is_transductive', False):
                correct_train_input = row.get('correct_train_input', [])
                if hasattr(correct_train_input, 'tolist'):
                    correct_train_input = correct_train_input.tolist()
                
                # Check if all-train-correct (opposite of refinement filtering)
                if isinstance(correct_train_input, list) and len(correct_train_input) > 0:
                    if all(correct_train_input):  # All-train-correct
                        task_counts[task_id] = task_counts.get(task_id, 0) + 1
                else:
                    # Single value case
                    if bool(correct_train_input):  # All-train-correct
                        task_counts[task_id] = task_counts.get(task_id, 0) + 1
        
        print(f"📊 Found {sum(task_counts.values())} all-train-correct non-transductive programs across {len(task_counts)} tasks for early stopping")
        
        return task_counts


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
