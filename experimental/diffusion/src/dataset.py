"""
Dataset and data loading utilities for ARC tasks.
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Any
import random
from pathlib import Path

from ..utils.grid_utils import grid_to_tokens, TaskAugmentation
from ..utils.task_filters import task_exceeds_max_size


class ARCDataset(Dataset):
    """
    Dataset for ARC tasks. Loads training examples from JSON files.
    Each task has a unique task_id and multiple train/test examples.
    """

    def __init__(
        self,
        data_paths: List[str],
        max_size: int = 30,
        augment: bool = True,
        include_training_test_examples: bool = True,
        subset_file: str = None,
        eval_subset_file: str = None,
        eval_weight: float = 1.0,
        max_val_examples: int = 128,
    ):
        """
        Args:
            data_paths: List of paths to JSON files containing ARC data
            max_size: Maximum grid size (grids will be padded to this size)
            augment: Whether to apply task-level data augmentation (uses all 71 D4 augmentations)
            include_training_test_examples: Whether to include test examples from training_challenges.json as training data
            subset_file: Optional path to text file containing training_challenges task IDs to include (one per line)
            eval_subset_file: Optional path to text file containing evaluation_challenges task IDs to include (one per line)
            eval_weight: Weight for sampling evaluation_challenges examples (for WeightedRandomSampler)
            max_val_examples: Maximum number of validation examples from evaluation test set
        """
        self.max_size = max_size
        self.augment = augment
        self.include_training_test_examples = include_training_test_examples
        self.eval_weight = eval_weight
        self.max_val_examples = max_val_examples

        # Load training subset task IDs if provided
        self.subset_task_ids = None
        if subset_file:
            with open(subset_file, 'r') as f:
                self.subset_task_ids = set(line.strip() for line in f if line.strip())
            print(f"📋 Using training subset from {subset_file}: {len(self.subset_task_ids)} tasks")

        # Load evaluation subset task IDs if provided
        self.eval_subset_task_ids = None
        if eval_subset_file:
            with open(eval_subset_file, 'r') as f:
                self.eval_subset_task_ids = set(line.strip() for line in f if line.strip())
            print(f"📋 Using evaluation subset from {eval_subset_file}: {len(self.eval_subset_task_ids)} tasks")

        # Load all data
        self.examples = []
        self.task_id_to_idx = {}  # Map task IDs to integer indices

        # Statistics for filtering
        self.filtered_tasks = 0
        self.total_tasks = 0

        self._load_data(data_paths)

        print(f"📊 Dataset Statistics:")
        print(f"  Total tasks processed: {self.total_tasks}")
        print(f"  Tasks filtered out (grid > {max_size}): {self.filtered_tasks}")
        print(f"  Tasks remaining: {len(self.task_id_to_idx)}")
        print(f"  Examples loaded: {len(self.examples)}")

    def _task_exceeds_max_size(self, task_examples: Dict) -> bool:
        """Check if any grid in the task exceeds max_size."""
        return task_exceeds_max_size(task_examples, self.max_size)

    def _load_data(self, data_paths: List[str]):
        """Load data from JSON files and apply task-level augmentation."""
        # First collect all tasks
        all_tasks = {}
        eval_task_ids = set()  # Track which tasks are from evaluation_challenges
        task_counter = 0

        # Infer dataset from data_paths (extract arc-prize-YYYY from first path)
        dataset_name = "arc-prize-2025"  # default
        if data_paths:
            # Extract dataset name from path like "data/arc-prize-2024/..."
            parts = data_paths[0].split('/')
            for part in parts:
                if part.startswith('arc-prize-'):
                    dataset_name = part
                    break

        # Load training solutions to get outputs for training test examples
        training_solutions = self._load_solutions(f"data/{dataset_name}/arc-agi_training_solutions.json")

        # Load evaluation solutions to get outputs for evaluation test examples (for validation)
        evaluation_solutions = self._load_solutions(f"data/{dataset_name}/arc-agi_evaluation_solutions.json")

        for data_path in data_paths:
            print(f"Loading data from {data_path}")
            with open(data_path, 'r') as f:
                data = json.load(f)

            # Determine if this is evaluation challenges (only use train examples)
            is_evaluation_challenges = "evaluation" in data_path

            for task_id, task_data in data.items():
                self.total_tasks += 1

                # Apply subset filtering based on dataset type
                is_training_challenges = "training_challenges" in data_path

                # Skip if not in training subset (if specified and this is training_challenges)
                if self.subset_task_ids and is_training_challenges and task_id not in self.subset_task_ids:
                    continue

                # Skip if not in evaluation subset (if specified and this is evaluation_challenges)
                if self.eval_subset_task_ids and is_evaluation_challenges and task_id not in self.eval_subset_task_ids:
                    continue

                # Create task structure for augmentation
                task_examples = {
                    'train': [],
                    'test': []
                }

                # Process training examples (always include these)
                for example in task_data.get('train', []):
                    task_examples['train'].append({
                        'input': np.array(example['input']),
                        'output': np.array(example['output'])
                    })

                # Process test examples
                if is_evaluation_challenges:
                    # For evaluation_challenges, test examples are for validation only
                    for i, example in enumerate(task_data.get('test', [])):
                        input_grid = np.array(example['input'])

                        # Get output from evaluation_solutions.json
                        if task_id in evaluation_solutions and i < len(evaluation_solutions[task_id]):
                            output_grid = np.array(evaluation_solutions[task_id][i])
                            task_examples['test'].append({
                                'input': input_grid,
                                'output': output_grid,
                                'is_validation': True  # Mark as validation-only
                            })
                elif self.include_training_test_examples:
                    # For training_challenges.json, use training_solutions.json for test outputs
                    for i, example in enumerate(task_data.get('test', [])):
                        input_grid = np.array(example['input'])

                        # Get output from training_solutions.json
                        if task_id in training_solutions and i < len(training_solutions[task_id]):
                            output_grid = np.array(training_solutions[task_id][i])
                            task_examples['test'].append({
                                'input': input_grid,
                                'output': output_grid
                            })
                        elif 'output' in example:
                            # Fallback if output is directly in the test example
                            output_grid = np.array(example['output'])
                            task_examples['test'].append({
                                'input': input_grid,
                                'output': output_grid
                            })

                # Check if task should be filtered out due to large grids
                if self._task_exceeds_max_size(task_examples):
                    self.filtered_tasks += 1
                    continue  # Skip this task

                all_tasks[task_id] = task_examples
                if is_evaluation_challenges:
                    eval_task_ids.add(task_id)

        # Now apply task-level augmentation and convert to examples
        print(f"Loaded {len(all_tasks)} tasks")

        if self.augment:
            self._apply_task_augmentation(all_tasks, eval_task_ids)  # Modifies all_tasks in-place
            print(f"Tasks after augmentation: {len(all_tasks)}")


        # Convert tasks to examples
        for task_id, task_data in all_tasks.items():
            # Map task_id to integer index
            if task_id not in self.task_id_to_idx:
                self.task_id_to_idx[task_id] = task_counter
                task_counter += 1

            task_idx = self.task_id_to_idx[task_id]

            # Convert all examples in this task
            for split in ['train', 'test']:
                for example in task_data[split]:
                    self.examples.append({
                        'task_id': task_id,
                        'task_idx': task_idx,
                        'input_grid': example['input'],
                        'output_grid': example['output'],
                        'split': split,
                        'd4_idx': example.get('d4_idx', 0),
                        'color_shift': example.get('color_shift', 0),
                        'is_validation': example.get('is_validation', False),
                        'from_eval_dataset': task_id in eval_task_ids  # Track which dataset
                    })

    def _generate_all_augmentations(self) -> List[tuple]:
        """
        Generate all unique augmentation combinations using D4 symmetry group.

        D4 (dihedral group of order 8) has 8 unique spatial transformations:
        0: identity
        1: rotate 90°
        2: rotate 180°
        3: rotate 270°
        4: flip horizontal
        5: flip vertical
        6: flip main diagonal
        7: flip anti-diagonal

        Combined with 9 color shifts (0-8), we get 8 × 9 = 72 total combinations.
        Minus 1 identity (d4=0, color=0) = 71 unique augmentations.

        Returns:
            List of (d4_idx, color_shift) tuples
        """
        d4_transformations = list(range(8))  # 8 D4 group elements
        color_shifts = list(range(9))  # 0-8

        # Generate all combinations deterministically
        all_augmentations = [
            (d4, color)
            for color in color_shifts
            for d4 in d4_transformations
        ]

        # Remove identity (no-op)
        all_augmentations = [aug for aug in all_augmentations if aug != (0, 0)]

        return all_augmentations

    def _apply_task_augmentation(self, tasks: Dict[str, Dict], eval_task_ids: set):
        """Apply task-level augmentation by adding all unique D4 augmentation combinations to each task."""
        # Generate all unique augmentations once
        all_augmentations = self._generate_all_augmentations()
        print(f"Applying {len(all_augmentations)} unique augmentations per task")

        for task_id, task_data in tasks.items():
            # Store original examples separately to avoid augmenting augmented examples
            # Skip validation examples (don't augment them)
            original_examples = {
                'train': list(task_data['train']),
                'test': [ex for ex in task_data['test'] if not ex.get('is_validation', False)]
            }

            for d4_idx, color_shift in all_augmentations:
                # Apply augmentation to all ORIGINAL examples in this task
                for split in ['train', 'test']:
                    for example in original_examples[split]:
                        # Augment grids using D4 transformation
                        input_grid = TaskAugmentation.apply_d4_augmentation(example['input'], d4_idx)
                        input_grid = TaskAugmentation.apply_color_cycle_augmentation(input_grid, color_shift)

                        output_grid = TaskAugmentation.apply_d4_augmentation(example['output'], d4_idx)
                        output_grid = TaskAugmentation.apply_color_cycle_augmentation(output_grid, color_shift)

                        # Append to existing task_data
                        task_data[split].append({
                            'input': input_grid,
                            'output': output_grid,
                            'd4_idx': d4_idx,
                            'color_shift': color_shift
                        })


    def _load_solutions(self, solutions_path: str) -> Dict[str, List[List[List[int]]]]:
        """Load solutions from JSON file."""
        try:
            with open(solutions_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Solutions file not found: {solutions_path}")
            return {}

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example."""
        example = self.examples[idx]

        # Convert to tensors and pad
        input_tokens, input_h, input_w = grid_to_tokens(example['input_grid'], self.max_size)
        output_tokens, output_h, output_w = grid_to_tokens(example['output_grid'], self.max_size)

        return {
            'input_grid': input_tokens,
            'output_grid': output_tokens,
            'task_idx': torch.tensor(example['task_idx'], dtype=torch.long),
            'height': torch.tensor(output_h, dtype=torch.long),
            'width': torch.tensor(output_w, dtype=torch.long),
            'd4_idx': torch.tensor(example['d4_idx'], dtype=torch.long),
            'color_shift': torch.tensor(example['color_shift'], dtype=torch.long),
            'task_id': example['task_id']  # Keep string ID for debugging
        }


    def get_task_info(self) -> Dict[str, int]:
        """Get information about tasks in the dataset."""
        return {
            'num_tasks': len(self.task_id_to_idx),
            'num_examples': len(self.examples),
            'task_id_to_idx': self.task_id_to_idx
        }



class ARCDataLoader:
    """Convenience class for creating data loaders."""

    @staticmethod
    def create_train_loader(
        train_data_paths: List[str],
        batch_size: int = 64,
        max_size: int = 30,
        augment: bool = True,
        num_workers: int = 4,
        shuffle: bool = True
    ) -> DataLoader:
        """Create training data loader."""
        dataset = ARCDataset(
            data_paths=train_data_paths,
            max_size=max_size,
            augment=augment,
            include_training_test_examples=True
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True  # For stable training
        )

    @staticmethod
    def create_eval_loader(
        eval_data_path: str,
        batch_size: int = 32,
        max_size: int = 30,
        num_workers: int = 2
    ) -> DataLoader:
        """Create evaluation data loader (for test examples with known outputs)."""
        dataset = ARCDataset(
            data_paths=[eval_data_path],
            max_size=max_size,
            augment=False,  # No augmentation for evaluation
            include_training_test_examples=False  # Only use actual train examples
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )


def load_arc_data_paths(data_dir: str = "data/arc-prize-2025", datasets: List[str] = None) -> Dict[str, List[str]]:
    """
    Get ARC data paths for training.

    Args:
        data_dir: Path to ARC data directory
        datasets: List of dataset names to include. Options: ['training_challenges', 'evaluation_challenges']
                 If None, defaults to both datasets.

    Returns:
        Dictionary with 'train' key containing list of dataset paths
    """
    data_dir = Path(data_dir)

    # Default to including both datasets
    if datasets is None:
        datasets = ['training_challenges', 'evaluation_challenges']

    train_paths = []

    # Map dataset names to file names
    dataset_files = {
        'training_challenges': 'arc-agi_training_challenges.json',
        'evaluation_challenges': 'arc-agi_evaluation_challenges.json'
    }

    for dataset in datasets:
        if dataset in dataset_files:
            train_paths.append(str(data_dir / dataset_files[dataset]))
        else:
            raise ValueError(f"Unknown dataset: {dataset}. Available: {list(dataset_files.keys())}")

    return {
        'train': train_paths
    }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching ARC examples.
    """
    # Stack tensors
    input_grids = torch.stack([item['input_grid'] for item in batch])
    output_grids = torch.stack([item['output_grid'] for item in batch])
    task_indices = torch.stack([item['task_idx'] for item in batch])
    heights = torch.stack([item['height'] for item in batch])
    widths = torch.stack([item['width'] for item in batch])
    d4_indices = torch.stack([item['d4_idx'] for item in batch])
    color_shifts = torch.stack([item['color_shift'] for item in batch])

    return {
        'input_grid': input_grids,
        'output_grid': output_grids,
        'task_idx': task_indices,
        'height': heights,
        'width': widths,
        'd4_idx': d4_indices,
        'color_shift': color_shifts,
        'task_ids': [item['task_id'] for item in batch]  # Keep string IDs for debugging
    }


# Utility functions for working with ARC data
def load_solutions(solutions_path: str) -> Dict[str, List[List[List[int]]]]:
    """Load solutions from JSON file."""
    with open(solutions_path, 'r') as f:
        return json.load(f)


def create_submission_format(predictions: Dict[str, List[np.ndarray]]) -> Dict[str, List[Dict[str, List[List[int]]]]]:
    """
    Convert predictions to ARC submission format.

    Args:
        predictions: Dict mapping task_id to list of predicted grids (as numpy arrays)

    Returns:
        Dictionary in ARC submission format
    """
    submission = {}
    for task_id, pred_grids in predictions.items():
        submission[task_id] = []
        for pred_grid in pred_grids:
            submission[task_id].append({
                "attempt_1": pred_grid.tolist(),
                "attempt_2": pred_grid.tolist()  # Duplicate for now
            })

    return submission