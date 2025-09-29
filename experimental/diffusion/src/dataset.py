"""
Dataset and data loading utilities for ARC tasks.
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import random
from pathlib import Path

from ..utils.grid_utils import grid_to_tokens, tokens_to_grid, GridAugmentation, TaskAugmentation
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
        n_augment: int = 3,
        include_training_test_examples: bool = True,
    ):
        """
        Args:
            data_paths: List of paths to JSON files containing ARC data
            max_size: Maximum grid size (grids will be padded to this size)
            augment: Whether to apply task-level data augmentation
            n_augment: Number of augmented versions per task (if augment=True)
            include_training_test_examples: Whether to include test examples from training_challenges.json as training data
        """
        self.max_size = max_size
        self.augment = augment
        self.n_augment = n_augment
        self.include_training_test_examples = include_training_test_examples

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
        task_counter = 0

        # Load training solutions to get outputs for training test examples
        training_solutions = self._load_solutions("data/arc-prize-2024/arc-agi_training_solutions.json")

        for data_path in data_paths:
            print(f"Loading data from {data_path}")
            with open(data_path, 'r') as f:
                data = json.load(f)

            # Determine if this is evaluation challenges (only use train examples)
            is_evaluation_challenges = "evaluation" in data_path

            for task_id, task_data in data.items():
                self.total_tasks += 1

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

                # Process test examples as training data
                if self.include_training_test_examples and not is_evaluation_challenges:
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

        # Now apply task-level augmentation and convert to examples
        print(f"Loaded {len(all_tasks)} tasks")

        if self.augment:
            print(f"Applying task-level augmentation with {self.n_augment} augmentations per task")
            augmented_tasks = self._apply_task_augmentation(all_tasks)
            all_tasks.update(augmented_tasks)
            print(f"Total tasks after augmentation: {len(all_tasks)}")

        # Use hard-coded global token distribution based on ARC dataset statistics
        self.global_distribution = self._get_hardcoded_distribution()

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
                        'split': split
                    })

    def _apply_task_augmentation(self, tasks: Dict[str, Dict]) -> Dict[str, Dict]:
        """Apply task-level augmentation to create new tasks."""
        augmented_tasks = {}

        for task_id, task_data in tasks.items():
            for aug_idx in range(self.n_augment):
                # Generate random augmentation parameters
                flip_type = random.choice(['none', 'none', 'horizontal', 'vertical'])  # 'none' has 2x weight
                rotation = random.choice([0, 90, 180, 270])
                color_cycle = random.choice(range(9))  # 0-8

                # Skip no-op augmentations (identical to original)
                is_no_op = (flip_type == 'none' and rotation == 0 and color_cycle == 0)
                if is_no_op:
                    continue

                # Create augmented task
                aug_suffix = f"_aug{aug_idx}_f{flip_type[0]}_r{rotation}_c{color_cycle}"
                aug_task_id = f"{task_id}{aug_suffix}"

                augmented_task = TaskAugmentation.augment_task(
                    task_data, flip_type, rotation, color_cycle, aug_suffix
                )

                augmented_tasks[aug_task_id] = augmented_task

        return augmented_tasks

    def _get_hardcoded_distribution(self) -> torch.Tensor:
        """
        Get hard-coded global token distribution based on ARC dataset statistics.

        Noise sampling uses only actual ARC colors 0-9 (no PAD tokens):
        - Black (token 0): keeps same relative frequency vs colors
        - Colors (tokens 1-9): equal distribution among colors
        - PAD token (10): excluded from noise sampling

        Returns:
            Global probability distribution over tokens 0-9 [vocab_size=10]
        """
        token_probs = torch.zeros(10)  # Only tokens 0-9 for noise sampling (no PAD)

        # Based on original ratios: Black was 5.43%, colors were 0.34% each
        # Ratio of black to single color: 5.43 / 0.34 = ~16:1
        # So black gets 16 parts, each color gets 1 part = 25 total parts

        black_weight = 16.0
        color_weight = 1.0
        total_weight = black_weight + 9 * color_weight  # 16 + 9 = 25

        token_probs[0] = black_weight / total_weight      # Black: 16/25 = 0.64
        token_probs[1:10] = color_weight / total_weight   # Colors: 1/25 = 0.04 each

        # Normalize to ensure exact sum of 1.0
        token_probs = token_probs / token_probs.sum()

        print(f"Using hard-coded global token distribution (no PAD):")
        for i in range(10):
            print(f"  Token {i}: {token_probs[i]:.4f}")

        return token_probs

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
            'task_id': example['task_id']  # Keep string ID for debugging
        }


    def get_task_info(self) -> Dict[str, int]:
        """Get information about tasks in the dataset."""
        return {
            'num_tasks': len(self.task_id_to_idx),
            'num_examples': len(self.examples),
            'task_id_to_idx': self.task_id_to_idx
        }

    def get_global_distribution(self) -> torch.Tensor:
        """Get the global token distribution for noise sampling."""
        return self.global_distribution


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


def load_arc_data_paths(data_dir: str = "data/arc-prize-2024", datasets: List[str] = None) -> Dict[str, List[str]]:
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

    return {
        'input_grid': input_grids,
        'output_grid': output_grids,
        'task_idx': task_indices,
        'height': heights,
        'width': widths,
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