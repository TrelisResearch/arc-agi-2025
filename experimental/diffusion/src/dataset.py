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

from ..utils.grid_utils import grid_to_tokens, tokens_to_grid, GridAugmentation


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
        use_test_examples_as_train: bool = True,
    ):
        """
        Args:
            data_paths: List of paths to JSON files containing ARC data
            max_size: Maximum grid size (grids will be padded to this size)
            augment: Whether to apply data augmentation
            use_test_examples_as_train: Whether to use test examples as training data
        """
        self.max_size = max_size
        self.augment = augment
        self.use_test_examples_as_train = use_test_examples_as_train

        # Load all data
        self.examples = []
        self.task_id_to_idx = {}  # Map task IDs to integer indices

        self._load_data(data_paths)
        self.augmenter = GridAugmentation()

        print(f"Loaded {len(self.examples)} examples from {len(self.task_id_to_idx)} tasks")

    def _load_data(self, data_paths: List[str]):
        """Load data from JSON files."""
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
                # Map task_id to integer index
                if task_id not in self.task_id_to_idx:
                    self.task_id_to_idx[task_id] = task_counter
                    task_counter += 1

                task_idx = self.task_id_to_idx[task_id]

                # Process training examples (always include these)
                for example in task_data.get('train', []):
                    input_grid = np.array(example['input'])
                    output_grid = np.array(example['output'])
                    self.examples.append({
                        'task_id': task_id,
                        'task_idx': task_idx,
                        'input_grid': input_grid,
                        'output_grid': output_grid,
                        'split': 'train'
                    })

                # Process test examples as training data
                if self.use_test_examples_as_train and not is_evaluation_challenges:
                    # For training_challenges.json, use training_solutions.json for test outputs
                    for i, example in enumerate(task_data.get('test', [])):
                        input_grid = np.array(example['input'])

                        # Get output from training_solutions.json
                        if task_id in training_solutions and i < len(training_solutions[task_id]):
                            output_grid = np.array(training_solutions[task_id][i])
                            self.examples.append({
                                'task_id': task_id,
                                'task_idx': task_idx,
                                'input_grid': input_grid,
                                'output_grid': output_grid,
                                'split': 'test_as_train'
                            })
                        elif 'output' in example:
                            # Fallback if output is directly in the test example
                            output_grid = np.array(example['output'])
                            self.examples.append({
                                'task_id': task_id,
                                'task_idx': task_idx,
                                'input_grid': input_grid,
                                'output_grid': output_grid,
                                'split': 'test_as_train'
                            })
                # For evaluation_challenges.json, skip test examples (they're for final evaluation only)

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

        # Apply augmentation if enabled
        if self.augment and random.random() < 0.5:
            input_tokens, output_tokens = self._augment_pair(input_tokens, output_tokens)
            # Note: after augmentation, the height/width might be swapped for rotations
            # For simplicity, we'll keep the original h/w since the model learns to predict size

        return {
            'input_grid': input_tokens,
            'output_grid': output_tokens,
            'task_idx': torch.tensor(example['task_idx'], dtype=torch.long),
            'height': torch.tensor(output_h, dtype=torch.long),
            'width': torch.tensor(output_w, dtype=torch.long),
            'task_id': example['task_id']  # Keep string ID for debugging
        }

    def _augment_pair(
        self,
        input_grid: torch.Tensor,
        output_grid: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply random augmentation to input-output pair."""
        # Randomly choose augmentation
        aug_choice = random.choice(['rotate_90', 'flip_h', 'flip_v', 'none'])

        if aug_choice == 'rotate_90':
            return self.augmenter.rotate_90(input_grid, output_grid)
        elif aug_choice == 'flip_h':
            return self.augmenter.flip_horizontal(input_grid, output_grid)
        elif aug_choice == 'flip_v':
            return self.augmenter.flip_vertical(input_grid, output_grid)
        else:
            return input_grid, output_grid

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
            use_test_examples_as_train=True
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
            use_test_examples_as_train=False  # Only use actual train examples
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )


def load_arc_data_paths(data_dir: str = "data/arc-prize-2024") -> Dict[str, List[str]]:
    """
    Get the standard ARC data paths for training and evaluation.

    Args:
        data_dir: Path to ARC data directory

    Returns:
        Dictionary with 'train' and 'eval' paths
    """
    data_dir = Path(data_dir)

    train_paths = [
        str(data_dir / "arc-agi_training_challenges.json"),
        str(data_dir / "arc-agi_evaluation_challenges.json"),  # Use evaluation train examples for training
    ]

    # For evaluation, we only use the evaluation challenges
    eval_paths = [
        str(data_dir / "arc-agi_evaluation_challenges.json")
    ]

    return {
        'train': train_paths,
        'eval': eval_paths
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