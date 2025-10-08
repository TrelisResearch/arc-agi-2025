#!/usr/bin/env python3
"""
ARC Diffusion Model Evaluation

Evaluates trained diffusion models with pass@2 scoring and detailed statistics:
- Accuracy metrics (pass@2, both correct, per-attempt accuracy)
- Copy behavior statistics (copy rate, edit accuracy, keep accuracy)
- Trajectory dynamics (delta-change curves, confidence, early-lock detection)
- Denoising visualizations for qualitative analysis

Usage:
    uv run python experimental/diffusion/evaluate.py --config experimental/diffusion/configs/smol_config.json
    uv run python experimental/diffusion/evaluate.py --config experimental/diffusion/configs/smol_config.json --limit 10
    uv run python experimental/diffusion/evaluate.py --config experimental/diffusion/configs/smol_config.json --dataset arc-prize-2025 --subset evaluation
"""
import json
import argparse
import datetime
import sys
import traceback
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Tuple, Any
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experimental.diffusion.src.model import ARCDiffusionModel
from experimental.diffusion.utils.noise_scheduler import DiscreteNoiseScheduler
from experimental.diffusion.src.training import ARCDiffusionSampler
from experimental.diffusion.src.dataset import ARCDataset, load_arc_data_paths
from experimental.diffusion.utils.grid_utils import grid_to_tokens, tokens_to_grid, TaskAugmentation
from experimental.diffusion.utils.task_filters import filter_tasks_by_max_size
from experimental.diffusion.utils.arc_colors import arc_cmap
import random
from collections import Counter


class AugmentationParams(TypedDict):
    """Parameters for a single D4 augmentation"""
    d4_idx: int  # D4 transformation index (0-7)
    color_cycle: int  # Color cycle offset (0-8)


def generate_augmentations() -> List[AugmentationParams]:
    """
    Generate all 72 D4 augmentations (8 D4 × 9 colors).

    Returns:
        List of augmentation parameters with identity (0, 0) first
    """
    aug_tuples = TaskAugmentation.generate_all_d4_augmentations()
    augmentations = [AugmentationParams(d4_idx=d4, color_cycle=color)
                     for d4, color in aug_tuples]
    return augmentations


def apply_augmentation(grid: np.ndarray, aug_params: AugmentationParams) -> np.ndarray:
    """Apply D4 augmentation to a grid."""
    grid = TaskAugmentation.apply_d4_augmentation(grid, aug_params['d4_idx'])
    grid = TaskAugmentation.apply_color_cycle_augmentation(grid, aug_params['color_cycle'])
    return grid


def reverse_augmentation(grid: np.ndarray, aug_params: AugmentationParams) -> np.ndarray:
    """Reverse D4 augmentation on a grid (de-augment)."""
    # Reverse color cycle first
    grid = TaskAugmentation.reverse_color_cycle_augmentation(grid, aug_params['color_cycle'])
    # Then reverse D4 transformation
    grid = TaskAugmentation.reverse_d4_augmentation(grid, aug_params['d4_idx'])
    return grid


def deaugment_size(height: int, width: int, aug_params: AugmentationParams) -> Tuple[int, int]:
    """
    De-augment size prediction based on D4 transformation.
    D4 transformations 1,3,6,7 swap dimensions.
    """
    return TaskAugmentation.deaugment_size_d4(height, width, aug_params['d4_idx'])


class TrajectoryStats(TypedDict):
    """Statistics about the sampling trajectory"""
    delta_change_curve: List[float]  # Fraction of cells that changed at each step
    confidence_curve: List[float]  # Mean top-1 probability at each step
    entropy_curve: List[float]  # Mean entropy at each step
    early_lock_step: Optional[int]  # First step where delta <= 1% and confidence >= 95%
    final_delta: float  # Delta at last step
    final_confidence: float  # Confidence at last step


class CopyStats(TypedDict):
    """Statistics about input copying behavior"""
    copy_rate: float  # Fraction of valid cells where pred == input
    edit_accuracy: float  # Accuracy on cells where target != input
    keep_accuracy: float  # Accuracy on cells where target == input
    num_edit_cells: int  # Number of cells where target != input
    num_keep_cells: int  # Number of cells where target == input
    num_valid_cells: int  # Total number of valid cells


class DiffusionResult(TypedDict):
    """Result of running diffusion model on a single test example"""
    test_idx: int
    input_grid: List[List[int]]  # Input grid for reference
    predicted: Optional[List[List[int]]]  # Predicted grid or None if error
    expected: List[List[int]]  # Expected output grid
    correct: bool  # Whether prediction matches expected
    error: Optional[str]  # Error message if execution failed
    pred_height: int  # Height of predicted grid
    pred_width: int  # Width of predicted grid
    size_source: str  # How the size was determined ("size_head", "ground_truth")

    # New: Copy and trajectory statistics
    copy_stats: Optional[CopyStats]  # Copy behavior statistics
    trajectory_stats: Optional[TrajectoryStats]  # Sampling trajectory statistics


class TestExampleResult(TypedDict):
    """Results for a single test example with pass@2 attempts"""
    test_idx: int
    attempt_1: DiffusionResult
    attempt_2: DiffusionResult
    pass_at_2: bool  # True if either attempt is correct
    both_correct: bool  # True if both attempts are correct


class TaskResult(TypedDict):
    """Result for a single ARC task with multiple test examples"""
    task_id: str
    timestamp: str

    # Results for each test example
    test_results: List[TestExampleResult]

    # Task-level scoring (partial credit)
    num_test_examples: int
    num_test_examples_passed: int  # Number of test examples where pass@2 = True
    task_score: float  # Fraction of test examples passed (partial credit)

    # Task metadata
    num_train_examples: int


class DiffusionInference:
    """Main inference class for ARC diffusion model"""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        num_inference_steps: Optional[int] = None,
        debug: bool = False,
        dataset: str = "arc-prize-2025",
        use_ema: bool = True
    ):
        self.model_path = model_path
        self.use_ema = use_ema
        # Set up device (prioritize CUDA > MPS > CPU)
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.num_inference_steps = num_inference_steps
        self.debug = debug

        print(f"🔥 Loading model from {model_path}")
        print(f"🖥️ Using device: {self.device}")

        # Load model
        self.model, self.config = self._load_model()
        self.noise_scheduler = DiscreteNoiseScheduler(
            num_timesteps=self.config['num_timesteps'],
            vocab_size=self.config['vocab_size'],
            schedule_type=self.config['schedule_type']
        )
        self.noise_scheduler.to(self.device)

        # Create dataset for task indexing (use the dataset being evaluated)
        data_paths = load_arc_data_paths(
            data_dir=f"data/{dataset}",
            datasets=["training_challenges", "evaluation_challenges"]
        )
        self.dataset = ARCDataset(
            data_paths=data_paths['train'],
            max_size=self.config['max_size'],
            augment=False,  # No augmentation for inference
            include_training_test_examples=True
        )
        print(f"📊 Loaded dataset with {len(self.dataset.task_id_to_idx)} tasks for task indexing")

        # Check for integrated size head in model
        if hasattr(self.model, 'include_size_head') and self.model.include_size_head:
            print(f"✓ Using integrated size head from model")

        self.sampler = ARCDiffusionSampler(
            self.model,
            self.noise_scheduler,
            self.device,
            dataset=self.dataset,
            debug=self.debug
        )

        if self.num_inference_steps is None:
            self.num_inference_steps = self.config['num_timesteps']

        print(f"✨ Model loaded: {self.model.__class__.__name__}")
        print(f"📊 Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"⚡ Inference steps: {self.num_inference_steps}")

    def _load_model(self) -> Tuple[ARCDiffusionModel, Dict]:
        """Load trained model from checkpoint"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        config = checkpoint['config']
        dataset_info = checkpoint['dataset_info']

        # Store task mapping for inference
        self.task_id_to_idx = dataset_info.get('task_id_to_idx', {})

        # Infer max_tasks from the actual model state_dict, not from dataset_info
        # The checkpoint's dataset_info might reflect a training subset, not the full model size
        state_dict = checkpoint['model_state_dict']
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        # Extract max_tasks from task_embedding shape in checkpoint
        if 'denoiser.task_embedding.weight' in state_dict:
            max_tasks = state_dict['denoiser.task_embedding.weight'].shape[0]
            print(f"Inferred max_tasks={max_tasks} from checkpoint task_embedding shape")
        else:
            max_tasks = dataset_info['num_tasks']
            print(f"Using max_tasks={max_tasks} from dataset_info (no task_embedding found)")

        self.max_tasks = max_tasks

        # Get auxiliary loss config for size head parameters
        aux_config = config.get('auxiliary_loss', {})
        include_size_head = aux_config.get('include_size_head', True)
        size_head_hidden_dim = aux_config.get('size_head_hidden_dim', None)

        # Create noise scheduler (needed for log(alpha_bar) timestep embedding)
        noise_scheduler = DiscreteNoiseScheduler(
            num_timesteps=config['num_timesteps'],
            vocab_size=config['vocab_size'],
            schedule_type=config['schedule_type']
        )

        # Recreate model with correct max_tasks
        model = ARCDiffusionModel(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            max_size=config['max_size'],
            max_tasks=max_tasks,
            embedding_dropout=config.get('embedding_dropout', 0.1),
            sc_dropout_prob=config.get('sc_dropout_prob', 0.5),
            include_size_head=include_size_head,
            size_head_hidden_dim=size_head_hidden_dim,
            noise_scheduler=noise_scheduler
        )

        # Load weights - always load base state_dict first, then apply EMA if needed
        model.load_state_dict(state_dict)

        if self.use_ema and 'ema_state_dict' in checkpoint:
            print(f"✓ Using EMA weights for inference")
            ema_state = checkpoint['ema_state_dict']
            # Apply EMA weights to model (overwrite all matching tensors with EMA values)
            model_state = model.state_dict()
            for name in model_state.keys():
                if name in ema_state:
                    # Convert EMA weight to model's dtype if needed (EMA stores in fp32)
                    target_dtype = model_state[name].dtype
                    model_state[name].copy_(ema_state[name].to(target_dtype))
        else:
            if self.use_ema and 'ema_state_dict' not in checkpoint:
                print(f"⚠️  EMA weights not found in checkpoint, using training weights")
            elif not self.use_ema:
                print(f"Using training weights (EMA disabled)")

        model.to(self.device)
        model.eval()

        return model, config

    def get_task_idx(self, task_id: str) -> int:
        """Get task index for a given task ID, handling unknown tasks."""
        if task_id in self.task_id_to_idx:
            task_idx = self.task_id_to_idx[task_id]
            # DEBUG: Print task embedding info (first call only)
            if not hasattr(self, '_debug_printed_task_embedding'):
                with torch.no_grad():
                    task_idx_tensor = torch.tensor([task_idx], device=self.device)
                    task_emb = self.model.denoiser.task_embedding(task_idx_tensor)
                    print(f"\n🔍 DEBUG: Inference task embedding verification")
                    print(f"  Task ID: {task_id}")
                    print(f"  Task index (from task_id_to_idx): {task_idx}")
                    print(f"  Task embedding (first 10 dims): {task_emb[0, :10].cpu().numpy()}")
                    print(f"  Task embedding norm: {task_emb.norm().item():.4f}\n")
                self._debug_printed_task_embedding = True
            return task_idx
        else:
            # For unknown tasks, use a default task index (0)
            # Could also raise an error or use a special "unknown" token
            print(f"Warning: Unknown task {task_id}, using default task index 0")
            return 0

    def predict_size_with_majority_voting(
        self,
        input_grid: np.ndarray,
        task_idx: int,
        augmentations: List[AugmentationParams],
        batch_size: int,
        print_stats: bool = False
    ) -> Tuple[int, int]:
        """
        Predict output size using majority voting over augmented inputs with batched inference.

        Args:
            input_grid: Original input grid
            task_idx: Task index
            augmentations: List of augmentation parameters
            batch_size: Batch size for inference
            print_stats: Whether to print histogram statistics

        Returns:
            (height, width): Majority-voted size
        """
        size_predictions = []

        # Process augmentations in batches
        for batch_start in range(0, len(augmentations), batch_size):
            batch_augs = augmentations[batch_start:batch_start + batch_size]
            batch_inputs = []
            batch_d4 = []
            batch_color_shifts = []

            # Apply augmentations to create batch
            for aug_params in batch_augs:
                aug_input = apply_augmentation(input_grid, aug_params)
                input_tokens, _, _ = grid_to_tokens(aug_input, max_size=self.config['max_size'])
                batch_inputs.append(input_tokens)

                # Get D4 and color shift indices
                d4_idx = aug_params['d4_idx']
                color_shift_idx = aug_params['color_cycle']

                batch_d4.append(d4_idx)
                batch_color_shifts.append(color_shift_idx)

            # Stack into batch
            input_batch = torch.stack(batch_inputs).to(self.device)
            task_ids = torch.tensor([task_idx] * len(batch_augs)).to(self.device)
            d4_batch = torch.tensor(batch_d4, dtype=torch.long).to(self.device)
            color_shift_batch = torch.tensor(batch_color_shifts, dtype=torch.long).to(self.device)

            # Get size predictions for batch with augmentation parameters
            with torch.no_grad():
                pred_heights, pred_widths = self.model.predict_sizes(
                    input_batch, task_ids, d4_batch, color_shift_batch
                )

            # De-augment each size prediction
            for i, aug_params in enumerate(batch_augs):
                pred_height = pred_heights[i].item()
                pred_width = pred_widths[i].item()
                deaug_height, deaug_width = deaugment_size(pred_height, pred_width, aug_params)
                size_predictions.append((deaug_height, deaug_width))

        # Majority vote
        size_counter = Counter(size_predictions)
        majority_size, count = size_counter.most_common(1)[0]

        if print_stats:
            print(f"  Size prediction histogram (n={len(augmentations)}):")
            for size, cnt in size_counter.most_common(5):
                print(f"    {size[0]}×{size[1]}: {cnt} ({cnt/len(augmentations):.1%})")

        return majority_size

    def predict_with_confidence_split_voting(
        self,
        input_grid: np.ndarray,
        task_idx: int,
        augmentations: List[AugmentationParams],
        batch_size: int,
        print_stats: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Smart majority voting that:
        1. Predicts sizes with all augmentations
        2. Takes top-2 size predictions
        3. Splits augmentations proportionally to confidence
        4. Generates outputs for each size
        5. Pools all predictions and returns top-2 distinct outputs

        Returns:
            (prediction_1, prediction_2): Top 2 distinct predictions
        """
        # Step 1: Get size predictions with majority voting
        size_predictions = []

        for batch_start in range(0, len(augmentations), batch_size):
            batch_augs = augmentations[batch_start:batch_start + batch_size]
            batch_inputs = []
            batch_d4 = []
            batch_color_shifts = []

            for aug_params in batch_augs:
                aug_input = apply_augmentation(input_grid, aug_params)
                input_tokens, _, _ = grid_to_tokens(aug_input, max_size=self.config['max_size'])
                batch_inputs.append(input_tokens)

                d4_idx = aug_params['d4_idx']
                color_shift_idx = aug_params['color_cycle']

                batch_d4.append(d4_idx)
                batch_color_shifts.append(color_shift_idx)

            input_batch = torch.stack(batch_inputs).to(self.device)
            task_ids = torch.tensor([task_idx] * len(batch_augs)).to(self.device)
            d4_batch = torch.tensor(batch_d4, dtype=torch.long).to(self.device)
            color_shift_batch = torch.tensor(batch_color_shifts, dtype=torch.long).to(self.device)

            with torch.no_grad():
                pred_heights, pred_widths = self.model.predict_sizes(
                    input_batch, task_ids, d4_batch, color_shift_batch
                )

            for i, aug_params in enumerate(batch_augs):
                pred_height = pred_heights[i].item()
                pred_width = pred_widths[i].item()
                deaug_height, deaug_width = deaugment_size(pred_height, pred_width, aug_params)
                size_predictions.append((deaug_height, deaug_width))

        # Step 2: Get top-2 size predictions and their counts
        size_counter = Counter(size_predictions)
        top_2_sizes = size_counter.most_common(2)

        if print_stats:
            print(f"  Size prediction histogram (n={len(augmentations)}):")
            for size, cnt in size_counter.most_common(5):
                print(f"    {size[0]}×{size[1]}: {cnt} ({cnt/len(augmentations):.1%})")

        # Handle case where there's only 1 unique size
        if len(top_2_sizes) == 1:
            size1, count1 = top_2_sizes[0]
            size2, count2 = size1, 0  # Duplicate the only size
        else:
            size1, count1 = top_2_sizes[0]
            size2, count2 = top_2_sizes[1]

        # Step 3: Split augmentations proportionally
        # Assign each augmentation to size1 or size2 based on their relative confidence
        size1_augmentations = []
        size2_augmentations = []

        for i, (aug, size_pred) in enumerate(zip(augmentations, size_predictions)):
            if size_pred == size1:
                size1_augmentations.append(aug)
            elif size_pred == size2:
                size2_augmentations.append(aug)
            else:
                # For other sizes, assign proportionally to top-2
                if len(size1_augmentations) < count1:
                    size1_augmentations.append(aug)
                else:
                    size2_augmentations.append(aug)

        if print_stats:
            print(f"  Split: {len(size1_augmentations)} augs for {size1[0]}×{size1[1]}, {len(size2_augmentations)} augs for {size2[0]}×{size2[1]}")

        # Step 4: Generate outputs for each size
        all_predictions = []

        for target_size, size_augs in [(size1, size1_augmentations), (size2, size2_augmentations)]:
            if not size_augs:
                continue

            pred_height, pred_width = target_size

            for batch_start in range(0, len(size_augs), batch_size):
                batch_augs = size_augs[batch_start:batch_start + batch_size]
                batch_inputs = []
                batch_d4 = []
                batch_color_shifts = []

                for aug_params in batch_augs:
                    aug_input = apply_augmentation(input_grid, aug_params)
                    input_tokens, _, _ = grid_to_tokens(aug_input, max_size=self.config['max_size'])
                    batch_inputs.append(input_tokens)

                    d4_idx = aug_params['d4_idx']
                    color_shift_idx = aug_params['color_cycle']

                    batch_d4.append(d4_idx)
                    batch_color_shifts.append(color_shift_idx)

                input_batch = torch.stack(batch_inputs).to(self.device)
                task_ids = torch.tensor([task_idx] * len(batch_augs)).to(self.device)
                d4_batch = torch.tensor(batch_d4, dtype=torch.long).to(self.device)
                color_shift_batch = torch.tensor(batch_color_shifts, dtype=torch.long).to(self.device)

                predicted_grids, _, _ = self.sample_with_steps(
                    input_batch, task_ids, pred_height, pred_width,
                    d4_idx=d4_batch, color_shift=color_shift_batch
                )

                for i, aug_params in enumerate(batch_augs):
                    pred_grid = predicted_grids[i].cpu().numpy()
                    pred_grid = pred_grid[:pred_height, :pred_width]
                    deaug_pred = reverse_augmentation(pred_grid, aug_params)
                    all_predictions.append(deaug_pred)

        # Step 5: Pool and get top-2 distinct predictions
        pred_tuples = [tuple(map(tuple, pred)) for pred in all_predictions]
        pred_counter = Counter(pred_tuples)

        if print_stats:
            unique_preds = len(pred_counter)
            print(f"  Output prediction histogram (n={len(all_predictions)}, unique={unique_preds}):")
            for i, (pred_tuple, cnt) in enumerate(pred_counter.most_common(min(5, unique_preds))):
                print(f"    Prediction {i+1}: {cnt} ({cnt/len(all_predictions):.1%})")

        # Get top 2 distinct predictions
        top_preds = pred_counter.most_common(2)
        pred1 = np.array(top_preds[0][0])
        pred2 = np.array(top_preds[1][0]) if len(top_preds) > 1 else pred1

        return pred1, pred2

    def predict_with_majority_voting(
        self,
        input_grid: np.ndarray,
        task_idx: int,
        pred_height: int,
        pred_width: int,
        augmentations: List[AugmentationParams],
        batch_size: int,
        print_stats: bool = False
    ) -> np.ndarray:
        """
        Generate predictions using majority voting over augmented inputs with batched inference.

        Args:
            input_grid: Original input grid
            task_idx: Task index
            pred_height: Predicted output height
            pred_width: Predicted output width
            augmentations: List of augmentation parameters
            batch_size: Batch size for inference
            print_stats: Whether to print histogram statistics

        Returns:
            Majority-voted prediction grid
        """
        all_predictions = []

        # Process augmentations in batches
        for batch_start in range(0, len(augmentations), batch_size):
            batch_augs = augmentations[batch_start:batch_start + batch_size]
            batch_inputs = []
            batch_d4 = []
            batch_color_shifts = []

            # Apply augmentations to create batch
            for aug_params in batch_augs:
                aug_input = apply_augmentation(input_grid, aug_params)
                input_tokens, _, _ = grid_to_tokens(aug_input, max_size=self.config['max_size'])
                batch_inputs.append(input_tokens)

                d4_idx = aug_params['d4_idx']
                color_shift_idx = aug_params['color_cycle']

                batch_d4.append(d4_idx)
                batch_color_shifts.append(color_shift_idx)

            # Stack into batch
            input_batch = torch.stack(batch_inputs).to(self.device)
            task_ids = torch.tensor([task_idx] * len(batch_augs)).to(self.device)
            d4_batch = torch.tensor(batch_d4, dtype=torch.long).to(self.device)
            color_shift_batch = torch.tensor(batch_color_shifts, dtype=torch.long).to(self.device)

            # Run diffusion sampling with augmentation parameters
            predicted_grids, _, _ = self.sample_with_steps(
                input_batch, task_ids, pred_height, pred_width,
                d4_idx=d4_batch, color_shift=color_shift_batch
            )

            # De-augment each prediction and store
            for i, aug_params in enumerate(batch_augs):
                pred_grid = predicted_grids[i].cpu().numpy()
                # Crop to predicted size
                pred_grid = pred_grid[:pred_height, :pred_width]
                # De-augment
                deaug_pred = reverse_augmentation(pred_grid, aug_params)
                all_predictions.append(deaug_pred)

        # Majority vote across all predictions
        # Convert each prediction to a tuple of tuples for hashing
        pred_tuples = [tuple(map(tuple, pred)) for pred in all_predictions]
        pred_counter = Counter(pred_tuples)
        majority_pred_tuple, count = pred_counter.most_common(1)[0]

        if print_stats:
            unique_preds = len(pred_counter)
            print(f"  Output prediction histogram (n={len(all_predictions)}, unique={unique_preds}):")
            for i, (pred_tuple, cnt) in enumerate(pred_counter.most_common(min(5, unique_preds))):
                print(f"    Prediction {i+1}: {cnt} ({cnt/len(all_predictions):.1%})")

        # Convert back to numpy array
        majority_pred = np.array(majority_pred_tuple)

        return majority_pred

    def _load_solutions(self, dataset: str) -> Dict[str, List[List[List[int]]]]:
        """Load solutions from appropriate solutions file."""
        # Extract dataset name from path like "arc-prize-2024/evaluation"
        dataset_name = dataset.split('/')[0] if '/' in dataset else "arc-prize-2025"

        if 'training' in dataset:
            solutions_path = f"data/{dataset_name}/arc-agi_training_solutions.json"
        elif 'evaluation' in dataset:
            solutions_path = f"data/{dataset_name}/arc-agi_evaluation_solutions.json"
        else:
            return {}  # No solutions for test set

        try:
            with open(solutions_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Solutions file not found: {solutions_path}")
            return {}

    def sample_with_steps(
        self,
        input_grids: torch.Tensor,
        task_indices: torch.Tensor,
        pred_height: int,
        pred_width: int,
        d4_idx: Optional[torch.Tensor] = None,
        color_shift: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[np.ndarray], TrajectoryStats]:
        """
        Sample with intermediate step capture and trajectory statistics.

        Returns:
            final_prediction: Final denoised output
            intermediate_steps: List of grids at different denoising timesteps
            trajectory_stats: Statistics about the sampling trajectory
        """
        self.model.eval()

        batch_size = input_grids.shape[0]
        max_size = input_grids.shape[1]
        num_inference_steps = self.num_inference_steps

        # Initialize with uniform random noise
        x_t = torch.randint(0, 10, (batch_size, max_size, max_size), device=self.device)
        x_prev = x_t.clone()

        # Storage for intermediate steps (grid, timestep) tuples
        intermediate_steps = []
        capture_interval = max(1, num_inference_steps // 6)  # Capture ~6 evenly spaced steps

        # Storage for trajectory statistics
        delta_change_curve = []
        confidence_curve = []
        entropy_curve = []
        early_lock_step = None

        # Denoising loop: create timesteps and compute logSNR for each
        T_train = self.noise_scheduler.num_timesteps  # e.g., 128
        timesteps = torch.round(
            torch.linspace(T_train - 1, 0, num_inference_steps)
        ).long().to(self.device)  # e.g., [127, 123, 119, ..., 4, 0] for 32 steps

        # Create valid region mask for statistics (only compute within predicted size)
        valid_mask = torch.zeros((batch_size, max_size, max_size), dtype=torch.bool, device=self.device)
        for b in range(batch_size):
            valid_mask[b, :pred_height, :pred_width] = True

        # Create float mask for model
        mask_float = valid_mask.float()

        # Initialize self-conditioning buffer
        sc_p0 = None

        with torch.no_grad():
            for i, t in enumerate(timesteps):
                t_batch = t.repeat(batch_size)

                # Compute logSNR from timestep for model input
                alpha_bars = self.noise_scheduler.alpha_bars[t_batch].clamp(1e-6, 1-1e-6).to(self.device)
                logsnr = torch.log(alpha_bars) - torch.log1p(-alpha_bars)

                # Calculate self-conditioning gain based on alpha_bar (noise level)
                from experimental.diffusion.utils.noise_scheduler import sc_gain_from_abar
                sc_gain = sc_gain_from_abar(t_batch, self.noise_scheduler)

                # Forward pass with self-conditioning (no augmentation during inference)
                logits = self.model(x_t, input_grids, task_indices, logsnr,
                                   d4_idx=d4_idx, color_shift=color_shift,
                                   masks=mask_float, sc_p0=sc_p0, sc_gain=sc_gain)

                # Get predicted probabilities
                probs = torch.softmax(logits, dim=-1)

                # Update self-conditioning buffer with current predictions
                sc_p0 = probs

                # Compute trajectory statistics (only on valid region)
                valid_probs = probs[valid_mask]  # [N_valid, vocab_size]

                # Confidence: mean top-1 probability
                top1_probs, _ = torch.max(valid_probs, dim=-1)
                mean_confidence = top1_probs.mean().item()
                confidence_curve.append(mean_confidence)

                # Entropy: mean entropy across valid cells
                # H = -sum(p * log(p))
                log_probs = torch.log(valid_probs + 1e-10)
                entropy = -(valid_probs * log_probs).sum(dim=-1)
                mean_entropy = entropy.mean().item()
                entropy_curve.append(mean_entropy)

                # Sample or argmax (use .item() for tensor comparison)
                if t.item() > 0:
                    x_t = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(x_t.shape)
                else:
                    x_t = torch.argmax(logits, dim=-1)

                # Apply size masking
                for b in range(batch_size):
                    if pred_height < max_size:
                        x_t[b, pred_height:, :] = 0
                    if pred_width < max_size:
                        x_t[b, :, pred_width:] = 0

                # Compute delta-change (fraction of cells that changed in valid region)
                if i > 0:
                    changed = (x_t != x_prev) & valid_mask
                    delta = changed.sum().item() / valid_mask.sum().item()
                    delta_change_curve.append(delta)

                    # Check for early-lock: delta <= 1% and confidence >= 95%
                    if early_lock_step is None and delta <= 0.01 and mean_confidence >= 0.95:
                        early_lock_step = i

                x_prev = x_t.clone()

                # Capture intermediate steps with their timestep
                if i % capture_interval == 0 or i == num_inference_steps - 1:
                    intermediate_steps.append((x_t[0].cpu().numpy().copy(), t.item()))

        # Create trajectory stats
        trajectory_stats = TrajectoryStats(
            delta_change_curve=delta_change_curve,
            confidence_curve=confidence_curve,
            entropy_curve=entropy_curve,
            early_lock_step=early_lock_step,
            final_delta=delta_change_curve[-1] if delta_change_curve else 0.0,
            final_confidence=confidence_curve[-1] if confidence_curve else 0.0
        )

        return x_t, intermediate_steps, trajectory_stats

    def visualize_denoising_progression(
        self,
        input_grid: np.ndarray,
        ground_truth: np.ndarray,
        intermediate_steps: List[Tuple[np.ndarray, int]],
        task_id: str,
        save_path: Optional[str] = None
    ):
        """
        Create visualization showing input, ground truth, and denoising progression.

        Args:
            input_grid: Input grid
            ground_truth: Expected output
            intermediate_steps: List of (grid, timestep) tuples
            task_id: Task identifier
            save_path: Path to save the visualization
        """
        # Handle None intermediate_steps
        if intermediate_steps is None:
            print(f"⚠️  No intermediate steps to visualize for {task_id}")
            return

        # Create figure with subplots
        n_steps = len(intermediate_steps)
        n_cols = min(n_steps + 2, 8)  # Input + GT + up to 6 intermediate steps
        n_rows = (n_steps + 2 + n_cols - 1) // n_cols  # Calculate rows needed

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        # Use ARC standard colors
        # Plot input
        ax = axes.flat[0]
        im = ax.imshow(input_grid, cmap=arc_cmap, vmin=0, vmax=9)
        ax.set_title(f"Input\n{input_grid.shape[0]}×{input_grid.shape[1]}", fontsize=10)
        ax.axis('off')

        # Plot ground truth
        ax = axes.flat[1]
        im = ax.imshow(ground_truth, cmap=arc_cmap, vmin=0, vmax=9)
        ax.set_title(f"Ground Truth\n{ground_truth.shape[0]}×{ground_truth.shape[1]}", fontsize=10)
        ax.axis('off')

        # Plot intermediate denoising steps
        step_indices = np.linspace(0, len(intermediate_steps)-1, min(n_cols-2, len(intermediate_steps)), dtype=int)
        for i, step_idx in enumerate(step_indices):
            ax = axes.flat[i + 2]
            grid, timestep = intermediate_steps[step_idx]
            im = ax.imshow(grid, cmap=arc_cmap, vmin=0, vmax=9)
            # Show the actual timestep value and predicted size
            ax.set_title(f"t={timestep}\n{grid.shape[0]}×{grid.shape[1]}", fontsize=10)
            ax.axis('off')

        # Hide any unused subplots
        for i in range(len(step_indices) + 2, len(axes.flat)):
            axes.flat[i].axis('off')

        # Add main title and colorbar
        fig.suptitle(f"Denoising Progression - Task: {task_id}", fontsize=12, y=1.02)

        # Add colorbar
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal',
                           pad=0.05, aspect=30, shrink=0.8)
        cbar.set_label('Token Value')
        cbar.set_ticks(range(10))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            print(f"📊 Saved denoising visualization to: {save_path}")

        plt.close()

    def compute_copy_stats(
        self,
        pred: np.ndarray,
        input_grid: np.ndarray,
        target: np.ndarray
    ) -> CopyStats:
        """
        Compute copy statistics comparing prediction to input and target.

        Args:
            pred: Predicted output grid
            input_grid: Input grid (conditioned on during sampling)
            target: Ground truth output grid

        Returns:
            CopyStats with copy_rate, edit_accuracy, keep_accuracy
        """
        # Ensure all grids have the same shape
        if pred.shape != target.shape:
            # If shapes don't match, we can't compute meaningful stats
            return CopyStats(
                copy_rate=0.0,
                edit_accuracy=0.0,
                keep_accuracy=0.0,
                num_edit_cells=0,
                num_keep_cells=0,
                num_valid_cells=0
            )

        # If input is different size, resize it to match (pad with 0s or crop)
        if input_grid.shape != target.shape:
            # Pad or crop input to match target size
            h, w = target.shape
            resized_input = np.zeros_like(target)
            h_min = min(h, input_grid.shape[0])
            w_min = min(w, input_grid.shape[1])
            resized_input[:h_min, :w_min] = input_grid[:h_min, :w_min]
            input_grid = resized_input

        # Create masks
        # Valid mask: all non-PAD cells in target (assuming target doesn't contain PAD)
        valid_mask = np.ones_like(target, dtype=bool)
        num_valid_cells = valid_mask.sum()

        if num_valid_cells == 0:
            return CopyStats(
                copy_rate=0.0,
                edit_accuracy=0.0,
                keep_accuracy=0.0,
                num_edit_cells=0,
                num_keep_cells=0,
                num_valid_cells=0
            )

        # Edit mask: cells where target != input (requires transformation)
        edit_mask = valid_mask & (target != input_grid)
        num_edit_cells = edit_mask.sum()

        # Keep mask: cells where target == input (should be preserved)
        keep_mask = valid_mask & (target == input_grid)
        num_keep_cells = keep_mask.sum()

        # Copy rate: fraction of valid cells where pred == input
        copy_rate = (pred[valid_mask] == input_grid[valid_mask]).mean()

        # Edit accuracy: accuracy on cells that need to be changed
        if num_edit_cells > 0:
            edit_accuracy = (pred[edit_mask] == target[edit_mask]).mean()
        else:
            edit_accuracy = 1.0  # No edits needed, vacuously true

        # Keep accuracy: accuracy on cells that should be preserved
        if num_keep_cells > 0:
            keep_accuracy = (pred[keep_mask] == target[keep_mask]).mean()
        else:
            keep_accuracy = 1.0  # No cells to preserve, vacuously true

        return CopyStats(
            copy_rate=float(copy_rate),
            edit_accuracy=float(edit_accuracy),
            keep_accuracy=float(keep_accuracy),
            num_edit_cells=int(num_edit_cells),
            num_keep_cells=int(num_keep_cells),
            num_valid_cells=int(num_valid_cells)
        )

    def predict_single(self, input_grid: np.ndarray, task_idx: int, task_id: str = None, expected_output: np.ndarray = None, capture_steps: bool = False) -> Tuple[np.ndarray, Optional[str], str]:
        """
        Run single prediction on input grid.

        Returns:
            predicted_grid: Predicted output grid (cropped to predicted/ground truth size)
            error: Error message if any
            size_source: How the size was determined ("size_head" or "ground_truth")
        """
        try:
            # Convert to tokens and add batch dimension
            input_tokens, _, _ = grid_to_tokens(input_grid, max_size=self.config['max_size'])
            input_batch = input_tokens.unsqueeze(0).to(self.device)  # [1, max_size, max_size]

            # Use task ID (ensure it's within trained range)
            task_ids = torch.tensor([task_idx]).to(self.device)

            # Get size predictions from integrated size head
            if hasattr(self.model, 'include_size_head') and self.model.include_size_head:
                # Use integrated size head
                with torch.no_grad():
                    pred_heights, pred_widths = self.model.predict_sizes(input_batch, task_ids)
                    pred_height, pred_width = pred_heights[0].item(), pred_widths[0].item()
                size_source = "integrated_size_head"
                print(f"Using integrated size head predictions: {pred_height}×{pred_width}")
            else:
                # Fallback to ground truth dimensions
                if expected_output is not None and len(expected_output) > 0:
                    pred_height, pred_width = expected_output.shape
                    print(f"⚠️  No size head available, using ground truth dimensions: {pred_height}×{pred_width}")
                    size_source = "ground_truth"
                else:
                    print("⚠️  No size head and no ground truth available, using max size")
                    pred_height, pred_width = self.config['max_size'], self.config['max_size']
                    size_source = "ground_truth"

            # Sample output - always capture steps for trajectory stats
            predicted_grids, intermediate_steps, trajectory_stats = self.sample_with_steps(
                input_batch, task_ids, pred_height, pred_width
            )

            # Crop final prediction
            full_grid = predicted_grids[0].cpu().numpy()
            predicted_grid = full_grid[:pred_height, :pred_width]

            # Also crop intermediate steps (maintaining timestep info)
            cropped_steps = []
            for grid, timestep in intermediate_steps:
                cropped = grid[:pred_height, :pred_width]
                cropped_steps.append((cropped, timestep))

            if capture_steps:
                return predicted_grid, None, size_source, cropped_steps, trajectory_stats
            else:
                return predicted_grid, None, size_source, None, trajectory_stats

        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            if self.debug:
                error_msg += f"\n{traceback.format_exc()}"
            return np.array([]), error_msg, "ground_truth", None, None

    def run_task(self, task_id: str, task_data: Dict[str, Any], dataset: str, visualize: bool = False, use_majority_voting: bool = False, print_stats: bool = False) -> TaskResult:
        """
        Run inference on a single ARC task with pass@2 scoring on all test examples.

        Args:
            task_id: Task identifier
            task_data: Task data containing train/test examples
            dataset: Dataset name to load appropriate solutions
            visualize: If True, visualize the first test example only

        Returns:
            TaskResult with results for all test examples and partial credit scoring
        """
        timestamp = datetime.datetime.now().isoformat()

        # Check for test examples
        if not task_data["test"]:
            raise ValueError(f"Task {task_id} has no test examples")

        # Get solutions for all test examples
        solutions = self._load_solutions(dataset)

        # Get correct task index
        task_idx = self.get_task_idx(task_id)

        # Process each test example
        test_results = []
        num_test_examples_passed = 0

        for test_idx, test_example in enumerate(task_data["test"]):
            input_grid = np.array(test_example["input"])

            # Get expected output for this test example
            if task_id in solutions and test_idx < len(solutions[task_id]):
                expected_output = np.array(solutions[task_id][test_idx])
            elif 'output' in test_example:
                # Fallback if solutions not found but output is in test data
                expected_output = np.array(test_example["output"])
            else:
                print(f"❌ No solution found for task {task_id}, test example {test_idx}")
                print(f"   - Solutions loaded: {len(solutions)} tasks")
                print(f"   - Task in solutions: {task_id in solutions}")
                print(f"   - Output in test_example: {'output' in test_example}")
                print(f"   - Dataset: {dataset}")
                raise ValueError(f"No solution found for task {task_id}, test example {test_idx}")

            # Run attempts for pass@2
            if use_majority_voting:
                # Use smart voting: one call generates both attempts via confidence splitting
                attempt_1, attempt_2 = self._run_smart_voting(
                    input_grid, expected_output, test_idx=test_idx, task_idx=task_idx,
                    task_id=task_id, print_stats=print_stats
                )
            else:
                # Run two independent attempts
                # Only visualize the first test example if requested
                if visualize and test_idx == 0:
                    attempt_1_result = self._run_attempt(input_grid, expected_output, test_idx=test_idx, task_idx=task_idx, task_id=task_id, capture_steps=True, use_majority_voting=False, print_stats=print_stats)
                    attempt_1, intermediate_steps = attempt_1_result

                    # Create visualization
                    output_dir = Path(self.config.get('output_dir', 'experimental/diffusion/outputs'))
                    output_dir.mkdir(parents=True, exist_ok=True)
                    vis_path = output_dir / f"denoising_progression_{task_id}_test{test_idx}.png"
                    self.visualize_denoising_progression(
                        input_grid,
                        expected_output,
                        intermediate_steps,
                        f"{task_id}_test{test_idx}",
                        save_path=str(vis_path)
                    )
                else:
                    attempt_1 = self._run_attempt(input_grid, expected_output, test_idx=test_idx, task_idx=task_idx, task_id=task_id, capture_steps=False, use_majority_voting=False, print_stats=print_stats)

                # For attempt 2, don't print stats to avoid duplicate output
                attempt_2 = self._run_attempt(input_grid, expected_output, test_idx=test_idx, task_idx=task_idx, task_id=task_id, capture_steps=False, use_majority_voting=False, print_stats=False)

            # Calculate pass@2 for this test example
            pass_at_2 = attempt_1["correct"] or attempt_2["correct"]
            both_correct = attempt_1["correct"] and attempt_2["correct"]

            if pass_at_2:
                num_test_examples_passed += 1

            test_results.append(TestExampleResult(
                test_idx=test_idx,
                attempt_1=attempt_1,
                attempt_2=attempt_2,
                pass_at_2=pass_at_2,
                both_correct=both_correct
            ))

        # Calculate task score (partial credit)
        num_test_examples = len(task_data["test"])
        task_score = num_test_examples_passed / num_test_examples if num_test_examples > 0 else 0.0

        return TaskResult(
            task_id=task_id,
            timestamp=timestamp,
            test_results=test_results,
            num_test_examples=num_test_examples,
            num_test_examples_passed=num_test_examples_passed,
            task_score=task_score,
            num_train_examples=len(task_data["train"])
        )

    def _run_smart_voting(self, input_grid: np.ndarray, expected_output: np.ndarray, test_idx: int, task_idx: int, task_id: str = None, print_stats: bool = False) -> tuple[DiffusionResult, DiffusionResult]:
        """
        Run smart confidence-split majority voting that returns top-2 predictions.
        Uses all 72 D4 augmentations.
        """
        # Generate all 72 augmentations
        augmentations = generate_augmentations()

        if print_stats:
            print(f"\n🔮 Smart majority voting for {task_id} test {test_idx} (72 augmentations):")

        # Get top-2 predictions using confidence-split voting
        pred1, pred2 = self.predict_with_confidence_split_voting(
            input_grid, task_idx, augmentations, batch_size=72, print_stats=print_stats
        )

        # Check correctness for both predictions
        def check_prediction(predicted_grid):
            correct = False
            error = None
            if len(expected_output) > 0 and len(predicted_grid) > 0:
                try:
                    if predicted_grid.shape == expected_output.shape:
                        correct = np.array_equal(predicted_grid, expected_output)
                    else:
                        error = f"Shape mismatch: predicted {predicted_grid.shape} vs expected {expected_output.shape}"
                except Exception as e:
                    error = f"Comparison failed: {str(e)}"
            elif len(predicted_grid) == 0:
                error = "No valid region extracted from prediction"

            # Compute copy statistics
            copy_stats = None
            if error is None and len(predicted_grid) > 0 and len(expected_output) > 0:
                copy_stats = self.compute_copy_stats(predicted_grid, input_grid, expected_output)

            return {
                "predicted_grid": predicted_grid.tolist() if len(predicted_grid) > 0 else [],
                "expected": expected_output.tolist(),
                "correct": correct,
                "error": error,
                "pred_height": predicted_grid.shape[0] if len(predicted_grid) > 0 else 0,
                "pred_width": predicted_grid.shape[1] if len(predicted_grid) > 0 else 0,
                "size_source": "confidence_split_voting",
                "copy_stats": copy_stats,
                "trajectory_stats": None
            }

        result1 = check_prediction(pred1)
        result2 = check_prediction(pred2)

        return result1, result2

    def _run_attempt(self, input_grid: np.ndarray, expected_output: np.ndarray, test_idx: int, task_idx: int, task_id: str = None, capture_steps: bool = False, use_majority_voting: bool = False, print_stats: bool = False) -> DiffusionResult:
        """Run a single diffusion attempt with copy and trajectory statistics"""
        if use_majority_voting:
            # Generate all 72 augmentations
            augmentations = generate_augmentations()

            if print_stats:
                print(f"\n🔮 Majority voting for {task_id} test {test_idx} (72 augmentations):")

            # Predict size with majority voting
            pred_height, pred_width = self.predict_size_with_majority_voting(
                input_grid, task_idx, augmentations, batch_size=72, print_stats=print_stats
            )

            # Predict output with majority voting
            predicted_grid = self.predict_with_majority_voting(
                input_grid, task_idx, pred_height, pred_width,
                augmentations, batch_size=72, print_stats=print_stats
            )

            error = None
            size_source = "majority_voting"
            intermediate_steps = None
            trajectory_stats = None
        else:
            result = self.predict_single(input_grid, task_idx, task_id, expected_output, capture_steps=capture_steps)
            if capture_steps:
                predicted_grid, error, size_source, intermediate_steps, trajectory_stats = result
            else:
                predicted_grid, error, size_source, intermediate_steps, trajectory_stats = result

        # Check correctness
        correct = False
        if error is None and len(expected_output) > 0 and len(predicted_grid) > 0:
            try:
                # Ensure both arrays have the same shape for comparison
                if predicted_grid.shape == expected_output.shape:
                    correct = np.array_equal(predicted_grid, expected_output)
                else:
                    error = f"Shape mismatch: predicted {predicted_grid.shape} vs expected {expected_output.shape}"
            except Exception as e:
                error = f"Comparison failed: {str(e)}"
        elif error is None and len(predicted_grid) == 0:
            error = "No valid region extracted from prediction"

        # Get predicted grid dimensions (default to 30x30 if extraction failed)
        # The predicted_grid is always cropped to the predicted size, so shape = predicted size
        pred_height = predicted_grid.shape[0] if len(predicted_grid) > 0 else 30
        pred_width = predicted_grid.shape[1] if len(predicted_grid) > 0 else 30

        # Compute copy statistics if we have valid prediction and target
        copy_stats = None
        if error is None and len(predicted_grid) > 0 and len(expected_output) > 0:
            copy_stats = self.compute_copy_stats(predicted_grid, input_grid, expected_output)

        result_dict = DiffusionResult(
            test_idx=test_idx,
            input_grid=input_grid.tolist(),
            predicted=predicted_grid.tolist() if len(predicted_grid) > 0 else None,
            expected=expected_output.tolist() if len(expected_output) > 0 else [],
            correct=correct,
            error=error,
            pred_height=pred_height,
            pred_width=pred_width,
            size_source=size_source,
            copy_stats=copy_stats,
            trajectory_stats=trajectory_stats
        )

        # Return intermediate steps if captured
        if capture_steps:
            return result_dict, intermediate_steps
        else:
            return result_dict


def calculate_metrics(results: List[TaskResult]) -> Dict[str, Any]:
    """Calculate pass@2 and other metrics from results with partial credit scoring"""
    if not results:
        return {"total_tasks": 0, "total_test_examples": 0}

    total_tasks = len(results)

    # Task-level metrics (with partial credit)
    total_task_score = sum(r["task_score"] for r in results)
    avg_task_score = total_task_score / total_tasks if total_tasks > 0 else 0.0
    perfect_tasks = sum(1 for r in results if r["task_score"] == 1.0)
    failed_tasks = sum(1 for r in results if r["task_score"] == 0.0)
    partial_tasks = sum(1 for r in results if 0.0 < r["task_score"] < 1.0)

    # Task-level Pass@2 (strict): task passes only if ALL test examples pass
    task_pass_at_2_count = perfect_tasks  # Same as perfect_tasks since all test examples must pass

    # Test example-level metrics
    all_test_results = [test_result for r in results for test_result in r["test_results"]]
    total_test_examples = len(all_test_results)

    test_pass_at_2_count = sum(1 for tr in all_test_results if tr["pass_at_2"])
    test_both_correct_count = sum(1 for tr in all_test_results if tr["both_correct"])
    test_attempt_1_correct = sum(1 for tr in all_test_results if tr["attempt_1"]["correct"])
    test_attempt_2_correct = sum(1 for tr in all_test_results if tr["attempt_2"]["correct"])

    # Error analysis
    test_attempt_1_errors = sum(1 for tr in all_test_results if tr["attempt_1"]["error"] is not None)
    test_attempt_2_errors = sum(1 for tr in all_test_results if tr["attempt_2"]["error"] is not None)

    # Size accuracy (from attempt 1 only, comparing predicted size to expected size)
    # Include ALL test examples with valid expected output (including those with errors)
    size_correct_count = 0
    size_mismatch_count = 0
    size_total_count = 0
    for tr in all_test_results:
        if len(tr["attempt_1"]["expected"]) > 0:
            pred_h = tr["attempt_1"]["pred_height"]
            pred_w = tr["attempt_1"]["pred_width"]
            expected_h = len(tr["attempt_1"]["expected"])
            expected_w = len(tr["attempt_1"]["expected"][0]) if expected_h > 0 else 0
            if pred_h == expected_h and pred_w == expected_w:
                size_correct_count += 1
            else:
                size_mismatch_count += 1
            size_total_count += 1

    # Aggregate copy statistics (from attempt 1 of all test examples)
    copy_stats_list = [tr["attempt_1"]["copy_stats"] for tr in all_test_results if tr["attempt_1"]["copy_stats"] is not None]
    if copy_stats_list:
        avg_copy_rate = np.mean([cs["copy_rate"] for cs in copy_stats_list])
        avg_edit_acc = np.mean([cs["edit_accuracy"] for cs in copy_stats_list])
        avg_keep_acc = np.mean([cs["keep_accuracy"] for cs in copy_stats_list])
        total_edit_cells = sum(cs["num_edit_cells"] for cs in copy_stats_list)
        total_keep_cells = sum(cs["num_keep_cells"] for cs in copy_stats_list)
    else:
        avg_copy_rate = avg_edit_acc = avg_keep_acc = 0.0
        total_edit_cells = total_keep_cells = 0

    # Aggregate trajectory statistics (from attempt 1 of all test examples)
    traj_stats_list = [tr["attempt_1"]["trajectory_stats"] for tr in all_test_results if tr["attempt_1"]["trajectory_stats"] is not None]
    if traj_stats_list:
        avg_final_delta = np.mean([ts["final_delta"] for ts in traj_stats_list])
        avg_final_confidence = np.mean([ts["final_confidence"] for ts in traj_stats_list])
        early_lock_count = sum(1 for ts in traj_stats_list if ts["early_lock_step"] is not None)
        avg_early_lock_step = np.mean([ts["early_lock_step"] for ts in traj_stats_list if ts["early_lock_step"] is not None]) if early_lock_count > 0 else None
    else:
        avg_final_delta = avg_final_confidence = 0.0
        early_lock_count = 0
        avg_early_lock_step = None

    # Per-task statistics
    avg_test_examples_per_task = total_test_examples / total_tasks if total_tasks > 0 else 0.0
    total_test_examples_passed = sum(r["num_test_examples_passed"] for r in results)
    avg_test_examples_passed_per_task = total_test_examples_passed / total_tasks if total_tasks > 0 else 0.0

    metrics = {
        # Task-level metrics (partial credit)
        "total_tasks": total_tasks,
        "task_pass_at_2": task_pass_at_2_count,
        "task_pass_at_2_rate": task_pass_at_2_count / total_tasks if total_tasks > 0 else 0.0,
        "avg_task_score": avg_task_score,
        "perfect_tasks": perfect_tasks,
        "partial_tasks": partial_tasks,
        "failed_tasks": failed_tasks,
        "perfect_task_rate": perfect_tasks / total_tasks if total_tasks > 0 else 0.0,

        # Test example-level metrics
        "total_test_examples": total_test_examples,
        "test_pass_at_2": test_pass_at_2_count,
        "test_pass_at_2_rate": test_pass_at_2_count / total_test_examples if total_test_examples > 0 else 0.0,
        "test_both_correct": test_both_correct_count,
        "test_both_correct_rate": test_both_correct_count / total_test_examples if total_test_examples > 0 else 0.0,
        "test_attempt_1_correct": test_attempt_1_correct,
        "test_attempt_1_accuracy": test_attempt_1_correct / total_test_examples if total_test_examples > 0 else 0.0,
        "test_attempt_2_correct": test_attempt_2_correct,
        "test_attempt_2_accuracy": test_attempt_2_correct / total_test_examples if total_test_examples > 0 else 0.0,
        "test_attempt_1_errors": test_attempt_1_errors,
        "test_attempt_2_errors": test_attempt_2_errors,
        "test_attempt_1_error_rate": test_attempt_1_errors / total_test_examples if total_test_examples > 0 else 0.0,
        "test_attempt_2_error_rate": test_attempt_2_errors / total_test_examples if total_test_examples > 0 else 0.0,

        # Size prediction accuracy
        "size_correct": size_correct_count,
        "size_mismatch": size_mismatch_count,
        "size_total": size_total_count,
        "size_accuracy": size_correct_count / size_total_count if size_total_count > 0 else 0.0,

        # Per-task breakdown
        "avg_test_examples_per_task": avg_test_examples_per_task,
        "avg_test_examples_passed_per_task": avg_test_examples_passed_per_task,

        # Copy statistics
        "avg_copy_rate": avg_copy_rate,
        "avg_edit_accuracy": avg_edit_acc,
        "avg_keep_accuracy": avg_keep_acc,
        "total_edit_cells": total_edit_cells,
        "total_keep_cells": total_keep_cells,

        # Trajectory statistics
        "avg_final_delta": avg_final_delta,
        "avg_final_confidence": avg_final_confidence,
        "early_lock_count": early_lock_count,
        "avg_early_lock_step": avg_early_lock_step,
    }

    return metrics


def print_metrics_report(metrics: Dict[str, Any], dataset: str, subset: str):
    """Print formatted metrics report with partial credit scoring"""
    total_tasks = metrics["total_tasks"]
    total_test_examples = metrics["total_test_examples"]

    print(f"\n{'='*80}")
    print(f"🎯 DIFFUSION MODEL EVALUATION RESULTS (PARTIAL CREDIT SCORING)")
    print(f"📊 Dataset: {dataset}, Subset: {subset}")
    print(f"{'='*80}")

    if total_tasks == 0:
        print("❌ No tasks saved as all returned errors.")
        return

    # Task-level metrics (partial credit with pass@2)
    print(f"\n🎯 TASK-LEVEL METRICS (Pass@2 with Partial Credit):")
    print(f"  Total Tasks: {total_tasks}")
    print(f"  Task Pass@2 Score (Partial Credit): {metrics['avg_task_score']:.1%} - average pass@2 score across all tasks")
    print(f"  Task Pass@2 (Strict): {metrics['task_pass_at_2']}/{total_tasks} ({metrics['task_pass_at_2_rate']:.1%}) - tasks where ALL test examples passed")
    print(f"  Perfect Tasks (100%): {metrics['perfect_tasks']}/{total_tasks} ({metrics['perfect_task_rate']:.1%}) - all test examples passed")
    print(f"  Partial Credit Tasks: {metrics['partial_tasks']}/{total_tasks} ({metrics['partial_tasks']/total_tasks:.1%}) - some test examples passed")
    print(f"  Failed Tasks (0%): {metrics['failed_tasks']}/{total_tasks} ({metrics['failed_tasks']/total_tasks:.1%}) - no test examples passed")

    # Test example-level metrics
    print(f"\n📊 TEST EXAMPLE-LEVEL METRICS:")
    print(f"  Total Test Examples: {total_test_examples} (avg {metrics['avg_test_examples_per_task']:.2f} per task)")
    print(f"  Pass@2 Rate: {metrics['test_pass_at_2']}/{total_test_examples} ({metrics['test_pass_at_2_rate']:.1%})")
    print(f"  Both Correct Rate: {metrics['test_both_correct']}/{total_test_examples} ({metrics['test_both_correct_rate']:.1%})")
    print(f"  Attempt 1 Accuracy: {metrics['test_attempt_1_correct']}/{total_test_examples} ({metrics['test_attempt_1_accuracy']:.1%})")
    print(f"  Attempt 2 Accuracy: {metrics['test_attempt_2_correct']}/{total_test_examples} ({metrics['test_attempt_2_accuracy']:.1%})")

    # Size prediction accuracy
    print(f"\n📏 SIZE PREDICTION ACCURACY:")
    print(f"  Correct Sizes: {metrics['size_correct']}/{metrics['size_total']} ({metrics['size_accuracy']:.1%})")

    # Per-task breakdown
    print(f"\n📈 PER-TASK BREAKDOWN:")
    print(f"  Avg Test Examples Passed Per Task: {metrics['avg_test_examples_passed_per_task']:.2f} / {metrics['avg_test_examples_per_task']:.2f}")

    # Copy statistics
    print(f"\n📋 COPY BEHAVIOR STATISTICS:")
    print(f"  Copy Rate: {metrics['avg_copy_rate']:.1%}")
    print(f"  Edit Accuracy: {metrics['avg_edit_accuracy']:.1%} (accuracy on cells requiring transformation)")
    print(f"  Keep Accuracy: {metrics['avg_keep_accuracy']:.1%} (accuracy on cells to preserve)")
    print(f"  Total Edit Cells: {metrics['total_edit_cells']:,}")
    print(f"  Total Keep Cells: {metrics['total_keep_cells']:,}")

    # Interpret copy behavior
    if metrics['avg_keep_accuracy'] > metrics['avg_edit_accuracy'] + 0.1:
        print(f"  ⚠️  Identity Attractor: Keep accuracy >> Edit accuracy")

    # Trajectory statistics
    print(f"\n🔄 SAMPLING TRAJECTORY STATISTICS:")
    print(f"  Final Δ-change: {metrics['avg_final_delta']:.2%} (fraction of cells changed in last step)")
    print(f"  Final Confidence: {metrics['avg_final_confidence']:.1%}")
    print(f"  Early-lock Count: {metrics['early_lock_count']}/{total_test_examples} ({metrics['early_lock_count']/total_test_examples:.1%})")
    if metrics['avg_early_lock_step'] is not None:
        print(f"  Avg Early-lock Step: {metrics['avg_early_lock_step']:.1f}")

    if metrics['early_lock_count'] > total_test_examples * 0.5 and metrics['avg_copy_rate'] > 0.7:
        print(f"  ⚠️  Input Gravity Confirmed: Early-lock + high copy rate")

    # Error analysis
    print(f"\n📋 ERROR ANALYSIS:")
    print(f"  Attempt 1 Errors: {metrics['test_attempt_1_errors']}/{total_test_examples} ({metrics['test_attempt_1_error_rate']:.1%})")
    print(f"  Attempt 2 Errors: {metrics['test_attempt_2_errors']}/{total_test_examples} ({metrics['test_attempt_2_error_rate']:.1%})")

    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Run ARC Diffusion Model Evaluation")

    # Required config file
    parser.add_argument("--config", required=True, help="Path to config file (contains model paths and output directory)")

    # Optional overrides
    parser.add_argument("--model-path", help="Override model path (defaults to final_model.pt in config output dir)")
    parser.add_argument("--prefer-best", action="store_true", help="Use best_model.pt instead of final_model.pt as default")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="auto", help="Device to use (default: auto)")
    parser.add_argument("--num-steps", type=int, default=32, help="Number of inference steps (default: 32)")

    # Data settings with defaults (config takes precedence if specified)
    parser.add_argument("--dataset", help="Dataset to use (overrides config if specified)")
    parser.add_argument("--subset", help="Subset to use (overrides config if specified)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of tasks to run (default: 0 for all)")

    # Majority voting ensemble
    parser.add_argument("--maj", action="store_true", help="Enable majority voting with all 72 D4 augmentations")
    parser.add_argument("--stats", action="store_true", help="Print histogram statistics for majority voting")

    # Output and debugging
    parser.add_argument("--output", help="Override output file path (defaults to config output dir)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--no-ema", action="store_true", help="Use training weights instead of EMA weights")

    args = parser.parse_args()

    # Load config file
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Extract output directory from config
    output_config = config.get('output', {})
    output_dir = Path(output_config.get('output_dir', 'experimental/diffusion/outputs/default'))

    # Determine model path
    if args.model_path:
        model_path = args.model_path
    else:
        final_model_path = output_dir / 'final_model.pt'
        best_model_path = output_dir / 'best_model.pt'

        # Choose model based on --prefer-best flag
        if args.prefer_best:
            # Prefer best_model.pt, fallback to final_model.pt
            if best_model_path.exists():
                model_path = str(best_model_path)
                print(f"Using best model: {model_path}")
            elif final_model_path.exists():
                model_path = str(final_model_path)
                print(f"Using final model (best_model.pt not found): {model_path}")
            else:
                model_path = str(best_model_path)
                print(f"Using default model path: {model_path}")
        else:
            # Default: prefer final_model.pt, fallback to best_model.pt
            if final_model_path.exists():
                model_path = str(final_model_path)
                print(f"Using final model: {model_path}")
            elif best_model_path.exists():
                model_path = str(best_model_path)
                print(f"Using best model (final_model.pt not found): {model_path}")
            else:
                model_path = str(final_model_path)
                print(f"Using default model path: {model_path}")

    # Determine dataset/subset early (needed for output path): command-line args override config, config overrides defaults
    # Priority: explicit command-line args > config > defaults
    data_config = config.get('data', {})
    config_data_dir = data_config.get('data_dir', config.get('data_dir'))

    # Extract dataset name from config's data_dir if it exists
    if config_data_dir:
        dataset_from_config = config_data_dir.split('/')[-1] if '/' in config_data_dir else config_data_dir
    else:
        dataset_from_config = None

    # Use command-line args if provided, otherwise config, otherwise defaults
    dataset = args.dataset if args.dataset else (dataset_from_config if dataset_from_config else "arc-prize-2025")
    subset = args.subset if args.subset else "evaluation"

    # Determine output file path
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = str(output_dir / f"evaluation_{dataset}_{subset}_{timestamp}.json")
        print(f"Using default output path: {output_file}")

    # Validate model path
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    # Handle limit parameter (0 means no limit)
    limit = args.limit if args.limit > 0 else None

    # Override device detection if specified
    if args.device != "auto":
        if args.device == "cpu":
            torch.cuda.is_available = lambda: False
        elif args.device == "cuda" and not torch.cuda.is_available():
            print("⚠️ CUDA requested but not available, falling back to CPU")

    print(f"🚀 ARC Diffusion Model Inference")
    print(f"📅 Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Config: {args.config}")
    print(f"🎲 Dataset: {dataset}/{subset}")
    if limit:
        print(f"⚡ Task limit: {limit}")
    else:
        print(f"⚡ Task limit: All tasks")

    try:
        # Initialize inference (pass dataset for correct task indexing)
        inference = DiffusionInference(
            model_path=model_path,
            device=args.device,
            num_inference_steps=args.num_steps,
            debug=args.debug,
            dataset=dataset,
            use_ema=not args.no_ema
        )

        # Load tasks directly from JSON files
        print(f"\n📂 Loading tasks from {dataset}/{subset}...")

        # Map dataset/subset to file paths
        data_file_map = {
            "arc-prize-2025/evaluation": "data/arc-prize-2025/arc-agi_evaluation_challenges.json",
            "arc-prize-2025/training": "data/arc-prize-2025/arc-agi_training_challenges.json",
            "arc-prize-2024/evaluation": "data/arc-prize-2024/arc-agi_evaluation_challenges.json",
            "arc-prize-2024/training": "data/arc-prize-2024/arc-agi_training_challenges.json"
        }

        dataset_key = f"{dataset}/{subset}"
        if dataset_key not in data_file_map:
            raise ValueError(f"Unknown dataset/subset: {dataset_key}")

        data_path = data_file_map[dataset_key]
        with open(data_path, 'r') as f:
            all_task_data = json.load(f)

        # Apply subset filtering BEFORE limit (subset takes precedence)
        subset_task_ids = None
        eval_subset_task_ids = None

        # Load training subset if specified (check both flattened and nested config)
        subset_file = config.get('subset_file') or config.get('data', {}).get('subset_file')
        if subset_file:
            with open(subset_file, 'r') as f:
                subset_task_ids = set(line.strip() for line in f if line.strip())
            print(f"📋 Using training subset from {subset_file}: {len(subset_task_ids)} tasks")

        # Load evaluation subset if specified (check both flattened and nested config)
        eval_subset_file = config.get('eval_subset_file') or config.get('data', {}).get('eval_subset_file')
        if eval_subset_file:
            with open(eval_subset_file, 'r') as f:
                eval_subset_task_ids = set(line.strip() for line in f if line.strip())
            print(f"📋 Using evaluation subset from {eval_subset_file}: {len(eval_subset_task_ids)} tasks")

        # Filter based on subset BEFORE applying limit
        is_evaluation = "evaluation" in data_path
        if is_evaluation and eval_subset_task_ids:
            # Filter evaluation tasks by eval_subset_file
            all_task_data = {task_id: task_data for task_id, task_data in all_task_data.items()
                           if task_id in eval_subset_task_ids}
            print(f"📋 Filtered to {len(all_task_data)} tasks from evaluation subset")
        elif not is_evaluation and subset_task_ids:
            # Filter training tasks by subset_file
            all_task_data = {task_id: task_data for task_id, task_data in all_task_data.items()
                           if task_id in subset_task_ids}
            print(f"📋 Filtered to {len(all_task_data)} tasks from training subset")

        # Convert to list of (task_id, task_data) tuples AFTER subset filtering
        all_tasks_list = [(task_id, task_data) for task_id, task_data in all_task_data.items()]

        # Apply limit if specified (limit applies AFTER subset filtering, so you can still limit within subset)
        if limit and limit < len(all_tasks_list):
            print(f"📋 Applying limit: {limit} tasks (from {len(all_tasks_list)} in subset)")
            all_tasks_list = all_tasks_list[:limit]

        # Convert to dict for filtering
        all_tasks_dict = {task_id: task_data for task_id, task_data in all_tasks_list}

        # Filter tasks by max_size using our utility
        filtered_tasks_dict, total_tasks, filtered_count = filter_tasks_by_max_size(
            all_tasks_dict,
            inference.config['max_size'],
            verbose=True
        )

        # Convert back to list
        filtered_tasks = [(task_id, task_data) for task_id, task_data in filtered_tasks_dict.items()]

        tasks = filtered_tasks
        print(f"📊 Task Filtering Results:")
        print(f"  Total tasks loaded: {total_tasks}")
        print(f"  Tasks filtered out (grid > {inference.config['max_size']}): {filtered_count}")
        print(f"  Tasks remaining: {len(tasks)}")

        if not tasks:
            print("❌ No tasks remaining after filtering")
            return

        # Run inference
        results = []
        errors = 0

        progress_bar = tqdm(tasks, desc="Running inference")
        for i, (task_id, task_data) in enumerate(progress_bar):
            try:
                # Visualize denoising for all tasks (unless using majority voting)
                visualize = not args.maj
                result = inference.run_task(
                    task_id,
                    task_data,
                    f"{dataset}/{subset}",
                    visualize=visualize,
                    use_majority_voting=args.maj,
                    print_stats=args.stats
                )
                results.append(result)

                # Update progress bar with current stats
                current_metrics = calculate_metrics(results)
                if current_metrics:
                    progress_bar.set_postfix({
                        'Avg Score': f"{current_metrics['avg_task_score']:.1%}",
                        'Test Pass@2': f"{current_metrics['test_pass_at_2_rate']:.1%}",
                        'Errors': errors
                    })

            except Exception as e:
                errors += 1
                # Always print first error for debugging
                if errors == 1 or args.debug:
                    print(f"\n❌ Error processing {task_id}: {str(e)}")
                    print(traceback.format_exc())

        # Calculate and display metrics
        metrics = calculate_metrics(results)
        print_metrics_report(metrics, dataset, subset)

        # Save results
        output_data = {
            "metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "config_path": args.config,
                "dataset": dataset,
                "subset": subset,
                "model_path": model_path,
                "num_inference_steps": args.num_steps or inference.config['num_timesteps'],
                "device": str(inference.device),
                "total_tasks": len(tasks),
                "completed_tasks": len(results),
                "errors": errors,
                "limit": limit
            },
            "metrics": metrics,
            "results": results
        }

        # Ensure output directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"💾 Results saved to {output_file}")

        print(f"\n✅ Inference completed: {len(results)}/{len(tasks)} tasks successful")

    except KeyboardInterrupt:
        print("\n⏹️ Inference interrupted by user")
    except Exception as e:
        print(f"\n💥 Inference failed: {str(e)}")
        if args.debug:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()