#!/usr/bin/env python3
"""
ARC Diffusion Model Evaluation

Runs diffusion model inference on ARC tasks with pass@2 scoring.
Uses config-driven parameters for simplified usage.

Usage:
    python experimental/diffusion/evaluate.py --config experimental/diffusion/configs/smol_config.json
    python experimental/diffusion/evaluate.py --config experimental/diffusion/configs/smol_config.json --limit 10
"""
import json
import argparse
import datetime
import sys
import time
import traceback
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Union, Tuple, Any
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experimental.diffusion.src.model import ARCDiffusionModel
from experimental.diffusion.src.training import ARCDiffusionSampler
from experimental.diffusion.src.dataset import ARCDataset, load_arc_data_paths
from experimental.diffusion.utils.noise_scheduler import DiscreteNoiseScheduler
from experimental.diffusion.utils.grid_utils import grid_to_tokens, tokens_to_grid
from experimental.diffusion.utils.task_filters import filter_tasks_by_max_size
from experimental.diffusion.utils.arc_colors import arc_cmap


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


class TaskResult(TypedDict):
    """Result for a single ARC task with pass@2 attempts"""
    task_id: str
    timestamp: str

    # Attempt results (2 attempts for pass@2)
    attempt_1: DiffusionResult
    attempt_2: DiffusionResult

    # Pass@2 scoring
    pass_at_2: bool  # True if either attempt is correct
    both_correct: bool  # True if both attempts are correct

    # Task metadata
    num_train_examples: int
    num_test_examples: int


class DiffusionInference:
    """Main inference class for ARC diffusion model"""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        num_inference_steps: Optional[int] = None,
        size_head_path: Optional[str] = None,
        debug: bool = False
    ):
        self.model_path = model_path
        self.size_head_path = size_head_path
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

        # Create dataset for task indexing
        data_paths = load_arc_data_paths(
            data_dir="data/arc-prize-2024",
            datasets=["training_challenges", "evaluation_challenges"]
        )
        self.dataset = ARCDataset(
            data_paths=data_paths['train'],
            max_size=self.config['max_size'],
            augment=False,  # No augmentation for inference
            include_training_test_examples=True
        )
        print(f"📊 Loaded dataset with {len(self.dataset.task_id_to_idx)} tasks for task indexing")

        # Check for integrated size head in model first
        self.size_head = None
        if hasattr(self.model, 'include_size_head') and self.model.include_size_head:
            print(f"✓ Using integrated size head from model")
            # No need to load external size head - model has it built in
        elif self.size_head_path:
            print(f"⚠️ External size head checkpoint provided but no longer supported.")
            print(f"⚠️ Model should have integrated size head. Ignoring size_head_checkpoint.")
            self.size_head = None

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

        checkpoint = torch.load(self.model_path, map_location=self.device)
        config = checkpoint['config']
        dataset_info = checkpoint['dataset_info']

        # Store task mapping for inference
        self.task_id_to_idx = dataset_info.get('task_id_to_idx', {})
        self.max_tasks = dataset_info['num_tasks']

        # Get auxiliary loss config for size head parameters
        aux_config = config.get('auxiliary_loss', {})
        include_size_head = aux_config.get('include_size_head', True)
        size_head_hidden_dim = aux_config.get('size_head_hidden_dim', None)

        # Recreate model with size head parameters
        model = ARCDiffusionModel(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            max_size=config['max_size'],
            max_tasks=dataset_info['num_tasks'],
            embedding_dropout=config.get('embedding_dropout', 0.1),
            include_size_head=include_size_head,
            size_head_hidden_dim=size_head_hidden_dim
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        return model, config

    def get_task_idx(self, task_id: str) -> int:
        """Get task index for a given task ID, handling unknown tasks."""
        if task_id in self.task_id_to_idx:
            return self.task_id_to_idx[task_id]
        else:
            # For unknown tasks, use a default task index (0)
            # Could also raise an error or use a special "unknown" token
            print(f"Warning: Unknown task {task_id}, using default task index 0")
            return 0

    def _load_solutions(self, dataset: str) -> Dict[str, List[List[List[int]]]]:
        """Load solutions from appropriate solutions file."""
        if 'training' in dataset:
            solutions_path = "data/arc-prize-2024/arc-agi_training_solutions.json"
        elif 'evaluation' in dataset:
            solutions_path = "data/arc-prize-2024/arc-agi_evaluation_solutions.json"
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
        pred_width: int
    ) -> Tuple[torch.Tensor, List[np.ndarray]]:
        """
        Sample with intermediate step capture for visualization.

        Returns:
            final_prediction: Final denoised output
            intermediate_steps: List of grids at different denoising timesteps
        """
        self.model.eval()

        batch_size = input_grids.shape[0]
        max_size = input_grids.shape[1]
        num_inference_steps = self.num_inference_steps

        # Initialize with uniform random noise
        x_t = torch.randint(0, 10, (batch_size, max_size, max_size), device=self.device)

        # Storage for intermediate steps (grid, timestep) tuples
        intermediate_steps = []
        capture_interval = max(1, num_inference_steps // 6)  # Capture ~6 evenly spaced steps

        # Denoising loop
        timesteps = torch.linspace(num_inference_steps - 1, 0, num_inference_steps, dtype=torch.long, device=self.device)

        with torch.no_grad():
            for i, t in enumerate(timesteps):
                t_batch = t.repeat(batch_size)

                # Forward pass
                logits = self.model(x_t, input_grids, task_indices, t_batch)

                # Get predicted probabilities
                probs = torch.softmax(logits, dim=-1)

                # Sample or argmax
                if t > 0:
                    x_t = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(x_t.shape)
                else:
                    x_t = torch.argmax(logits, dim=-1)

                # Apply size masking
                for b in range(batch_size):
                    if pred_height < max_size:
                        x_t[b, pred_height:, :] = 0
                    if pred_width < max_size:
                        x_t[b, :, pred_width:] = 0

                # Capture intermediate steps with their timestep
                if i % capture_interval == 0 or i == num_inference_steps - 1:
                    intermediate_steps.append((x_t[0].cpu().numpy().copy(), t.item()))

        return x_t, intermediate_steps

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

            # Get size predictions - check integrated first, then external
            if hasattr(self.model, 'include_size_head') and self.model.include_size_head:
                # Use integrated size head
                with torch.no_grad():
                    pred_heights, pred_widths = self.model.predict_sizes(input_batch, task_ids)
                    pred_height, pred_width = pred_heights[0].item(), pred_widths[0].item()
                size_source = "integrated_size_head"
                print(f"Using integrated size head predictions: {pred_height}×{pred_width}")
            elif self.size_head is not None:
                # Use external size head
                with torch.no_grad():
                    pred_heights, pred_widths = self.size_head.predict_sizes(input_batch, task_ids)
                    pred_height, pred_width = pred_heights[0].item(), pred_widths[0].item()
                size_source = "external_size_head"
                print(f"Using external size head predictions: {pred_height}×{pred_width}")
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

            # Sample output
            if capture_steps:
                # Sample with intermediate step capture for visualization
                predicted_grids, intermediate_steps = self.sample_with_steps(
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

                return predicted_grid, None, size_source, cropped_steps
            else:
                # Normal sampling without capturing steps
                with torch.no_grad():
                    predicted_grids = self.sampler.sample(
                        input_grids=input_batch,
                        task_indices=task_ids,
                        num_inference_steps=self.num_inference_steps
                    )

                # Crop to predicted dimensions (no more region detection!)
                full_grid = predicted_grids[0].cpu().numpy()
                predicted_grid = full_grid[:pred_height, :pred_width]

                return predicted_grid, None, size_source

        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            if self.debug:
                error_msg += f"\n{traceback.format_exc()}"
            return np.array([]), error_msg, "ground_truth"

    def run_task(self, task_id: str, task_data: Dict[str, Any], dataset: str, visualize: bool = False) -> TaskResult:
        """
        Run inference on a single ARC task with pass@2 scoring.

        Args:
            task_id: Task identifier
            task_data: Task data containing train/test examples
            dataset: Dataset name to load appropriate solutions

        Returns:
            TaskResult with both attempts and scoring
        """
        timestamp = datetime.datetime.now().isoformat()

        # Use the first test example for inference
        if not task_data["test"]:
            raise ValueError(f"Task {task_id} has no test examples")

        test_example = task_data["test"][0]
        input_grid = np.array(test_example["input"])

        # Get expected output from solutions file
        solutions = self._load_solutions(dataset)
        if task_id in solutions and len(solutions[task_id]) > 0:
            expected_output = np.array(solutions[task_id][0])
        elif 'output' in test_example:
            # Fallback if solutions not found but output is in test data
            expected_output = np.array(test_example["output"])
        else:
            raise ValueError(f"No solution found for task {task_id}, test example 0")

        # Get correct task index
        task_idx = self.get_task_idx(task_id)

        # Run two attempts for pass@2
        # For the first attempt, capture steps if visualization is requested
        if visualize:
            attempt_1_result = self._run_attempt(input_grid, expected_output, test_idx=0, task_idx=task_idx, task_id=task_id, capture_steps=True)
            attempt_1, intermediate_steps = attempt_1_result

            # Create visualization
            output_dir = Path(self.config.get('output_dir', 'experimental/diffusion/outputs'))
            output_dir.mkdir(parents=True, exist_ok=True)
            vis_path = output_dir / f"denoising_progression_{task_id}.png"
            self.visualize_denoising_progression(
                input_grid,
                expected_output,
                intermediate_steps,
                task_id,
                save_path=str(vis_path)
            )
        else:
            attempt_1 = self._run_attempt(input_grid, expected_output, test_idx=0, task_idx=task_idx, task_id=task_id, capture_steps=False)

        attempt_2 = self._run_attempt(input_grid, expected_output, test_idx=0, task_idx=task_idx, task_id=task_id, capture_steps=False)

        # Calculate pass@2 metrics
        pass_at_2 = attempt_1["correct"] or attempt_2["correct"]
        both_correct = attempt_1["correct"] and attempt_2["correct"]

        return TaskResult(
            task_id=task_id,
            timestamp=timestamp,
            attempt_1=attempt_1,
            attempt_2=attempt_2,
            pass_at_2=pass_at_2,
            both_correct=both_correct,
            num_train_examples=len(task_data["train"]),
            num_test_examples=len(task_data["test"])
        )

    def _run_attempt(self, input_grid: np.ndarray, expected_output: np.ndarray, test_idx: int, task_idx: int, task_id: str = None, capture_steps: bool = False) -> DiffusionResult:
        """Run a single diffusion attempt"""
        result = self.predict_single(input_grid, task_idx, task_id, expected_output, capture_steps=capture_steps)
        if capture_steps:
            predicted_grid, error, size_source, intermediate_steps = result
        else:
            predicted_grid, error, size_source = result
            intermediate_steps = None

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
        pred_height = predicted_grid.shape[0] if len(predicted_grid) > 0 else 30
        pred_width = predicted_grid.shape[1] if len(predicted_grid) > 0 else 30

        result_dict = DiffusionResult(
            test_idx=test_idx,
            input_grid=input_grid.tolist(),
            predicted=predicted_grid.tolist() if len(predicted_grid) > 0 else None,
            expected=expected_output.tolist() if len(expected_output) > 0 else [],
            correct=correct,
            error=error,
            pred_height=pred_height,
            pred_width=pred_width,
            size_source=size_source
        )

        # Return intermediate steps if captured
        if capture_steps:
            return result_dict, intermediate_steps
        else:
            return result_dict


def calculate_metrics(results: List[TaskResult]) -> Dict[str, Any]:
    """Calculate pass@2 and other metrics from results"""
    if not results:
        return {}

    total_tasks = len(results)

    # Pass@2 metrics
    pass_at_2_count = sum(1 for r in results if r["pass_at_2"])
    both_correct_count = sum(1 for r in results if r["both_correct"])

    # Attempt-level accuracy
    attempt_1_correct = sum(1 for r in results if r["attempt_1"]["correct"])
    attempt_2_correct = sum(1 for r in results if r["attempt_2"]["correct"])

    # Error analysis
    attempt_1_errors = sum(1 for r in results if r["attempt_1"]["error"] is not None)
    attempt_2_errors = sum(1 for r in results if r["attempt_2"]["error"] is not None)

    metrics = {
        "total_tasks": total_tasks,
        "pass_at_2": pass_at_2_count,
        "both_correct": both_correct_count,
        "attempt_1_correct": attempt_1_correct,
        "attempt_2_correct": attempt_2_correct,
        "attempt_1_errors": attempt_1_errors,
        "attempt_2_errors": attempt_2_errors,

        # Percentages
        "pass_at_2_rate": pass_at_2_count / total_tasks if total_tasks > 0 else 0.0,
        "both_correct_rate": both_correct_count / total_tasks if total_tasks > 0 else 0.0,
        "attempt_1_accuracy": attempt_1_correct / total_tasks if total_tasks > 0 else 0.0,
        "attempt_2_accuracy": attempt_2_correct / total_tasks if total_tasks > 0 else 0.0,
        "attempt_1_error_rate": attempt_1_errors / total_tasks if total_tasks > 0 else 0.0,
        "attempt_2_error_rate": attempt_2_errors / total_tasks if total_tasks > 0 else 0.0,
    }

    return metrics


def print_metrics_report(metrics: Dict[str, Any], dataset: str, subset: str):
    """Print formatted metrics report similar to soar style"""
    total = metrics["total_tasks"]

    print(f"\n{'='*80}")
    print(f"🎯 DIFFUSION MODEL INFERENCE RESULTS")
    print(f"📊 Dataset: {dataset}, Subset: {subset}")
    print(f"📋 Total Tasks: {total}")
    print(f"{'='*80}")

    if total == 0:
        print("❌ No tasks processed")
        return

    # Main metrics
    print(f"🎲 Pass@2: {metrics['pass_at_2']}/{total} ({metrics['pass_at_2_rate']:.1%})")
    print(f"🎯 Both Correct: {metrics['both_correct']}/{total} ({metrics['both_correct_rate']:.1%})")
    print(f"🥇 Attempt 1: {metrics['attempt_1_correct']}/{total} ({metrics['attempt_1_accuracy']:.1%})")
    print(f"🥈 Attempt 2: {metrics['attempt_2_correct']}/{total} ({metrics['attempt_2_accuracy']:.1%})")

    # Error analysis
    print(f"\n📋 Error Analysis:")
    print(f"  Attempt 1 Errors: {metrics['attempt_1_errors']}/{total} ({metrics['attempt_1_error_rate']:.1%})")
    print(f"  Attempt 2 Errors: {metrics['attempt_2_errors']}/{total} ({metrics['attempt_2_error_rate']:.1%})")

    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Run ARC Diffusion Model Evaluation")

    # Required config file
    parser.add_argument("--config", required=True, help="Path to config file (contains model paths and output directory)")

    # Optional overrides
    parser.add_argument("--model-path", help="Override model path (defaults to best model in config output dir)")
    parser.add_argument("--size-head-path", help="Override size head path (defaults to best size head in config output dir)")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="auto", help="Device to use (default: auto)")
    parser.add_argument("--num-steps", type=int, help="Number of inference steps (default: use training steps)")

    # Data settings with defaults
    parser.add_argument("--dataset", default="arc-prize-2024", help="Dataset to use (default: arc-prize-2024)")
    parser.add_argument("--subset", default="evaluation", help="Subset to use (default: evaluation)")
    parser.add_argument("--limit", type=int, default=5, help="Limit number of tasks to run (default: 5, use 0 for all)")

    # Output and debugging
    parser.add_argument("--output", help="Override output file path (defaults to config output dir)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

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

    # Determine model paths
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = str(output_dir / 'best_model.pt')
        print(f"Using default model path: {model_path}")

    if args.size_head_path:
        size_head_path = args.size_head_path
    else:
        size_head_path = str(output_dir / 'best_size_head.pt')
        if Path(size_head_path).exists():
            print(f"Using default size head path: {size_head_path}")
        else:
            size_head_path = None
            print(f"No size head found at {output_dir / 'best_size_head.pt'}, proceeding without size head")

    # Determine output file path
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = str(output_dir / f"evaluation_{args.dataset}_{args.subset}_{timestamp}.json")
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
    print(f"🎲 Dataset: {args.dataset}/{args.subset}")
    if limit:
        print(f"⚡ Task limit: {limit}")
    else:
        print(f"⚡ Task limit: All tasks")

    try:
        # Initialize inference
        inference = DiffusionInference(
            model_path=model_path,
            device=args.device,
            num_inference_steps=args.num_steps,
            size_head_path=size_head_path,
            debug=args.debug
        )

        # Load tasks directly from JSON files
        print(f"\n📂 Loading tasks from {args.dataset}/{args.subset}...")

        # Map dataset/subset to file paths
        data_file_map = {
            "arc-prize-2024/evaluation": "data/arc-prize-2024/arc-agi_evaluation_challenges.json",
            "arc-prize-2024/training": "data/arc-prize-2024/arc-agi_training_challenges.json"
        }

        dataset_key = f"{args.dataset}/{args.subset}"
        if dataset_key not in data_file_map:
            raise ValueError(f"Unknown dataset/subset: {dataset_key}")

        data_path = data_file_map[dataset_key]
        with open(data_path, 'r') as f:
            all_task_data = json.load(f)

        # Convert to list of (task_id, task_data) tuples
        all_tasks_list = [(task_id, task_data) for task_id, task_data in all_task_data.items()]

        # Apply limit if specified
        if limit:
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
                # Visualize denoising for the first task
                visualize = (i == 0)
                result = inference.run_task(task_id, task_data, f"{args.dataset}/{args.subset}", visualize=visualize)
                results.append(result)

                # Update progress bar with current stats
                current_metrics = calculate_metrics(results)
                if current_metrics:
                    progress_bar.set_postfix({
                        'Pass@2': f"{current_metrics['pass_at_2_rate']:.1%}",
                        'Errors': errors
                    })

            except Exception as e:
                errors += 1
                if args.debug:
                    print(f"\n❌ Error processing {task_id}: {str(e)}")
                    print(traceback.format_exc())

        # Calculate and display metrics
        metrics = calculate_metrics(results)
        print_metrics_report(metrics, args.dataset, args.subset)

        # Save results
        output_data = {
            "metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "config_path": args.config,
                "dataset": args.dataset,
                "subset": args.subset,
                "model_path": model_path,
                "size_head_path": size_head_path,
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