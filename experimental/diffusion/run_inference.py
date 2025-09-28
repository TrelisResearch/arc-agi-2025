#!/usr/bin/env python3
"""
ARC Diffusion Model Inference Script

Runs diffusion model inference on ARC tasks with pass@2 scoring.
Follows the same pattern as run_arc_tasks_soar.py for consistency.
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

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experimental.diffusion.src.model import ARCDiffusionModel, GridSizePredictionHead
from experimental.diffusion.src.training import ARCDiffusionSampler
from experimental.diffusion.src.dataset import ARCDataset, load_arc_data_paths
from experimental.diffusion.utils.noise_scheduler import DiscreteNoiseScheduler
from experimental.diffusion.utils.grid_utils import grid_to_tokens, tokens_to_grid, detect_valid_region
from llm_python.utils.task_loader import TaskData, get_task_loader


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
        device: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        size_head_path: Optional[str] = None,
        debug: bool = False
    ):
        self.model_path = model_path
        self.size_head_path = size_head_path
        # Set up device (prioritize CUDA > MPS > CPU)
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
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

        # Create dataset for task-specific distributions
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
        print(f"📊 Loaded dataset with {len(self.dataset.task_id_to_idx)} tasks for task-specific noise distributions")

        # Load size prediction head if provided
        self.size_head = None
        if self.size_head_path:
            print(f"🧠 Loading size prediction head from {self.size_head_path}")
            self.size_head = GridSizePredictionHead(
                diffusion_model=self.model,
                hidden_dim=256,
                max_size=self.config['max_size']
            )
            self.size_head.load_state_dict(torch.load(self.size_head_path, map_location=self.device))
            self.size_head.to(self.device)
            self.size_head.eval()
            print(f"✓ Size head loaded with {sum(p.numel() for p in self.size_head.parameters() if p.requires_grad):,} parameters")

        self.sampler = ARCDiffusionSampler(
            self.model,
            self.noise_scheduler,
            self.device,
            dataset=self.dataset,
            size_predictor=self.size_head,
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

        # Recreate model
        model = ARCDiffusionModel(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            max_size=config['max_size'],
            max_tasks=dataset_info['num_tasks']
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

    def predict_single(self, input_grid: np.ndarray, task_idx: int, task_id: str = None) -> Tuple[np.ndarray, Optional[str], str]:
        """
        Run single prediction on input grid.

        Returns:
            predicted_grid: Predicted output grid (extracted from PAD tokens)
            error: Error message if any
            size_source: How the size was determined ("size_head" or "ground_truth")
        """
        try:
            # Convert to tokens and add batch dimension
            input_tokens, _, _ = grid_to_tokens(input_grid, max_size=self.config['max_size'])
            input_batch = input_tokens.unsqueeze(0).to(self.device)  # [1, max_size, max_size]

            # Use task ID (ensure it's within trained range)
            task_ids = torch.tensor([task_idx]).to(self.device)

            # Sample output
            with torch.no_grad():
                predicted_grids = self.sampler.sample(
                    input_grids=input_batch,
                    task_indices=task_ids,
                    num_inference_steps=self.num_inference_steps
                )

            # Extract valid region from predicted grid (find non-PAD tokens)
            predicted_grid, region_error = detect_valid_region(predicted_grids[0].cpu().numpy())

            if region_error:
                return np.array([]), f"Region detection failed: {region_error}", "ground_truth"

            # Determine size source - if size head was used in sampling, it's "size_head", otherwise "ground_truth"
            size_source = "size_head" if self.size_head is not None else "ground_truth"

            return predicted_grid, None, size_source

        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            if self.debug:
                error_msg += f"\n{traceback.format_exc()}"
            return np.array([]), error_msg, "ground_truth"

    def run_task(self, task_id: str, task_data: TaskData, dataset: str) -> TaskResult:
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
        attempt_1 = self._run_attempt(input_grid, expected_output, test_idx=0, task_idx=task_idx, task_id=task_id)
        attempt_2 = self._run_attempt(input_grid, expected_output, test_idx=0, task_idx=task_idx, task_id=task_id)

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

    def _run_attempt(self, input_grid: np.ndarray, expected_output: np.ndarray, test_idx: int, task_idx: int, task_id: str = None) -> DiffusionResult:
        """Run a single diffusion attempt"""
        predicted_grid, error, size_source = self.predict_single(input_grid, task_idx, task_id)

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

        return DiffusionResult(
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
    parser = argparse.ArgumentParser(description="Run ARC Diffusion Model Inference")

    # Model and inference settings
    parser.add_argument("--model-path", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--size-head-path", help="Path to trained size prediction head (optional)")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto", help="Device to use")
    parser.add_argument("--num-steps", type=int, help="Number of inference steps (default: use training steps)")

    # Data settings (following soar pattern)
    parser.add_argument("--dataset", default="arc-prize-2024", help="Dataset to use")
    parser.add_argument("--subset", default="evaluation", help="Subset to use")
    parser.add_argument("--limit", type=int, help="Limit number of tasks to run")

    # Output and debugging
    parser.add_argument("--output", help="Output file to save results")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    # Override device detection if specified
    if args.device != "auto":
        if args.device == "cpu":
            torch.cuda.is_available = lambda: False
        elif args.device == "cuda" and not torch.cuda.is_available():
            print("⚠️ CUDA requested but not available, falling back to CPU")

    print(f"🚀 ARC Diffusion Model Inference")
    print(f"📅 Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎲 Dataset: {args.dataset}/{args.subset}")
    if args.limit:
        print(f"⚡ Task limit: {args.limit}")

    try:
        # Initialize inference
        inference = DiffusionInference(
            model_path=args.model_path,
            device=args.device if args.device != "auto" else None,
            num_inference_steps=args.num_steps,
            size_head_path=args.size_head_path,
            debug=args.debug
        )

        # Load tasks
        print(f"\n📂 Loading tasks from {args.dataset}/{args.subset}...")
        task_loader = get_task_loader()
        tasks = task_loader.get_dataset_subset(f"{args.dataset}/{args.subset}", max_rows=args.limit)

        print(f"📋 Found {len(tasks)} tasks")

        if not tasks:
            print("❌ No tasks found")
            return

        # Run inference
        results = []
        errors = 0

        progress_bar = tqdm(tasks, desc="Running inference")
        for task_id, task_data in progress_bar:
            try:
                result = inference.run_task(task_id, task_data, f"{args.dataset}/{args.subset}")
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

        # Save results if requested
        if args.output:
            output_data = {
                "metadata": {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "dataset": args.dataset,
                    "subset": args.subset,
                    "model_path": args.model_path,
                    "num_inference_steps": args.num_steps or inference.config['num_timesteps'],
                    "device": str(inference.device),
                    "total_tasks": len(tasks),
                    "completed_tasks": len(results),
                    "errors": errors
                },
                "metrics": metrics,
                "results": results
            }

            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"💾 Results saved to {args.output}")

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