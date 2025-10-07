#!/usr/bin/env python3
"""
ARC Iterative Refiner Model Evaluation

Evaluates trained iterative refinement models with pass@2 scoring.

Usage:
    uv run python itransformer/evaluate.py --config itransformer/configs/test_config.json
    uv run python itransformer/evaluate.py --config itransformer/configs/test_config.json --limit 10
    uv run python itransformer/evaluate.py --config itransformer/configs/test_config.json --inference-steps 4
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
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from itransformer.src.model import ARCIterativeModel
from itransformer.src.dataset import ARCDataset, load_arc_data_paths
from itransformer.utils.grid_utils import grid_to_tokens, tokens_to_grid, TaskAugmentation
from itransformer.utils.task_filters import filter_tasks_by_max_size
from itransformer.utils.arc_colors import arc_cmap


class AugmentationParams(TypedDict):
    """Parameters for a single D4 augmentation"""
    d4_idx: int
    color_cycle: int


def generate_augmentations() -> List[AugmentationParams]:
    """Generate all 72 D4 augmentations (8 D4 × 9 colors)."""
    aug_tuples = TaskAugmentation.generate_all_d4_augmentations()
    return [AugmentationParams(d4_idx=d4, color_cycle=color) for d4, color in aug_tuples]


def apply_augmentation(grid: np.ndarray, aug_params: AugmentationParams) -> np.ndarray:
    """Apply D4 augmentation to a grid."""
    grid = TaskAugmentation.apply_d4_augmentation(grid, aug_params['d4_idx'])
    grid = TaskAugmentation.apply_color_cycle_augmentation(grid, aug_params['color_cycle'])
    return grid


def reverse_augmentation(grid: np.ndarray, aug_params: AugmentationParams) -> np.ndarray:
    """Reverse D4 augmentation on a grid."""
    grid = TaskAugmentation.reverse_color_cycle_augmentation(grid, aug_params['color_cycle'])
    grid = TaskAugmentation.reverse_d4_augmentation(grid, aug_params['d4_idx'])
    return grid


def deaugment_size(height: int, width: int, aug_params: AugmentationParams) -> Tuple[int, int]:
    """De-augment size prediction based on D4 transformation."""
    return TaskAugmentation.deaugment_size_d4(height, width, aug_params['d4_idx'])


class IterativeResult(TypedDict):
    """Result of running iterative model on a single test example"""
    test_idx: int
    input_grid: List[List[int]]
    predicted: Optional[List[List[int]]]
    expected: List[List[int]]
    correct: bool
    error: Optional[str]
    pred_height: int
    pred_width: int
    size_source: str


class TestExampleResult(TypedDict):
    """Results for a single test example with pass@2 attempts"""
    test_idx: int
    attempt_1: IterativeResult
    attempt_2: IterativeResult
    pass_at_2: bool
    both_correct: bool


class TaskResult(TypedDict):
    """Result for a single ARC task with multiple test examples"""
    task_id: str
    timestamp: str
    test_results: List[TestExampleResult]
    num_test_examples: int
    num_test_examples_passed: int
    task_score: float
    num_train_examples: int


class IterativeInference:
    """Main inference class for ARC iterative refinement model"""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        K: Optional[int] = None,
        debug: bool = False,
        dataset: str = "arc-prize-2025"
    ):
        self.model_path = model_path
        # Set up device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.K = K
        self.debug = debug

        print(f"🔥 Loading model from {model_path}")
        print(f"🖥️ Using device: {self.device}")

        # Load model
        self.model, self.config = self._load_model()

        # Create dataset for task indexing
        data_paths = load_arc_data_paths(
            data_dir=f"data/{dataset}",
            datasets=["training_challenges", "evaluation_challenges"]
        )
        self.dataset = ARCDataset(
            data_paths=data_paths['train'],
            max_size=self.config['max_size'],
            augment=False,
            include_training_test_examples=True
        )
        print(f"📊 Loaded dataset with {len(self.dataset.task_id_to_idx)} tasks for task indexing")

        if hasattr(self.model, 'include_size_head') and self.model.include_size_head:
            print(f"✓ Using integrated size head from model")

        if self.K is None:
            self.K = self.config.get('K', 8)

        print(f"✨ Model loaded: {self.model.__class__.__name__}")
        print(f"📊 Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"⚡ Refinement steps (K): {self.K}")

    def _load_model(self) -> Tuple[ARCIterativeModel, Dict]:
        """Load trained model from checkpoint"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        config = checkpoint['config']
        dataset_info = checkpoint['dataset_info']

        # Store task mapping
        self.task_id_to_idx = dataset_info.get('task_id_to_idx', {})

        # Extract max_tasks from task_embedding shape
        state_dict = checkpoint['model_state_dict']
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

        if 'refiner.task_embedding.weight' in state_dict:
            max_tasks = state_dict['refiner.task_embedding.weight'].shape[0]
            print(f"Inferred max_tasks={max_tasks} from checkpoint")
        else:
            max_tasks = dataset_info['num_tasks']
            print(f"Using max_tasks={max_tasks} from dataset_info")

        self.max_tasks = max_tasks

        # Infer max_steps from checkpoint step_embedding shape
        if 'refiner.step_embedding.weight' in state_dict:
            max_steps = state_dict['refiner.step_embedding.weight'].shape[0]
            print(f"Inferred max_steps={max_steps} from checkpoint")
        else:
            max_steps = config.get('max_steps', 8)
            print(f"Using max_steps={max_steps} from config")

        # Get auxiliary loss config
        aux_config = config.get('auxiliary_loss', {})
        include_size_head = aux_config.get('include_size_head', True)
        size_head_hidden_dim = aux_config.get('size_head_hidden_dim', None)

        # Recreate model
        model = ARCIterativeModel(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            max_size=config['max_size'],
            max_steps=max_steps,
            max_tasks=max_tasks,
            embedding_dropout=config.get('embedding_dropout', 0.1),
            include_size_head=include_size_head,
            size_head_hidden_dim=size_head_hidden_dim
        )

        # Load weights
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        return model, config

    def get_task_idx(self, task_id: str) -> int:
        """Get task index for a given task ID."""
        if task_id in self.task_id_to_idx:
            return self.task_id_to_idx[task_id]
        else:
            print(f"Warning: Unknown task {task_id}, using default task index 0")
            return 0

    def sample_iterative(
        self,
        input_grids: torch.Tensor,
        task_indices: torch.Tensor,
        pred_height: int,
        pred_width: int,
        d4_idx: Optional[torch.Tensor] = None,
        color_shift: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Run K-step iterative refinement (all argmax for inference).

        Returns:
            final_prediction: Final refined output
        """
        self.model.eval()

        batch_size = input_grids.shape[0]
        max_size = input_grids.shape[1]

        # Initialize with zeros
        x_current = torch.zeros((batch_size, max_size, max_size), dtype=torch.long, device=self.device)

        # Create valid region mask
        valid_mask = torch.zeros((batch_size, max_size, max_size), dtype=torch.bool, device=self.device)
        for b in range(batch_size):
            valid_mask[b, :pred_height, :pred_width] = True
        mask_float = valid_mask.float()

        with torch.no_grad():
            for step in range(self.K):
                step_tensor = torch.full((batch_size,), step, dtype=torch.long, device=self.device)

                # Forward pass
                logits = self.model(
                    x_current, input_grids, task_indices, step_tensor,
                    d4_idx=d4_idx, color_shift=color_shift, masks=mask_float
                )

                # Argmax for next prediction
                x_current = torch.argmax(logits, dim=-1)

                # Apply size masking
                for b in range(batch_size):
                    if pred_height < max_size:
                        x_current[b, pred_height:, :] = 0
                    if pred_width < max_size:
                        x_current[b, :, pred_width:] = 0

        return x_current

    def predict_single(
        self,
        input_grid: np.ndarray,
        task_idx: int,
        expected_output: np.ndarray = None
    ) -> Tuple[np.ndarray, Optional[str], str]:
        """
        Run single prediction on input grid.

        Returns:
            predicted_grid, error, size_source
        """
        try:
            # Convert to tokens
            input_tokens, _, _ = grid_to_tokens(input_grid, max_size=self.config['max_size'])
            input_batch = input_tokens.unsqueeze(0).to(self.device)
            task_ids = torch.tensor([task_idx]).to(self.device)

            # Get size prediction
            if hasattr(self.model, 'include_size_head') and self.model.include_size_head:
                with torch.no_grad():
                    pred_heights, pred_widths = self.model.predict_sizes(input_batch, task_ids)
                    pred_height, pred_width = pred_heights[0].item(), pred_widths[0].item()
                size_source = "integrated_size_head"
            else:
                # Fallback to ground truth dimensions
                if expected_output is not None and len(expected_output) > 0:
                    pred_height, pred_width = expected_output.shape
                    size_source = "ground_truth"
                else:
                    pred_height, pred_width = self.config['max_size'], self.config['max_size']
                    size_source = "ground_truth"

            # Run iterative refinement
            predicted_grids = self.sample_iterative(
                input_batch, task_ids, pred_height, pred_width
            )

            # Crop to predicted size
            full_grid = predicted_grids[0].cpu().numpy()
            predicted_grid = full_grid[:pred_height, :pred_width]

            return predicted_grid, None, size_source

        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            if self.debug:
                error_msg += f"\n{traceback.format_exc()}"
            return np.array([]), error_msg, "ground_truth"

    def _load_solutions(self, dataset: str) -> Dict[str, List[List[List[int]]]]:
        """Load solutions from appropriate solutions file."""
        dataset_name = dataset.split('/')[0] if '/' in dataset else "arc-prize-2025"

        if 'training' in dataset:
            solutions_path = f"data/{dataset_name}/arc-agi_training_solutions.json"
        elif 'evaluation' in dataset:
            solutions_path = f"data/{dataset_name}/arc-agi_evaluation_solutions.json"
        else:
            return {}

        try:
            with open(solutions_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Solutions file not found: {solutions_path}")
            return {}

    def run_task(
        self,
        task_id: str,
        task_data: Dict[str, Any],
        dataset: str
    ) -> TaskResult:
        """Run inference on a single ARC task with pass@2 scoring."""
        timestamp = datetime.datetime.now().isoformat()

        if not task_data["test"]:
            raise ValueError(f"Task {task_id} has no test examples")

        solutions = self._load_solutions(dataset)
        task_idx = self.get_task_idx(task_id)

        test_results = []
        num_test_examples_passed = 0

        for test_idx, test_example in enumerate(task_data["test"]):
            input_grid = np.array(test_example["input"])

            # Get expected output
            if task_id in solutions and test_idx < len(solutions[task_id]):
                expected_output = np.array(solutions[task_id][test_idx])
            elif 'output' in test_example:
                expected_output = np.array(test_example["output"])
            else:
                raise ValueError(f"No solution found for task {task_id}, test example {test_idx}")

            # Run two independent attempts
            attempt_1 = self._run_attempt(input_grid, expected_output, test_idx, task_idx)
            attempt_2 = self._run_attempt(input_grid, expected_output, test_idx, task_idx)

            # Calculate pass@2
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

    def _run_attempt(
        self,
        input_grid: np.ndarray,
        expected_output: np.ndarray,
        test_idx: int,
        task_idx: int
    ) -> IterativeResult:
        """Run a single iterative refinement attempt"""
        predicted_grid, error, size_source = self.predict_single(input_grid, task_idx, expected_output)

        # Check correctness
        correct = False
        if error is None and len(expected_output) > 0 and len(predicted_grid) > 0:
            try:
                if predicted_grid.shape == expected_output.shape:
                    correct = np.array_equal(predicted_grid, expected_output)
                else:
                    error = f"Shape mismatch: predicted {predicted_grid.shape} vs expected {expected_output.shape}"
            except Exception as e:
                error = f"Comparison failed: {str(e)}"
        elif error is None and len(predicted_grid) == 0:
            error = "No valid region extracted from prediction"

        pred_height = predicted_grid.shape[0] if len(predicted_grid) > 0 else 30
        pred_width = predicted_grid.shape[1] if len(predicted_grid) > 0 else 30

        return IterativeResult(
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
        return {"total_tasks": 0, "total_test_examples": 0}

    total_tasks = len(results)

    # Task-level metrics
    total_task_score = sum(r["task_score"] for r in results)
    avg_task_score = total_task_score / total_tasks if total_tasks > 0 else 0.0
    perfect_tasks = sum(1 for r in results if r["task_score"] == 1.0)
    failed_tasks = sum(1 for r in results if r["task_score"] == 0.0)
    partial_tasks = sum(1 for r in results if 0.0 < r["task_score"] < 1.0)

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

    # Size accuracy
    size_correct_count = 0
    size_total_count = 0
    for tr in all_test_results:
        if len(tr["attempt_1"]["expected"]) > 0:
            pred_h = tr["attempt_1"]["pred_height"]
            pred_w = tr["attempt_1"]["pred_width"]
            expected_h = len(tr["attempt_1"]["expected"])
            expected_w = len(tr["attempt_1"]["expected"][0]) if expected_h > 0 else 0
            if pred_h == expected_h and pred_w == expected_w:
                size_correct_count += 1
            size_total_count += 1

    metrics = {
        "total_tasks": total_tasks,
        "task_pass_at_2": perfect_tasks,
        "task_pass_at_2_rate": perfect_tasks / total_tasks if total_tasks > 0 else 0.0,
        "avg_task_score": avg_task_score,
        "perfect_tasks": perfect_tasks,
        "partial_tasks": partial_tasks,
        "failed_tasks": failed_tasks,
        "total_test_examples": total_test_examples,
        "test_pass_at_2": test_pass_at_2_count,
        "test_pass_at_2_rate": test_pass_at_2_count / total_test_examples if total_test_examples > 0 else 0.0,
        "test_both_correct": test_both_correct_count,
        "test_attempt_1_correct": test_attempt_1_correct,
        "test_attempt_1_accuracy": test_attempt_1_correct / total_test_examples if total_test_examples > 0 else 0.0,
        "test_attempt_2_correct": test_attempt_2_correct,
        "test_attempt_2_accuracy": test_attempt_2_correct / total_test_examples if total_test_examples > 0 else 0.0,
        "test_attempt_1_errors": test_attempt_1_errors,
        "test_attempt_2_errors": test_attempt_2_errors,
        "size_correct": size_correct_count,
        "size_total": size_total_count,
        "size_accuracy": size_correct_count / size_total_count if size_total_count > 0 else 0.0,
    }

    return metrics


def print_metrics_report(metrics: Dict[str, Any], dataset: str, subset: str):
    """Print formatted metrics report"""
    total_tasks = metrics["total_tasks"]
    total_test_examples = metrics["total_test_examples"]

    print(f"\n{'='*80}")
    print(f"🎯 ITERATIVE REFINER EVALUATION RESULTS")
    print(f"📊 Dataset: {dataset}, Subset: {subset}")
    print(f"{'='*80}")

    if total_tasks == 0:
        print("❌ No tasks evaluated")
        return

    print(f"\n🎯 TASK-LEVEL METRICS (Pass@2):")
    print(f"  Total Tasks: {total_tasks}")
    print(f"  Avg Task Score: {metrics['avg_task_score']:.1%}")
    print(f"  Perfect Tasks: {metrics['perfect_tasks']}/{total_tasks} ({metrics['task_pass_at_2_rate']:.1%})")
    print(f"  Partial Tasks: {metrics['partial_tasks']}/{total_tasks}")
    print(f"  Failed Tasks: {metrics['failed_tasks']}/{total_tasks}")

    print(f"\n📊 TEST EXAMPLE-LEVEL METRICS:")
    print(f"  Total Test Examples: {total_test_examples}")
    print(f"  Pass@2 Rate: {metrics['test_pass_at_2']}/{total_test_examples} ({metrics['test_pass_at_2_rate']:.1%})")
    print(f"  Both Correct: {metrics['test_both_correct']}/{total_test_examples}")
    print(f"  Attempt 1 Accuracy: {metrics['test_attempt_1_correct']}/{total_test_examples} ({metrics['test_attempt_1_accuracy']:.1%})")
    print(f"  Attempt 2 Accuracy: {metrics['test_attempt_2_correct']}/{total_test_examples} ({metrics['test_attempt_2_accuracy']:.1%})")

    print(f"\n📏 SIZE PREDICTION:")
    print(f"  Correct: {metrics['size_correct']}/{metrics['size_total']} ({metrics['size_accuracy']:.1%})")

    print(f"\n📋 ERRORS:")
    print(f"  Attempt 1: {metrics['test_attempt_1_errors']}/{total_test_examples}")
    print(f"  Attempt 2: {metrics['test_attempt_2_errors']}/{total_test_examples}")

    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Run ARC Iterative Refiner Evaluation")

    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--model-path", help="Override model path")
    parser.add_argument("--prefer-best", action="store_true", help="Use best_model.pt instead of final_model.pt")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="auto")
    parser.add_argument("--inference-steps", type=int, help="Number of refinement steps at inference (default: K from config)")
    parser.add_argument("--K", type=int, help="Deprecated: use --inference-steps instead")
    parser.add_argument("--dataset", help="Dataset to use")
    parser.add_argument("--subset", help="Subset to use")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of tasks")
    parser.add_argument("--output", help="Override output file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = json.load(f)

    output_config = config.get('output', {})
    output_dir = Path(output_config.get('output_dir', 'itransformer/outputs/default'))

    # Determine model path
    if args.model_path:
        model_path = args.model_path
    else:
        final_model_path = output_dir / 'final_model.pt'
        best_model_path = output_dir / 'best_model.pt'

        if args.prefer_best:
            model_path = str(best_model_path if best_model_path.exists() else final_model_path)
        else:
            model_path = str(final_model_path if final_model_path.exists() else best_model_path)

    # Determine dataset/subset
    data_config = config.get('data', {})
    config_data_dir = data_config.get('data_dir')
    dataset_from_config = config_data_dir.split('/')[-1] if config_data_dir and '/' in config_data_dir else None

    dataset = args.dataset if args.dataset else (dataset_from_config if dataset_from_config else "arc-prize-2025")
    subset = args.subset if args.subset else "evaluation"

    # Output path
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = str(output_dir / f"evaluation_{dataset}_{subset}_{timestamp}.json")

    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    limit = args.limit if args.limit > 0 else None

    print(f"🚀 ARC Iterative Refiner Evaluation")
    print(f"📅 Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Config: {args.config}")
    print(f"🎲 Dataset: {dataset}/{subset}")
    print(f"⚡ Task limit: {limit if limit else 'All'}")

    try:
        # Determine inference steps (prefer --inference-steps, fallback to --K for backward compat)
        inference_steps = args.inference_steps if args.inference_steps is not None else args.K

        # Initialize inference
        inference = IterativeInference(
            model_path=model_path,
            device=args.device,
            K=inference_steps,
            debug=args.debug,
            dataset=dataset
        )

        # Load tasks
        print(f"\n📂 Loading tasks from {dataset}/{subset}...")

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

        all_tasks_list = [(task_id, task_data) for task_id, task_data in all_task_data.items()]

        if limit and limit < len(all_tasks_list):
            all_tasks_list = all_tasks_list[:limit]

        all_tasks_dict = {task_id: task_data for task_id, task_data in all_tasks_list}

        # Filter by max_size
        filtered_tasks_dict, total_tasks, filtered_count = filter_tasks_by_max_size(
            all_tasks_dict,
            inference.config['max_size'],
            verbose=True
        )

        tasks = [(task_id, task_data) for task_id, task_data in filtered_tasks_dict.items()]
        print(f"📊 Tasks remaining: {len(tasks)}")

        if not tasks:
            print("❌ No tasks remaining after filtering")
            return

        # Run inference
        results = []
        errors = 0

        progress_bar = tqdm(tasks, desc="Running inference")
        for i, (task_id, task_data) in enumerate(progress_bar):
            try:
                result = inference.run_task(task_id, task_data, f"{dataset}/{subset}")
                results.append(result)

                current_metrics = calculate_metrics(results)
                if current_metrics:
                    progress_bar.set_postfix({
                        'Avg Score': f"{current_metrics['avg_task_score']:.1%}",
                        'Pass@2': f"{current_metrics['test_pass_at_2_rate']:.1%}",
                        'Errors': errors
                    })

            except Exception as e:
                errors += 1
                if errors == 1 or args.debug:
                    print(f"\n❌ Error processing {task_id}: {str(e)}")
                    print(traceback.format_exc())

        # Calculate metrics
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
                "K": inference.K,
                "device": str(inference.device),
                "total_tasks": len(tasks),
                "completed_tasks": len(results),
                "errors": errors,
                "limit": limit
            },
            "metrics": metrics,
            "results": results
        }

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
