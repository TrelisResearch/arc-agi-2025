#!/usr/bin/env python3
"""
ARC Diffusion Model Inference Script with Majority Voting

Generates 64 samples in parallel and uses majority voting to select the top 2 attempts.
This is much more efficient than sequential sampling and should improve accuracy.
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
from collections import Counter

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experimental.diffusion.src.model import ARCDiffusionModel
from experimental.diffusion.src.training import ARCDiffusionSampler
from experimental.diffusion.src.dataset import ARCDataset, load_arc_data_paths
from experimental.diffusion.utils.noise_scheduler import DiscreteNoiseScheduler
from experimental.diffusion.utils.grid_utils import grid_to_tokens, tokens_to_grid, detect_valid_region
from llm_python.utils.task_loader import TaskData, get_task_loader


class MajorityVoteResult(TypedDict):
    """Result of running majority vote diffusion model on a single test example"""
    test_idx: int
    input_grid: List[List[int]]  # Input grid for reference
    predicted: Optional[List[List[int]]]  # Predicted grid or None if error
    expected: List[List[int]]  # Expected output grid
    correct: bool  # Whether prediction matches expected
    error: Optional[str]  # Error message if execution failed
    pred_height: int  # Height of predicted grid
    pred_width: int  # Width of predicted grid
    vote_count: int  # Number of samples that voted for this result
    total_samples: int  # Total number of samples generated


class TaskResult(TypedDict):
    """Result for a single ARC task with majority vote attempts"""
    task_id: str
    timestamp: str

    # Attempt results (top 2 from majority vote)
    attempt_1: MajorityVoteResult
    attempt_2: MajorityVoteResult

    # Overall metrics
    pass_at_2: bool
    both_correct: bool
    num_train_examples: int
    num_test_examples: int


class MajorityVoteDiffusionInference:
    """Main inference class for ARC diffusion model with majority voting"""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        num_samples: int = 64,
        debug: bool = False
    ):
        self.model_path = model_path
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
        self.num_samples = num_samples
        self.debug = debug

        print(f"🔥 Loading model from {model_path}")
        print(f"🖥️ Using device: {self.device}")
        print(f"🎲 Generating {num_samples} samples per task for majority voting")

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

        self.sampler = ARCDiffusionSampler(self.model, self.noise_scheduler, self.device, dataset=self.dataset, debug=False)

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
            max_tasks=self.max_tasks,
            embedding_dropout=config.get('embedding_dropout', 0.1)
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        return model, config

    def get_task_idx(self, task_id: str) -> int:
        """Get task index from task ID, with fallback for unknown tasks"""
        if task_id in self.task_id_to_idx:
            return self.task_id_to_idx[task_id]
        else:
            # Fallback for unknown tasks (use hash modulo)
            return hash(task_id) % self.max_tasks

    def predict_with_majority_vote(self, input_grid: np.ndarray, task_idx: int, task_id: str = None) -> Tuple[List[np.ndarray], List[int], Optional[str]]:
        """
        Run majority vote prediction on input grid.

        Returns:
            predicted_grids: List of top candidate grids ranked by vote count
            vote_counts: List of vote counts for each candidate
            error: Error message if any
        """
        try:
            # Convert to tokens and create batch
            input_tokens, _, _ = grid_to_tokens(input_grid, max_size=self.config['max_size'])
            input_batch = input_tokens.unsqueeze(0).repeat(self.num_samples, 1, 1).to(self.device)  # [num_samples, max_size, max_size]

            # Use task ID (ensure it's within trained range)
            task_ids = torch.tensor([task_idx]).repeat(self.num_samples).to(self.device)

            # Sample outputs in parallel
            with torch.no_grad():
                if self.debug:
                    print(f"🎲 Generating {self.num_samples} samples in parallel...")

                predicted_grids = self.sampler.sample(
                    input_grids=input_batch,
                    task_indices=task_ids,
                    task_ids=[task_id] * self.num_samples if task_id is not None else None,
                    num_inference_steps=self.num_inference_steps
                )

            # Extract valid regions and collect candidates
            candidates = []
            extraction_errors = []

            for i in range(self.num_samples):
                predicted_grid, region_error = detect_valid_region(predicted_grids[i].cpu().numpy())
                if region_error:
                    extraction_errors.append(region_error)
                    continue

                # Convert to tuple for hashing in Counter
                if len(predicted_grid) > 0:
                    grid_tuple = tuple(tuple(row) for row in predicted_grid)
                    candidates.append(grid_tuple)

            if not candidates:
                error_summary = f"All {self.num_samples} samples failed region extraction. Errors: {set(extraction_errors)}"
                return [], [], error_summary

            # Count votes for each unique grid
            vote_counter = Counter(candidates)

            if self.debug:
                print(f"📊 Found {len(vote_counter)} unique candidates from {len(candidates)} valid samples")
                for i, (grid_tuple, count) in enumerate(vote_counter.most_common(5)):
                    print(f"  Rank {i+1}: {count} votes ({count/len(candidates)*100:.1f}%)")

            # Get top 2 candidates by vote count
            top_candidates = vote_counter.most_common(2)

            # Convert back to numpy arrays
            result_grids = []
            result_votes = []

            for grid_tuple, vote_count in top_candidates:
                grid_array = np.array([list(row) for row in grid_tuple])
                result_grids.append(grid_array)
                result_votes.append(vote_count)

            return result_grids, result_votes, None

        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            if self.debug:
                error_msg += f"\nTraceback: {traceback.format_exc()}"
            return [], [], error_msg

    def run_task(
        self,
        task_id: str,
        task_data: TaskData,
        dataset: str = "evaluation"
    ) -> TaskResult:
        """
        Run inference on a single ARC task using majority voting.

        Args:
            task_id: ARC task identifier
            task_data: Task data containing train/test examples
            dataset: Dataset name to load appropriate solutions

        Returns:
            TaskResult with both attempts and scoring
        """
        timestamp = datetime.datetime.now().isoformat()

        # TEMPORARY: Use the first training example instead of test example
        # This is for debugging - normally we'd use test examples
        if not task_data["train"]:
            raise ValueError(f"Task {task_id} has no training examples")

        train_example = task_data["train"][0]
        input_grid = np.array(train_example["input"])

        # Use the training example's output as expected (we have ground truth)
        expected_output = np.array(train_example["output"])

        # Get correct task index
        task_idx = self.get_task_idx(task_id)

        # Generate candidates using majority voting
        candidate_grids, vote_counts, error = self.predict_with_majority_vote(input_grid, task_idx, task_id)

        if error or len(candidate_grids) == 0:
            # Create failed attempts
            failed_attempt = MajorityVoteResult(
                test_idx=0,
                input_grid=input_grid.tolist(),
                predicted=None,
                expected=expected_output.tolist(),
                correct=False,
                error=error or "No valid candidates generated",
                pred_height=0,
                pred_width=0,
                vote_count=0,
                total_samples=self.num_samples
            )
            return TaskResult(
                task_id=task_id,
                timestamp=timestamp,
                attempt_1=failed_attempt,
                attempt_2=failed_attempt,
                pass_at_2=False,
                both_correct=False,
                num_train_examples=len(task_data["train"]),
                num_test_examples=len(task_data["test"])
            )

        # Create attempts from top candidates
        attempts = []
        for i, (predicted_grid, vote_count) in enumerate(zip(candidate_grids, vote_counts)):
            # Check correctness
            correct = False
            attempt_error = None
            if len(expected_output) > 0 and len(predicted_grid) > 0:
                try:
                    if predicted_grid.shape == expected_output.shape:
                        correct = np.array_equal(predicted_grid, expected_output)
                    else:
                        attempt_error = f"Shape mismatch: predicted {predicted_grid.shape} vs expected {expected_output.shape}"
                except Exception as e:
                    attempt_error = f"Comparison failed: {str(e)}"

            attempts.append(MajorityVoteResult(
                test_idx=0,
                input_grid=input_grid.tolist(),
                predicted=predicted_grid.tolist(),
                expected=expected_output.tolist(),
                correct=correct,
                error=attempt_error,
                pred_height=predicted_grid.shape[0],
                pred_width=predicted_grid.shape[1],
                vote_count=vote_count,
                total_samples=self.num_samples
            ))

        # Ensure we have exactly 2 attempts (pad with copy if needed)
        if len(attempts) == 1:
            attempts.append(attempts[0])  # Duplicate the only candidate

        # Calculate pass@2 metrics
        pass_at_2 = attempts[0]["correct"] or attempts[1]["correct"]
        both_correct = attempts[0]["correct"] and attempts[1]["correct"]

        return TaskResult(
            task_id=task_id,
            timestamp=timestamp,
            attempt_1=attempts[0],
            attempt_2=attempts[1],
            pass_at_2=pass_at_2,
            both_correct=both_correct,
            num_train_examples=len(task_data["train"]),
            num_test_examples=len(task_data["test"])
        )


def main():
    parser = argparse.ArgumentParser(description="Run ARC Diffusion Model Inference with Majority Voting")
    parser.add_argument("--model-path", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto", help="Device to use")
    parser.add_argument("--num-steps", type=int, help="Number of inference steps (default: use training steps)")
    parser.add_argument("--num-samples", type=int, default=64, help="Number of samples to generate for majority voting")
    parser.add_argument("--dataset", default="arc-prize-2024/evaluation", help="Dataset to use")
    parser.add_argument("--subset", help="Subset to use")
    parser.add_argument("--limit", type=int, help="Limit number of tasks to run")
    parser.add_argument("--output", help="Output file to save results")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    print("🚀 ARC Diffusion Model Inference with Majority Voting")
    print(f"📅 Started at: {datetime.datetime.now().isoformat()}")
    print(f"🎲 Dataset: {args.dataset}")
    if args.limit:
        print(f"⚡ Task limit: {args.limit}")

    try:
        # Initialize inference
        device = None if args.device == "auto" else args.device
        inference = MajorityVoteDiffusionInference(
            model_path=args.model_path,
            device=device,
            num_inference_steps=args.num_steps,
            num_samples=args.num_samples,
            debug=args.debug
        )

        # Load tasks
        subset_path = f"{args.dataset}/{args.subset}" if args.subset else args.dataset
        print(f"\n📂 Loading tasks from {subset_path}...")
        task_loader = get_task_loader()
        tasks = task_loader.get_dataset_subset(subset_path, max_rows=args.limit)

        print(f"📋 Found {len(tasks)} tasks")

        if len(tasks) == 0:
            print("❌ No tasks found!")
            return

        # Run inference
        results = []
        total_pass_at_2 = 0
        total_both_correct = 0
        total_attempt_1 = 0
        total_attempt_2 = 0
        total_errors = 0

        progress_bar = tqdm(tasks, desc="Running inference")
        for task_id, task_data in progress_bar:
            try:
                result = inference.run_task(task_id, task_data, dataset=args.dataset)
                results.append(result)

                # Update metrics
                if result["pass_at_2"]:
                    total_pass_at_2 += 1
                if result["both_correct"]:
                    total_both_correct += 1
                if result["attempt_1"]["correct"]:
                    total_attempt_1 += 1
                if result["attempt_2"]["correct"]:
                    total_attempt_2 += 1
                if result["attempt_1"]["error"] or result["attempt_2"]["error"]:
                    total_errors += 1

                # Update progress bar
                progress_bar.set_postfix({
                    'Pass@2': f'{total_pass_at_2/len(results)*100:.1f}%',
                    'Errors': total_errors
                })

            except Exception as e:
                print(f"❌ Failed to process task {task_id}: {str(e)}")
                if args.debug:
                    traceback.print_exc()
                total_errors += 1

        # Final results
        total_tasks = len(results)
        if total_tasks > 0:
            print("\n" + "="*80)
            print("🎯 MAJORITY VOTE DIFFUSION MODEL INFERENCE RESULTS")
            print(f"📊 Dataset: {args.dataset}")
            print(f"🎲 Samples per task: {args.num_samples}")
            print(f"📋 Total Tasks: {total_tasks}")
            print("="*80)
            print(f"🎲 Pass@2: {total_pass_at_2}/{total_tasks} ({total_pass_at_2/total_tasks*100:.1f}%)")
            print(f"🎯 Both Correct: {total_both_correct}/{total_tasks} ({total_both_correct/total_tasks*100:.1f}%)")
            print(f"🥇 Attempt 1: {total_attempt_1}/{total_tasks} ({total_attempt_1/total_tasks*100:.1f}%)")
            print(f"🥈 Attempt 2: {total_attempt_2}/{total_tasks} ({total_attempt_2/total_tasks*100:.1f}%)")
            print()
            print(f"📋 Error Analysis:")
            print(f"  Tasks with errors: {total_errors}/{total_tasks} ({total_errors/total_tasks*100:.1f}%)")
            print("="*80)

        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            output_data = {
                "metadata": {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "dataset": args.dataset,
                    "subset": args.subset,
                    "model_path": args.model_path,
                    "num_samples": args.num_samples,
                    "num_inference_steps": args.num_steps or inference.config['num_timesteps'],
                    "device": str(inference.device),
                    "total_tasks": len(tasks),
                    "completed_tasks": total_tasks,
                    "errors": total_errors
                },
                "metrics": {
                    "pass_at_2": total_pass_at_2 / total_tasks if total_tasks > 0 else 0,
                    "both_correct": total_both_correct / total_tasks if total_tasks > 0 else 0,
                    "attempt_1_correct": total_attempt_1 / total_tasks if total_tasks > 0 else 0,
                    "attempt_2_correct": total_attempt_2 / total_tasks if total_tasks > 0 else 0,
                    "error_rate": total_errors / total_tasks if total_tasks > 0 else 0
                },
                "results": results
            }

            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)

            print(f"💾 Results saved to: {output_path}")

        print(f"\n✅ Inference completed: {len(results)}/{len(tasks)} tasks successful")

    except Exception as e:
        print(f"\n💥 Inference failed: {str(e)}")
        if args.debug:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()