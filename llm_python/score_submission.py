#!/usr/bin/env python3
"""
Score an ARC submission file against a reference dataset.
Provides Pass@1 and Pass@2 accuracy metrics.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from llm_python.utils.task_loader import get_task_loader
from llm_python.utils.submission_validator import validate_submission_file


def load_submission(submission_path: str) -> Dict:
    """Load submission file"""
    with open(submission_path, 'r') as f:
        return json.load(f)


def score_submission(submission: Dict, dataset: str, subset: str) -> Dict:
    """Score submission against reference dataset"""
    
    # Load reference dataset
    task_loader = get_task_loader()
    reference_tasks = task_loader.get_subset_tasks(f"{dataset}/{subset}")
    reference_dict = {task_id: task_data for task_id, task_data in reference_tasks}
    
    # Initialize metrics
    total_tasks = 0
    total_predictions = 0
    pass_at_1_count = 0
    pass_at_2_count = 0
    
    # Track missing and extra tasks
    submission_task_ids = set(submission.keys())
    reference_task_ids = set(reference_dict.keys())
    missing_tasks = reference_task_ids - submission_task_ids
    extra_tasks = submission_task_ids - reference_task_ids
    
    # Per-task analysis
    task_scores = {}
    
    for task_id in reference_task_ids:
        if task_id not in submission:
            print(f"⚠️ Missing task: {task_id}")
            continue
            
        reference_task = reference_dict[task_id]
        submission_predictions = submission[task_id]
        
        # Get ground truth test outputs
        test_outputs = [test_case["output"] for test_case in reference_task["test"]]
        
        if len(test_outputs) != len(submission_predictions):
            print(f"⚠️ Task {task_id}: Expected {len(test_outputs)} predictions, got {len(submission_predictions)}")
            continue
        
        total_tasks += 1
        
        # Score each test output
        test_results = []
        for i, (ground_truth, prediction_obj) in enumerate(zip(test_outputs, submission_predictions)):
            if not isinstance(prediction_obj, dict) or "attempt_1" not in prediction_obj or "attempt_2" not in prediction_obj:
                print(f"⚠️ Task {task_id}[{i}]: Invalid prediction format")
                continue
                
            attempt_1 = prediction_obj["attempt_1"]
            attempt_2 = prediction_obj["attempt_2"]
            total_predictions += 2  # Two attempts per prediction
            
            # Check if either attempt matches ground truth
            attempt_1_correct = (attempt_1 == ground_truth)
            attempt_2_correct = (attempt_2 == ground_truth)
            
            # Pass@1: First attempt correct
            if attempt_1_correct:
                pass_at_1_count += 1
            
            # Pass@2: Either attempt correct
            if attempt_1_correct or attempt_2_correct:
                pass_at_2_count += 1
            
            test_results.append({
                "test_idx": i,
                "ground_truth": ground_truth,
                "attempt_1": attempt_1,
                "attempt_2": attempt_2,
                "attempt_1_correct": attempt_1_correct,
                "attempt_2_correct": attempt_2_correct,
                "pass_at_2": attempt_1_correct or attempt_2_correct
            })
        
        # Task-level scoring
        task_pass_at_1 = all(result["attempt_1_correct"] for result in test_results)
        task_pass_at_2 = all(result["pass_at_2"] for result in test_results)
        
        task_scores[task_id] = {
            "num_test_outputs": len(test_outputs),
            "test_results": test_results,
            "task_pass_at_1": task_pass_at_1,
            "task_pass_at_2": task_pass_at_2
        }
    
    # Calculate final metrics
    
    pass_at_1_rate = pass_at_1_count / total_predictions if total_predictions > 0 else 0.0
    pass_at_2_rate = pass_at_2_count / total_predictions if total_predictions > 0 else 0.0
    
    # Task-level metrics
    task_pass_at_1_count = sum(1 for scores in task_scores.values() if scores["task_pass_at_1"])
    task_pass_at_2_count = sum(1 for scores in task_scores.values() if scores["task_pass_at_2"])
    
    task_pass_at_1_rate = task_pass_at_1_count / total_tasks if total_tasks > 0 else 0.0
    task_pass_at_2_rate = task_pass_at_2_count / total_tasks if total_tasks > 0 else 0.0
    
    return {
        "dataset": dataset,
        "subset": subset,
        "total_tasks_in_reference": len(reference_task_ids),
        "total_tasks_scored": total_tasks,
        "total_predictions": total_predictions,
        "missing_tasks": list(missing_tasks),
        "extra_tasks": list(extra_tasks),
        
        # Prediction-level metrics (individual test outputs)
        "pass_at_1_count": pass_at_1_count,
        "pass_at_2_count": pass_at_2_count,
        "pass_at_1_rate": pass_at_1_rate,
        "pass_at_2_rate": pass_at_2_rate,
        
        # Task-level metrics (all test outputs must be correct)
        "task_pass_at_1_count": task_pass_at_1_count,
        "task_pass_at_2_count": task_pass_at_2_count,
        "task_pass_at_1_rate": task_pass_at_1_rate,
        "task_pass_at_2_rate": task_pass_at_2_rate,
        
        "task_scores": task_scores
    }


def print_results(results: Dict, verbose: bool = False):
    """Print scoring results"""
    print("=" * 60)
    print("SUBMISSION SCORING RESULTS")
    print("=" * 60)
    print(f"Dataset: {results['dataset']}")
    print(f"Subset: {results['subset']}")
    print(f"Reference tasks: {results['total_tasks_in_reference']}")
    print(f"Tasks scored: {results['total_tasks_scored']}")
    print(f"Total predictions: {results['total_predictions']}")
    
    if results['missing_tasks']:
        print(f"Missing tasks: {len(results['missing_tasks'])}")
        if verbose:
            print(f"  {results['missing_tasks']}")
    
    if results['extra_tasks']:
        print(f"Extra tasks: {len(results['extra_tasks'])}")
        if verbose:
            print(f"  {results['extra_tasks']}")
    
    print(f"\n📊 PREDICTION-LEVEL METRICS:")
    print(f"  Pass@1 (first attempt): {results['pass_at_1_count']}/{results['total_predictions']} ({results['pass_at_1_rate']:.1%})")
    print(f"  Pass@2 (either attempt): {results['pass_at_2_count']}/{results['total_predictions']} ({results['pass_at_2_rate']:.1%})")
    
    print(f"\n📊 TASK-LEVEL METRICS:")
    print(f"  Tasks Pass@1 (all outputs correct on first attempt): {results['task_pass_at_1_count']}/{results['total_tasks_scored']} ({results['task_pass_at_1_rate']:.1%})")
    print(f"  Tasks Pass@2 (all outputs correct on either attempt): {results['task_pass_at_2_count']}/{results['total_tasks_scored']} ({results['task_pass_at_2_rate']:.1%})")
    
    if verbose:
        print(f"\n📋 DETAILED TASK RESULTS:")
        for task_id, scores in results['task_scores'].items():
            status_1 = "✅" if scores['task_pass_at_1'] else "❌"
            status_2 = "✅" if scores['task_pass_at_2'] else "❌"
            print(f"  {task_id}: {status_1} Pass@1 | {status_2} Pass@2 | {scores['num_test_outputs']} outputs")
            
            if not scores['task_pass_at_2']:  # Show details for failed tasks
                for result in scores['test_results']:
                    if not result['pass_at_2']:
                        idx = result['test_idx']
                        print(f"    Output {idx}: ❌ Both attempts incorrect")


def main():
    parser = argparse.ArgumentParser(description="Score ARC submission file against reference dataset")
    parser.add_argument("--submission", default="submission.json", help="Path to submission.json file")
    parser.add_argument("--dataset", default="arc-prize-2025", choices=["arc-agi-1", "arc-agi-1r", "arc-agi-2", "arc-prize-2024", "arc-prize-2025"], help="Reference dataset")
    parser.add_argument("--subset", default="evaluation", help="Reference subset")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed results")
    parser.add_argument("--output", "-o", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Validate submission file exists
    if not Path(args.submission).exists():
        print(f"❌ Submission file not found: {args.submission}")
        return 1
    
    try:
        # Load reference dataset to get expected task IDs for validation
        task_loader = get_task_loader()
        reference_tasks = task_loader.get_subset_tasks(f"{args.dataset}/{args.subset}")
        expected_task_ids = [task_id for task_id, _ in reference_tasks]
        
        # Validate submission file structure
        print(f"🔍 Validating submission file: {args.submission}")
        validate_submission_file(args.submission, expected_task_ids, challenges_path=None)
        
        # Load and score submission
        print(f"📂 Loading submission: {args.submission}")
        submission = load_submission(args.submission)
        
        print(f"🔍 Scoring against {args.dataset}/{args.subset}")
        results = score_submission(submission, args.dataset, args.subset)
        
        # Print results
        print_results(results, verbose=args.verbose)
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())