#!/usr/bin/env python3
"""
Size Head Evaluation Script

Evaluates a trained size prediction head on ARC tasks.
Measures accuracy on the first test output grid of each task.

Usage:
    python experimental/diffusion/eval_size_head.py \
        --diffusion-model experimental/diffusion/outputs/gpu/best_model.pt \
        --size-head experimental/diffusion/outputs/gpu/size_head.pt \
        --dataset data/arc-prize-2024/arc-agi_evaluation_challenges.json
"""
import json
import argparse
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, Any, List, Tuple

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experimental.diffusion.src.model import ARCDiffusionModel, GridSizePredictionHead
from experimental.diffusion.utils.grid_utils import grid_to_tokens
from llm_python.utils.task_loader import get_task_loader


def load_arc_tasks_with_solutions(dataset_path: str) -> List[Tuple[str, Dict]]:
    """Load ARC tasks from JSON file and merge with solutions if available."""
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path, 'r') as f:
        data = json.load(f)

    # Try to load solutions file
    solutions = {}
    dataset_dir = Path(dataset_path).parent
    dataset_name = Path(dataset_path).stem

    # Handle different naming patterns
    if "training_challenges" in dataset_name:
        solutions_path = dataset_dir / "arc-agi_training_solutions.json"
    elif "evaluation_challenges" in dataset_name:
        solutions_path = dataset_dir / "arc-agi_evaluation_solutions.json"
    else:
        solutions_path = dataset_dir / f"{dataset_name}_solutions.json"

    if solutions_path.exists():
        print(f"📝 Loading solutions from: {solutions_path}")
        with open(solutions_path, 'r') as f:
            solutions = json.load(f)
        print(f"✓ Loaded solutions for {len(solutions)} tasks")
    else:
        print(f"⚠️ No solutions file found at: {solutions_path}")

    # Merge solutions into task data
    tasks = []
    for task_id, task_data in data.items():
        # Add solutions to test examples if available
        if task_id in solutions:
            for i, test_example in enumerate(task_data['test']):
                if i < len(solutions[task_id]):
                    test_example['output'] = solutions[task_id][i]

        tasks.append((task_id, task_data))

    return tasks


def evaluate_size_head(
    size_head_path: str,
    diffusion_model_path: str,
    dataset_path: str,
    device: str = "auto"
) -> Dict[str, Any]:
    """
    Evaluate size head performance on ARC tasks.

    Returns:
        Dictionary with evaluation metrics
    """
    # Set up device
    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)

    print(f"🖥️ Using device: {device}")

    # Load diffusion model
    print(f"🔥 Loading diffusion model: {diffusion_model_path}")
    checkpoint = torch.load(diffusion_model_path, map_location=device)
    model_config = checkpoint['config']
    dataset_info = checkpoint['dataset_info']

    diffusion_model = ARCDiffusionModel(
        vocab_size=model_config['vocab_size'],
        d_model=model_config['d_model'],
        nhead=model_config['nhead'],
        num_layers=model_config['num_layers'],
        max_size=model_config['max_size'],
        max_tasks=dataset_info['num_tasks'],
        embedding_dropout=model_config.get('embedding_dropout', 0.1)
    )
    diffusion_model.load_state_dict(checkpoint['model_state_dict'])
    diffusion_model.to(device)
    diffusion_model.eval()

    # Load size head
    print(f"🧠 Loading size head: {size_head_path}")
    size_head = GridSizePredictionHead(
        diffusion_model=diffusion_model,
        hidden_dim=256,
        max_size=model_config['max_size']
    )
    size_head.load_state_dict(torch.load(size_head_path, map_location=device))
    size_head.to(device)
    size_head.eval()

    # Load tasks with solutions
    print(f"📊 Loading tasks: {dataset_path}")
    tasks = load_arc_tasks_with_solutions(dataset_path)
    print(f"✓ Loaded {len(tasks)} tasks")

    # Evaluation metrics
    metrics = {
        'total_tasks': len(tasks),
        'height_correct': 0,
        'width_correct': 0,
        'both_correct': 0,
        'height_errors': [],
        'width_errors': [],
        'size_distribution': defaultdict(int),
        'error_distribution': defaultdict(int)
    }

    print(f"\\n🎯 Evaluating size head performance...")

    # Evaluate each task
    with torch.no_grad():
        for task_id, task_data in tqdm(tasks, desc="Evaluating tasks"):
            # Get first test example (solutions should be loaded)
            test_examples = task_data['test']
            if not test_examples:
                continue

            test_example = test_examples[0]  # First test example only

            # Get input and expected output
            input_grid = np.array(test_example['input'])
            expected_output = np.array(test_example['output']) if 'output' in test_example else None

            if expected_output is None:
                continue  # Skip tasks without ground truth

            # Convert input to tensor
            input_tokens, input_h, input_w = grid_to_tokens(input_grid, model_config['max_size'])
            input_tensor = input_tokens.unsqueeze(0).to(device)

            # Dummy task ID (we don't use task-specific predictions here)
            task_tensor = torch.tensor([0], device=device)

            # Predict sizes
            pred_heights, pred_widths = size_head.predict_sizes(input_tensor, task_tensor)

            # Get actual sizes
            true_height, true_width = expected_output.shape
            pred_height, pred_width = pred_heights[0].item(), pred_widths[0].item()

            # Update metrics
            height_correct = (pred_height == true_height)
            width_correct = (pred_width == true_width)

            if height_correct:
                metrics['height_correct'] += 1
            else:
                metrics['height_errors'].append((task_id, true_height, pred_height))

            if width_correct:
                metrics['width_correct'] += 1
            else:
                metrics['width_errors'].append((task_id, true_width, pred_width))

            if height_correct and width_correct:
                metrics['both_correct'] += 1

            # Track size distributions
            metrics['size_distribution'][f"{true_height}x{true_width}"] += 1

            # Track error patterns
            height_error = abs(pred_height - true_height)
            width_error = abs(pred_width - true_width)
            total_error = height_error + width_error
            metrics['error_distribution'][total_error] += 1

    # Calculate final metrics
    total_tasks = metrics['total_tasks']
    height_accuracy = metrics['height_correct'] / total_tasks if total_tasks > 0 else 0
    width_accuracy = metrics['width_correct'] / total_tasks if total_tasks > 0 else 0
    both_accuracy = metrics['both_correct'] / total_tasks if total_tasks > 0 else 0

    results = {
        'total_tasks': total_tasks,
        'height_accuracy': height_accuracy,
        'width_accuracy': width_accuracy,
        'both_correct_accuracy': both_accuracy,
        'height_errors': len(metrics['height_errors']),
        'width_errors': len(metrics['width_errors']),
        'size_distribution': dict(metrics['size_distribution']),
        'error_distribution': dict(metrics['error_distribution'])
    }

    # Print results
    print(f"\\n📈 Evaluation Results:")
    print(f"=" * 50)
    print(f"Total tasks: {total_tasks}")
    print(f"Height accuracy: {height_accuracy:.3f} ({metrics['height_correct']}/{total_tasks})")
    print(f"Width accuracy: {width_accuracy:.3f} ({metrics['width_correct']}/{total_tasks})")
    print(f"Both correct: {both_accuracy:.3f} ({metrics['both_correct']}/{total_tasks})")
    print()

    # Show most common sizes
    print(f"🎯 Most common output sizes:")
    sorted_sizes = sorted(metrics['size_distribution'].items(), key=lambda x: x[1], reverse=True)
    for size, count in sorted_sizes[:10]:
        print(f"  {size}: {count} tasks")
    print()

    # Show error distribution
    print(f"❌ Error distribution (total absolute error):")
    sorted_errors = sorted(metrics['error_distribution'].items())
    for error, count in sorted_errors[:10]:
        print(f"  Error {error}: {count} tasks")
    print()

    # Show some examples of errors
    if metrics['height_errors']:
        print(f"📏 Height error examples:")
        for task_id, true_h, pred_h in metrics['height_errors'][:5]:
            print(f"  {task_id}: True={true_h}, Pred={pred_h}")
        print()

    if metrics['width_errors']:
        print(f"📐 Width error examples:")
        for task_id, true_w, pred_w in metrics['width_errors'][:5]:
            print(f"  {task_id}: True={true_w}, Pred={pred_w}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ARC Size Prediction Head")

    parser.add_argument("--diffusion-model", required=True, help="Path to trained diffusion model")
    parser.add_argument("--size-head", required=True, help="Path to trained size head")
    parser.add_argument("--dataset", required=True, help="Path to ARC dataset JSON file")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="auto", help="Device to use")
    parser.add_argument("--output", help="Path to save results JSON (optional)")

    args = parser.parse_args()

    # Validate paths
    if not Path(args.diffusion_model).exists():
        print(f"❌ Diffusion model not found: {args.diffusion_model}")
        sys.exit(1)

    if not Path(args.size_head).exists():
        print(f"❌ Size head not found: {args.size_head}")
        sys.exit(1)

    if not Path(args.dataset).exists():
        print(f"❌ Dataset not found: {args.dataset}")
        sys.exit(1)

    print("🎯 Size Head Evaluation")
    print("=" * 50)
    print(f"Diffusion model: {args.diffusion_model}")
    print(f"Size head: {args.size_head}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print("=" * 50)

    # Run evaluation
    try:
        results = evaluate_size_head(
            size_head_path=args.size_head,
            diffusion_model_path=args.diffusion_model,
            dataset_path=args.dataset,
            device=args.device
        )

        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\\n💾 Results saved to: {output_path}")

        print("\\n✅ Evaluation complete!")

    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()