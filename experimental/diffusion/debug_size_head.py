#!/usr/bin/env python3
"""
Debug script to see what the size head is actually predicting.
"""
import json
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experimental.diffusion.src.model import ARCDiffusionModel, GridSizePredictionHead
from experimental.diffusion.utils.grid_utils import grid_to_tokens


def debug_size_head_predictions(
    size_head_path: str,
    diffusion_model_path: str,
    dataset_path: str,
    num_examples: int = 10
):
    """Debug what the size head is predicting."""
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"🖥️ Using device: {device}")

    # Load models
    checkpoint = torch.load(diffusion_model_path, map_location=device)

    # Handle both old and new checkpoint formats
    if 'config' in checkpoint:
        model_config = checkpoint['config']
        dataset_info = checkpoint['dataset_info']
    else:
        # Old format - checkpoint is the model state dict directly
        print("⚠️ Old checkpoint format detected - using default config")
        model_config = {
            'vocab_size': 11,
            'd_model': 256,
            'nhead': 4,
            'num_layers': 8,
            'max_size': 30,
            'embedding_dropout': 0.25
        }
        dataset_info = {'num_tasks': 1000}  # Default

    diffusion_model = ARCDiffusionModel(
        vocab_size=model_config['vocab_size'],
        d_model=model_config['d_model'],
        nhead=model_config['nhead'],
        num_layers=model_config['num_layers'],
        max_size=model_config['max_size'],
        max_tasks=dataset_info['num_tasks'],
        embedding_dropout=model_config.get('embedding_dropout', 0.1)
    )
    # Load weights based on checkpoint format
    if 'model_state_dict' in checkpoint:
        diffusion_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        diffusion_model.load_state_dict(checkpoint)
    diffusion_model.to(device)
    diffusion_model.eval()

    size_head = GridSizePredictionHead(
        diffusion_model=diffusion_model,
        hidden_dim=256,
        max_size=model_config['max_size']
    )
    size_head.load_state_dict(torch.load(size_head_path, map_location=device))
    size_head.to(device)
    size_head.eval()

    # Load tasks
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    tasks = list(data.items())[:num_examples]

    print(f"\\n🔍 Size Head Predictions Debug:")
    print(f"=" * 60)

    with torch.no_grad():
        for i, (task_id, task_data) in enumerate(tasks):
            test_example = task_data['test'][0]
            input_grid = np.array(test_example['input'])
            expected_output = np.array(test_example['output']) if 'output' in test_example else None

            if expected_output is None:
                continue

            # Convert input to tensor
            input_tokens, input_h, input_w = grid_to_tokens(input_grid, model_config['max_size'])
            input_tensor = input_tokens.unsqueeze(0).to(device)
            task_tensor = torch.tensor([0], device=device)

            # Get predictions
            pred_heights, pred_widths = size_head.predict_sizes(input_tensor, task_tensor)
            pred_height, pred_width = pred_heights[0].item(), pred_widths[0].item()

            # Get ground truth
            true_height, true_width = expected_output.shape

            print(f"Task {i+1}: {task_id}")
            print(f"  Input size: {input_grid.shape}")
            print(f"  Expected output: {true_height} x {true_width}")
            print(f"  Predicted output: {pred_height} x {pred_width}")
            print(f"  Height correct: {pred_height == true_height}")
            print(f"  Width correct: {pred_width == true_width}")
            print()

    # Also check what the raw logits look like for a few examples
    print(f"\\n🧠 Raw Logits Analysis:")
    print(f"=" * 60)

    with torch.no_grad():
        for i, (task_id, task_data) in enumerate(tasks[:3]):
            test_example = task_data['test'][0]
            input_grid = np.array(test_example['input'])
            expected_output = np.array(test_example['output']) if 'output' in test_example else None

            if expected_output is None:
                continue

            input_tokens, input_h, input_w = grid_to_tokens(input_grid, model_config['max_size'])
            input_tensor = input_tokens.unsqueeze(0).to(device)
            task_tensor = torch.tensor([0], device=device)

            # Get raw logits
            height_logits, width_logits = size_head(input_tensor, task_tensor)

            # Convert to probabilities
            height_probs = torch.softmax(height_logits, dim=-1)
            width_probs = torch.softmax(width_logits, dim=-1)

            # Show top predictions
            height_top_vals, height_top_idx = torch.topk(height_probs[0], 5)
            width_top_vals, width_top_idx = torch.topk(width_probs[0], 5)

            true_height, true_width = expected_output.shape

            print(f"Task {i+1}: {task_id}")
            print(f"  Ground truth: {true_height} x {true_width}")
            print(f"  Top 5 height predictions:")
            for j, (idx, prob) in enumerate(zip(height_top_idx, height_top_vals)):
                size = idx.item() + 1  # Convert 0-indexed to 1-indexed
                marker = "✓" if size == true_height else " "
                print(f"    {marker} {size}: {prob.item():.3f}")

            print(f"  Top 5 width predictions:")
            for j, (idx, prob) in enumerate(zip(width_top_idx, width_top_vals)):
                size = idx.item() + 1  # Convert 0-indexed to 1-indexed
                marker = "✓" if size == true_width else " "
                print(f"    {marker} {size}: {prob.item():.3f}")
            print()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python debug_size_head.py <diffusion_model> <size_head> <dataset>")
        sys.exit(1)

    debug_size_head_predictions(sys.argv[1], sys.argv[2], sys.argv[3])