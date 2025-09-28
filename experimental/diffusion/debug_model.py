#!/usr/bin/env python3
"""
Debug script to check if the diffusion model is responding to different timesteps.
"""
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experimental.diffusion.src.model import ARCDiffusionModel
from experimental.diffusion.utils.noise_scheduler import DiscreteNoiseScheduler


def debug_model_timestep_sensitivity(model_path: str):
    """Check if model outputs different things for different timesteps."""
    print(f"🔍 Loading model: {model_path}")

    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"🖥️ Using device: {device}")

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['config']
    dataset_info = checkpoint['dataset_info']

    model = ARCDiffusionModel(
        vocab_size=model_config['vocab_size'],
        d_model=model_config['d_model'],
        nhead=model_config['nhead'],
        num_layers=model_config['num_layers'],
        max_size=model_config['max_size'],
        max_tasks=dataset_info['num_tasks'],
        embedding_dropout=model_config.get('embedding_dropout', 0.1)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✓ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create test inputs
    batch_size = 1
    max_size = model_config['max_size']

    # Simple test input and noisy output
    input_grid = torch.zeros(batch_size, max_size, max_size, dtype=torch.long, device=device)  # All black
    input_grid[0, :3, :3] = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.long)  # Small pattern

    noisy_output = torch.randint(0, 10, (batch_size, max_size, max_size), dtype=torch.long, device=device)
    task_ids = torch.tensor([0], device=device)

    print(f"📊 Testing timestep sensitivity...")
    print(f"Input grid (top 5x5):")
    print(input_grid[0, :5, :5].cpu().numpy())
    print(f"Noisy output (top 5x5):")
    print(noisy_output[0, :5, :5].cpu().numpy())

    # Test different timesteps
    test_timesteps = [31, 20, 10, 5, 1, 0]  # High to low noise

    outputs = []
    with torch.no_grad():
        for t in test_timesteps:
            timestep = torch.tensor([t], device=device)
            logits = model(noisy_output, input_grid, task_ids, timestep)
            prediction = torch.argmax(logits, dim=-1)

            # Check a small region for differences
            pred_region = prediction[0, :5, :5].cpu().numpy()
            outputs.append(pred_region)

            print(f"\\nTimestep {t:2d} prediction (top 5x5):")
            print(pred_region)

            # Check if output is all the same
            unique_values = torch.unique(prediction[0]).cpu().numpy()
            print(f"  Unique values in prediction: {unique_values}")

    # Check if outputs are different across timesteps
    all_same = True
    for i in range(1, len(outputs)):
        if not np.array_equal(outputs[0], outputs[i]):
            all_same = False
            break

    if all_same:
        print("\\n❌ WARNING: Model produces identical outputs for all timesteps!")
        print("   This suggests the model is not using timestep information properly.")
    else:
        print("\\n✅ Model produces different outputs for different timesteps.")
        print("   The model appears to be responding to timestep conditioning.")

    # Check if model is just predicting one value everywhere
    final_prediction = outputs[-1]  # t=0 prediction
    if len(np.unique(final_prediction)) == 1:
        print(f"\\n⚠️  Model is predicting the same value ({final_prediction[0,0]}) everywhere.")
        print("   This might indicate the model has collapsed or is undertrained.")
    else:
        print(f"\\n✅ Model predicts diverse values: {np.unique(final_prediction)}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python debug_model.py <model_path>")
        print("Example: python debug_model.py experimental/diffusion/outputs/mps/best_model.pt")
        sys.exit(1)

    model_path = sys.argv[1]
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)

    debug_model_timestep_sensitivity(model_path)