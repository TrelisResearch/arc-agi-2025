#!/usr/bin/env python3
"""
Test script for size head training functionality.
"""
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experimental.diffusion.src.model import ARCDiffusionModel, GridSizePredictionHead
from experimental.diffusion.src.dataset import ARCDataset, load_arc_data_paths
from experimental.diffusion.utils.noise_scheduler import DiscreteNoiseScheduler


def test_size_head():
    """Test size head creation and basic functionality."""
    print("Testing GridSizePredictionHead...")

    # Create a small diffusion model for testing
    diffusion_model = ARCDiffusionModel(
        vocab_size=11,
        d_model=64,
        nhead=2,
        num_layers=2,
        max_size=10,
        max_tasks=10,
        embedding_dropout=0.0
    )

    # Create size head
    size_head = GridSizePredictionHead(
        diffusion_model=diffusion_model,
        hidden_dim=32,  # Small for testing
        max_size=10
    )

    print(f"✓ Created size head with {sum(p.numel() for p in size_head.parameters() if p.requires_grad):,} trainable parameters")

    # Test forward pass
    batch_size = 2
    input_grids = torch.randint(0, 11, (batch_size, 10, 10))
    task_ids = torch.tensor([0, 1])
    target_heights = torch.tensor([5, 7])
    target_widths = torch.tensor([6, 8])

    # Test prediction
    pred_heights, pred_widths = size_head.predict_sizes(input_grids, task_ids)
    print(f"✓ Predictions: heights={pred_heights.tolist()}, widths={pred_widths.tolist()}")

    # Test loss computation
    loss = size_head.compute_size_loss(input_grids, task_ids, target_heights, target_widths)
    print(f"✓ Loss computation: {loss.item():.4f}")

    # Test that diffusion model is frozen
    diffusion_params_frozen = all(not p.requires_grad for p in diffusion_model.parameters())
    print(f"✓ Diffusion model frozen: {diffusion_params_frozen}")

    return size_head


def test_integration():
    """Test integration with dataset."""
    print("\\nTesting dataset integration...")

    try:
        # Load small dataset
        data_paths = load_arc_data_paths(
            data_dir="data/arc-prize-2024",
            datasets=["training_challenges"]
        )

        dataset = ARCDataset(
            data_paths=data_paths['train'][:1],  # Only one file for testing
            max_size=10,
            augment=False,
            include_training_test_examples=False
        )

        print(f"✓ Loaded test dataset with {len(dataset)} examples")

        # Get one example
        example = dataset[0]
        print(f"✓ Example shape: input={example['input_grid'].shape}, target size=({example['height'].item()}, {example['width'].item()})")

        return True

    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False


def main():
    print("Size Head Testing")
    print("=" * 40)

    # Test 1: Basic functionality
    size_head = test_size_head()

    # Test 2: Dataset integration
    dataset_ok = test_integration()

    print("\\n" + "=" * 40)
    if dataset_ok:
        print("✅ All tests passed! Size head is ready for training.")
        print("\\nNext steps:")
        print("1. Train diffusion model: uv run python experimental/diffusion/train_with_config.py --config configs/gpu_config.json")
        print("2. Train size head: uv run python experimental/diffusion/train_size_head.py --diffusion-model outputs/gpu/best_model.pt --config configs/gpu_config.json --output outputs/gpu/size_head.pt")
        print("3. Run inference: uv run python experimental/diffusion/run_inference.py --model-path outputs/gpu/best_model.pt --size-head-path outputs/gpu/size_head.pt")
    else:
        print("❌ Some tests failed - check data directory exists")


if __name__ == "__main__":
    main()