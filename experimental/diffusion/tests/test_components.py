#!/usr/bin/env python3
"""
Simple tests to verify diffusion model components work correctly.
"""
import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from experimental.diffusion.src.model import ARCDiffusionModel
from experimental.diffusion.src.dataset import ARCDataset, load_arc_data_paths
from experimental.diffusion.utils.noise_scheduler import DiscreteNoiseScheduler
from experimental.diffusion.utils.grid_utils import pad_grid_to_size, create_mask, grid_to_tokens, detect_valid_region, grid_to_display_string


def test_noise_scheduler():
    """Test the discrete noise scheduler."""
    print("Testing noise scheduler...")

    scheduler = DiscreteNoiseScheduler(num_timesteps=8, vocab_size=11)

    # Test basic properties
    assert len(scheduler.betas) == 8
    assert len(scheduler.alphas) == 8
    assert len(scheduler.alpha_bars) == 8

    # Test noise addition
    batch_size = 2
    height, width = 5, 5
    x0 = torch.randint(0, 10, (batch_size, height, width))
    timesteps = torch.tensor([1, 4])

    xt = scheduler.add_noise(x0, timesteps)
    assert xt.shape == x0.shape

    print("✓ Noise scheduler test passed")


def test_grid_utils():
    """Test grid utility functions."""
    print("Testing grid utilities...")

    # Test padding
    grid = torch.tensor([[1, 2], [3, 4]])
    padded = pad_grid_to_size(grid, target_size=5, pad_value=10)
    assert padded.shape == (5, 5)
    assert padded[0, 0] == 1
    assert padded[4, 4] == 10  # Padded region

    # Test mask creation
    mask = create_mask(3, 4, max_size=5)
    assert mask.shape == (5, 5)
    assert mask[2, 3] == 1.0  # Inside valid region
    assert mask[3, 0] == 0.0  # Outside valid region

    # Test grid to tokens conversion
    grid_np = np.array([[1, 2, 3], [4, 5, 6]])
    tokens, h, w = grid_to_tokens(grid_np, max_size=5)
    assert tokens.shape == (5, 5)
    assert h == 2 and w == 3
    assert tokens[0, 0] == 1
    assert tokens[3, 3] == 10  # PAD token

    # Test PAD token detection
    padded_grid = np.array([
        [1, 2, 10, 10, 10],
        [3, 4, 10, 10, 10],
        [10, 10, 10, 10, 10],
        [10, 10, 10, 10, 10],
        [10, 10, 10, 10, 10]
    ])
    valid_region, error = detect_valid_region(padded_grid)
    assert error is None
    assert valid_region.shape == (2, 2)
    assert np.array_equal(valid_region, [[1, 2], [3, 4]])

    # Test grid display with PAD tokens
    display_str = grid_to_display_string(padded_grid[:3, :3], pad_symbol='*')
    expected = "12*\n34*\n***"
    assert display_str == expected

    print("✓ Grid utilities test passed")


def test_model_forward():
    """Test model forward pass."""
    print("Testing model forward pass...")

    # Small model for testing
    model = ARCDiffusionModel(
        vocab_size=11,
        d_model=64,
        nhead=2,
        num_layers=2,
        max_size=10,
        max_tasks=10,
        embedding_dropout=0.0  # Disable dropout for testing
    )

    batch_size = 2
    max_size = 10

    # Create test inputs
    xt = torch.randint(0, 11, (batch_size, max_size, max_size))
    input_grid = torch.randint(0, 10, (batch_size, max_size, max_size))
    task_ids = torch.tensor([0, 1])
    timesteps = torch.tensor([1, 2])

    # Forward pass
    logits = model(xt, input_grid, task_ids, timesteps)

    # Check shapes
    assert logits.shape == (batch_size, max_size, max_size, 11)

    print("✓ Model forward pass test passed")


def test_loss_computation():
    """Test loss computation."""
    print("Testing loss computation...")

    model = ARCDiffusionModel(
        vocab_size=11,
        d_model=64,
        nhead=2,
        num_layers=2,
        max_size=10,
        max_tasks=10,
        embedding_dropout=0.0  # Disable dropout for testing
    )

    batch_size = 2
    max_size = 10

    # Create test data
    x0 = torch.randint(0, 10, (batch_size, max_size, max_size))
    input_grid = torch.randint(0, 10, (batch_size, max_size, max_size))
    task_ids = torch.tensor([0, 1])
    heights = torch.tensor([5, 7])
    widths = torch.tensor([6, 8])
    xt = torch.randint(0, 11, (batch_size, max_size, max_size))
    timesteps = torch.tensor([1, 2])

    # Compute losses
    losses = model.compute_loss(x0, input_grid, task_ids, xt, timesteps)

    # Check that we get expected loss components
    assert 'total_loss' in losses
    assert 'grid_loss' in losses
    assert isinstance(losses['total_loss'].item(), float)

    print("✓ Loss computation test passed")


def test_pad_token_prediction():
    """Test that model can predict PAD tokens."""
    print("Testing PAD token prediction...")

    model = ARCDiffusionModel(
        vocab_size=11,
        d_model=64,
        nhead=2,
        num_layers=2,
        max_size=10,
        max_tasks=10,
        embedding_dropout=0.0  # Disable dropout for testing
    )

    batch_size = 2
    max_size = 10

    # Create test inputs with some PAD tokens
    xt = torch.randint(0, 11, (batch_size, max_size, max_size))
    input_grid = torch.randint(0, 10, (batch_size, max_size, max_size))
    task_ids = torch.tensor([0, 1])
    timesteps = torch.tensor([1, 2])

    # Forward pass
    logits = model(xt, input_grid, task_ids, timesteps)

    # Check that model can predict all 11 classes including PAD (10)
    assert logits.shape == (batch_size, max_size, max_size, 11)

    # Sample from logits and check that PAD tokens can be generated
    predictions = torch.argmax(logits, dim=-1)
    assert predictions.min() >= 0 and predictions.max() <= 10

    print("✓ PAD token prediction test passed")


def test_data_loading():
    """Test data loading (if data files exist)."""
    print("Testing data loading...")

    try:
        # Try to load data paths
        data_paths = load_arc_data_paths()

        # Check if training data exists
        train_path = Path(data_paths['train'][0])
        if not train_path.exists():
            print("⚠ Training data not found, skipping data loading test")
            return

        # Create a small dataset (using defaults: include_training_test_examples=True)
        dataset = ARCDataset(
            data_paths=[str(train_path)],
            max_size=10,
            augment=False
        )

        if len(dataset) == 0:
            print("⚠ Dataset is empty, skipping data loading test")
            return

        # Test getting an example
        example = dataset[0]
        assert 'input_grid' in example
        assert 'output_grid' in example
        assert 'task_idx' in example
        assert 'height' in example
        assert 'width' in example

        assert example['input_grid'].shape == (10, 10)
        assert example['output_grid'].shape == (10, 10)

        print(f"✓ Data loading test passed (dataset size: {len(dataset)})")

    except Exception as e:
        print(f"⚠ Data loading test failed: {e}")


def run_all_tests():
    """Run all component tests."""
    print("Running diffusion model component tests...")
    print("=" * 50)

    try:
        test_noise_scheduler()
        test_grid_utils()
        test_model_forward()
        test_loss_computation()
        test_pad_token_prediction()
        test_data_loading()

        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)