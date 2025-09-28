#!/usr/bin/env python3
"""
Grid Size Prediction Head Training

Trains a size prediction head on top of a pre-trained diffusion model.
The diffusion model weights are frozen during this training.

Usage:
    python experimental/diffusion/train_size_head.py \
        --diffusion-model experimental/diffusion/outputs/gpu/best_model.pt \
        --config experimental/diffusion/configs/gpu_config.json \
        --output experimental/diffusion/outputs/gpu/size_head.pt
"""
import json
import argparse
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experimental.diffusion.src.model import ARCDiffusionModel, GridSizePredictionHead
from experimental.diffusion.src.dataset import ARCDataLoader, load_arc_data_paths
from experimental.diffusion.utils.noise_scheduler import DiscreteNoiseScheduler


def train_size_head(
    diffusion_model_path: str,
    config_path: str,
    output_path: str,
    device: str = "auto"
) -> GridSizePredictionHead:
    """
    Train grid size prediction head on frozen diffusion model.

    Args:
        diffusion_model_path: Path to trained diffusion model
        config_path: Path to config file (contains size head training params)
        output_path: Where to save trained size head
        device: Device to use ("auto", "cuda", "mps", "cpu")

    Returns:
        Trained GridSizePredictionHead
    """

    # Set up device
    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)

    print(f"Using device: {device}")

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Extract size head training parameters
    size_head_config = config.get('size_head', {})
    epochs = size_head_config.get('epochs', 100)
    learning_rate = size_head_config.get('learning_rate', 1e-3)
    batch_size = size_head_config.get('batch_size', 32)
    hidden_dim = size_head_config.get('hidden_dim', 256)
    weight_decay = size_head_config.get('weight_decay', 1e-4)

    print(f"Size head training config:")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Weight decay: {weight_decay}")

    # Load pre-trained diffusion model
    print(f"Loading diffusion model from {diffusion_model_path}")
    checkpoint = torch.load(diffusion_model_path, map_location=device)

    # Extract model config and dataset info from checkpoint
    model_config = checkpoint['config']
    dataset_info = checkpoint['dataset_info']

    # Create model with the correct parameters
    diffusion_model = ARCDiffusionModel(
        vocab_size=model_config['vocab_size'],
        d_model=model_config['d_model'],
        nhead=model_config['nhead'],
        num_layers=model_config['num_layers'],
        max_size=model_config['max_size'],
        max_tasks=dataset_info['num_tasks'],
        embedding_dropout=model_config.get('embedding_dropout', 0.1)
    )

    # Load the model weights
    diffusion_model.load_state_dict(checkpoint['model_state_dict'])
    diffusion_model.to(device)
    diffusion_model.eval()

    print(f"✓ Loaded diffusion model with {sum(p.numel() for p in diffusion_model.parameters()):,} parameters")

    # Create size prediction head
    size_head = GridSizePredictionHead(
        diffusion_model=diffusion_model,
        hidden_dim=hidden_dim,
        max_size=model_config['max_size']
    ).to(device)

    print(f"✓ Created size head with {sum(p.numel() for p in size_head.parameters() if p.requires_grad):,} trainable parameters")

    # Set up data loading
    data_paths = load_arc_data_paths(
        data_dir=config['data']['data_dir'],
        datasets=config['data']['datasets']
    )

    train_loader = ARCDataLoader.create_train_loader(
        train_data_paths=data_paths['train'],
        batch_size=batch_size,
        max_size=config['model']['max_size'],
        augment=False,  # No augmentation for size prediction
        num_workers=2,
        shuffle=True
    )

    # Validation loader (use subset for faster evaluation)
    val_loader = ARCDataLoader.create_eval_loader(
        eval_data_path=data_paths['train'][0],  # Use training data for validation
        batch_size=batch_size,
        max_size=config['model']['max_size'],
        num_workers=2
    )

    print(f"✓ Loaded training data: {len(train_loader.dataset)} examples")

    # Set up optimizer
    optimizer = torch.optim.AdamW(
        [p for p in size_head.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.1
    )

    # Training loop
    print(f"\\nStarting size head training for {epochs} epochs...")

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training phase
        size_head.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            # Move batch to device
            input_grids = batch['input_grid'].to(device)
            target_heights = batch['height'].to(device)
            target_widths = batch['width'].to(device)
            task_ids = batch['task_idx'].to(device)

            # Forward pass
            optimizer.zero_grad()
            loss = size_head.compute_size_loss(
                input_grids, task_ids, target_heights, target_widths
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(size_head.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()

        # Validation phase
        size_head.eval()
        val_losses = []
        correct_heights = 0
        correct_widths = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                input_grids = batch['input_grid'].to(device)
                target_heights = batch['height'].to(device)
                target_widths = batch['width'].to(device)
                task_ids = batch['task_idx'].to(device)

                # Compute loss
                loss = size_head.compute_size_loss(
                    input_grids, task_ids, target_heights, target_widths
                )
                val_losses.append(loss.item())

                # Compute accuracy
                pred_heights, pred_widths = size_head.predict_sizes(input_grids, task_ids)
                correct_heights += (pred_heights == target_heights).sum().item()
                correct_widths += (pred_widths == target_widths).sum().item()
                total_samples += len(target_heights)

        # Calculate metrics
        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)
        height_acc = correct_heights / total_samples
        width_acc = correct_widths / total_samples

        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Height Acc: {height_acc:.3f}")
        print(f"  Width Acc: {width_acc:.3f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  ✓ New best validation loss: {val_loss:.4f}")

            # Save the size head
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(size_head.state_dict(), output_path)
            print(f"  ✓ Saved size head to {output_path}")

    print(f"\\n✅ Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Size head saved to: {output_path}")

    # Load best model
    size_head.load_state_dict(torch.load(output_path, map_location=device))

    return size_head


def main():
    parser = argparse.ArgumentParser(description="Train grid size prediction head")
    parser.add_argument(
        "--diffusion-model",
        required=True,
        help="Path to trained diffusion model (.pt file)"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config file (same as used for diffusion training)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for trained size head (.pt file)"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use (default: auto)"
    )

    args = parser.parse_args()

    # Validate inputs
    diffusion_path = Path(args.diffusion_model)
    config_path = Path(args.config)

    if not diffusion_path.exists():
        print(f"❌ Diffusion model not found: {diffusion_path}")
        sys.exit(1)

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    print("Grid Size Prediction Head Training")
    print("=" * 50)
    print(f"Diffusion model: {args.diffusion_model}")
    print(f"Config: {args.config}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print("=" * 50)

    # Train size head
    size_head = train_size_head(
        diffusion_model_path=args.diffusion_model,
        config_path=args.config,
        output_path=args.output,
        device=args.device
    )

    print("\\n🎉 Size head training completed successfully!")


if __name__ == "__main__":
    main()