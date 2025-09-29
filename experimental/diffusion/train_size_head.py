#!/usr/bin/env python3
"""
Grid Size Prediction Head Training

Trains a size prediction head on top of a pre-trained diffusion model.
The diffusion model weights are frozen during this training.

Usage:
    python experimental/diffusion/train_size_head.py \
        --config experimental/diffusion/configs/smol_config.json \
        [--diffusion-model path/to/model.pt]  # Optional, defaults to best model in output dir
"""
import json
import argparse
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, logging disabled")

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experimental.diffusion.src.model import ARCDiffusionModel, GridSizePredictionHead
from experimental.diffusion.src.dataset import ARCDataLoader, load_arc_data_paths
from experimental.diffusion.utils.noise_scheduler import DiscreteNoiseScheduler


def train_size_head(
    config_path: str,
    diffusion_model_path: str = None,
    device: str = "auto"
) -> GridSizePredictionHead:
    """
    Train grid size prediction head on frozen diffusion model.

    Args:
        config_path: Path to config file (contains size head training params)
        diffusion_model_path: Path to trained diffusion model (optional, defaults to best model in output dir)
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

    # Extract output directory and determine paths
    output_config = config.get('output', {})
    output_dir = Path(output_config.get('output_dir', 'experimental/diffusion/outputs/default'))
    save_best = output_config.get('save_best', True)
    save_final = output_config.get('save_final', True)

    # Determine diffusion model path
    if diffusion_model_path is None:
        diffusion_model_path = output_dir / 'best_model.pt'
        print(f"Using default diffusion model: {diffusion_model_path}")
    else:
        diffusion_model_path = Path(diffusion_model_path)

    # Set up size head output paths
    best_size_head_path = output_dir / 'best_size_head.pt'
    final_size_head_path = output_dir / 'final_size_head.pt'

    # Extract size head training parameters
    size_head_config = config.get('size_head', {})

    # Extract step-based training parameters
    if 'optimizer_steps' not in size_head_config:
        raise KeyError(
            "Size head config missing 'optimizer_steps'. Please update your config file to use 'optimizer_steps' instead of 'epochs'. "
            "Example: 'size_head': {'optimizer_steps': 1000, ...}"
        )
    optimizer_steps = size_head_config['optimizer_steps']
    learning_rate = size_head_config.get('learning_rate', 1e-3)
    batch_size = size_head_config.get('batch_size', 32)
    hidden_dim = size_head_config.get('hidden_dim', 256)
    weight_decay = size_head_config.get('weight_decay', 1e-4)

    print(f"Size head training config:")
    print(f"  Optimizer steps: {optimizer_steps}")
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

    # Calculate estimated epochs from optimizer steps
    steps_per_epoch = len(train_loader)
    estimated_epochs = optimizer_steps / steps_per_epoch
    print(f"Training for {optimizer_steps} optimizer steps (~{estimated_epochs:.1f} epochs at {steps_per_epoch} steps/epoch)")

    # Set up optimizer
    optimizer = torch.optim.AdamW(
        [p for p in size_head.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=optimizer_steps, eta_min=learning_rate * 0.1
    )

    # Initialize wandb
    if WANDB_AVAILABLE:
        wandb.init(
            project="arc-prize-2025-size-head",
            config={
                "optimizer_steps": optimizer_steps,
                "estimated_epochs": estimated_epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "hidden_dim": hidden_dim,
                "weight_decay": weight_decay,
                "max_size": model_config['max_size'],
                "diffusion_model_path": diffusion_model_path,
                "dataset_size": len(train_loader.dataset),
                "device": str(device),
                "model_version": model_config.get('model_version', 'unknown')
            }
        )
        print(f"✓ Initialized wandb logging to project: arc-prize-2025-size-head")
    else:
        print("⚠️ Wandb not available, training metrics won't be logged")

    # Training loop
    print(f"\\nStarting size head training for {optimizer_steps} optimizer steps...")

    best_val_loss = float('inf')

    # Create infinite data loader
    def infinite_dataloader(dataloader):
        while True:
            for batch in dataloader:
                yield batch

    data_iter = infinite_dataloader(train_loader)

    # Progress bar for optimizer steps
    pbar = tqdm(range(optimizer_steps), desc="Training")

    # Track epoch-level metrics
    epoch_train_losses = []
    current_epoch = 0

    for step in pbar:
        batch = next(data_iter)
        size_head.train()

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
        scheduler.step()

        epoch_train_losses.append(loss.item())
        current_epoch_approx = step / steps_per_epoch

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'epoch': f"{current_epoch_approx:.1f}"
        })

        # Run validation at intervals (every epoch or specified steps)
        val_every_steps = size_head_config.get('val_every_steps', steps_per_epoch)
        if (step + 1) % val_every_steps == 0:
            current_epoch = (step + 1) // steps_per_epoch

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
            recent_train_losses = epoch_train_losses[-val_every_steps:] if len(epoch_train_losses) >= val_every_steps else epoch_train_losses
            train_loss = sum(recent_train_losses) / len(recent_train_losses)
            val_loss = sum(val_losses) / len(val_losses)
            height_acc = correct_heights / total_samples
            width_acc = correct_widths / total_samples

            print(f"\nStep {step + 1} (Epoch {current_epoch}):")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Height Acc: {height_acc:.3f}")
            print(f"  Width Acc: {width_acc:.3f}")
            print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

            # Log to wandb
            if WANDB_AVAILABLE:
                wandb.log({
                    "step": step + 1,
                    "epoch": current_epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "height_accuracy": height_acc,
                    "width_accuracy": width_acc,
                    "overall_accuracy": (height_acc + width_acc) / 2,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "best_val_loss": best_val_loss
                }, step=step + 1)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"  ✓ New best validation loss: {val_loss:.4f}")

                # Save the best size head
                if save_best:
                    best_size_head_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(size_head.state_dict(), best_size_head_path)
                    print(f"  ✓ Saved best size head to {best_size_head_path}")

    # Save final model
    if save_final:
        final_size_head_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(size_head.state_dict(), final_size_head_path)
        print(f"✓ Saved final size head to {final_size_head_path}")

    print(f"\\n✅ Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    if save_best:
        print(f"Best size head saved to: {best_size_head_path}")
    if save_final:
        print(f"Final size head saved to: {final_size_head_path}")

    # Finish wandb logging
    if WANDB_AVAILABLE:
        wandb.log({"final_best_val_loss": best_val_loss})
        wandb.finish()
        print("✓ Wandb logging finished")

    # Load best model if available, otherwise final model
    if save_best and best_size_head_path.exists():
        size_head.load_state_dict(torch.load(best_size_head_path, map_location=device))
        print(f"✓ Loaded best model for return")
    elif save_final and final_size_head_path.exists():
        size_head.load_state_dict(torch.load(final_size_head_path, map_location=device))
        print(f"✓ Loaded final model for return")

    return size_head


def main():
    parser = argparse.ArgumentParser(description="Train grid size prediction head")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config file (contains training params and output directory)"
    )
    parser.add_argument(
        "--diffusion-model",
        help="Path to trained diffusion model (.pt file). If not provided, uses best_model.pt from output directory"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use (default: auto)"
    )

    args = parser.parse_args()

    # Validate inputs
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    # Validate diffusion model if provided
    if args.diffusion_model:
        diffusion_path = Path(args.diffusion_model)
        if not diffusion_path.exists():
            print(f"❌ Diffusion model not found: {diffusion_path}")
            sys.exit(1)

    print("Grid Size Prediction Head Training")
    print("=" * 50)
    print(f"Config: {args.config}")
    if args.diffusion_model:
        print(f"Diffusion model: {args.diffusion_model}")
    else:
        print(f"Diffusion model: [will use best model from output directory]")
    print(f"Device: {args.device}")
    print("=" * 50)

    # Train size head
    size_head = train_size_head(
        config_path=args.config,
        diffusion_model_path=args.diffusion_model,
        device=args.device
    )

    print("\\n🎉 Size head training completed successfully!")


if __name__ == "__main__":
    main()