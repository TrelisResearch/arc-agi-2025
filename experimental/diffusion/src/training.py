"""
Training loop and sampling for the ARC diffusion model.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import random
from tqdm import tqdm
import wandb
import os
from pathlib import Path

from .model import ARCDiffusionModel
from .dataset import ARCDataset, ARCDataLoader, load_arc_data_paths, collate_fn
from ..utils.noise_scheduler import DiscreteNoiseScheduler
from ..utils.grid_utils import clamp_outside_mask, batch_create_masks, extract_valid_region, grid_to_display_string
from torch.utils.data import DataLoader


class ARCDiffusionTrainer:
    """Training class for the ARC diffusion model."""

    def __init__(
        self,
        model: ARCDiffusionModel,
        noise_scheduler: DiscreteNoiseScheduler,
        device: torch.device,
        learning_rate: float = 3e-4,
        cfg_drop_prob: float = 0.15,
    ):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.device = device
        self.cfg_drop_prob = cfg_drop_prob

        # Move model to device
        self.model.to(device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=10000,
            eta_min=learning_rate * 0.1
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one training step."""
        self.model.train()

        # Move batch to device
        input_grids = batch['input_grid'].to(self.device)  # [batch_size, max_size, max_size]
        output_grids = batch['output_grid'].to(self.device)  # [batch_size, max_size, max_size]
        task_indices = batch['task_idx'].to(self.device)  # [batch_size]

        batch_size = input_grids.shape[0]

        # Sample random timesteps (0-indexed for array access)
        timesteps = torch.randint(0, self.noise_scheduler.num_timesteps, (batch_size,), device=self.device)

        # Add noise to clean output grids
        noisy_grids = self.noise_scheduler.add_noise(output_grids, timesteps)

        # Apply classifier-free guidance training: randomly drop conditioning
        if self.cfg_drop_prob > 0:
            # Create masks for dropping conditioning
            drop_task_mask = torch.rand(batch_size, device=self.device) < self.cfg_drop_prob
            drop_input_mask = torch.rand(batch_size, device=self.device) < self.cfg_drop_prob

            # Replace with "null" conditioning
            task_indices_cond = torch.where(drop_task_mask, 0, task_indices)  # Use task 0 as null
            input_grids_cond = torch.where(
                drop_input_mask.unsqueeze(-1).unsqueeze(-1),
                torch.zeros_like(input_grids),
                input_grids
            )
        else:
            task_indices_cond = task_indices
            input_grids_cond = input_grids

        # Forward pass
        losses = self.model.compute_loss(
            x0=output_grids,
            input_grid=input_grids_cond,
            task_ids=task_indices_cond,
            xt=noisy_grids,
            timesteps=timesteps
        )

        # Backward pass
        total_loss = losses['total_loss']
        self.optimizer.zero_grad()
        total_loss.backward()

        # Compute gradient norm before clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float('inf'))

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.scheduler.step()

        # Return losses and grad norm as Python floats
        losses['grad_norm'] = grad_norm.item()
        return {key: value.item() if hasattr(value, 'item') else value for key, value in losses.items()}

    def validate(self, val_loader: torch.utils.data.DataLoader, num_batches: int = 10) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_losses = {
            'total_loss': 0.0,
            'grid_loss': 0.0
        }
        num_samples = 0

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= num_batches:
                    break

                # Move batch to device
                input_grids = batch['input_grid'].to(self.device)
                output_grids = batch['output_grid'].to(self.device)
                task_indices = batch['task_idx'].to(self.device)

                batch_size = input_grids.shape[0]

                # Sample random timesteps (0-indexed for array access)
                timesteps = torch.randint(0, self.noise_scheduler.num_timesteps, (batch_size,), device=self.device)

                # Add noise
                noisy_grids = self.noise_scheduler.add_noise(output_grids, timesteps)

                # Forward pass (no CFG during validation)
                losses = self.model.compute_loss(
                    x0=output_grids,
                    input_grid=input_grids,
                    task_ids=task_indices,
                    xt=noisy_grids,
                    timesteps=timesteps
                )

                # Accumulate losses
                for key, value in losses.items():
                    total_losses[key] += value.item() * batch_size

                num_samples += batch_size

        # Average losses
        avg_losses = {key: total / num_samples for key, total in total_losses.items()}
        return avg_losses


class ARCDiffusionSampler:
    """Sampling class for the ARC diffusion model."""

    def __init__(
        self,
        model: ARCDiffusionModel,
        noise_scheduler: DiscreteNoiseScheduler,
        device: torch.device,
        debug: bool = False
    ):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.device = device
        self.debug = debug

    @torch.no_grad()
    def sample(
        self,
        input_grids: torch.Tensor,
        task_indices: torch.Tensor,
        guidance_scale: float = 2.0,
        num_inference_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Sample outputs for given inputs using DDPM sampling.

        Args:
            input_grids: [batch_size, max_size, max_size]
            task_indices: [batch_size]
            guidance_scale: CFG guidance scale
            num_inference_steps: Number of denoising steps (default: use scheduler's num_timesteps)

        Returns:
            predictions: [batch_size, max_size, max_size] - predicted output grids
        """
        self.model.eval()

        batch_size = input_grids.shape[0]
        max_size = input_grids.shape[1]

        if num_inference_steps is None:
            num_inference_steps = self.noise_scheduler.num_timesteps

        # Initialize with random noise
        x_t = torch.randint(
            0, self.noise_scheduler.vocab_size,
            (batch_size, max_size, max_size),
            device=self.device
        )

        # Denoising loop (0-indexed timesteps)
        timesteps = torch.linspace(num_inference_steps - 1, 0, num_inference_steps, dtype=torch.long, device=self.device)

        for i, t in enumerate(tqdm(timesteps, desc="Sampling", disable=(not self.debug))):
            t_batch = t.repeat(batch_size)

            if guidance_scale > 1.0:
                # Classifier-free guidance
                # Conditional prediction
                logits_cond = self.model(x_t, input_grids, task_indices, t_batch)

                # Unconditional prediction (using null conditioning)
                null_task_indices = torch.zeros_like(task_indices)
                null_input_grids = torch.zeros_like(input_grids)
                logits_uncond = self.model(x_t, null_input_grids, null_task_indices, t_batch)

                # Apply CFG
                logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
            else:
                logits = self.model(x_t, input_grids, task_indices, t_batch)

            # Sample next step (greedy for now)
            x_t = torch.argmax(logits, dim=-1)

            # Debug printing
            if self.debug:
                print(f"\n=== Timestep {t.item()} (step {i+1}/{len(timesteps)}) ===")

                # Show first 10x10 of the grid with PAD tokens as *
                display_size = min(10, max_size)
                valid_grid = x_t[0, :display_size, :display_size].cpu().numpy()
                print(f"Grid content (showing {display_size}x{display_size}):")
                print(grid_to_display_string(valid_grid, pad_symbol='*'))
                print("---")

        return x_t


def train_arc_diffusion(config: Dict[str, Any]) -> ARCDiffusionModel:
    """
    Main training function for ARC diffusion model.

    Args:
        config: Training configuration dict with all settings

    Returns:
        Trained model
    """
    # Set up device (prioritize CUDA > MPS > CPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        # MPS memory optimization
        torch.mps.set_per_process_memory_fraction(0.7)  # Use max 70% of memory
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize W&B if enabled
    if config.get('use_wandb', False):
        wandb.init(
            project="arc-prize-2025-diffusion",
            config=config,
            save_code=True
        )

    # Load data paths
    data_paths = load_arc_data_paths(
        data_dir=config.get('data_dir', 'data/arc-prize-2024'),
        datasets=config.get('datasets', None)
    )

    # Create full dataset first
    full_dataset = ARCDataset(
        data_paths=data_paths['train'],
        max_size=config['max_size'],
        augment=config['augment'],
        n_augment=config.get('n_augment', 3),
        include_training_test_examples=config.get('include_training_test_examples', True)
    )

    # Split into train and validation
    total_examples = len(full_dataset)
    max_val_examples = config.get('max_val_examples', 32)
    val_size = min(int(0.1 * total_examples), max_val_examples)
    train_size = total_examples - val_size

    print(f"Splitting {total_examples} examples: {train_size} train, {val_size} validation")

    # Create train/val split
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )

    # Create data loaders
    # Disable pin_memory for MPS to avoid warnings and reduce memory usage
    use_pin_memory = device.type == 'cuda'

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=use_pin_memory,
        drop_last=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'] // 2,
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin_memory,
        collate_fn=collate_fn
    )

    print(f"Created train loader with {len(train_loader)} batches")
    print(f"Created val loader with {len(val_loader)} batches")

    # Get dataset info from the original full dataset
    dataset_info = full_dataset.get_task_info()
    print(f"Dataset info: {dataset_info}")

    # Create model
    model = ARCDiffusionModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        max_size=config['max_size'],
        max_tasks=dataset_info['num_tasks']
    )

    # Create noise scheduler
    noise_scheduler = DiscreteNoiseScheduler(
        num_timesteps=config['num_timesteps'],
        vocab_size=config['vocab_size'],
        schedule_type=config['schedule_type']
    )
    noise_scheduler.to(device)

    # Create trainer
    trainer = ARCDiffusionTrainer(
        model=model,
        noise_scheduler=noise_scheduler,
        device=device,
        learning_rate=config['learning_rate'],
        cfg_drop_prob=config['cfg_drop_prob']
    )

    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Training loop
    step = 0
    best_val_loss = float('inf')

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

        # Training
        epoch_losses = {
            'total_loss': 0.0,
            'grid_loss': 0.0,
            'grad_norm': 0.0
        }
        num_train_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        for batch in progress_bar:
            losses = trainer.train_step(batch)

            # Accumulate losses
            for key in epoch_losses:
                if key in losses:
                    epoch_losses[key] += losses[key]
            num_train_batches += 1
            step += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{losses['total_loss']:.4f}",
                'grid': f"{losses['grid_loss']:.4f}",
                'grad': f"{losses['grad_norm']:.2f}"
            })

            # Log to wandb
            if config.get('use_wandb', False) and step % config.get('log_every', 50) == 0:
                wandb.log({
                    f"train/{key}": value for key, value in losses.items()
                }, step=step)

            # Limit training batches for CPU testing
            if num_train_batches >= config.get('max_train_batches', float('inf')):
                break

        # Average training losses
        avg_train_losses = {key: total / num_train_batches for key, total in epoch_losses.items()}

        # Validation
        if (epoch + 1) % config.get('val_every', 1) == 0:
            print("Running validation...")
            val_losses = trainer.validate(val_loader, num_batches=config.get('max_val_batches', 10))

            print(f"Validation losses: {val_losses}")

            # Log validation losses
            if config.get('use_wandb', False):
                wandb.log({
                    f"val/{key}": value for key, value in val_losses.items()
                }, step=step)

            # Save best model
            if val_losses['total_loss'] < best_val_loss:
                best_val_loss = val_losses['total_loss']
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'scheduler_state_dict': trainer.scheduler.state_dict(),
                    'epoch': epoch,
                    'step': step,
                    'val_loss': val_losses['total_loss'],
                    'config': config,
                    'dataset_info': dataset_info
                }, output_dir / 'best_model.pt')
                print(f"Saved best model with val loss: {best_val_loss:.4f}")

        # Print epoch summary
        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_losses['total_loss']:.4f}, Grad Norm: {avg_train_losses['grad_norm']:.3f}")

        # Early stopping for CPU testing
        if config.get('early_stop_epochs') and epoch >= config['early_stop_epochs']:
            print(f"Early stopping after {epoch + 1} epochs")
            break

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': trainer.scheduler.state_dict(),
        'epoch': epoch,
        'step': step,
        'config': config,
        'dataset_info': dataset_info
    }, output_dir / 'final_model.pt')

    if config.get('use_wandb', False):
        wandb.finish()

    print("Training completed!")
    return model