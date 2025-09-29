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
from ..utils.visualization import create_training_visualization, create_denoising_progression_visualization
from torch.utils.data import DataLoader


class ARCDiffusionTrainer:
    """Training class for the ARC diffusion model."""

    def __init__(
        self,
        model: ARCDiffusionModel,
        noise_scheduler: DiscreteNoiseScheduler,
        device: torch.device,
        dataset,  # Need dataset reference to get task info
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        use_mixed_precision: bool = True,
        pixel_noise_prob: float = 0.15,
        pixel_noise_rate: float = 0.02,
        total_steps: int = 10000,
        auxiliary_size_loss_weight: float = 0.1
    ):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.device = device
        self.dataset = dataset
        self.use_mixed_precision = use_mixed_precision
        self.pixel_noise_prob = pixel_noise_prob
        self.pixel_noise_rate = pixel_noise_rate
        self.auxiliary_size_loss_weight = auxiliary_size_loss_weight

        # Set up mixed precision
        if use_mixed_precision and device.type in ['cuda', 'mps']:
            self.use_mixed_precision = True
            # Use bfloat16 for modern hardware
            if device.type == 'cuda' and torch.cuda.is_bf16_supported():
                self.amp_dtype = torch.bfloat16
                print("Using bfloat16 mixed precision")
                self.scaler = None  # bfloat16 doesn't need scaling
            else:
                self.amp_dtype = torch.float16
                print("Using float16 mixed precision")
                # Only use scaler for CUDA float16
                if device.type == 'cuda':
                    self.scaler = torch.amp.GradScaler(device.type)
                else:
                    self.scaler = None
        else:
            self.use_mixed_precision = False
            self.amp_dtype = torch.float32
            self.scaler = None
            print("Using float32 precision")

        # Move model to device and ensure parameters stay in float32
        self.model.to(device)
        self.model.float()  # Always keep parameters in fp32

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=learning_rate * 0.1
        )

    def apply_pixel_noise(self, grids: torch.Tensor) -> torch.Tensor:
        """
        Apply pixel noise to input grids: randomly swap black pixels (0) with colors (1-9).

        Args:
            grids: Input grids [batch_size, height, width]

        Returns:
            Grids with noise applied
        """
        if self.pixel_noise_prob <= 0 or self.pixel_noise_rate <= 0:
            return grids

        batch_size = grids.shape[0]
        grids_noisy = grids.clone()

        for i in range(batch_size):
            # Apply noise to this example with probability pixel_noise_prob
            if torch.rand(1).item() < self.pixel_noise_prob:
                grid = grids_noisy[i]

                # Find all black pixels (value 0)
                black_mask = (grid == 0)
                black_indices = torch.where(black_mask)

                if len(black_indices[0]) > 0:
                    # Determine how many black pixels to flip
                    num_black = len(black_indices[0])
                    num_to_flip = max(1, int(num_black * self.pixel_noise_rate))

                    # Randomly select which black pixels to flip
                    perm = torch.randperm(num_black)[:num_to_flip]
                    flip_row_idx = black_indices[0][perm]
                    flip_col_idx = black_indices[1][perm]

                    # Replace with random colors 1-9 (avoid 0 and 10/PAD)
                    random_colors = torch.randint(1, 10, (num_to_flip,), device=grids.device)
                    grids_noisy[i, flip_row_idx, flip_col_idx] = random_colors

        return grids_noisy

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one training step."""
        self.model.train()

        # Move batch to device
        input_grids = batch['input_grid'].to(self.device)  # [batch_size, max_size, max_size]
        output_grids = batch['output_grid'].to(self.device)  # [batch_size, max_size, max_size]
        task_indices = batch['task_idx'].to(self.device)  # [batch_size]
        heights = batch['height'].to(self.device)  # [batch_size] - grid heights
        widths = batch['width'].to(self.device)   # [batch_size] - grid widths

        batch_size = input_grids.shape[0]

        # Apply pixel noise to input grids only (not outputs)
        input_grids = self.apply_pixel_noise(input_grids)

        # Sample random timesteps from {1, ..., T} (1-indexed as specified)
        # Convert to 0-indexed for array access by subtracting 1
        timesteps = torch.randint(1, self.noise_scheduler.num_timesteps + 1, (batch_size,), device=self.device) - 1

        # Add noise to clean output grids using uniform distribution over {0..9}
        noisy_grids = self.noise_scheduler.add_noise(output_grids, timesteps)

        # Forward pass with mixed precision
        if self.use_mixed_precision and self.device.type in ['cuda', 'mps']:
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                losses = self.model.compute_loss(
                    x0=output_grids,
                    input_grid=input_grids,
                    task_ids=task_indices,
                    xt=noisy_grids,
                    timesteps=timesteps,
                    heights=heights,
                    widths=widths,
                    auxiliary_size_loss_weight=self.auxiliary_size_loss_weight
                )
        else:
            losses = self.model.compute_loss(
                x0=output_grids,
                input_grid=input_grids,
                task_ids=task_indices,
                xt=noisy_grids,
                timesteps=timesteps,
                heights=heights,
                widths=widths,
                auxiliary_size_loss_weight=self.auxiliary_size_loss_weight
            )

        # Backward pass with mixed precision
        total_loss = losses['total_loss']
        self.optimizer.zero_grad()

        if self.scaler is not None:
            # CUDA with float16 and gradient scaling
            self.scaler.scale(total_loss).backward()

            # Compute gradient norm before clipping
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float('inf'))

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # MPS, CPU, or bfloat16 without gradient scaling
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
        total_losses = {}  # Will be initialized from first batch
        num_samples = 0

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= num_batches:
                    break

                # Move batch to device
                input_grids = batch['input_grid'].to(self.device)
                output_grids = batch['output_grid'].to(self.device)
                task_indices = batch['task_idx'].to(self.device)
                heights = batch['height'].to(self.device)
                widths = batch['width'].to(self.device)

                batch_size = input_grids.shape[0]

                # Sample random timesteps (0-indexed for array access)
                timesteps = torch.randint(0, self.noise_scheduler.num_timesteps, (batch_size,), device=self.device)

                # Add noise using uniform distribution over {0..9}
                noisy_grids = self.noise_scheduler.add_noise(output_grids, timesteps)

                # Forward pass (no CFG during validation) with mixed precision
                if self.use_mixed_precision and self.device.type in ['cuda', 'mps']:
                    with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                        losses = self.model.compute_loss(
                            x0=output_grids,
                            input_grid=input_grids,
                            task_ids=task_indices,
                            xt=noisy_grids,
                            timesteps=timesteps,
                            heights=heights,
                            widths=widths
                        )
                else:
                    losses = self.model.compute_loss(
                        x0=output_grids,
                        input_grid=input_grids,
                        task_ids=task_indices,
                        xt=noisy_grids,
                        timesteps=timesteps,
                        heights=heights,
                        widths=widths
                    )

                # Initialize total_losses on first batch
                if not total_losses:
                    total_losses = {key: 0.0 for key in losses.keys()}

                # Accumulate losses - handle missing keys gracefully
                for key, value in losses.items():
                    if key not in total_losses:
                        total_losses[key] = 0.0

                    if isinstance(value, torch.Tensor):
                        total_losses[key] += value.item() * batch_size
                    else:
                        total_losses[key] += value * batch_size

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
        dataset=None,  # Need dataset for task info
        debug: bool = False
    ):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.device = device
        self.dataset = dataset
        self.debug = debug

    @torch.no_grad()
    def sample(
        self,
        input_grids: torch.Tensor,
        task_indices: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Sample outputs for given inputs using DDPM sampling.

        Args:
            input_grids: [batch_size, max_size, max_size]
            task_indices: [batch_size]
            num_inference_steps: Number of denoising steps (default: use scheduler's num_timesteps)

        Returns:
            predictions: [batch_size, max_size, max_size] - predicted output grids
        """
        self.model.eval()

        batch_size = input_grids.shape[0]
        max_size = input_grids.shape[1]

        if num_inference_steps is None:
            num_inference_steps = self.noise_scheduler.num_timesteps

        # Predict output grid sizes if available
        predicted_heights = None
        predicted_widths = None
        # First check for integrated size head in model
        if hasattr(self.model, 'include_size_head') and self.model.include_size_head:
            predicted_heights, predicted_widths = self.model.predict_sizes(input_grids, task_indices)
            print(f"Predicted sizes (integrated): heights={predicted_heights.cpu().tolist()}, widths={predicted_widths.cpu().tolist()}")
        # Fallback to external size predictor if provided
            print(f"Predicted sizes (external): heights={predicted_heights.cpu().tolist()}, widths={predicted_widths.cpu().tolist()}")

        # Initialize with uniform random noise over {0..9}
        x_t = torch.randint(
            0, 10,  # Uniform over {0..9}
            (batch_size, max_size, max_size),
            device=self.device
        )

        # Denoising loop (0-indexed timesteps)
        timesteps = torch.linspace(num_inference_steps - 1, 0, num_inference_steps, dtype=torch.long, device=self.device)

        for i, t in enumerate(tqdm(timesteps, desc="Sampling", disable=(not self.debug))):
            t_batch = t.repeat(batch_size)

            # Forward pass
            logits = self.model(x_t, input_grids, task_indices, t_batch)

            # Apply temperature scaling
            logits_scaled = logits / temperature

            # Get predicted probabilities
            probs = F.softmax(logits_scaled, dim=-1)

            # Sample from categorical distribution instead of greedy argmax
            # This adds stochasticity which helps avoid mode collapse
            if t > 0:  # Sample for all but the last step
                x_t = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(x_t.shape)
            else:  # Use argmax only for final step
                x_t = torch.argmax(logits_scaled, dim=-1)

            # Apply masking if size predictor provided - set tokens outside predicted bounds to black (0)
            if predicted_heights is not None and predicted_widths is not None:
                for b in range(batch_size):
                    h, w = predicted_heights[b].item(), predicted_widths[b].item()
                    # Set positions outside [0:h, 0:w] to black (token 0)
                    if h < max_size:
                        x_t[b, h:, :] = 0  # Set rows beyond height to black
                    if w < max_size:
                        x_t[b, :, w:] = 0  # Set columns beyond width to black

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
        torch.mps.set_per_process_memory_fraction(0.5)  # Use max 70% of memory
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

    # Get auxiliary loss config
    aux_config = config.get('auxiliary_loss', {})
    include_size_head = aux_config.get('include_size_head', True)
    size_head_hidden_dim = aux_config.get('size_head_hidden_dim', None)

    # Create model
    model = ARCDiffusionModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        max_size=config['max_size'],
        max_tasks=dataset_info['num_tasks'],
        embedding_dropout=config.get('embedding_dropout', 0.1),
        include_size_head=include_size_head,
        size_head_hidden_dim=size_head_hidden_dim
    )

    # Create noise scheduler
    noise_scheduler = DiscreteNoiseScheduler(
        num_timesteps=config['num_timesteps'],
        vocab_size=config['vocab_size'],
        schedule_type=config['schedule_type']
    )
    noise_scheduler.to(device)

    # Get optimizer steps from config
    if 'optimizer_steps' not in config:
        raise KeyError(
            "Config missing 'optimizer_steps'. Please update your config file to use 'optimizer_steps' instead of 'num_epochs'. "
            "Example: 'optimizer_steps': 3000"
        )
    optimizer_steps = config['optimizer_steps']
    steps_per_epoch = len(train_loader)
    estimated_epochs = optimizer_steps / steps_per_epoch
    print(f"Training setup: {optimizer_steps} optimizer steps (~{estimated_epochs:.1f} epochs at {steps_per_epoch} steps/epoch)")

    # Create trainer
    trainer = ARCDiffusionTrainer(
        model=model,
        noise_scheduler=noise_scheduler,
        device=device,
        dataset=full_dataset,  # Pass dataset for task info
        learning_rate=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.01),
        use_mixed_precision=config.get('use_mixed_precision', True),
        pixel_noise_prob=config.get('pixel_noise_prob', 0.15),
        pixel_noise_rate=config.get('pixel_noise_rate', 0.02),
        total_steps=optimizer_steps,
        auxiliary_size_loss_weight=aux_config.get('auxiliary_size_loss_weight', 0.1)
    )

    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create training data visualization before starting training
    create_training_visualization(
        dataset=full_dataset,
        noise_scheduler=noise_scheduler,
        device=device,
        output_dir=output_dir,
        config=config
    )

    # Training loop
    step = 0
    best_val_loss = float('inf')

    # Training loop using steps instead of epochs
    steps_per_epoch = len(train_loader)
    current_epoch = 0

    # Create infinite data loader
    def infinite_dataloader(dataloader):
        while True:
            for batch in dataloader:
                yield batch

    data_iter = infinite_dataloader(train_loader)

    # Progress bar for optimizer steps
    progress_bar = tqdm(range(optimizer_steps), desc="Training")

    # Track epoch losses for validation
    epoch_losses = {
        'total_loss': 0.0,
        'grid_loss': 0.0,
        'grad_norm': 0.0
    }
    num_batches_this_epoch = 0

    for step_idx in progress_bar:
        batch = next(data_iter)
        losses = trainer.train_step(batch)

        # Accumulate losses for epoch average
        for key in epoch_losses:
            if key in losses:
                epoch_losses[key] += losses[key]
        num_batches_this_epoch += 1
        step = step_idx + 1

        # Update progress bar
        current_epoch_approx = step / steps_per_epoch
        progress_bar.set_postfix({
            'loss': f"{losses['total_loss']:.4f}",
            'grid': f"{losses['grid_loss']:.4f}",
            'acc': f"{losses.get('accuracy', 0.0):.3f}",
            'adj_acc': f"{losses.get('chance_adjusted_accuracy', -0.10):.3f}",
            'grad': f"{losses['grad_norm']:.2f}",
            'epoch': f"{current_epoch_approx:.1f}"
        })

        # Log to wandb
        if config.get('use_wandb', False) and step % config.get('log_every', 50) == 0:
            log_dict = {f"train/{key}": value for key, value in losses.items()}
            log_dict["train/learning_rate"] = trainer.scheduler.get_last_lr()[0]
            log_dict["train/step"] = step
            log_dict["train/epoch"] = current_epoch_approx
            wandb.log(log_dict, step=step)

        # Check if we've completed an epoch for printing
        if step % steps_per_epoch == 0:
            current_epoch = step // steps_per_epoch
            print(f"\nCompleted epoch {current_epoch} (step {step})")

        # Validation based on step intervals (not epoch boundaries)
        val_every_steps = config.get('val_every_steps', steps_per_epoch)  # Default to every epoch
        if step % val_every_steps == 0:
            # Calculate average training losses for this validation period
            avg_train_losses = {key: total / num_batches_this_epoch for key, total in epoch_losses.items()}

            print("Running validation...")
            val_losses = trainer.validate(val_loader, num_batches=config.get('max_val_batches', 10))

            # Print key validation metrics in a readable format
            print(f"Validation losses: {val_losses}")

            # Print timestep bucket summary if available
            if any(key.endswith('_count') for key in val_losses.keys()):
                print("Timestep bucket breakdown:")
                for bucket in ['low_noise', 'mid_noise', 'high_noise']:
                    if f'{bucket}_count' in val_losses:
                        acc = val_losses.get(f'{bucket}_accuracy', 0.0)
                        adj_acc = val_losses.get(f'{bucket}_chance_adj_acc', -0.10)
                        count = val_losses.get(f'{bucket}_count', 0)
                        print(f"  {bucket:10s}: acc={acc:.3f}, adj_acc={adj_acc:+.3f}, count={count:4.0f}")

            overall_acc = val_losses.get('accuracy', 0.0)
            overall_adj = val_losses.get('chance_adjusted_accuracy', -0.10)
            print(f"Overall: acc={overall_acc:.3f}, chance_adj_acc={overall_adj:+.3f}")

            # Log validation losses
            if config.get('use_wandb', False):
                val_log_dict = {f"val/{key}": value for key, value in val_losses.items()}
                val_log_dict["val/learning_rate"] = trainer.scheduler.get_last_lr()[0]
                val_log_dict["val/step"] = step
                val_log_dict["val/epoch"] = current_epoch_approx
                wandb.log(val_log_dict, step=step)

            # Save best model
            if val_losses['total_loss'] < best_val_loss:
                best_val_loss = val_losses['total_loss']
                # Save model in bfloat16 without modifying the original
                model_state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
                torch.save({
                    'model_state_dict': model_state_dict_bf16,
                    'optimizer_state_dict': trainer.optimizer.state_dict(),
                    'scheduler_state_dict': trainer.scheduler.state_dict(),
                    'epoch': current_epoch,
                    'step': step,
                    'val_loss': val_losses['total_loss'],
                    'config': config,
                    'dataset_info': dataset_info
                }, output_dir / 'best_model.pt')
                print(f"Saved best model with val loss: {best_val_loss:.4f}")

            # Create denoising progression visualization
            vis_every_steps = config.get('vis_every_steps', val_every_steps)  # Default to same as validation
            if step % vis_every_steps == 0:
                try:
                    create_denoising_progression_visualization(
                        model=model,
                        noise_scheduler=noise_scheduler,
                        val_dataset=val_dataset,
                        device=device,
                        output_dir=output_dir,
                        step=step,
                        config=config
                    )
                except Exception as e:
                    print(f"⚠️ Failed to create denoising visualization: {e}")

            # Print validation summary
            current_lr = trainer.scheduler.get_last_lr()[0]
            print(f"Step {step} - Train Loss: {avg_train_losses['total_loss']:.4f}, Grad Norm: {avg_train_losses['grad_norm']:.3f}, LR: {current_lr:.6f}")

            # Reset validation period tracking
            epoch_losses = {
                'total_loss': 0.0,
                'grid_loss': 0.0,
                'grad_norm': 0.0
            }
            num_batches_this_epoch = 0

        # Early stopping based on steps if configured
        if config.get('early_stop_steps') and step >= config['early_stop_steps']:
            print(f"Early stopping after {step} steps")
            break

    # Save final model
    # Save final model in bfloat16 without modifying the original
    final_epoch = step // steps_per_epoch
    model_state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
    torch.save({
        'model_state_dict': model_state_dict_bf16,
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'scheduler_state_dict': trainer.scheduler.state_dict(),
        'epoch': final_epoch,
        'step': step,
        'config': config,
        'dataset_info': dataset_info
    }, output_dir / 'final_model.pt')

    if config.get('use_wandb', False):
        wandb.finish()

    print("Training completed!")
    return model