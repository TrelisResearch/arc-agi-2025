"""
Training loop and sampling for the ARC iterative refinement model.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import wandb
from pathlib import Path

from .model import ARCIterativeModel
from .dataset import ARCDataset, load_arc_data_paths, collate_fn
from ..utils.grid_utils import grid_to_display_string
from torch.utils.data import DataLoader


class ARCIterativeTrainer:
    """Training class for the ARC iterative refinement model."""

    def __init__(
        self,
        model: ARCIterativeModel,
        device: torch.device,
        dataset,  # Need dataset reference to get task info
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        use_mixed_precision: bool = True,
        pixel_noise_prob: float = 0.0,
        pixel_noise_rate: float = 0.0,
        total_steps: int = 10000,
        auxiliary_size_loss_weight: float = 0.1,
        gradient_accumulation_steps: int = 1,
        lr_warmup_steps: int = None,
        K: int = 8,  # Number of refinement steps
    ):
        self.model = model
        self.device = device
        self.dataset = dataset
        self.use_mixed_precision = use_mixed_precision
        self.pixel_noise_prob = pixel_noise_prob
        self.pixel_noise_rate = pixel_noise_rate
        self.auxiliary_size_loss_weight = auxiliary_size_loss_weight
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.K = K

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

        # Optimizer with fused kernels for CUDA
        use_fused = device.type == 'cuda'
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            fused=use_fused
        )
        if use_fused:
            print("Using fused AdamW optimizer")

        # Learning rate scheduler with linear warmup
        if lr_warmup_steps is None:
            warmup_steps = int(0.05 * total_steps)  # 5% warmup (fallback)
        else:
            warmup_steps = lr_warmup_steps

        # Create warmup scheduler (linear from 0 to max_lr)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.01,  # Start at 1% of max LR
            end_factor=1.0,     # End at max LR
            total_iters=warmup_steps
        )

        # Create cosine annealing scheduler (after warmup)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=learning_rate * 0.1
        )

        # Combine with SequentialLR
        self.scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        print(f"LR Scheduler: warmup_steps={warmup_steps}, cosine_T_max={total_steps - warmup_steps}, eta_min={learning_rate * 0.1}, initial_lr={learning_rate}")

        # Initialize global step counter
        self.global_step = 0
        self.optimizer_steps = total_steps
        self.accumulation_step = 0  # Track steps within accumulation cycle

        if gradient_accumulation_steps > 1:
            print(f"Using gradient accumulation: {gradient_accumulation_steps} steps")

    def apply_pixel_noise(self, grids: torch.Tensor) -> torch.Tensor:
        """
        Apply pixel noise: randomly replace % of cells with random colors 0-9 (different from current).

        Args:
            grids: Grids [batch_size, height, width]

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
                total_cells = grid.numel()
                num_to_flip = max(1, int(total_cells * self.pixel_noise_rate))

                # Get all positions
                flat_grid = grid.view(-1)
                perm = torch.randperm(total_cells)[:num_to_flip]

                # For each position, sample a DIFFERENT random color (0-9)
                current_colors = flat_grid[perm]
                new_colors = torch.randint(0, 10, (num_to_flip,), device=grids.device)
                # Ensure different: if same, add 1 mod 10
                same_mask = (new_colors == current_colors)
                new_colors[same_mask] = (new_colors[same_mask] + 1) % 10

                flat_grid[perm] = new_colors

        return grids_noisy

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute one training step with K-step rollout."""
        self.model.train()

        # Move batch to device
        input_grids = batch['input_grid'].to(self.device)  # [batch_size, max_size, max_size]
        output_grids = batch['output_grid'].to(self.device)  # [batch_size, max_size, max_size]
        task_indices = batch['task_idx'].to(self.device)  # [batch_size]
        heights = batch['height'].to(self.device)  # [batch_size]
        widths = batch['width'].to(self.device)   # [batch_size]
        d4_indices = batch['d4_idx'].to(self.device)  # [batch_size]
        color_shifts = batch['color_shift'].to(self.device)  # [batch_size]

        batch_size = input_grids.shape[0]

        # Create masks for valid regions
        from ..utils.grid_utils import batch_create_masks
        masks = batch_create_masks(heights, widths, self.model.max_size)

        # Initialize x_current as zeros
        x_current = torch.zeros_like(output_grids)

        # Apply pixel noise to x_current (not input_grid)
        x_current = self.apply_pixel_noise(x_current)

        # Zero gradients only at the start of accumulation cycle
        if self.accumulation_step == 0:
            self.optimizer.zero_grad()

        # K-step rollout loop
        total_loss = 0.0
        step_losses = []
        step_accuracies = []
        step_changes = []
        prev_pred = None

        for step in range(self.K):
            step_tensor = torch.full((batch_size,), step, dtype=torch.long, device=self.device)

            # Forward pass with mixed precision
            if self.use_mixed_precision and self.device.type in ['cuda', 'mps']:
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                    losses = self.model.compute_loss(
                        x0=output_grids,
                        input_grid=input_grids,
                        task_ids=task_indices,
                        x_prev=x_current,
                        step_idx=step_tensor,
                        d4_idx=d4_indices,
                        color_shift=color_shifts,
                        heights=heights,
                        widths=widths,
                        auxiliary_size_loss_weight=self.auxiliary_size_loss_weight,
                    )
            else:
                losses = self.model.compute_loss(
                    x0=output_grids,
                    input_grid=input_grids,
                    task_ids=task_indices,
                    x_prev=x_current,
                    step_idx=step_tensor,
                    d4_idx=d4_indices,
                    color_shift=color_shifts,
                    heights=heights,
                    widths=widths,
                    auxiliary_size_loss_weight=self.auxiliary_size_loss_weight,
                )

            # Backward pass with equal weight for all steps (loss / K)
            step_loss = losses['total_loss'] / self.K
            scaled_loss = step_loss / self.gradient_accumulation_steps

            if self.scaler is not None:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            step_losses.append(losses['grid_loss'].item())
            step_accuracies.append(losses['accuracy'])
            total_loss += step_loss.item()

            # Sample next prediction
            with torch.no_grad():
                # Get logits for sampling
                if self.use_mixed_precision and self.device.type in ['cuda', 'mps']:
                    with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                        logits = self.model(
                            x_prev=x_current,
                            input_grid=input_grids,
                            task_ids=task_indices,
                            step_idx=step_tensor,
                            d4_idx=d4_indices,
                            color_shift=color_shifts,
                            masks=masks,
                        )
                else:
                    logits = self.model(
                        x_prev=x_current,
                        input_grid=input_grids,
                        task_ids=task_indices,
                        step_idx=step_tensor,
                        d4_idx=d4_indices,
                        color_shift=color_shifts,
                        masks=masks,
                    )

                # Sampling strategy:
                # - Train: categorical sampling for steps 0..K-2, argmax at K-1
                # - This teaches robustness to stochastic errors while keeping target sharp
                if step < self.K - 1:
                    # Categorical sampling from softmax
                    probs = F.softmax(logits, dim=-1)  # [B, H, W, 10]
                    x_next = torch.multinomial(
                        probs.view(batch_size, -1, 10).view(-1, 10),
                        num_samples=1
                    ).view(batch_size, self.model.max_size, self.model.max_size)
                else:
                    # Argmax for final step
                    x_next = torch.argmax(logits, dim=-1)

                # Track % cells changed
                if prev_pred is not None:
                    mask_flat = masks.view(batch_size, -1).bool()
                    changed = (x_next.view(batch_size, -1) != prev_pred.view(batch_size, -1))[mask_flat]
                    pct_changed = changed.float().mean().item() * 100
                    step_changes.append(pct_changed)
                else:
                    step_changes.append(100.0)  # First step: all "changed"

                prev_pred = x_next.clone()

                # Detach for next iteration (KEY!)
                x_current = x_next.detach()

        # Increment accumulation step
        self.accumulation_step += 1

        # Only update weights after accumulating gradients
        grad_norm = 0.0
        if self.accumulation_step >= self.gradient_accumulation_steps:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            # Compute gradient norm and clip
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.scheduler.step()

            # Increment global step counter
            self.global_step += 1

            # Reset accumulation counter
            self.accumulation_step = 0

        # Compute per-step delta-improvement
        step_delta_acc = [0.0] + [step_accuracies[i] - step_accuracies[i-1] for i in range(1, self.K)]

        # Return metrics
        return {
            'loss': total_loss,
            'accuracy': step_accuracies[-1],  # Final step accuracy (headline)
            'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            'step_losses': step_losses,
            'step_accuracies': step_accuracies,
            'step_delta_acc': step_delta_acc,
            'step_changes_pct': step_changes,
        }

    def validate(self, val_loader: torch.utils.data.DataLoader, num_batches: int = 10) -> Dict[str, float]:
        """Run validation with K-step refinement (all argmax)."""
        self.model.eval()
        total_metrics = {}
        num_samples = 0

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= num_batches:
                    break

                input_grids = batch['input_grid'].to(self.device)
                output_grids = batch['output_grid'].to(self.device)
                task_indices = batch['task_idx'].to(self.device)
                heights = batch['height'].to(self.device)
                widths = batch['width'].to(self.device)
                d4_indices = batch['d4_idx'].to(self.device)
                color_shifts = batch['color_shift'].to(self.device)

                batch_size = input_grids.shape[0]

                # Create masks
                from ..utils.grid_utils import batch_create_masks
                masks = batch_create_masks(heights, widths, self.model.max_size)

                # K-step refinement (all argmax for eval)
                x_current = torch.zeros_like(output_grids)
                step_accuracies = []

                for step in range(self.K):
                    step_tensor = torch.full((batch_size,), step, dtype=torch.long, device=self.device)

                    losses = self.model.compute_loss(
                        x0=output_grids,
                        input_grid=input_grids,
                        task_ids=task_indices,
                        x_prev=x_current,
                        step_idx=step_tensor,
                        d4_idx=d4_indices,
                        color_shift=color_shifts,
                        heights=heights,
                        widths=widths,
                        auxiliary_size_loss_weight=self.auxiliary_size_loss_weight,
                    )

                    step_accuracies.append(losses['accuracy'])

                    # Argmax for next step
                    logits = self.model(
                        x_prev=x_current,
                        input_grid=input_grids,
                        task_ids=task_indices,
                        step_idx=step_tensor,
                        d4_idx=d4_indices,
                        color_shift=color_shifts,
                        masks=masks,
                    )
                    x_current = torch.argmax(logits, dim=-1)

                # Accumulate metrics (use final step)
                for key, value in losses.items():
                    if key in total_metrics:
                        total_metrics[key] += value * batch_size
                    else:
                        total_metrics[key] = value * batch_size

                num_samples += batch_size

        # Average metrics
        avg_metrics = {key: value / num_samples for key, value in total_metrics.items()}

        self.model.train()
        return avg_metrics


def train_arc_iterative(config: Dict[str, Any]) -> ARCIterativeModel:
    """
    Main training function for ARC iterative refinement model.

    Args:
        config: Training configuration dict with all settings

    Returns:
        Trained model
    """
    # Set up device (prioritize CUDA > MPS > CPU)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # Enable PyTorch optimizations for CUDA
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        print("Enabled CUDA optimizations: TF32, Flash Attention, Memory-Efficient SDPA")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        # MPS memory optimization
        torch.mps.set_per_process_memory_fraction(0.7)
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config to output directory
    import json
    config_save_path = output_dir / 'config.json'
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_save_path}")

    # Construct wandb run name from model_version and tag
    model_version = config.get('model_version', 'unknown')
    tag = config.get('tag', 'default')
    wandb_run_name = f"{model_version}_{tag}"

    # Initialize W&B if enabled
    if config.get('use_wandb', False):
        wandb.init(
            project="arc-prize-2025-itransformer",
            name=wandb_run_name,
            config=config,
            save_code=True
        )

    # Load data paths
    data_paths = load_arc_data_paths(
        data_dir=config.get('data_dir', 'data/arc-prize-2025'),
        datasets=config.get('datasets', None)
    )

    # Create full dataset first
    max_val_examples = config.get('max_val_examples', 128)
    eval_weight = config.get('eval_weight', 1.0)

    full_dataset = ARCDataset(
        data_paths=data_paths['train'],
        max_size=config['max_size'],
        augment=config['augment'],
        include_training_test_examples=config.get('include_training_test_examples', True),
        subset_file=config.get('subset_file', None),
        eval_subset_file=config.get('eval_subset_file', None),
        eval_weight=eval_weight,
        max_val_examples=max_val_examples
    )

    # Split into train and validation based on is_validation flag
    import random

    total_examples = len(full_dataset)
    val_indices = [i for i in range(total_examples) if full_dataset.examples[i].get('is_validation', False)]
    train_indices = [i for i in range(total_examples) if i not in set(val_indices)]

    # Cap validation examples
    random.seed(42)
    if len(val_indices) > max_val_examples:
        val_indices = sorted(random.sample(val_indices, max_val_examples))

    print(f"Splitting {total_examples} examples: {len(train_indices)} train, {len(val_indices)} validation")

    # Create train loader with weighted sampling
    from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

    train_dataset = Subset(full_dataset, train_indices)

    # Create weights for training examples (upweight eval dataset)
    train_weights = []
    for idx in train_indices:
        if full_dataset.examples[idx]['from_eval_dataset']:
            train_weights.append(eval_weight)
        else:
            train_weights.append(1.0)

    train_sampler = WeightedRandomSampler(train_weights, len(train_indices), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )

    print(f"Created train loader with {len(train_loader)} batches")
    print(f"Training set upweighting: eval_weight={eval_weight}")

    # Create validation loader
    val_dataset = Subset(full_dataset, val_indices)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=(device.type == 'cuda')
    )
    print(f"Created val loader with {len(val_loader)} batches")

    # Get dataset info for logging
    dataset_info = full_dataset.get_task_info()

    # Get auxiliary loss config
    aux_config = config.get('auxiliary_loss', {})

    # Create model
    model = ARCIterativeModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        max_size=config['max_size'],
        max_tasks=len(dataset_info['task_id_to_idx']),
        max_steps=config.get('K', 8),
        embedding_dropout=config['embedding_dropout'],
        input_grid_dropout=config.get('input_grid_dropout', 0.0),
        include_size_head=aux_config.get('include_size_head', True),
        size_head_hidden_dim=aux_config.get('size_head_hidden_dim', None),
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Training setup
    optimizer_steps = config['optimizer_steps']
    print(f"Training setup: {optimizer_steps} optimizer steps (~{optimizer_steps / len(train_loader):.1f} epochs at {len(train_loader)} steps/epoch)")
    print(f"Using float16 mixed precision" if config.get('use_mixed_precision', True) else "Using float32 precision")

    # Create trainer
    trainer = ARCIterativeTrainer(
        model=model,
        device=device,
        dataset=full_dataset,
        learning_rate=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.01),
        use_mixed_precision=config.get('use_mixed_precision', True),
        pixel_noise_prob=config.get('pixel_noise_prob', 0.0),
        pixel_noise_rate=config.get('pixel_noise_rate', 0.0),
        total_steps=optimizer_steps,
        auxiliary_size_loss_weight=config.get('auxiliary_size_loss_weight', 0.1),
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
        lr_warmup_steps=config.get('lr_warmup_steps', None),
        K=config.get('K', 8),
    )

    print(f"Model has {num_params:,} parameters")

    # Training loop
    best_val_loss = float('inf')
    pbar = tqdm(total=optimizer_steps, desc="Training")

    for step in range(optimizer_steps):
        # Get batch
        try:
            batch = next(train_iter)
        except (StopIteration, NameError):
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Train step
        metrics = trainer.train_step(batch)

        # Update progress bar
        pbar.update(1)
        pbar.set_postfix({
            'loss': f"{metrics['loss']:.4f}",
            'acc': f"{metrics['accuracy']:.3f}",
            'grad': f"{metrics['grad_norm']:.2f}",
        })

        # Log to W&B
        if config.get('use_wandb', False) and step % config.get('log_every', 10) == 0:
            log_dict = {
                'train/loss': metrics['loss'],
                'train/accuracy': metrics['accuracy'],
                'train/grad_norm': metrics['grad_norm'],
                'train/lr': trainer.scheduler.get_last_lr()[0],
                'train/step': step,
            }
            # Log per-step metrics
            for k in range(trainer.K):
                log_dict[f'train/step_{k}_acc'] = metrics['step_accuracies'][k]
                log_dict[f'train/step_{k}_delta_acc'] = metrics['step_delta_acc'][k]
                log_dict[f'train/step_{k}_changes_pct'] = metrics['step_changes_pct'][k]

            wandb.log(log_dict)

        # Validation
        if step % config.get('val_every_steps', 500) == 0 and step > 0:
            val_metrics = trainer.validate(val_loader, num_batches=len(val_loader))
            print(f"\nStep {step} - Val Loss: {val_metrics['total_loss']:.4f}, Val Acc: {val_metrics['accuracy']:.3f}")

            if config.get('use_wandb', False):
                wandb.log({
                    'val/loss': val_metrics['total_loss'],
                    'val/accuracy': val_metrics['accuracy'],
                    'val/step': step,
                })

            # Save best model
            if val_metrics['total_loss'] < best_val_loss and config.get('save_best', True):
                best_val_loss = val_metrics['total_loss']
                save_path = output_dir / 'best_model.pt'
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'dataset_info': dataset_info
                }
                torch.save(checkpoint, save_path)
                print(f"Saved best model with val loss: {best_val_loss:.4f}")

    pbar.close()

    # Save final model
    if config.get('save_final', True):
        final_save_path = output_dir / 'final_model.pt'
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': config,
            'dataset_info': dataset_info
        }
        torch.save(checkpoint, final_save_path)
        print(f"Saved final model to {final_save_path}")

    print("Training completed!")

    return model
