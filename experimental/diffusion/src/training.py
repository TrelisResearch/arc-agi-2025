"""
Training loop and sampling for the ARC diffusion model.
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import wandb
from pathlib import Path

from .model import ARCDiffusionModel
from .dataset import ARCDataset, load_arc_data_paths, collate_fn
from ..utils.noise_scheduler import DiscreteNoiseScheduler
from ..utils.grid_utils import grid_to_display_string
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
        auxiliary_size_loss_weight: float = 0.1,
        use_ema: bool = True,
        ema_decay: float = 0.9995,
        ema_warmup_steps: int = 1000
    ):
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.device = device
        self.dataset = dataset
        self.use_mixed_precision = use_mixed_precision
        self.pixel_noise_prob = pixel_noise_prob
        self.pixel_noise_rate = pixel_noise_rate
        self.auxiliary_size_loss_weight = auxiliary_size_loss_weight
        self.use_ema = use_ema

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
        warmup_steps = int(0.05 * total_steps)  # 5% warmup

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

        # Set up EMA if enabled
        self.ema = None
        if use_ema:
            from experimental.diffusion.utils.ema import EMA
            self.ema = EMA(
                model=model,
                decay=ema_decay,
                warmup_steps=ema_warmup_steps,
                device=device
            )
            print(f"Using EMA with decay={ema_decay}, warmup={ema_warmup_steps} steps")

        # Initialize global step counter for self-conditioning gain scheduling
        self.global_step = 0
        self.optimizer_steps = total_steps

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
        rotations = batch['rotation'].to(self.device)  # [batch_size] - rotation indices
        flips = batch['flip'].to(self.device)  # [batch_size] - flip indices
        color_shifts = batch['color_shift'].to(self.device)  # [batch_size] - color shift values

        batch_size = input_grids.shape[0]

        # Apply pixel noise to input grids only (not outputs)
        input_grids = self.apply_pixel_noise(input_grids)

        # Sample random timesteps (0-indexed: [0, num_timesteps))
        timesteps = torch.randint(0, self.noise_scheduler.num_timesteps, (batch_size,), device=self.device)

        # Create masks for valid regions
        from experimental.diffusion.utils.grid_utils import batch_create_masks
        masks = batch_create_masks(heights, widths, self.model.max_size).to(self.device)

        # Add noise to clean output grids using uniform distribution over {0..9}
        # Only noise valid regions, clamp invalid regions to 0
        noisy_grids = self.noise_scheduler.add_noise(output_grids, timesteps, masks)

        # Calculate self-conditioning gain (ramp from 0.3 to 1.0 over training)
        progress = min(1.0, self.global_step / (0.8 * self.optimizer_steps))  # Reach 1.0 at 80% of training
        sc_gain = 0.3 + 0.7 * progress

        # First pass: Generate p0_prev without gradients for self-conditioning
        sc_p0 = None
        if torch.rand(1).item() > 0.5:  # 50% dropout for self-conditioning
            with torch.no_grad():
                if self.use_mixed_precision and self.device.type in ['cuda', 'mps']:
                    with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                        logits_prev = self.model(
                            xt=noisy_grids,
                            input_grid=input_grids,
                            task_ids=task_indices,
                            timesteps=timesteps,
                            rotation=rotations,
                            flip=flips,
                            color_shift=color_shifts,
                            masks=masks,
                            sc_p0=None,  # No self-conditioning in first pass
                            sc_gain=0.0
                        )
                else:
                    logits_prev = self.model(
                        xt=noisy_grids,
                        input_grid=input_grids,
                        task_ids=task_indices,
                        timesteps=timesteps,
                        rotation=rotations,
                        flip=flips,
                        color_shift=color_shifts,
                        masks=masks,
                        sc_p0=None,  # No self-conditioning in first pass
                        sc_gain=0.0
                    )
                # Convert logits to probabilities
                sc_p0 = torch.softmax(logits_prev, dim=-1)

        # Second pass: Forward with self-conditioning and compute loss
        if self.use_mixed_precision and self.device.type in ['cuda', 'mps']:
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                losses = self.model.compute_loss(
                    x0=output_grids,
                    input_grid=input_grids,
                    task_ids=task_indices,
                    xt=noisy_grids,
                    timesteps=timesteps,
                    rotation=rotations,
                    flip=flips,
                    color_shift=color_shifts,
                    heights=heights,
                    widths=widths,
                    auxiliary_size_loss_weight=self.auxiliary_size_loss_weight,
                    sc_p0=sc_p0,
                    sc_gain=sc_gain
                )
        else:
            losses = self.model.compute_loss(
                x0=output_grids,
                input_grid=input_grids,
                task_ids=task_indices,
                xt=noisy_grids,
                timesteps=timesteps,
                rotation=rotations,
                flip=flips,
                color_shift=color_shifts,
                heights=heights,
                widths=widths,
                auxiliary_size_loss_weight=self.auxiliary_size_loss_weight,
                sc_p0=sc_p0,
                sc_gain=sc_gain
            )

        # Backward pass with mixed precision
        total_loss = losses['total_loss']
        self.optimizer.zero_grad()

        if self.scaler is not None:
            # CUDA with float16 and gradient scaling
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)

            # Compute gradient norm and clip
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # MPS, CPU, or bfloat16 without gradient scaling
            total_loss.backward()

            # Compute gradient norm and clip
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

        self.scheduler.step()

        # Update EMA after optimizer step
        if self.ema is not None:
            self.ema.update(self.model)

        # Increment global step counter
        self.global_step += 1

        # Return losses and grad norm as Python floats
        losses['grad_norm'] = grad_norm.item()
        return {key: value.item() if hasattr(value, 'item') else value for key, value in losses.items()}

    def validate(self, val_loader: torch.utils.data.DataLoader, num_batches: int = 10, use_ema: bool = True) -> Dict[str, float]:
        """Run validation."""
        # Use EMA weights if available and requested
        if use_ema and self.ema is not None:
            from experimental.diffusion.utils.ema import ModelWithEMA
            with ModelWithEMA(self.model, self.ema) as ema_model:
                return self._validate_impl(val_loader, num_batches)
        else:
            return self._validate_impl(val_loader, num_batches)

    def _validate_impl(self, val_loader: torch.utils.data.DataLoader, num_batches: int = 10) -> Dict[str, float]:
        """Internal validation implementation."""
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

                # Create masks for valid regions
                from experimental.diffusion.utils.grid_utils import batch_create_masks
                masks = batch_create_masks(heights, widths, self.model.max_size).to(self.device)

                # Add noise using uniform distribution over {0..9}
                # Only noise valid regions, clamp invalid regions to 0
                noisy_grids = self.noise_scheduler.add_noise(output_grids, timesteps, masks)

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

    def discrete_reverse_step(
        self,
        x_t: torch.Tensor,
        logits_x0: torch.Tensor,
        t: int,
        mask: Optional[torch.Tensor] = None,
        deterministic: bool = True
    ) -> torch.Tensor:
        """
        Perform discrete reverse diffusion step using uniform kernel.

        Args:
            x_t: Current noised state [batch_size, H, W] with values in {0..9}
            logits_x0: Model's prediction of clean data [batch_size, H, W, 10]
            t: Current timestep (0-indexed)
            mask: Valid region mask [batch_size, H, W]
            deterministic: If True, use argmax; if False, sample

        Returns:
            x_{t-1}: Less noisy state [batch_size, H, W]
        """
        batch_size, H, W, _ = logits_x0.shape
        device = logits_x0.device

        # Get model's p(x0|xt)
        p0 = torch.softmax(logits_x0, dim=-1)  # [B, H, W, 10]

        # Get schedule parameters
        beta = self.noise_scheduler.betas[t].to(device).view(1, 1, 1, 1)

        # alpha_bar_{t-1}
        if t > 0:
            abar_tm1 = self.noise_scheduler.alpha_bars[t-1].to(device).view(1, 1, 1, 1)
        else:
            abar_tm1 = torch.tensor(1.0, device=device).view(1, 1, 1, 1)

        # Uniform kernel over 10 colors
        K = 0.1

        # Compute A(x0) = (1 - alpha_bar_{t-1}) * K + alpha_bar_{t-1} * p(x0)
        A = (1 - abar_tm1) * K + abar_tm1 * p0  # [B, H, W, 10]

        # Initialize mass with base term: beta * K * A(x0)
        mass = beta * K * A  # [B, H, W, 10]

        # Add inertia term: (1 - beta) * A(x_t) for current token
        s = x_t.unsqueeze(-1)  # [B, H, W, 1] - current token indices
        A_s = A.gather(-1, s)  # [B, H, W, 1] - A values at current tokens
        mass.scatter_(-1, s, (1 - beta) * A_s)  # Update mass at current token positions

        # Normalize to get probabilities
        probs = mass / mass.sum(-1, keepdim=True).clamp_min(1e-8)  # [B, H, W, 10]

        # Sample or take argmax
        if deterministic:
            x_tm1 = probs.argmax(-1)  # [B, H, W]
        else:
            # Sample from categorical distribution
            x_tm1 = torch.multinomial(probs.view(-1, 10), 1).view(batch_size, H, W)

        # Apply mask to keep invalid regions fixed at 0
        if mask is not None:
            x_tm1 = torch.where(mask, x_tm1, torch.zeros_like(x_tm1))

        return x_tm1

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
        # Check for integrated size head in model
        if hasattr(self.model, 'include_size_head') and self.model.include_size_head:
            predicted_heights, predicted_widths = self.model.predict_sizes(input_grids, task_indices)
            print(f"Predicted sizes (integrated): heights={predicted_heights.cpu().tolist()}, widths={predicted_widths.cpu().tolist()}")

        # Build mask from predicted sizes (or full grid if no size prediction)
        mask = torch.zeros((batch_size, max_size, max_size), dtype=torch.bool, device=self.device)
        if predicted_heights is not None and predicted_widths is not None:
            for b in range(batch_size):
                h, w = predicted_heights[b].item(), predicted_widths[b].item()
                mask[b, :h, :w] = True
        else:
            # If no size prediction, assume full grid is valid
            mask[:] = True

        # Create float mask for model
        mask_float = mask.float()

        # Initialize with uniform random noise over {0..9}, black outside mask
        x_t = torch.randint(
            0, 10,  # Uniform over {0..9}
            (batch_size, max_size, max_size),
            device=self.device
        )
        x_t = torch.where(mask, x_t, torch.zeros_like(x_t))

        # Initialize self-conditioning buffer
        sc_p0 = None

        # Denoising loop (reverse from T-1 to 0)
        for t in reversed(range(num_inference_steps)):
            if self.debug:
                print(f"\n=== Timestep {t} (step {num_inference_steps - t}/{num_inference_steps}) ===")

            # Create batch of current timestep
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            # Calculate self-conditioning gain (full gain during inference)
            sc_gain = 1.0

            # Forward pass with masking and self-conditioning (no augmentation during inference)
            logits = self.model(x_t, input_grids, task_indices, t_batch,
                               rotation=None, flip=None, color_shift=None,  # No augmentation
                               masks=mask_float, sc_p0=sc_p0, sc_gain=sc_gain)

            # Apply temperature scaling
            logits_scaled = logits / temperature

            # Update self-conditioning buffer with current predictions
            sc_p0 = torch.softmax(logits_scaled, dim=-1)

            # Perform discrete reverse step
            x_t = self.discrete_reverse_step(
                x_t, logits_scaled, t, mask=mask,
                deterministic=True
                # deterministic=(t == 0)  # Only deterministic at final step
            )

            # Debug printing
            if self.debug:
                # Show first 10x10 of the grid
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

    # Create data loaders with optimized settings
    # Disable pin_memory for MPS to avoid warnings and reduce memory usage
    use_pin_memory = device.type == 'cuda'
    num_workers = 4 if device.type == 'cuda' else 0
    use_persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=use_persistent_workers,
        prefetch_factor=4 if num_workers > 0 else None,
        pin_memory=use_pin_memory,
        drop_last=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'] // 2,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=use_persistent_workers,
        prefetch_factor=4 if num_workers > 0 else None,
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
        input_grid_dropout=config.get('input_grid_dropout', 0.0),
        include_size_head=include_size_head,
        size_head_hidden_dim=size_head_hidden_dim
    )

    # Compile model with torch.compile for CUDA optimization
    if device.type == 'cuda':
        print("Compiling model with torch.compile(mode='max-autotune')...")
        model = torch.compile(model, mode="max-autotune")
        print("Model compiled successfully")

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
        auxiliary_size_loss_weight=aux_config.get('auxiliary_size_loss_weight', 0.1),
        use_ema=config.get('use_ema', True),
        ema_decay=config.get('ema_decay', 0.9995),
        ema_warmup_steps=config.get('ema_warmup_steps', 1000)
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
                        count = val_losses.get(f'{bucket}_count', 0)
                        print(f"  {bucket:10s}: acc={acc:.3f}, count={count:4.0f}")

            overall_acc = val_losses.get('accuracy', 0.0)
            print(f"Overall: acc={overall_acc:.3f}")

            # Log validation losses
            if config.get('use_wandb', False):
                val_log_dict = {f"val/{key}": value for key, value in val_losses.items()}
                val_log_dict["val/learning_rate"] = trainer.scheduler.get_last_lr()[0]
                val_log_dict["val/step"] = step
                val_log_dict["val/epoch"] = current_epoch_approx
                wandb.log(val_log_dict, step=step)

            # Save best model (weights only to save space)
            if val_losses['total_loss'] < best_val_loss:
                best_val_loss = val_losses['total_loss']
                # Save model in bfloat16 without modifying the original
                model_state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
                save_dict = {
                    'model_state_dict': model_state_dict_bf16,
                    'config': config,
                    'dataset_info': dataset_info
                }
                # Save EMA state if available
                if trainer.ema is not None:
                    save_dict['ema_state_dict'] = trainer.ema.state_dict()
                torch.save(save_dict, output_dir / 'best_model.pt')
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

    # Save final model (weights only to save space)
    model_state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
    save_dict = {
        'model_state_dict': model_state_dict_bf16,
        'config': config,
        'dataset_info': dataset_info
    }
    # Save EMA state if available
    if trainer.ema is not None:
        save_dict['ema_state_dict'] = trainer.ema.state_dict()
    torch.save(save_dict, output_dir / 'final_model.pt')

    if config.get('use_wandb', False):
        wandb.finish()

    print("Training completed!")
    return model