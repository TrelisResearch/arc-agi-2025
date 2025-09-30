"""
Visualization utilities for ARC diffusion model training.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path


# ARC color palette (matching viewer.html)
ARC_COLORS = [
    '#000000',  # 0: Black
    '#0074D9',  # 1: Blue
    '#FF4136',  # 2: Red
    '#2ECC40',  # 3: Green
    '#FFDC00',  # 4: Yellow
    '#AAAAAA',  # 5: Gray
    '#F012BE',  # 6: Magenta
    '#FF851B',  # 7: Orange
    '#7FDBFF',  # 8: Aqua
    '#870C25',  # 9: Brown
    '#CCCCCC',  # 10: PAD (light gray)
    '#FFFFFF'   # 11: Loss mask indicator (white)
]

# Create colormap
ARC_COLORMAP = ListedColormap(ARC_COLORS)


def render_grid(grid: np.ndarray, ax: plt.Axes, title: str = "",
                valid_height: Optional[int] = None, valid_width: Optional[int] = None,
                show_loss_mask: bool = False) -> None:
    """
    Render a single ARC grid with proper colors, padding visualization, and loss masking.

    Args:
        grid: 2D numpy array representing the grid
        ax: Matplotlib axes to render on
        title: Title for the grid
        valid_height: Height of valid (non-padded) region
        valid_width: Width of valid (non-padded) region
        show_loss_mask: Whether to show loss masking for output grids
    """
    height, width = grid.shape

    # Create a copy for visualization
    vis_grid = grid.copy()

    # Mark loss-masked areas (for output grids)
    if show_loss_mask and valid_height is not None and valid_width is not None:
        # Areas outside valid region should show loss masking
        mask = np.zeros_like(vis_grid, dtype=bool)
        mask[valid_height:, :] = True  # Below valid region
        mask[:, valid_width:] = True   # Right of valid region
        vis_grid[mask] = 11  # Use white color for loss-masked areas

    # Display the grid with ARC colors
    ax.imshow(vis_grid, cmap=ARC_COLORMAP, vmin=0, vmax=11, aspect='equal')

    # Add grid lines
    for i in range(height + 1):
        ax.axhline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
    for j in range(width + 1):
        ax.axvline(j - 0.5, color='black', linewidth=0.5, alpha=0.3)

    # Add thicker lines to show valid region boundary
    if valid_height is not None and valid_width is not None:
        # Horizontal line at bottom of valid region
        if valid_height < height:
            ax.axhline(valid_height - 0.5, color='red', linewidth=2, alpha=0.8)
        # Vertical line at right of valid region
        if valid_width < width:
            ax.axvline(valid_width - 0.5, color='red', linewidth=2, alpha=0.8)


    # Set title
    ax.set_title(title, fontsize=10, pad=5)

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Set aspect ratio to be square
    ax.set_aspect('equal')


def visualize_noise_schedule(
    input_grid_full: np.ndarray,
    output_grid_full: np.ndarray,
    input_valid_height: int,
    input_valid_width: int,
    output_valid_height: int,
    output_valid_width: int,
    noise_scheduler,
    device: torch.device,
    output_path: str,
    num_timesteps: int = 5
) -> None:
    """
    Visualize how noise affects grids across different timesteps, showing full padded grids.

    Args:
        input_grid_full: Full padded input grid [max_size, max_size]
        output_grid_full: Full padded output grid [max_size, max_size]
        input_valid_height: Height of valid (non-padded) region in input
        input_valid_width: Width of valid (non-padded) region in input
        output_valid_height: Height of valid (non-padded) region in output
        output_valid_width: Width of valid (non-padded) region in output
        noise_scheduler: The noise scheduler used for training
        device: PyTorch device
        output_path: Path to save the visualization PNG
        num_timesteps: Number of timesteps to visualize (default 5)
    """
    # Convert grids to tensors
    output_tensor = torch.from_numpy(output_grid_full).to(device)

    # Create timesteps: 0%, 25%, 50%, 75%, 100% of max timesteps
    max_t = noise_scheduler.num_timesteps - 1
    timesteps = [int(max_t * p) for p in [0.0, 0.25, 0.5, 0.75, 1.0]]

    # Create figure with layout: top row (clean input, target), bottom row (5 noised versions)
    fig, axes = plt.subplots(2, max(2, num_timesteps), figsize=(2.5 * max(2, num_timesteps), 6))

    # Handle case where we have fewer axes in top row
    if axes.shape[1] > 2:
        # Hide extra axes in top row
        for i in range(2, axes.shape[1]):
            axes[0, i].axis('off')

    # Top row: Clean input and target output (show full padded grids)
    render_grid(input_grid_full, axes[0, 0], "Input Grid\n(Full Padded)",
               valid_height=input_valid_height, valid_width=input_valid_width,
               show_loss_mask=False)

    render_grid(output_grid_full, axes[0, 1], "Target Output\n(With Loss Mask)",
               valid_height=output_valid_height, valid_width=output_valid_width,
               show_loss_mask=True)

    # Bottom row: Noised versions at different timesteps (show full padded grids)
    for i, t in enumerate(timesteps):
        # Add noise to the target output (this is what the model tries to denoise)
        timestep_tensor = torch.tensor([t], device=device)
        noisy_tensor = noise_scheduler.add_noise(
            output_tensor.unsqueeze(0),
            timestep_tensor
        )
        noisy_grid = noisy_tensor[0].cpu().numpy()

        # Calculate actual noise level for 10-class discrete diffusion
        alpha_bar_t = noise_scheduler.alpha_bars[t].item()
        # P(correct) = P(kept) + P(replaced & random correct) = alpha_bar_t + (1-alpha_bar_t) * 0.1
        prob_correct = alpha_bar_t + (1 - alpha_bar_t) * 0.1
        actual_noise_pct = (1 - prob_correct) * 100
        title = f"Noisy Target t={t}\n({actual_noise_pct:.0f}% noise)"

        render_grid(noisy_grid, axes[1, i], title,
                   valid_height=output_valid_height, valid_width=output_valid_width,
                   show_loss_mask=True)

    # Adjust layout
    plt.tight_layout()

    # Add title text in top right corner instead of centered suptitle
    fig.text(0.98, 0.95, "Diffusion Training Data Visualization\n(Red lines = valid region boundary, Gray = padding, White = loss masked)",
             fontsize=10, ha='right', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    # Save the visualization
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"📊 Saved noise schedule visualization to: {output_path}")


def create_denoising_progression_visualization(
    model,
    noise_scheduler,
    val_dataset,
    device: torch.device,
    output_dir: Path,
    step: int,
    config: Dict
) -> None:
    """
    Create a visualization showing the denoising progression during training.
    Shows input grid, ground truth, and denoising steps at different timesteps.

    Args:
        model: The diffusion model
        noise_scheduler: The noise scheduler
        val_dataset: Validation dataset to sample from
        device: PyTorch device
        output_dir: Directory to save visualization
        step: Current training step (for filename)
        config: Training configuration
    """
    print(f"🎨 Creating denoising progression visualization at step {step}...")

    model.eval()

    # Get a random validation example
    import random
    val_idx = random.randint(0, len(val_dataset) - 1)
    example = val_dataset[val_idx]

    # Extract data and move to device
    input_grid = example['input_grid'].unsqueeze(0).to(device)  # [1, max_size, max_size]
    output_grid = example['output_grid'].unsqueeze(0).to(device)  # [1, max_size, max_size]
    task_idx = example['task_idx'].unsqueeze(0).to(device)  # [1]
    height = example['height'].item()
    width = example['width'].item()

    # Define timesteps to visualize (as fractions of total timesteps)
    timestep_fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
    max_timesteps = noise_scheduler.num_timesteps
    timesteps_to_viz = [int(frac * (max_timesteps - 1)) for frac in timestep_fractions]

    # Create figure: 2 rows, len(timesteps_to_viz) + 2 columns
    # First row: Input, Ground Truth, then denoising steps
    # Second row: Noisy versions at each timestep
    fig, axes = plt.subplots(2, len(timesteps_to_viz) + 2, figsize=(3 * (len(timesteps_to_viz) + 2), 6))

    # Row 1, Col 1: Input grid
    input_np = input_grid[0].cpu().numpy()
    render_grid(input_np, axes[0, 0], "Input Grid",
                valid_height=height, valid_width=width)

    # Row 1, Col 2: Ground truth
    output_np = output_grid[0].cpu().numpy()
    render_grid(output_np, axes[0, 1], "Ground Truth",
                valid_height=height, valid_width=width, show_loss_mask=True)

    # Row 2, Cols 1-2: Empty (or repeat input/gt for reference)
    axes[1, 0].axis('off')
    axes[1, 1].axis('off')

    with torch.no_grad():
        # Create mask for valid region
        from experimental.diffusion.utils.grid_utils import batch_create_masks
        heights = torch.tensor([height], device=device)
        widths = torch.tensor([width], device=device)
        mask = batch_create_masks(heights, widths, output_grid.shape[1]).to(device)

        for i, t in enumerate(timesteps_to_viz):
            col_idx = i + 2  # Offset by 2 for input and ground truth columns

            # Create noisy version at timestep t with masking
            t_tensor = torch.tensor([t], device=device)
            noisy_grid = noise_scheduler.add_noise(output_grid, t_tensor, mask)

            # Show noisy grid in bottom row
            noisy_np = noisy_grid[0].cpu().numpy()
            alpha_bar_t = noise_scheduler.alpha_bars[t].item()
            prob_correct = alpha_bar_t + (1 - alpha_bar_t) * 0.1
            noise_pct = (1 - prob_correct) * 100

            render_grid(noisy_np, axes[1, col_idx], f"Noisy t={t}\n({noise_pct:.0f}% noise)",
                       valid_height=height, valid_width=width)

            # Generate denoised prediction from this timestep with masking
            logits = model(noisy_grid, input_grid, task_idx, t_tensor, mask.float())
            predicted_grid = torch.argmax(logits, dim=-1)

            # Debug: Check prediction statistics
            if i == 0:  # Only print for first timestep to avoid spam
                unique_preds, counts = torch.unique(predicted_grid[0, :height, :width], return_counts=True)
                pred_stats = {pred.item(): count.item() for pred, count in zip(unique_preds, counts)}
                print(f"Debug - Step {step}, t={t}: Prediction distribution in valid region: {pred_stats}")

            # Apply ground truth masking (only show predictions within valid region)
            masked_pred = predicted_grid.clone()
            masked_pred[0, height:, :] = 10  # Set invalid rows to PAD
            masked_pred[0, :, width:] = 10   # Set invalid cols to PAD

            pred_np = masked_pred[0].cpu().numpy()

            # Show denoised prediction in top row
            title = f"Denoised t={t}\n(frac={timestep_fractions[i]:.2f})"
            render_grid(pred_np, axes[0, col_idx], title,
                       valid_height=height, valid_width=width, show_loss_mask=True)

    # Adjust layout and add title
    plt.tight_layout()
    task_id = example.get('task_id', 'unknown')
    fig.suptitle(f"Denoising Progression - Step {step} - Task: {task_id}", fontsize=14, y=1.02)

    # Create output path
    output_path = output_dir / f"denoising_progression_step_{step:06d}.png"

    # Save visualization
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"📊 Saved denoising progression to: {output_path}")

    model.train()  # Return to training mode


def create_training_visualization(
    dataset,
    noise_scheduler,
    device: torch.device,
    output_dir: Path,
    config: Dict
) -> None:
    """
    Create and save a visualization of the training data before training starts.

    Args:
        dataset: The training dataset
        noise_scheduler: The noise scheduler used for training
        device: PyTorch device
        output_dir: Directory to save the visualization
        config: Training configuration
    """
    print("🎨 Creating training data visualization...")

    # Get the first example from the dataset
    example = dataset[0]

    # Extract full padded grids (what the model actually sees)
    input_grid_full = example['input_grid'].numpy()  # [max_size, max_size]
    output_grid_full = example['output_grid'].numpy()  # [max_size, max_size]

    # Get actual dimensions (for valid region marking)
    output_height = example['height'].item()
    output_width = example['width'].item()

    # Need to get the original input dimensions - we'll look at the raw task data
    task_id = example['task_id']

    # Find the input dimensions by looking at the raw task
    # We need to access the original dataset's examples to find input dimensions
    input_height, input_width = None, None
    for task_examples in [dataset.examples]:
        for ex in task_examples:
            if ex['task_id'] == task_id:
                # Get the input grid from the example
                raw_input = ex['input_grid']
                # Find actual content bounds
                non_pad_mask = raw_input != 10  # PAD token is 10
                if non_pad_mask.any():
                    rows_with_content = non_pad_mask.any(axis=1)
                    cols_with_content = non_pad_mask.any(axis=0)
                    input_height = rows_with_content.sum()
                    input_width = cols_with_content.sum()
                    break
        if input_height is not None:
            break

    # Fallback: use output dimensions if we can't find input dimensions
    if input_height is None or input_width is None:
        input_height, input_width = output_height, output_width
        print("⚠️  Could not determine input dimensions, using output dimensions")


    # Create output path
    output_path = output_dir / "training_noise_visualization.png"

    # Generate visualization
    visualize_noise_schedule(
        input_grid_full=input_grid_full,
        output_grid_full=output_grid_full,
        input_valid_height=input_height,
        input_valid_width=input_width,
        output_valid_height=output_height,
        output_valid_width=output_width,
        noise_scheduler=noise_scheduler,
        device=device,
        output_path=str(output_path),
        num_timesteps=5
    )

    # Print info about the example
    print(f"📝 Visualization shows example from task: {task_id}")
    print(f"📐 Input dimensions: {input_height}×{input_width}")
    print(f"📐 Output dimensions: {output_height}×{output_width}")
    print(f"📐 Full grid size: {config['max_size']}×{config['max_size']}")
    print(f"🔧 Noise schedule: {config['schedule_type']} with {config['num_timesteps']} timesteps")