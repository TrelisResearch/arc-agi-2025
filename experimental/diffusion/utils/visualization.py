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

    # Note: No longer using global distribution, noise scheduler uses uniform over {0..9}

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