"""
Utilities for handling ARC grids, padding, masking, and size operations.
"""
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
import numpy as np


def pad_grid_to_size(grid: torch.Tensor, target_size: int = 30, pad_value: int = 10) -> torch.Tensor:
    """
    Pad a grid to target_size x target_size with pad_value.

    Args:
        grid: Input grid [H, W] or [batch, H, W]
        target_size: Target size (assumes square grids)
        pad_value: Value to pad with (PAD token = 10)

    Returns:
        Padded grid [target_size, target_size] or [batch, target_size, target_size]
    """
    if grid.dim() == 2:
        h, w = grid.shape
        pad_h = target_size - h
        pad_w = target_size - w
        # Pad: (left, right, top, bottom)
        return F.pad(grid, (0, pad_w, 0, pad_h), mode='constant', value=pad_value)
    elif grid.dim() == 3:
        batch_size, h, w = grid.shape
        pad_h = target_size - h
        pad_w = target_size - w
        return F.pad(grid, (0, pad_w, 0, pad_h), mode='constant', value=pad_value)
    else:
        raise ValueError(f"Grid must be 2D or 3D, got shape {grid.shape}")


def create_mask(height: int, width: int, max_size: int = 30) -> torch.Tensor:
    """
    Create a mask that is 1 for valid regions [0:height, 0:width] and 0 elsewhere.

    Args:
        height: True height of the grid
        width: True width of the grid
        max_size: Maximum grid size (30x30)

    Returns:
        Mask tensor [max_size, max_size] with 1s in valid region
    """
    mask = torch.zeros(max_size, max_size, dtype=torch.float32)
    mask[:height, :width] = 1.0
    return mask


def batch_create_masks(heights: torch.Tensor, widths: torch.Tensor, max_size: int = 30) -> torch.Tensor:
    """
    Create masks for a batch of grids.

    Args:
        heights: Heights for each item in batch [batch_size]
        widths: Widths for each item in batch [batch_size]
        max_size: Maximum grid size

    Returns:
        Masks [batch_size, max_size, max_size]
    """
    batch_size = len(heights)
    masks = torch.zeros(batch_size, max_size, max_size, dtype=torch.float32)

    for i, (h, w) in enumerate(zip(heights, widths)):
        masks[i, :h, :w] = 1.0

    return masks


def clamp_outside_mask(grid: torch.Tensor, mask: torch.Tensor, pad_value: int = 10) -> torch.Tensor:
    """
    Clamp tokens outside the valid region (where mask=0) to pad_value.

    Args:
        grid: Grid tensor [..., H, W]
        mask: Mask tensor [..., H, W] with 1s for valid regions
        pad_value: Value to set outside valid region

    Returns:
        Grid with tokens outside mask set to pad_value
    """
    return torch.where(mask.bool(), grid, pad_value)


def extract_valid_region(grid: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Extract the valid [0:height, 0:width] region from a padded grid.

    Args:
        grid: Padded grid [..., max_size, max_size]
        height: True height
        width: True width

    Returns:
        Extracted region [..., height, width]
    """
    if grid.dim() == 2:
        return grid[:height, :width]
    elif grid.dim() == 3:
        return grid[:, :height, :width]
    else:
        return grid[..., :height, :width]


def grid_to_tokens(grid: np.ndarray, max_size: int = 30) -> Tuple[torch.Tensor, int, int]:
    """
    Convert a numpy grid to padded tokens with size info.

    Args:
        grid: Numpy grid [H, W] with values 0-9
        max_size: Target padded size

    Returns:
        tokens: Padded tokens [max_size, max_size] (padded with black/0, no PAD token)
        height: Original height
        width: Original width
    """
    height, width = grid.shape
    tokens = torch.from_numpy(grid.astype(np.int64))
    padded_tokens = pad_grid_to_size(tokens, max_size, pad_value=10)  # Use PAD token for input grids
    return padded_tokens, height, width


def tokens_to_grid(tokens: torch.Tensor, height: int, width: int) -> np.ndarray:
    """
    Convert padded tokens back to a numpy grid.

    Args:
        tokens: Padded tokens [max_size, max_size]
        height: True height
        width: True width

    Returns:
        grid: Numpy grid [height, width]
    """
    valid_tokens = extract_valid_region(tokens, height, width)
    return valid_tokens.cpu().numpy().astype(np.int32)


def detect_valid_region(grid: np.ndarray, pad_value: int = 10) -> Tuple[np.ndarray, Optional[str]]:
    """
    Detect and extract the valid region from a grid by finding PAD token boundaries.

    Args:
        grid: Predicted grid [max_size, max_size] with PAD tokens at boundaries
        pad_value: Value used for PAD tokens (default: 10)

    Returns:
        valid_grid: Extracted valid region, or empty array if detection fails
        error: Error message if detection fails, None otherwise
    """
    try:
        if grid.size == 0:
            return np.array([]), "Empty grid"

        # If grid is all PAD tokens, return minimum 1x1 grid with first cell
        if np.all(grid == pad_value):
            return grid[:1, :1], None

        # Find the rightmost non-PAD column
        valid_cols = []
        for col in range(grid.shape[1]):
            if not np.all(grid[:, col] == pad_value):
                valid_cols.append(col)

        # Find the bottommost non-PAD row
        valid_rows = []
        for row in range(grid.shape[0]):
            if not np.all(grid[row, :] == pad_value):
                valid_rows.append(row)

        # If no valid content found (shouldn't happen after all-PAD check above)
        if not valid_rows or not valid_cols:
            return grid[:1, :1], None

        # Extract rectangular region from (0,0) to the furthest non-PAD token
        min_row, max_row = 0, max(valid_rows) + 1
        min_col, max_col = 0, max(valid_cols) + 1

        valid_region = grid[min_row:max_row, min_col:max_col]

        # Always return the extracted region, regardless of PAD tokens inside
        # The model might predict PAD tokens incorrectly, but we want to see the result
        return valid_region, None

    except Exception as e:
        # If anything fails, return at least a 1x1 grid to avoid crashes
        try:
            return grid[:1, :1], None
        except:
            return np.array([[0]]), f"Fallback after error: {str(e)}"


def grid_to_display_string(grid: np.ndarray, pad_symbol: str = '*') -> str:
    """
    Convert grid to display string, replacing PAD tokens with symbols.

    Args:
        grid: Grid to display
        pad_symbol: Symbol to use for PAD tokens (value 10)

    Returns:
        String representation of grid
    """
    if grid.size == 0:
        return "(empty grid)"

    lines = []
    for row in grid:
        line = ''.join(pad_symbol if cell == 10 else str(int(cell)) for cell in row)
        lines.append(line)
    return '\n'.join(lines)


class TaskAugmentation:
    """Task-level augmentation utilities for ARC data."""

    @staticmethod
    def apply_flip_augmentation(grid: np.ndarray, flip_type: str) -> np.ndarray:
        """Apply flip augmentation to a grid.

        Args:
            grid: Input grid [H, W]
            flip_type: 'horizontal', 'vertical', or 'none'

        Returns:
            Augmented grid [H, W]
        """
        if flip_type == 'horizontal':
            return np.fliplr(grid)
        elif flip_type == 'vertical':
            return np.flipud(grid)
        else:  # 'none'
            return grid.copy()

    @staticmethod
    def apply_rotation_augmentation(grid: np.ndarray, rotation: int) -> np.ndarray:
        """Apply rotation augmentation to a grid.

        Args:
            grid: Input grid [H, W]
            rotation: 0, 90, 180, or 270 degrees

        Returns:
            Rotated grid
        """
        if rotation == 90:
            return np.rot90(grid, k=-1)  # 90 clockwise
        elif rotation == 180:
            return np.rot90(grid, k=2)
        elif rotation == 270:
            return np.rot90(grid, k=1)   # 270 clockwise = 90 counter-clockwise
        else:  # 0
            return grid.copy()

    @staticmethod
    def apply_color_cycle_augmentation(grid: np.ndarray, cycle_offset: int) -> np.ndarray:
        """Apply color cycling augmentation to a grid.

        Args:
            grid: Input grid [H, W] with values 0-9
            cycle_offset: Number of positions to cycle colors (0-8)

        Returns:
            Color-cycled grid [H, W]
        """
        if cycle_offset == 0:
            return grid.copy()

        # Create color mapping (keep black/0 unchanged, cycle 1-9)
        color_map = np.arange(10, dtype=np.int64)

        # Cycle colors 1-9 by offset
        for i in range(1, 10):
            color_map[i] = ((i - 1 + cycle_offset) % 9) + 1

        # Apply mapping
        return color_map[grid]

    @staticmethod
    def augment_task(task_data: dict, flip_type: str, rotation: int, color_cycle: int, task_suffix: str) -> dict:
        """Apply consistent augmentation to an entire task.

        Args:
            task_data: Original task with 'train' and 'test' lists
            flip_type: 'horizontal', 'vertical', or 'none'
            rotation: 0, 90, 180, or 270 degrees
            color_cycle: 0-8 color cycle offset
            task_suffix: Suffix to add to task ID

        Returns:
            Augmented task data
        """
        augmented_task = {
            'train': [],
            'test': []
        }

        # Apply same augmentation to all examples
        for split in ['train', 'test']:
            for example in task_data[split]:
                # Augment input grid
                input_grid = example['input']
                input_grid = TaskAugmentation.apply_flip_augmentation(input_grid, flip_type)
                input_grid = TaskAugmentation.apply_rotation_augmentation(input_grid, rotation)
                input_grid = TaskAugmentation.apply_color_cycle_augmentation(input_grid, color_cycle)

                # Augment output grid
                output_grid = example['output']
                output_grid = TaskAugmentation.apply_flip_augmentation(output_grid, flip_type)
                output_grid = TaskAugmentation.apply_rotation_augmentation(output_grid, rotation)
                output_grid = TaskAugmentation.apply_color_cycle_augmentation(output_grid, color_cycle)

                augmented_task[split].append({
                    'input': input_grid,
                    'output': output_grid
                })

        return augmented_task


class GridAugmentation:
    """Simple augmentation for ARC grids: rotations and reflections."""

    @staticmethod
    def rotate_90(input_grid: torch.Tensor, output_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rotate both grids 90 degrees clockwise."""
        # torch.rot90 rotates counter-clockwise, so use k=-1 for clockwise
        input_rot = torch.rot90(input_grid, k=-1, dims=(-2, -1))
        output_rot = torch.rot90(output_grid, k=-1, dims=(-2, -1))
        return input_rot, output_rot

    @staticmethod
    def flip_horizontal(input_grid: torch.Tensor, output_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flip both grids horizontally."""
        input_flip = torch.flip(input_grid, dims=(-1,))
        output_flip = torch.flip(output_grid, dims=(-1,))
        return input_flip, output_flip

    @staticmethod
    def flip_vertical(input_grid: torch.Tensor, output_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flip both grids vertically."""
        input_flip = torch.flip(input_grid, dims=(-2,))
        output_flip = torch.flip(output_grid, dims=(-2,))
        return input_flip, output_flip