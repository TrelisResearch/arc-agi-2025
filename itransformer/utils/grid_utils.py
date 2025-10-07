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
        pad_value: Value to pad with (10/PAD for invalid regions)

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
    Create masks for a batch of grids (vectorized, no Python loops).

    Args:
        heights: Heights for each item in batch [batch_size]
        widths: Widths for each item in batch [batch_size]
        max_size: Maximum grid size

    Returns:
        Masks [batch_size, max_size, max_size]
    """
    batch_size = heights.shape[0]
    device = heights.device

    # Create coordinate grids: [max_size, max_size]
    row_indices = torch.arange(max_size, device=device).view(1, max_size, 1)
    col_indices = torch.arange(max_size, device=device).view(1, 1, max_size)

    # Broadcast heights and widths: [batch_size, 1, 1]
    heights_expanded = heights.view(batch_size, 1, 1)
    widths_expanded = widths.view(batch_size, 1, 1)

    # Create masks: [batch_size, max_size, max_size]
    # True where row < height AND col < width
    masks = (row_indices < heights_expanded) & (col_indices < widths_expanded)

    return masks.float()


def clamp_outside_mask(grid: torch.Tensor, mask: torch.Tensor, pad_value: int = 10) -> torch.Tensor:
    """
    Clamp tokens outside the valid region (where mask=0) to pad_value.

    Args:
        grid: Grid tensor [..., H, W]
        mask: Mask tensor [..., H, W] with 1s for valid regions
        pad_value: Value to set outside valid region (10/PAD)

    Returns:
        Grid with tokens outside mask set to pad_value
    """
    return torch.where(mask.bool(), grid, 0 if pad_value == 0 else pad_value)


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
        tokens: Padded tokens [max_size, max_size] (padded with black/0)
        height: Original height
        width: Original width
    """
    height, width = grid.shape
    tokens = torch.from_numpy(grid.astype(np.int64))
    padded_tokens = pad_grid_to_size(tokens, max_size, pad_value=10)  # Use PAD/10 for padding
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
    def apply_d4_augmentation(grid: np.ndarray, d4_idx: int) -> np.ndarray:
        """Apply D4 dihedral group transformation to a grid.

        D4 (dihedral group of order 8) has 8 unique spatial transformations:
        0: identity
        1: rotate 90° clockwise
        2: rotate 180°
        3: rotate 270° clockwise (= 90° counter-clockwise)
        4: flip horizontal
        5: flip vertical
        6: flip main diagonal (transpose)
        7: flip anti-diagonal

        Args:
            grid: Input grid [H, W]
            d4_idx: D4 transformation index (0-7)

        Returns:
            Transformed grid [H, W]
        """
        if d4_idx == 0:  # identity
            return grid.copy()
        elif d4_idx == 1:  # rotate 90° clockwise
            return np.rot90(grid, k=-1)
        elif d4_idx == 2:  # rotate 180°
            return np.rot90(grid, k=2)
        elif d4_idx == 3:  # rotate 270° clockwise
            return np.rot90(grid, k=1)
        elif d4_idx == 4:  # flip horizontal
            return np.fliplr(grid)
        elif d4_idx == 5:  # flip vertical
            return np.flipud(grid)
        elif d4_idx == 6:  # flip main diagonal (transpose)
            return np.transpose(grid)
        elif d4_idx == 7:  # flip anti-diagonal
            return np.rot90(np.transpose(grid), k=2)
        else:
            raise ValueError(f"Invalid d4_idx: {d4_idx}. Must be 0-7.")

    @staticmethod
    def reverse_d4_augmentation(grid: np.ndarray, d4_idx: int) -> np.ndarray:
        """Reverse (invert) a D4 dihedral group transformation.

        D4 inverse mapping:
        0 (identity) → 0
        1 (rot90) → 3 (rot270)
        2 (rot180) → 2 (rot180, self-inverse)
        3 (rot270) → 1 (rot90)
        4 (flip_h) → 4 (self-inverse)
        5 (flip_v) → 5 (self-inverse)
        6 (flip_diag) → 6 (self-inverse)
        7 (flip_anti) → 7 (self-inverse)

        Args:
            grid: Input grid [H, W]
            d4_idx: D4 transformation index (0-7) to reverse

        Returns:
            Grid with inverse transformation applied [H, W]
        """
        # D4 inverse mapping
        inverse_map = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4, 5: 5, 6: 6, 7: 7}
        inverse_idx = inverse_map.get(d4_idx)
        if inverse_idx is None:
            raise ValueError(f"Invalid d4_idx: {d4_idx}. Must be 0-7.")
        return TaskAugmentation.apply_d4_augmentation(grid, inverse_idx)

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
    def reverse_color_cycle_augmentation(grid: np.ndarray, cycle_offset: int) -> np.ndarray:
        """Reverse color cycling augmentation.

        Args:
            grid: Input grid [H, W] with values 0-9
            cycle_offset: Original cycle offset (0-8) to reverse

        Returns:
            Grid with reversed color cycling [H, W]
        """
        if cycle_offset == 0:
            return grid.copy()

        # To reverse a cycle, apply the inverse cycle: (9 - offset)
        reverse_offset = 9 - cycle_offset
        return TaskAugmentation.apply_color_cycle_augmentation(grid, reverse_offset)

    @staticmethod
    def generate_all_d4_augmentations() -> List[Tuple[int, int]]:
        """Generate all 72 D4 × color augmentations (8 D4 × 9 colors).

        Returns:
            List of (d4_idx, color_cycle) tuples, with identity (0, 0) first
        """
        augmentations = []
        # Identity first
        augmentations.append((0, 0))
        # All other combinations
        for d4_idx in range(8):
            for color_cycle in range(9):
                if d4_idx == 0 and color_cycle == 0:
                    continue  # Skip identity, already added
                augmentations.append((d4_idx, color_cycle))
        return augmentations

    @staticmethod
    def deaugment_size_d4(height: int, width: int, d4_idx: int) -> Tuple[int, int]:
        """De-augment size prediction based on D4 transformation.

        D4 transformations that swap dimensions:
        - d4=1 (rot90): swaps
        - d4=3 (rot270): swaps
        - d4=6 (flip_diag/transpose): swaps
        - d4=7 (flip_anti): swaps

        Args:
            height: Predicted height
            width: Predicted width
            d4_idx: D4 transformation index (0-7)

        Returns:
            (de_augmented_height, de_augmented_width)
        """
        if d4_idx in [1, 3, 6, 7]:  # Transformations that swap dimensions
            return width, height
        else:
            return height, width

