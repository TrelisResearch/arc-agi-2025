from abc import ABC, abstractmethod
from copy import deepcopy
from llm_python.utils.task_loader import TaskData, Grid


class GridAugmentation(ABC):
    """Base class for grid augmentations with forward and backward transformations."""
    
    @abstractmethod
    def forward_grid(self, grid: Grid) -> Grid:
        """Apply the augmentation to a single grid.
        
        Args:
            grid: 2D list of integers representing the grid
            
        Returns:
            Augmented grid
        """
        pass
    
    @abstractmethod
    def backward_grid(self, grid: Grid) -> Grid:
        """Apply the reverse augmentation to a single grid.
        
        Args:
            grid: 2D list of integers representing the augmented grid
            
        Returns:
            Original grid (reverse of forward_grid)
        """
        pass
    
    def forward_task(self, task_data: TaskData) -> TaskData:
        """Apply the augmentation to all grids in a TaskData.
        
        Args:
            task_data: Complete task data structure
            
        Returns:
            Augmented task data
        """
        augmented_task = deepcopy(task_data)
        
        # Apply to all training examples
        for example in augmented_task['train']:
            example['input'] = self.forward_grid(example['input'])
            if example['output'] is not None:
                example['output'] = self.forward_grid(example['output'])
        
        # Apply to all test examples
        for example in augmented_task['test']:
            example['input'] = self.forward_grid(example['input'])
            if example['output'] is not None:
                example['output'] = self.forward_grid(example['output'])
        
        return augmented_task
    
    def backward_task(self, task_data: TaskData) -> TaskData:
        """Apply the reverse augmentation to all grids in a TaskData.
        
        Args:
            task_data: Augmented task data structure
            
        Returns:
            Original task data (reverse of forward_task)
        """
        original_task = deepcopy(task_data)
        
        # Apply backward to all training examples
        for example in original_task['train']:
            example['input'] = self.backward_grid(example['input'])
            if example['output'] is not None:
                example['output'] = self.backward_grid(example['output'])
        
        # Apply backward to all test examples
        for example in original_task['test']:
            example['input'] = self.backward_grid(example['input'])
            if example['output'] is not None:
                example['output'] = self.backward_grid(example['output'])
        
        return original_task


class VerticalFlipAugmentation(GridAugmentation):
    """Vertical flip augmentation - flips the grid vertically (top-bottom)."""
    
    def forward_grid(self, grid: Grid) -> Grid:
        """Flip the grid vertically."""
        return list(reversed(grid))
    
    def backward_grid(self, grid: Grid) -> Grid:
        """Flip the grid vertically (same as forward for vertical flip)."""
        return list(reversed(grid))


class HorizontalFlipAugmentation(GridAugmentation):
    """Horizontal flip augmentation - flips the grid horizontally (left-right)."""
    
    def forward_grid(self, grid: Grid) -> Grid:
        """Flip the grid horizontally."""
        return [list(reversed(row)) for row in grid]
    
    def backward_grid(self, grid: Grid) -> Grid:
        """Flip the grid horizontally (same as forward for horizontal flip)."""
        return [list(reversed(row)) for row in grid]


class ColorRotationAugmentation(GridAugmentation):
    """Color rotation augmentation - rotates color values by a fixed offset.
    
    Colors are integers 0-9. Rotation is done with modulo 10.
    """
    
    def __init__(self, offset: int = 1):
        """Initialize with a color rotation offset.
        
        Args:
            offset: Number of positions to rotate colors (default: 1)
        """
        self.offset = offset % 10  # Ensure offset is in valid range
    
    def forward_grid(self, grid: Grid) -> Grid:
        """Rotate all color values by the offset."""
        return [[(cell + self.offset) % 10 for cell in row] for row in grid]
    
    def backward_grid(self, grid: Grid) -> Grid:
        """Rotate all color values by the negative offset."""
        return [[(cell - self.offset) % 10 for cell in row] for row in grid]
