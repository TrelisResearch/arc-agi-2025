from llm_python.augmentations import (
    VerticalFlipAugmentation,
    HorizontalFlipAugmentation,
    ColorRotationAugmentation
)


class TestVerticalFlipAugmentation:
    """Test vertical flip augmentation."""
    
    def test_forward_grid_simple(self):
        """Test vertical flip on a simple grid."""
        aug = VerticalFlipAugmentation()
        grid = [[1, 2], [3, 4]]
        expected = [[3, 4], [1, 2]]
        assert aug.forward_grid(grid) == expected
    
    def test_backward_grid_simple(self):
        """Test backward vertical flip on a simple grid."""
        aug = VerticalFlipAugmentation()
        grid = [[1, 2], [3, 4]]
        expected = [[3, 4], [1, 2]]
        assert aug.backward_grid(grid) == expected
    
    def test_forward_backward_identity(self):
        """Test that forward followed by backward returns original."""
        aug = VerticalFlipAugmentation()
        original = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = aug.backward_grid(aug.forward_grid(original))
        assert result == original
    
    def test_single_row(self):
        """Test vertical flip on single row."""
        aug = VerticalFlipAugmentation()
        grid = [[1, 2, 3]]
        expected = [[1, 2, 3]]
        assert aug.forward_grid(grid) == expected
    
    def test_single_column(self):
        """Test vertical flip on single column."""
        aug = VerticalFlipAugmentation()
        grid = [[1], [2], [3]]
        expected = [[3], [2], [1]]
        assert aug.forward_grid(grid) == expected
    
    def test_empty_grid(self):
        """Test vertical flip on empty grid."""
        aug = VerticalFlipAugmentation()
        grid = []
        expected = []
        assert aug.forward_grid(grid) == expected


class TestHorizontalFlipAugmentation:
    """Test horizontal flip augmentation."""
    
    def test_forward_grid_simple(self):
        """Test horizontal flip on a simple grid."""
        aug = HorizontalFlipAugmentation()
        grid = [[1, 2], [3, 4]]
        expected = [[2, 1], [4, 3]]
        assert aug.forward_grid(grid) == expected
    
    def test_backward_grid_simple(self):
        """Test backward horizontal flip on a simple grid."""
        aug = HorizontalFlipAugmentation()
        grid = [[1, 2], [3, 4]]
        expected = [[2, 1], [4, 3]]
        assert aug.backward_grid(grid) == expected
    
    def test_forward_backward_identity(self):
        """Test that forward followed by backward returns original."""
        aug = HorizontalFlipAugmentation()
        original = [[1, 2, 3], [4, 5, 6]]
        result = aug.backward_grid(aug.forward_grid(original))
        assert result == original
    
    def test_single_row(self):
        """Test horizontal flip on single row."""
        aug = HorizontalFlipAugmentation()
        grid = [[1, 2, 3]]
        expected = [[3, 2, 1]]
        assert aug.forward_grid(grid) == expected
    
    def test_single_column(self):
        """Test horizontal flip on single column."""
        aug = HorizontalFlipAugmentation()
        grid = [[1], [2], [3]]
        expected = [[1], [2], [3]]
        assert aug.forward_grid(grid) == expected
    
    def test_empty_grid(self):
        """Test horizontal flip on empty grid."""
        aug = HorizontalFlipAugmentation()
        grid = []
        expected = []
        assert aug.forward_grid(grid) == expected


class TestColorRotationAugmentation:
    """Test color rotation augmentation."""
    
    def test_forward_grid_simple(self):
        """Test color rotation on a simple grid."""
        aug = ColorRotationAugmentation(offset=1)
        grid = [[0, 1], [2, 9]]
        expected = [[1, 2], [3, 0]]  # 9+1 = 10 % 10 = 0
        assert aug.forward_grid(grid) == expected
    
    def test_backward_grid_simple(self):
        """Test backward color rotation on a simple grid."""
        aug = ColorRotationAugmentation(offset=1)
        grid = [[1, 2], [3, 0]]
        expected = [[0, 1], [2, 9]]  # 0-1 = -1 % 10 = 9
        assert aug.backward_grid(grid) == expected
    
    def test_forward_backward_identity(self):
        """Test that forward followed by backward returns original."""
        aug = ColorRotationAugmentation(offset=3)
        original = [[0, 1, 2], [7, 8, 9]]
        result = aug.backward_grid(aug.forward_grid(original))
        assert result == original
    
    def test_offset_zero(self):
        """Test color rotation with zero offset."""
        aug = ColorRotationAugmentation(offset=0)
        grid = [[0, 5, 9]]
        expected = [[0, 5, 9]]
        assert aug.forward_grid(grid) == expected
        assert aug.backward_grid(grid) == expected
    
    def test_offset_large(self):
        """Test color rotation with large offset (should wrap)."""
        aug = ColorRotationAugmentation(offset=15)  # 15 % 10 = 5
        grid = [[0, 1]]
        expected = [[5, 6]]
        assert aug.forward_grid(grid) == expected
    
    def test_offset_negative(self):
        """Test color rotation with negative offset."""
        aug = ColorRotationAugmentation(offset=-3)  # -3 % 10 = 7
        grid = [[0, 1]]
        expected = [[7, 8]]
        assert aug.forward_grid(grid) == expected
    
    def test_all_colors(self):
        """Test color rotation on all possible color values."""
        aug = ColorRotationAugmentation(offset=1)
        grid = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        expected = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]]
        assert aug.forward_grid(grid) == expected
    
    def test_empty_grid(self):
        """Test color rotation on empty grid."""
        aug = ColorRotationAugmentation(offset=5)
        grid = []
        expected = []
        assert aug.forward_grid(grid) == expected


class TestTaskAugmentations:
    """Test augmentations on full TaskData structures."""
    
    def create_test_task(self):
        """Create a simple test task."""
        return {
            'train': [
                {
                    'input': [[0, 1], [2, 3]],
                    'output': [[1, 0], [3, 2]]
                }
            ],
            'test': [
                {
                    'input': [[4, 5], [6, 7]],
                    'output': None
                }
            ]
        }
    
    def test_vertical_flip_task(self):
        """Test vertical flip on entire task."""
        aug = VerticalFlipAugmentation()
        task = self.create_test_task()
        
        result = aug.forward_task(task)
        
        # Check train example
        assert result['train'][0]['input'] == [[2, 3], [0, 1]]
        assert result['train'][0]['output'] == [[3, 2], [1, 0]]
        
        # Check test example
        assert result['test'][0]['input'] == [[6, 7], [4, 5]]
        assert result['test'][0]['output'] is None
    
    def test_horizontal_flip_task(self):
        """Test horizontal flip on entire task."""
        aug = HorizontalFlipAugmentation()
        task = self.create_test_task()
        
        result = aug.forward_task(task)
        
        # Check train example
        assert result['train'][0]['input'] == [[1, 0], [3, 2]]
        assert result['train'][0]['output'] == [[0, 1], [2, 3]]
        
        # Check test example
        assert result['test'][0]['input'] == [[5, 4], [7, 6]]
        assert result['test'][0]['output'] is None
    
    def test_color_rotation_task(self):
        """Test color rotation on entire task."""
        aug = ColorRotationAugmentation(offset=2)
        task = self.create_test_task()
        
        result = aug.forward_task(task)
        
        # Check train example
        assert result['train'][0]['input'] == [[2, 3], [4, 5]]
        assert result['train'][0]['output'] == [[3, 2], [5, 4]]
        
        # Check test example
        assert result['test'][0]['input'] == [[6, 7], [8, 9]]
        assert result['test'][0]['output'] is None
    
    def test_task_backward_identity(self):
        """Test that forward followed by backward returns original task."""
        aug = ColorRotationAugmentation(offset=7)
        original_task = self.create_test_task()
        
        augmented = aug.forward_task(original_task)
        restored = aug.backward_task(augmented)
        
        assert restored == original_task


class TestConvenienceFunctions:
    """Test convenience factory functions."""
    
    def test_vertical_flip_factory(self):
        """Test vertical flip factory function."""
        aug = VerticalFlipAugmentation()
        assert isinstance(aug, VerticalFlipAugmentation)
        
        grid = [[1, 2], [3, 4]]
        expected = [[3, 4], [1, 2]]
        assert aug.forward_grid(grid) == expected
    
    def test_horizontal_flip_factory(self):
        """Test horizontal flip factory function."""
        aug = HorizontalFlipAugmentation()
        assert isinstance(aug, HorizontalFlipAugmentation)
        
        grid = [[1, 2], [3, 4]]
        expected = [[2, 1], [4, 3]]
        assert aug.forward_grid(grid) == expected
    
    def test_color_rotation_factory_default(self):
        """Test color rotation factory function with default offset."""
        aug = ColorRotationAugmentation()
        assert isinstance(aug, ColorRotationAugmentation)
        assert aug.offset == 1
        
        grid = [[0, 9]]
        expected = [[1, 0]]
        assert aug.forward_grid(grid) == expected
    
    def test_color_rotation_factory_custom(self):
        """Test color rotation factory function with custom offset."""
        aug = ColorRotationAugmentation(offset=5)
        assert isinstance(aug, ColorRotationAugmentation)
        assert aug.offset == 5
        
        grid = [[0, 5]]
        expected = [[5, 0]]  # 5+5 = 10 % 10 = 0
        assert aug.forward_grid(grid) == expected


class TestAllAugmentations:
    """Test the available augmentations."""
    
    def test_all_augmentations_work(self):
        """Test that all augmentations work correctly."""
        test_grid = [[0, 1], [2, 3]]
        augmentations = [
            VerticalFlipAugmentation(),
            HorizontalFlipAugmentation(), 
            ColorRotationAugmentation()
        ]
        
        for aug in augmentations:
            # Should not raise any exceptions
            result = aug.forward_grid(test_grid)
            assert isinstance(result, list)
            assert len(result) == len(test_grid)
            if len(result) > 0:
                assert len(result[0]) == len(test_grid[0])
            
            # Test backward
            restored = aug.backward_grid(result)
            assert restored == test_grid


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_rows(self):
        """Test augmentations on grid with empty rows."""
        grid = [[], []]
        
        v_aug = VerticalFlipAugmentation()
        h_aug = HorizontalFlipAugmentation()
        c_aug = ColorRotationAugmentation()
        
        # Should not raise exceptions
        assert v_aug.forward_grid(grid) == [[], []]
        assert h_aug.forward_grid(grid) == [[], []]
        assert c_aug.forward_grid(grid) == [[], []]
    
    def test_irregular_grid(self):
        """Test augmentations on irregular grids (different row lengths)."""
        grid = [[1, 2, 3], [4, 5]]
        
        v_aug = VerticalFlipAugmentation()
        h_aug = HorizontalFlipAugmentation()
        c_aug = ColorRotationAugmentation()
        
        # Vertical flip should work
        assert v_aug.forward_grid(grid) == [[4, 5], [1, 2, 3]]
        
        # Horizontal flip should work
        assert h_aug.forward_grid(grid) == [[3, 2, 1], [5, 4]]
        
        # Color rotation should work
        assert c_aug.forward_grid(grid) == [[2, 3, 4], [5, 6]]
