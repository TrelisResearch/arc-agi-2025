#!/usr/bin/env python3

import pytest
from llm_python.utils.program_filters import (
    has_multi_color_ground_truth,
    is_single_color_grid,
    is_pass_through_program,
    has_single_color_predictions_with_multi_color_truth,
    should_filter_program,
    filter_programs
)


class TestHasMultiColorGroundTruth:
    """Test the has_multi_color_ground_truth function"""

    def test_multi_color_grid(self):
        """Test grid with multiple non-black colors"""
        grid = [[1, 2], [3, 0]]
        assert has_multi_color_ground_truth(grid)

    def test_single_non_black_color_grid(self):
        """Test grid with only one non-black color"""
        grid = [[1, 1], [0, 1]]
        assert not has_multi_color_ground_truth(grid)

    def test_all_black_grid(self):
        """Test grid with only black/0 colors"""
        grid = [[0, 0], [0, 0]]
        assert not has_multi_color_ground_truth(grid)

    def test_empty_grid(self):
        """Test empty or invalid grids"""
        assert not has_multi_color_ground_truth([])
        assert not has_multi_color_ground_truth(None)
        assert not has_multi_color_ground_truth("invalid")

    def test_numpy_array_input(self):
        """Test with numpy array input"""
        import numpy as np
        grid = np.array([[1, 2], [3, 0]])
        assert has_multi_color_ground_truth(grid)


class TestIsSingleColorGrid:
    """Test the is_single_color_grid function"""

    def test_single_color_grid(self):
        """Test grid with only one color"""
        grid = [[1, 1], [1, 1]]
        assert is_single_color_grid(grid)

    def test_multi_color_grid(self):
        """Test grid with multiple colors"""
        grid = [[1, 2], [1, 1]]
        assert not is_single_color_grid(grid)

    def test_all_black_grid(self):
        """Test grid with only black/0 colors"""
        grid = [[0, 0], [0, 0]]
        assert is_single_color_grid(grid)

    def test_empty_grid(self):
        """Test empty or invalid grids"""
        assert is_single_color_grid([])
        assert is_single_color_grid(None)
        assert is_single_color_grid("invalid")

    def test_numpy_array_input(self):
        """Test with numpy array input"""
        import numpy as np
        grid = np.array([[1, 1], [1, 1]])
        assert is_single_color_grid(grid)


class TestIsPassThroughProgram:
    """Test the is_pass_through_program function"""

    def test_pass_through_program(self):
        """Test program where all predicted outputs equal inputs"""
        program = {
            'predicted_train_output': [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]]
            ]
        }
        task_data = {
            'train': [
                {'input': [[1, 2], [3, 4]]},
                {'input': [[5, 6], [7, 8]]}
            ]
        }
        assert is_pass_through_program(program, task_data)

    def test_non_pass_through_program(self):
        """Test program where at least one predicted output differs from input"""
        program = {
            'predicted_train_output': [
                [[0, 1], [2, 3]],  # Different from input
                [[5, 6], [7, 8]]   # Same as input
            ]
        }
        task_data = {
            'train': [
                {'input': [[1, 2], [3, 4]]},
                {'input': [[5, 6], [7, 8]]}
            ]
        }
        assert not is_pass_through_program(program, task_data)

    def test_without_task_data(self):
        """Test that function returns False without task data"""
        program = {
            'predicted_train_output': [[[1, 1], [1, 1]]]
        }
        assert not is_pass_through_program(program, None)
        assert not is_pass_through_program(program, {})

    def test_missing_predicted_output(self):
        """Test program without predicted train output"""
        program = {}
        task_data = {'train': [{'input': [[1, 2], [3, 4]]}]}
        assert not is_pass_through_program(program, task_data)


class TestHasSingleColorPredictionsWithMultiColorTruth:
    """Test the has_single_color_predictions_with_multi_color_truth function"""

    def test_single_color_predictions_multi_color_truth(self):
        """Test single-color predictions with multi-colored ground truth"""
        program = {
            'predicted_train_output': [
                [[1, 1], [1, 1]],  # Single color
                [[2, 2], [2, 2]]   # Single color
            ]
        }
        task_data = {
            'train': [
                {'output': [[1, 2], [3, 4]]},  # Multi-color ground truth
                {'output': [[5, 6], [7, 8]]}   # Multi-color ground truth
            ]
        }
        assert has_single_color_predictions_with_multi_color_truth(program, task_data)

    def test_single_color_predictions_single_color_truth(self):
        """Test single-color predictions with single-colored ground truth"""
        program = {
            'predicted_train_output': [
                [[1, 1], [1, 1]],  # Single color
                [[2, 2], [2, 2]]   # Single color
            ]
        }
        task_data = {
            'train': [
                {'output': [[1, 1], [1, 1]]},  # Single-color ground truth
                {'output': [[2, 2], [2, 2]]}   # Single-color ground truth
            ]
        }
        assert not has_single_color_predictions_with_multi_color_truth(program, task_data)

    def test_multi_color_predictions(self):
        """Test multi-color predictions (should not be filtered)"""
        program = {
            'predicted_train_output': [
                [[1, 2], [3, 4]],  # Multi-color
                [[5, 6], [7, 8]]   # Multi-color
            ]
        }
        task_data = {
            'train': [
                {'output': [[1, 2], [3, 4]]},  # Multi-color ground truth
                {'output': [[5, 6], [7, 8]]}   # Multi-color ground truth
            ]
        }
        assert not has_single_color_predictions_with_multi_color_truth(program, task_data)


class TestShouldFilterProgram:
    """Test the should_filter_program function"""

    def test_filter_transductive_program(self):
        """Test that transductive programs are filtered"""
        program = {
            'is_transductive': True,
            'correct_train_input': [True, False]
        }
        assert should_filter_program(program)

    def test_filter_perfect_program(self):
        """Test that 100% correct programs are filtered"""
        program = {
            'is_transductive': False,
            'correct_train_input': [True, True, True]
        }
        assert should_filter_program(program)

    def test_filter_pass_through_program(self):
        """Test that pass-through programs are filtered"""
        program = {
            'is_transductive': False,
            'correct_train_input': [False, False],
            'predicted_train_output': [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]]
            ]
        }
        task_data = {
            'train': [
                {'input': [[1, 2], [3, 4]], 'output': [[0, 1], [2, 3]]},
                {'input': [[5, 6], [7, 8]], 'output': [[4, 5], [6, 7]]}
            ]
        }
        assert should_filter_program(program, task_data)

    def test_filter_single_color_predictions_with_multi_color_truth(self):
        """Test that single-color predictions with multi-color truth are filtered"""
        program = {
            'is_transductive': False,
            'correct_train_input': [False, False],
            'predicted_train_output': [
                [[1, 1], [1, 1]],  # Single color
                [[2, 2], [2, 2]]   # Single color
            ]
        }
        task_data = {
            'train': [
                {'input': [[0, 1], [2, 3]], 'output': [[1, 2], [3, 4]]},  # Multi-color ground truth
                {'input': [[0, 0], [0, 0]], 'output': [[5, 6], [7, 8]]}   # Multi-color ground truth
            ]
        }
        assert should_filter_program(program, task_data)

    def test_keep_valid_program(self):
        """Test that valid programs are not filtered"""
        program = {
            'is_transductive': False,
            'correct_train_input': [True, False, True],  # Partially correct
            'predicted_train_output': [
                [[1, 2], [3, 4]],  # Multi-color
                [[5, 6], [7, 8]]   # Multi-color
            ]
        }
        task_data = {
            'train': [
                {'input': [[0, 1], [2, 3]], 'output': [[1, 2], [3, 4]]},
                {'input': [[4, 5], [6, 7]], 'output': [[5, 6], [7, 8]]}
            ]
        }
        assert not should_filter_program(program, task_data)

    def test_numpy_array_correctness(self):
        """Test with numpy array correctness data"""
        import numpy as np
        program = {
            'is_transductive': False,
            'correct_train_input': np.array([True, False, True])
        }
        assert not should_filter_program(program)

    def test_single_boolean_correctness(self):
        """Test with single boolean correctness value"""
        # Perfect program (single True)
        program_perfect = {
            'is_transductive': False,
            'correct_train_input': True
        }
        assert should_filter_program(program_perfect)

        # Imperfect program (single False)
        program_imperfect = {
            'is_transductive': False,
            'correct_train_input': False
        }
        assert not should_filter_program(program_imperfect)


class TestFilterPrograms:
    """Test the filter_programs function"""

    def test_filter_multiple_programs(self):
        """Test filtering a list of programs"""
        programs = [
            {  # Should be kept (partially correct)
                'is_transductive': False,
                'correct_train_input': [True, False]
            },
            {  # Should be filtered (transductive)
                'is_transductive': True,
                'correct_train_input': [True, False]
            },
            {  # Should be filtered (perfect)
                'is_transductive': False,
                'correct_train_input': [True, True]
            },
            {  # Should be kept (zero correct)
                'is_transductive': False,
                'correct_train_input': [False, False]
            }
        ]

        filtered = filter_programs(programs)
        assert len(filtered) == 2
        assert not filtered[0]['is_transductive']
        assert filtered[0]['correct_train_input'] == [True, False]
        assert not filtered[1]['is_transductive']
        assert filtered[1]['correct_train_input'] == [False, False]

    def test_filter_empty_list(self):
        """Test filtering an empty list"""
        assert filter_programs([]) == []

    def test_filter_with_task_data(self):
        """Test filtering with task data for additional checks"""
        programs = [
            {  # Should be filtered (pass-through)
                'is_transductive': False,
                'correct_train_input': [False, False],
                'predicted_train_output': [
                    [[1, 2], [3, 4]],
                    [[5, 6], [7, 8]]
                ]
            },
            {  # Should be kept (non-pass-through)
                'is_transductive': False,
                'correct_train_input': [False, False],
                'predicted_train_output': [
                    [[0, 1], [2, 3]],  # Different from input
                    [[5, 6], [7, 8]]   # Same as input
                ]
            }
        ]

        task_data = {
            'train': [
                {'input': [[1, 2], [3, 4]], 'output': [[0, 1], [2, 3]]},
                {'input': [[5, 6], [7, 8]], 'output': [[4, 5], [6, 7]]}
            ]
        }

        filtered = filter_programs(programs, task_data)
        assert len(filtered) == 1
        assert filtered[0]['predicted_train_output'][0] == [[0, 1], [2, 3]]


class TestParquetFilteringIntegration:
    """Test integration with parquet data loading for fine-tuning"""

    def test_parquet_filtering_integration(self):
        """Test that the filtering works with parquet data loading utilities"""
        from llm_python.utils.program_filters import filter_programs

        # Create sample program data
        programs = [
            {  # Should be kept (partially correct, non-transductive)
                'task_id': 'task1',
                'is_transductive': False,
                'correct_train_input': [True, False],
                'predicted_train_output': [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                'code': 'def transform(grid): return grid'
            },
            {  # Should be filtered (transductive)
                'task_id': 'task2',
                'is_transductive': True,
                'correct_train_input': [True, False],
                'predicted_train_output': [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                'code': 'def transform(grid): return grid'
            },
            {  # Should be filtered (perfect)
                'task_id': 'task3',
                'is_transductive': False,
                'correct_train_input': [True, True],
                'predicted_train_output': [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                'code': 'def transform(grid): return grid'
            }
        ]

        filtered = filter_programs(programs)
        assert len(filtered) == 1
        assert filtered[0]['task_id'] == 'task1'