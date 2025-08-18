import pytest
from llm_python.utils.transduction import detect_transduction_augmentation
from llm_python.utils.arc_tester import ArcTester


class TestTransductionDetectionAugmentation:
    """Test the augmentation-based transduction detector with table-driven tests."""
    
    def setup_method(self):
        """Setup ArcTester for each test method."""
        self.arc_tester = ArcTester(timeout=2)
    
    def teardown_method(self):
        """Cleanup ArcTester after each test method."""
        ArcTester.cleanup_executor()
    
    # Test data tasks
    SIMPLE_TASK = {
        'train': [
            {
                'input': [[0, 1], [2, 3]],
                'output': [[1, 0], [3, 2]]  # horizontal flip pattern
            }
        ],
        'test': [
            {
                'input': [[4, 5], [6, 7]],
                'output': None
            }
        ]
    }
    
    MULTI_EXAMPLE_TASK = {
        'train': [
            {
                'input': [[0, 1], [2, 3]],
                'output': [[1, 0], [3, 2]]  # horizontal flip pattern
            },
            {
                'input': [[4, 5, 6], [7, 8, 9]],
                'output': [[6, 5, 4], [9, 8, 7]]  # horizontal flip pattern
            }
        ],
        'test': [
            {
                'input': [[1, 2], [3, 4]],
                'output': None
            }
        ]
    }
    
    COLOR_TASK = {
        'train': [
            {
                'input': [[0, 1, 2], [3, 4, 5]],
                'output': [[1, 2, 3], [4, 5, 6]]  # color +1 pattern
            },
            {
                'input': [[6, 7, 8], [9, 0, 1]],
                'output': [[7, 8, 9], [0, 1, 2]]  # color +1 pattern
            }
        ],
        'test': [
            {
                'input': [[2, 3, 4], [5, 6, 7]],
                'output': None
            }
        ]
    }
    
    VERTICAL_FLIP_TASK = {
        'train': [
            {
                'input': [[1, 2], [3, 4]],
                'output': [[3, 4], [1, 2]]  # vertical flip
            }
        ],
        'test': [
            {
                'input': [[5, 6], [7, 8]],
                'output': None
            }
        ]
    }
    
    IDENTITY_TASK = {
        'train': [
            {
                'input': [[1, 2], [3, 4]],
                'output': [[1, 2], [3, 4]]  # identity
            }
        ],
        'test': [
            {
                'input': [[5, 6], [7, 8]],
                'output': None
            }
        ]
    }
    
    # Test cases: (test_name, program_code, task, expected_transductive, expected_reason_contains)
    TEST_CASES = [
        # === NON-TRANSDUCTIVE PROGRAMS (Pattern-based) ===
        (
            "horizontal_flip_not_transductive",
            '''
def transform(grid):
    return [list(reversed(row)) for row in grid]
            ''',
            SIMPLE_TASK,
            False,
            "horizontal_flip"
        ),
        (
            "vertical_flip_not_transductive",
            '''
def transform(grid):
    return list(reversed(grid))
            ''',
            VERTICAL_FLIP_TASK,
            False,
            "vertical_flip"
        ),
        (
            "color_increment_not_transductive",
            '''
def transform(grid):
    return [[(cell + 1) % 10 for cell in row] for row in grid]
            ''',
            COLOR_TASK,
            False,
            "color_rotation"
        ),
        (
            "identity_not_transductive",
            '''
def transform(grid):
    return [row[:] for row in grid]
            ''',
            IDENTITY_TASK,
            False,
            "invariant"
        ),
        (
            "complex_pattern_not_transductive",
            '''
def transform(grid):
    # Transpose the grid
    return [[grid[j][i] for j in range(len(grid))] for i in range(len(grid[0]))]
            ''',
            {
                'train': [
                    {
                        'input': [[1, 2], [3, 4]],
                        'output': [[1, 3], [2, 4]]  # transpose
                    }
                ],
                'test': [
                    {
                        'input': [[5, 6], [7, 8]],
                        'output': None
                    }
                ]
            },
            False,
            "invariant"
        ),
        
        # === TRANSDUCTIVE PROGRAMS (Hardcoded/Memorized) ===
        (
            "completely_hardcoded_transductive",
            '''
def transform(grid):
    # Always return the same hardcoded output regardless of input
    return [[1, 0], [3, 2]]
            ''',
            SIMPLE_TASK,
            True,
            "not invariant to any augmentation"
        ),
        (
            "input_size_based_transductive",
            '''
def transform(grid):
    rows, cols = len(grid), len(grid[0]) if grid else 0
    
    # Use grid size to determine specific hardcoded output
    if rows == 2 and cols == 2:
        return [[1, 0], [3, 2]]
    else:
        return [[9, 9], [9, 9]]
            ''',
            SIMPLE_TASK,
            True,
            "not invariant to any augmentation"
        ),
        (
            "specific_cell_matching_transductive",
            '''
def transform(grid):
    # Check specific cell values to determine output
    if len(grid) > 0 and len(grid[0]) > 0:
        if grid[0][0] == 0:  # top-left cell is 0
            return [[1, 0], [3, 2]]
        else:
            return [[9, 9], [9, 9]]
    else:
        return [[0]]
            ''',
            SIMPLE_TASK,
            True,
            "not invariant to any augmentation"
        ),
        (
            "line_by_line_hardcoded_transductive",
            '''
def transform(grid):
    # Build output line by line with hardcoded values
    line1 = [1, 0]
    line2 = [3, 2]
    return [line1, line2]
            ''',
            SIMPLE_TASK,
            True,
            "not invariant to any augmentation"
        ),
        (
            "copy_and_overwrite_transductive",
            '''
def transform(grid):
    # Copy input and overwrite specific cells
    output = [row[:] for row in grid]
    if len(output) >= 2 and len(output[0]) >= 2:
        output[0][0] = 1  # overwrite to match expected
        output[0][1] = 0
        output[1][0] = 3
        output[1][1] = 2
    return output
            ''',
            SIMPLE_TASK,
            True,
            "not invariant to any augmentation"
        ),
        (
            "input_pattern_matching_transductive",
            '''
def transform(grid):
    # Hardcoded input-output mappings
    if str(grid) == str([[0, 1], [2, 3]]):
        return [[1, 0], [3, 2]]
    else:
        return [[0, 0], [0, 0]]
            ''',
            SIMPLE_TASK,
            True,
            "not invariant to any augmentation"
        ),
        (
            "multi_example_hardcoded_transductive",
            '''
def transform(grid):
    # Multiple hardcoded mappings
    if len(grid) == 2 and len(grid[0]) == 2:
        return [[1, 0], [3, 2]]
    elif len(grid) == 2 and len(grid[0]) == 3:
        return [[6, 5, 4], [9, 8, 7]]
    else:
        return [[9, 9], [9, 9]]
            ''',
            MULTI_EXAMPLE_TASK,
            True,
            "not invariant to any augmentation"
        ),
        
        # === ERROR CASES ===
        (
            "failing_program_transductive",
            '''
def transform(grid):
    raise ValueError("Program execution failed")
            ''',
            SIMPLE_TASK,
            True,
            "failed to produce any valid outputs"
        ),
        (
            "no_function_transductive",
            '''
x = 5
y = 10
result = x + y
            ''',
            SIMPLE_TASK,
            True,
            "failed to produce any valid outputs"
        ),
        (
            "wrong_function_name_not_transductive",
            '''
def process(grid):
    return [list(reversed(row)) for row in grid]
            ''',
            SIMPLE_TASK,
            False,
            "invariant to"
        ),
        (
            "invalid_return_type_transductive",
            '''
def transform(grid):
    return "not a grid"
            ''',
            SIMPLE_TASK,
            True,
            "failed to produce any valid outputs"
        ),
        
        # === EDGE CASES ===
        (
            "partially_correct_pattern_not_transductive",
            '''
def transform(grid):
    # This program implements a pattern that works for this specific case
    # but would generalize (it's doing horizontal flip)
    return [list(reversed(row)) for row in grid]
            ''',
            SIMPLE_TASK,
            False,
            "horizontal_flip"
        ),
        (
            "single_cell_pattern_not_transductive",
            '''
def transform(grid):
    # Add 1 to each cell (works for single cells and multi-cell grids)
    return [[(cell + 1) % 10 for cell in row] for row in grid]
            ''',
            {
                'train': [
                    {
                        'input': [[0]],
                        'output': [[1]]
                    },
                    {
                        'input': [[2]],
                        'output': [[3]]
                    }
                ],
                'test': [
                    {
                        'input': [[5]],
                        'output': None
                    }
                ]
            },
            False,
            "color_rotation"
        ),
    ]
    
    @pytest.mark.parametrize("test_name,program_code,task,expected_transductive,expected_reason_contains", TEST_CASES)
    def test_transduction_detection(self, test_name, program_code, task, expected_transductive, expected_reason_contains):
        """Test transduction detection with various program types."""
        is_transductive, reason = detect_transduction_augmentation(
            program_code.strip(), 
            task, 
            self.arc_tester
        )
        
        # Assert the expected transductive result
        assert is_transductive == expected_transductive, f"Test {test_name}: Expected transductive={expected_transductive}, got {is_transductive}. Reason: {reason}"
        
        # Assert the reason contains expected text
        assert expected_reason_contains.lower() in reason.lower(), f"Test {test_name}: Expected reason to contain '{expected_reason_contains}', got: {reason}"
    
    def test_debug_output(self):
        """Test that debug output works correctly."""
        program = '''
def transform(grid):
    return [list(reversed(row)) for row in grid]
        '''
        
        # Test with debug=True (should print debug info)
        is_transductive, reason = detect_transduction_augmentation(
            program, 
            self.SIMPLE_TASK, 
            self.arc_tester, 
            debug=True
        )
        
        assert not is_transductive
        assert "horizontal_flip" in reason
    
    def test_empty_task_handling(self):
        """Test handling of tasks with no training examples."""
        empty_task = {
            'train': [],
            'test': [
                {
                    'input': [[1, 2], [3, 4]],
                    'output': None
                }
            ]
        }
        
        program = '''
def transform(grid):
    return [list(reversed(row)) for row in grid]
        '''
        
        is_transductive, reason = detect_transduction_augmentation(
            program, 
            empty_task, 
            self.arc_tester
        )
        
        # Should not be transductive because there are no training outputs to compare
        assert not is_transductive
        assert "no training outputs" in reason.lower()
    
    def test_augmentation_failure_resilience(self):
        """Test that the detector handles augmentation failures gracefully."""
        # This program should work on normal inputs but might fail on some augmented inputs
        program = '''
def transform(grid):
    # This will work normally but might have issues with certain augmented inputs
    return [list(reversed(row)) for row in grid]
        '''
        
        # Use a task with values that could cause issues after augmentation
        edge_case_task = {
            'train': [
                {
                    'input': [[8, 9], [0, 1]],  # values near the boundary for color rotation
                    'output': [[9, 8], [1, 0]]
                }
            ],
            'test': [
                {
                    'input': [[2, 3], [4, 5]],
                    'output': None
                }
            ]
        }
        
        is_transductive, reason = detect_transduction_augmentation(
            program, 
            edge_case_task, 
            self.arc_tester
        )
        
        # Should still be able to determine invariance based on successful augmentations
        assert isinstance(is_transductive, bool)
        assert isinstance(reason, str)
        assert len(reason) > 0
