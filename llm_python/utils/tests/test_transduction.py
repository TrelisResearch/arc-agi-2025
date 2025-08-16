import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import pytest
from utils.transduction import detect_transduction


class TestTransductionDetection:
    """Tests for transductive cheating detection"""
    
    def test_non_transductive_program(self):
        """Test that a legitimate program passes transduction detection"""
        # Real program from the log file that correctly solves the task
        legitimate_program = """def transform(grid: list[list[int]]) -> list[list[int]]:
    rotated_grid = [row[::-1] for row in grid[::-1]]
    return rotated_grid"""
        
        # Task data from the log file (6150a2bd)
        task_data = {
            "train": [
                {
                    "input": [[3, 3, 8], [3, 7, 0], [5, 0, 0]],
                    "output": [[0, 0, 5], [0, 7, 3], [8, 3, 3]]
                },
                {
                    "input": [[5, 5, 2], [1, 0, 0], [0, 0, 0]],
                    "output": [[0, 0, 0], [0, 0, 1], [2, 5, 5]]
                }
            ],
            "test": [
                {
                    "input": [[6, 3, 5], [6, 8, 0], [4, 0, 0]],
                    "output": [[0, 0, 4], [0, 8, 6], [5, 3, 6]]
                }
            ]
        }
        
        is_cheat, reason, _, _ = detect_transduction(legitimate_program, task_data)
        
        # Should NOT be detected as cheating
        assert is_cheat is False, f"Legitimate program incorrectly detected as cheating: {reason}"
        assert reason == "", f"Legitimate program should have empty reason, got: {reason}"
    
    def test_transductive_program_hardcoded_output(self):
        """Test that a program with hardcoded outputs is detected as transductive"""
        # Transductive program that hardcodes the exact output values
        transductive_program = """def transform(grid: list[list[int]]) -> list[list[int]]:
    # This is cheating - hardcoding the exact outputs
    if grid == [[3, 3, 8], [3, 7, 0], [5, 0, 0]]:
        return [[0, 0, 5], [0, 7, 3], [8, 3, 3]]
    elif grid == [[5, 5, 2], [1, 0, 0], [0, 0, 0]]:
        return [[0, 0, 0], [0, 0, 1], [2, 5, 5]]
    else:
        return [[0, 0, 4], [0, 8, 6], [5, 3, 6]]"""
        
        task_data = {
            "train": [
                {
                    "input": [[3, 3, 8], [3, 7, 0], [5, 0, 0]],
                    "output": [[0, 0, 5], [0, 7, 3], [8, 3, 3]]
                },
                {
                    "input": [[5, 5, 2], [1, 0, 0], [0, 0, 0]],
                    "output": [[0, 0, 0], [0, 0, 1], [2, 5, 5]]
                }
            ],
            "test": [
                {
                    "input": [[6, 3, 5], [6, 8, 0], [4, 0, 0]],
                    "output": [[0, 0, 4], [0, 8, 6], [5, 3, 6]]
                }
            ]
        }
        
        is_cheat, reason, _, _ = detect_transduction(transductive_program, task_data)
        
        # Should be detected as cheating
        assert is_cheat is True, "Transductive program with hardcoded outputs should be detected as cheating"
        assert "hardcoded" in reason.lower(), f"Reason should mention hardcoding, got: {reason}"
    
    def test_transductive_program_long_line(self):
        """Test that a program with very long lines is detected as transductive"""
        # Transductive program with a very long line (likely hardcoded array)
        transductive_program = """def transform(grid: list[list[int]]) -> list[list[int]]:
    # This line is extremely long and likely contains hardcoded values
    hardcoded_outputs = [[[0, 0, 5], [0, 7, 3], [8, 3, 3]], [[0, 0, 0], [0, 0, 1], [2, 5, 5]], [[0, 0, 4], [0, 8, 6], [5, 3, 6]], [[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[9, 8, 7], [6, 5, 4], [3, 2, 1]], [[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]], [[7, 7, 7], [8, 8, 8], [9, 9, 9]], [[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[8, 7, 6], [5, 4, 3], [2, 1, 0]]]
    return hardcoded_outputs[0]"""
        
        task_data = {
            "train": [
                {
                    "input": [[3, 3, 8], [3, 7, 0], [5, 0, 0]],
                    "output": [[0, 0, 5], [0, 7, 3], [8, 3, 3]]
                }
            ],
            "test": [
                {
                    "input": [[6, 3, 5], [6, 8, 0], [4, 0, 0]],
                    "output": [[0, 0, 4], [0, 8, 6], [5, 3, 6]]
                }
            ]
        }
        
        is_cheat, reason, _, _ = detect_transduction(transductive_program, task_data)
        
        # Should be detected as cheating due to long line
        assert is_cheat is True, "Transductive program with long line should be detected as cheating"
        assert "200 characters" in reason or "long line" in reason.lower(), f"Reason should mention line length, got: {reason}"
    
    def test_another_legitimate_program(self):
        """Test another legitimate program that implements a different transformation"""
        # A different legitimate program that implements a valid transformation
        legitimate_program2 = """def transform(grid: list[list[int]]) -> list[list[int]]:
    # Flip the grid horizontally and vertically
    rows = len(grid)
    cols = len(grid[0])
    result = [[0 for _ in range(cols)] for _ in range(rows)]
    
    for i in range(rows):
        for j in range(cols):
            result[rows-1-i][cols-1-j] = grid[i][j]
    
    return result"""
        
        task_data = {
            "train": [
                {
                    "input": [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    "output": [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
                }
            ],
            "test": [
                {
                    "input": [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
                    "output": [[8, 7, 6], [5, 4, 3], [2, 1, 0]]
                }
            ]
        }
        
        is_cheat, reason, _, _ = detect_transduction(legitimate_program2, task_data)
        
        # Should NOT be detected as cheating
        assert is_cheat is False, f"Legitimate program 2 incorrectly detected as cheating: {reason}"
        assert reason == "", f"Legitimate program 2 should have empty reason, got: {reason}"
    
    def test_edge_case_empty_task_data(self):
        """Test edge case with empty task data"""
        program = "def transform(grid): return grid"
        empty_task_data = {"train": [], "test": []}
        
        is_cheat, reason, _, _ = detect_transduction(program, empty_task_data)
        
        # Should not be detected as cheating when no outputs to check against
        assert is_cheat is False, "Empty task data should not trigger cheating detection"
        assert reason == "", "Empty task data should have empty reason" 