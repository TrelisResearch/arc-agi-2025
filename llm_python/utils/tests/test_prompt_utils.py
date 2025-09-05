"""
Tests for prompt utility functions.
"""

import unittest
from unittest.mock import Mock

from llm_python.utils import (
    create_arc_prompt,
    extract_python_code,
    format_grid_for_prompt,
    get_grid_shape_string
)
from llm_python.utils.prompt_utils import generate_output_diff


class TestPromptUtils(unittest.TestCase):
    
    def test_format_grid_for_prompt(self):
        """Test grid formatting for prompts"""
        grid = [[1, 2], [3, 4]]
        expected = "[[1 2] [3 4]]"
        result = format_grid_for_prompt(grid)
        self.assertEqual(result, expected)
    
    def test_get_grid_shape_string(self):
        """Test grid shape string generation"""
        grid = [[1, 2, 3], [4, 5, 6]]
        expected = "3 by 2"
        result = get_grid_shape_string(grid)
        self.assertEqual(result, expected)
        
        # Test empty grid
        empty_grid = []
        result = get_grid_shape_string(empty_grid)
        self.assertEqual(result, "0 by 0")
    
    def test_extract_python_code(self):
        """Test Python code extraction from text"""
        text = """Here is some code:
        
```python
def solve(grid):
    return grid
```

That's the solution."""
        
        expected = "def solve(grid):\n    return grid"
        result = extract_python_code(text)
        self.assertEqual(result, expected)
        
        # Test with no code
        text_no_code = "No code here"
        result = extract_python_code(text_no_code)
        self.assertEqual(result, "")
    
    def test_extract_python_code_from_response_content(self):
        """Test Python code extraction from response content"""
        response_content = """Here is the solution:
        
```python
def solve(grid):
    return [[0, 1], [1, 0]]
```

This should work."""
        
        expected = "def solve(grid):\n    return [[0, 1], [1, 0]]"
        result = extract_python_code(response_content)
        self.assertEqual(result, expected)
    
    def test_create_arc_prompt(self):
        """Test ARC prompt creation"""
        # Mock task data
        task_data = {
            'train': [
                {
                    'input': [[1, 0], [0, 1]],
                    'output': [[0, 1], [1, 0]]
                }
            ],
            'test': [
                {
                    'input': [[1, 1], [0, 0]]
                }
            ]
        }
        
        # Mock prompt loader
        mock_prompt_loader = Mock()
        mock_prompt_loader.get_system_message.return_value = "You are an AI assistant."
        mock_prompt_loader.get_initial_turn_prompt.return_value = "Solve this task:\n{task_content}"
        
        system_content, user_content = create_arc_prompt(task_data, mock_prompt_loader, "soar")
        
        self.assertEqual(system_content, "You are an AI assistant.")
        self.assertIn("Input 1 (grid shape: 2 by 2)", user_content)
        self.assertIn("Output 1 (grid shape: 2 by 2)", user_content)
        self.assertIn("Test Input 1 (grid shape: 2 by 2)", user_content)
        self.assertIn("[[1 0] [0 1]]", user_content)  # Input grid formatted
        self.assertIn("[[0 1] [1 0]]", user_content)  # Output grid formatted
        
        # Verify prompt loader was called with correct version
        mock_prompt_loader.get_system_message.assert_called_with("soar")
        mock_prompt_loader.get_initial_turn_prompt.assert_called_with("soar")

    def test_generate_output_diff_perfect_match(self):
        """Test diff generation for perfect match"""
        expected = [[0, 1, 2], [3, 4, 5]]
        predicted = [[0, 1, 2], [3, 4, 5]]
        result = generate_output_diff(expected, predicted)
        
        self.assertIn("ACCURACY: 6/6 cells correct (100.0%)", result)
        self.assertIn("✓ ✓ ✓", result)
        self.assertNotIn("✗", result)
    
    def test_generate_output_diff_some_errors(self):
        """Test diff generation with some incorrect cells"""
        expected = [[0, 1, 2], [3, 4, 5]]
        predicted = [[0, 2, 2], [3, 4, 9]]
        result = generate_output_diff(expected, predicted)
        
        self.assertIn("ACCURACY: 4/6 cells correct (66.7%)", result)
        self.assertIn("✓ ✗(1→2) ✓", result)
        self.assertIn("✓ ✓ ✗(5→9)", result)
    
    def test_generate_output_diff_shape_mismatch(self):
        """Test diff generation with shape mismatch"""
        expected = [[0, 1, 2], [3, 4, 5]]
        predicted = [[0, 1], [3, 4]]
        result = generate_output_diff(expected, predicted)
        
        self.assertIn("SHAPE MISMATCH: Expected 3 by 2, got 2 by 2", result)
        self.assertIn("EXPECTED: [[0 1 2] [3 4 5]]", result)
        self.assertIn("PREDICTED: [[0 1] [3 4]]", result)
    
    def test_generate_output_diff_none_prediction(self):
        """Test diff generation with None prediction (execution failed)"""
        expected = [[0, 1, 2], [3, 4, 5]]
        predicted = None
        result = generate_output_diff(expected, predicted)
        
        self.assertEqual(result, "PREDICTED: None (execution failed)")
    
    def test_create_arc_prompt_refinement_mode_basic(self):
        """Test unified function in refinement mode without predicted outputs"""
        task_data = {
            'train': [
                {
                    'input': [[1, 0], [0, 1]],
                    'output': [[0, 1], [1, 0]]
                }
            ],
            'test': [
                {
                    'input': [[1, 1], [0, 0]]
                }
            ]
        }
        
        draft_program = "def transform(grid):\n    return grid"
        
        # Mock prompt loader  
        mock_prompt_loader = Mock()
        mock_prompt_loader.get_system_message.return_value = "You are an AI assistant."
        mock_prompt_loader.get_initial_turn_prompt.return_value = "{refinement_instructions}{draft_program_section}\n{task_content}"
        
        system_content, user_content = create_arc_prompt(
            task_data, mock_prompt_loader, "soar", draft_program=draft_program
        )
        
        self.assertEqual(system_content, "You are an AI assistant.")
        self.assertIn("Input 1 (grid shape: 2 by 2)", user_content)
        self.assertIn("Output 1 (grid shape: 2 by 2)", user_content)
        self.assertIn("Test Input 1 (grid shape: 2 by 2)", user_content)
        self.assertIn("def transform(grid):", user_content)
        self.assertNotIn("Draft Program's Output", user_content)  # No predicted outputs
        
        # Verify prompt loader was called
        mock_prompt_loader.get_system_message.assert_called_with("soar")
        mock_prompt_loader.get_initial_turn_prompt.assert_called_with("soar")
    
    def test_create_arc_prompt_refinement_with_full_outputs(self):
        """Test unified function in refinement mode with full predicted outputs"""
        task_data = {
            'train': [
                {
                    'input': [[1, 0], [0, 1]],
                    'output': [[0, 1], [1, 0]]
                }
            ],
            'test': [
                {
                    'input': [[1, 1], [0, 0]]
                }
            ]
        }
        
        draft_program = "def transform(grid):\n    return grid"
        
        predicted_outputs = {
            'train': [[[1, 0], [0, 1]]]  # Wrong - should be [[0, 1], [1, 0]]
        }
        
        # Mock prompt loader  
        mock_prompt_loader = Mock()
        mock_prompt_loader.get_system_message.return_value = "You are an AI assistant."
        mock_prompt_loader.get_initial_turn_prompt.return_value = "{refinement_instructions}{draft_program_section}\n{task_content}"
        
        system_content, user_content = create_arc_prompt(
            task_data, mock_prompt_loader, "soar", 
            draft_program=draft_program,
            predicted_outputs=predicted_outputs, 
            output_mode="full"
        )
        
        self.assertEqual(system_content, "You are an AI assistant.")
        self.assertIn("Draft Program's Output 1 (grid shape: 2 by 2)", user_content)
        self.assertIn("[[1 0] [0 1]]", user_content)  # Predicted output
    
    def test_create_arc_prompt_refinement_with_diff_outputs(self):
        """Test unified function in refinement mode with diff output mode - verifies diff approach works correctly"""
        task_data = {
            'train': [
                {
                    'input': [[1, 0], [0, 1]],
                    'output': [[0, 1], [1, 0]]
                }
            ],
            'test': [
                {
                    'input': [[1, 1], [0, 0]]
                }
            ]
        }
        
        draft_program = "def transform(grid):\n    return grid"
        
        predicted_outputs = {
            'train': [[[1, 0], [0, 1]]]  # Wrong - should be [[0, 1], [1, 0]]
        }
        
        # Mock prompt loader  
        mock_prompt_loader = Mock()
        mock_prompt_loader.get_system_message.return_value = "You are an AI assistant."
        mock_prompt_loader.get_initial_turn_prompt.return_value = "{refinement_instructions}{draft_program_section}\n{task_content}"
        
        system_content, user_content = create_arc_prompt(
            task_data, mock_prompt_loader, "soar", 
            draft_program=draft_program,
            predicted_outputs=predicted_outputs, 
            output_mode="diff"
        )
        
        self.assertEqual(system_content, "You are an AI assistant.")
        self.assertIn("Draft Program vs Expected Output 1:", user_content)
        self.assertIn("ACCURACY: 0/4 cells correct (0.0%)", user_content)
        self.assertIn("✗(0→1) ✗(1→0)", user_content)
        self.assertIn("✗(1→0) ✗(0→1)", user_content)
    
    def test_create_arc_prompt_regular_mode_unchanged(self):
        """Test that regular mode (no draft program) works exactly as before"""
        task_data = {
            'train': [
                {
                    'input': [[1, 0], [0, 1]],
                    'output': [[0, 1], [1, 0]]
                }
            ],
            'test': [
                {
                    'input': [[1, 1], [0, 0]]
                }
            ]
        }
        
        # Mock prompt loader
        mock_prompt_loader = Mock()
        mock_prompt_loader.get_system_message.return_value = "You are an AI assistant."
        mock_prompt_loader.get_initial_turn_prompt.return_value = "Solve this task:\n{task_content}"
        
        system_content, user_content = create_arc_prompt(task_data, mock_prompt_loader, "soar")
        
        self.assertEqual(system_content, "You are an AI assistant.")
        self.assertIn("Input 1 (grid shape: 2 by 2)", user_content)
        self.assertIn("Output 1 (grid shape: 2 by 2)", user_content)
        self.assertIn("Test Input 1 (grid shape: 2 by 2)", user_content)
        self.assertIn("[[1 0] [0 1]]", user_content)  # Input grid formatted
        self.assertIn("[[0 1] [1 0]]", user_content)  # Output grid formatted
        self.assertNotIn("Draft Program", user_content)  # No refinement content
        
        # Verify prompt loader was called with regular template
        mock_prompt_loader.get_system_message.assert_called_with("soar")
        mock_prompt_loader.get_initial_turn_prompt.assert_called_with("soar")
    
    def test_diff_approach_comprehensive_verification(self):
        """Comprehensive test of diff generation approach with various scenarios"""
        # Test 1: Mixed correct/incorrect cells
        expected = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        predicted = [[0, 9, 2], [3, 4, 1], [6, 7, 8]]
        result = generate_output_diff(expected, predicted)
        
        self.assertIn("ACCURACY: 7/9 cells correct (77.8%)", result)
        self.assertIn("✓ ✗(1→9) ✓", result)  # First row
        self.assertIn("✓ ✓ ✗(5→1)", result)  # Second row
        self.assertIn("✓ ✓ ✓", result)  # Third row - all correct
        
        # Test 2: Complete failure
        expected = [[0, 1], [2, 3]]
        predicted = [[9, 8], [7, 6]]
        result = generate_output_diff(expected, predicted)
        
        self.assertIn("ACCURACY: 0/4 cells correct (0.0%)", result)
        self.assertIn("✗(0→9) ✗(1→8)", result)
        self.assertIn("✗(2→7) ✗(3→6)", result)
        
        # Test 3: Shape mismatch detection
        expected = [[0, 1, 2]]
        predicted = [[0], [1]]  # Different shape
        result = generate_output_diff(expected, predicted)
        
        self.assertIn("SHAPE MISMATCH: Expected 3 by 1, got 1 by 2", result)
        self.assertIn("EXPECTED: [[0 1 2]]", result)
        self.assertIn("PREDICTED: [[0] [1]]", result)
        
        # Test 4: Execution failure (None prediction)
        expected = [[0, 1], [2, 3]]
        predicted = None
        result = generate_output_diff(expected, predicted)
        
        self.assertEqual(result, "PREDICTED: None (execution failed)")
    
    def test_unified_template_generates_correct_refinement_content(self):
        """Test that the unified template generates correct refinement-specific content"""
        task_data = {
            'train': [{'input': [[1, 0]], 'output': [[0, 1]]}],
            'test': [{'input': [[1, 1]]}]
        }
        
        draft_program = "def transform(grid): return grid"
        
        mock_prompt_loader = Mock()
        mock_prompt_loader.get_system_message.return_value = "System message"
        mock_prompt_loader.get_initial_turn_prompt.return_value = "Analyzing and refining existing Python code.{refinement_instructions}{draft_program_section}\n{task_content}"
        
        system_content, user_content = create_arc_prompt(
            task_data, mock_prompt_loader, "soar", draft_program=draft_program
        )
        
        self.assertEqual(system_content, "System message")
        self.assertIn("Analyzing and refining existing", user_content)
        self.assertIn("Draft program to refine:", user_content)
        self.assertIn("def transform(grid): return grid", user_content)
        self.assertIn("You should analyze:", user_content)


if __name__ == '__main__':
    unittest.main() 