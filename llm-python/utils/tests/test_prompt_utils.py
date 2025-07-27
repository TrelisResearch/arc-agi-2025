"""
Tests for prompt utility functions.
"""

import unittest
from unittest.mock import Mock, MagicMock
from ..prompt_utils import (
    create_arc_prompt,
    extract_python_code_from_response,
    extract_python_code_from_text,
    format_grid_for_prompt,
    get_grid_shape_string
)


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
    
    def test_extract_python_code_from_text(self):
        """Test Python code extraction from text"""
        text = """Here is some code:
        
```python
def solve(grid):
    return grid
```

That's the solution."""
        
        expected = "def solve(grid):\n    return grid"
        result = extract_python_code_from_text(text)
        self.assertEqual(result, expected)
        
        # Test with no code
        text_no_code = "No code here"
        result = extract_python_code_from_text(text_no_code)
        self.assertEqual(result, "")
    
    def test_extract_python_code_from_response(self):
        """Test Python code extraction from API response"""
        # Mock response object
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        
        mock_message.content = """Here is the solution:
        
```python
def solve(grid):
    return [[0, 1], [1, 0]]
```

This should work."""
        
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        
        expected = "def solve(grid):\n    return [[0, 1], [1, 0]]"
        result = extract_python_code_from_response(mock_response)
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


if __name__ == '__main__':
    unittest.main() 