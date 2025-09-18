"""
Tests for prompt utility functions.
"""

import unittest
from unittest.mock import Mock

from llm_python.utils.prompt_utils import (
    create_arc_prompt,
    extract_python_code,
    _format_grid_for_prompt as format_grid_for_prompt,
    _get_grid_shape_string as get_grid_shape_string
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
    
    def test_create_arc_prompt_basic(self):
        """Test basic ARC prompt creation"""
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

    def test_create_arc_prompt_with_splitter(self):
        """Test ARC prompt creation with splitter enabled"""
        task_data = {
            'train': [
                {'input': [[1, 0]], 'output': [[0, 1]]},
                {'input': [[0, 1]], 'output': [[1, 0]]},
                {'input': [[1, 1]], 'output': [[0, 0]]}
            ],
            'test': [
                {'input': [[0, 0]]}
            ]
        }
        
        mock_prompt_loader = Mock()
        mock_prompt_loader.get_system_message.return_value = "You are an AI assistant."
        mock_prompt_loader.get_initial_turn_prompt.return_value = "Solve this task:\n{task_content}"
        
        # With splitter enabled, we should get a subset of training examples
        system_content, user_content = create_arc_prompt(
            task_data, mock_prompt_loader, "soar", splitter=True
        )
        
        self.assertEqual(system_content, "You are an AI assistant.")
        self.assertIn("Test Input 1", user_content)  # Test input should always be present
        # Training examples should be present but may be a subset due to random selection

    def test_create_arc_prompt_multiple_examples(self):
        """Test ARC prompt creation with multiple train and test examples"""
        task_data = {
            'train': [
                {'input': [[1, 0]], 'output': [[0, 1]]},
                {'input': [[0, 1]], 'output': [[1, 0]]}
            ],
            'test': [
                {'input': [[1, 1]]},
                {'input': [[0, 0]]}
            ]
        }
        
        mock_prompt_loader = Mock()
        mock_prompt_loader.get_system_message.return_value = "You are an AI assistant."
        mock_prompt_loader.get_initial_turn_prompt.return_value = "Solve this task:\n{task_content}"
        
        system_content, user_content = create_arc_prompt(task_data, mock_prompt_loader)
        
        # Should have both training examples
        self.assertIn("Input 1 (grid shape: 2 by 1)", user_content)
        self.assertIn("Input 2 (grid shape: 2 by 1)", user_content)
        self.assertIn("Output 1 (grid shape: 2 by 1)", user_content)
        self.assertIn("Output 2 (grid shape: 2 by 1)", user_content)
        
        # Should have both test examples
        self.assertIn("Test Input 1 (grid shape: 2 by 1)", user_content)
        self.assertIn("Test Input 2 (grid shape: 2 by 1)", user_content)

    def test_create_arc_prompt_refinement_basic(self):
        """Test ARC prompt creation in refinement mode with draft program"""
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

        # Mock prompt loader for refinement mode
        mock_prompt_loader = Mock()
        mock_prompt_loader.get_system_message.return_value = "You are an AI assistant specialized in refinement."
        mock_prompt_loader.get_initial_turn_prompt.return_value = "Refinement prompt: {task_content}"

        system_content, user_content = create_arc_prompt(
            task_data, mock_prompt_loader, "soar", draft_program=draft_program
        )

        self.assertEqual(system_content, "You are an AI assistant specialized in refinement.")
        self.assertIn("Input 1 (grid shape: 2 by 2)", user_content)
        self.assertIn("Output 1 (grid shape: 2 by 2)", user_content)
        self.assertIn("Test Input 1 (grid shape: 2 by 2)", user_content)
        self.assertIn("def transform(grid):", user_content)
        self.assertIn("Previous implementation:", user_content)

        # Verify prompt loader was called with soar-refine for refinement mode
        mock_prompt_loader.get_system_message.assert_called_with("soar-refine")
        mock_prompt_loader.get_initial_turn_prompt.assert_called_with("soar-refine")

    def test_create_arc_prompt_refinement_with_full_outputs(self):
        """Test refinement mode with full predicted outputs"""
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

        # Mock prompt loader for refinement mode
        mock_prompt_loader = Mock()
        mock_prompt_loader.get_system_message.return_value = "You are an AI assistant specialized in refinement."
        mock_prompt_loader.get_initial_turn_prompt.return_value = "Refinement prompt: {task_content}"

        # Test with correct_train_input for the new format
        correct_train_input = [False]  # Wrong prediction

        system_content, user_content = create_arc_prompt(
            task_data, mock_prompt_loader, "soar",
            draft_program=draft_program,
            predicted_outputs=predicted_outputs,
            output_mode="full",
            correct_train_input=correct_train_input
        )

        self.assertEqual(system_content, "You are an AI assistant specialized in refinement.")
        self.assertIn("Previous implementation:", user_content)
        self.assertIn("[[1 0] [0 1]]", user_content)  # Predicted output
        self.assertIn("incorrect", user_content)  # Should show incorrect result



if __name__ == '__main__':
    unittest.main()