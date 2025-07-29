"""
Utility functions for prompt preparation and code extraction from LLM responses.
"""

import re
from typing import Dict, List, Tuple

def _format_grid_for_prompt(grid: List[List[int]]) -> str:
    """
    Format a grid for inclusion in prompts.
    
    Args:
        grid: 2D list representing the grid
    
    Returns:
        Formatted string representation of the grid
    """
    return str(grid).replace(',', '')


def _get_grid_shape_string(grid: List[List[int]]) -> str:
    """
    Get a human-readable shape string for a grid.
    
    Args:
        grid: 2D list representing the grid
    
    Returns:
        Shape string in format "width by height"
    """
    if not grid or not grid[0]:
        return "0 by 0"
    return f"{len(grid[0])} by {len(grid)}"

def create_arc_prompt(task_data: Dict, prompt_loader, prompt_version: str = "soar") -> Tuple[str, str]:
    """
    Create a prompt for the model to solve an ARC task.
    
    Args:
        task_data: Dictionary containing 'train' and 'test' examples
        prompt_loader: PromptLoader instance for getting system message and prompt template
        prompt_version: Version of prompts to use (default: "soar")
    
    Returns:
        Tuple of (system_content, user_content)
    """
    # Format the task data
    task_content = ""
    
    # Add training examples
    for i, example in enumerate(task_data['train'], 1):
        input_grid = example['input']
        output_grid = example['output']
        input_shape = _get_grid_shape_string(input_grid)
        output_shape = _get_grid_shape_string(output_grid)
        input_str = _format_grid_for_prompt(input_grid)
        output_str = _format_grid_for_prompt(output_grid)
        task_content += f"## Input {i} (grid shape: {input_shape}):\n{input_str}\n"
        task_content += f"## Output {i} (grid shape: {output_shape}):\n{output_str}\n\n"
    
    # Add test examples (now handles multiple test examples!)
    for i, example in enumerate(task_data['test'], 1):
        input_grid = example['input']
        input_shape = _get_grid_shape_string(input_grid)
        input_str = _format_grid_for_prompt(input_grid)
        task_content += f"## Test Input {i} (grid shape: {input_shape}):\n{input_str}\n"
        
        # Optionally include expected output for training data (if available)
        if 'output' in example:
            output_grid = example['output']
            output_shape = _get_grid_shape_string(output_grid)
            output_str = _format_grid_for_prompt(output_grid)
            task_content += f"## Expected Test Output {i} (grid shape: {output_shape}):\n{output_str}\n"
    
    # Get the system message and prompt template
    system_content = prompt_loader.get_system_message(prompt_version)
    prompt_template = prompt_loader.get_initial_turn_prompt(prompt_version)
    
    # Handle template formatting
    user_content = prompt_template.format(task_content=task_content)
    
    return system_content, user_content

def extract_python_code(text: str, debug: bool = False) -> str:
    """
    Extract Python code from text (looks for the last ```python code block).
    
    Args:
        text: Text content to extract code from
        debug: Whether to print debug information
    
    Returns:
        Extracted Python code as string, or empty string if no code found
    """
    if debug and len(text) > 0:
        print(f"🔍 Text content: {len(text)} chars")
    
    # Look for python code blocks
    python_blocks = re.findall(r'```python\s*\n(.*?)\n```', text, re.DOTALL)
    if python_blocks:
        return python_blocks[-1].strip()
    
    return ""


 