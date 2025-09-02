"""
Utility functions for prompt preparation and code extraction from LLM responses.
"""

import re
import random
from typing import Dict, List, Tuple, Optional

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
    if not grid:
        return "0 by 0"
    return f"{len(grid[0])} by {len(grid)}"

def create_arc_prompt(
    task_data: Dict, 
    prompt_loader, 
    prompt_version: str = "soar", 
    splitter: bool = False,
    draft_program: Optional[str] = None,
    predicted_outputs: Optional[Dict] = None,
    output_mode: Optional[str] = None
) -> Tuple[str, str]:
    """
    Create a unified prompt for the model to solve an ARC task (both regular and refinement modes).
    
    Args:
        task_data: Dictionary containing 'train' and 'test' examples
        prompt_loader: PromptLoader instance for getting system message and prompt template
        prompt_version: Version of prompts to use (default: "soar")
        splitter: Whether to randomly select and shuffle a subset of training examples
        draft_program: Existing program code to be refined (enables refinement mode)
        predicted_outputs: Dict containing 'train' predicted outputs from draft program
        output_mode: How to display outputs ('full', 'diff', or None)
    
    Returns:
        Tuple of (system_content, user_content)
    """
    # Determine if we're in refinement mode
    refinement_mode = draft_program is not None
    
    # Format the task data
    task_content = ""
    
    # Get training examples
    train_examples = task_data['train'].copy()
    
    # Apply splitter logic if enabled
    if splitter and len(train_examples) > 1:
        # Select a random number of examples (between 1 and total available)
        num_to_select = random.randint(1, len(train_examples))
        # Randomly select and shuffle the examples
        train_examples = random.sample(train_examples, num_to_select)
    
    # Add training examples
    for i, example in enumerate(train_examples, 1):
        input_grid = example['input']
        output_grid = example['output']
        input_shape = _get_grid_shape_string(input_grid)
        output_shape = _get_grid_shape_string(output_grid)
        input_str = _format_grid_for_prompt(input_grid)
        output_str = _format_grid_for_prompt(output_grid)
        task_content += f"## Input {i} (grid shape: {input_shape}):\n{input_str}\n"
        task_content += f"## Output {i} (grid shape: {output_shape}):\n{output_str}\n"
        
        # Add predicted outputs if available and requested (refinement mode)
        if refinement_mode and predicted_outputs and output_mode and 'train' in predicted_outputs:
            predicted_train = predicted_outputs['train']
            if i <= len(predicted_train):
                predicted_grid = predicted_train[i-1]  # Convert to 0-based index
                if output_mode == "full":
                    if predicted_grid is not None:
                        predicted_str = _format_grid_for_prompt(predicted_grid)
                        predicted_shape = _get_grid_shape_string(predicted_grid)
                        task_content += f"## Draft Program's Output {i} (grid shape: {predicted_shape}):\n{predicted_str}\n"
                    else:
                        task_content += f"## Draft Program's Output {i}:\nNone (execution failed)\n"
                elif output_mode == "diff":
                    diff_result = generate_output_diff(output_grid, predicted_grid)
                    task_content += f"## Draft Program vs Expected Output {i}:\n{diff_result}\n"
        
        task_content += "\n"
    
    # Add test examples
    for i, example in enumerate(task_data['test'], 1):
        input_grid = example['input']
        input_shape = _get_grid_shape_string(input_grid)
        input_str = _format_grid_for_prompt(input_grid)
        task_content += f"## Test Input {i} (grid shape: {input_shape}):\n{input_str}\n"
    
    # Get the system message and unified template
    system_content = prompt_loader.get_system_message(prompt_version)
    prompt_template = prompt_loader.get_initial_turn_prompt(prompt_version)
    
    # Simple template placeholders - fill if refinement mode, empty if not
    if refinement_mode:
        refinement_instructions = """
You should analyze:
1. The task input-output patterns to understand the correct transformation rule
2. The provided draft program to identify its errors or shortcomings
3. How to correct and improve the draft to properly solve the task"""
        
        refinement_requirements = " The code should fix bugs in the original draft."
        
        draft_program_section = f"""
# Draft program to refine:
```python
{draft_program}
```"""
        
    else:
        refinement_instructions = ""
        refinement_requirements = ""
        draft_program_section = ""
    
    # Simple template formatting
    user_content = prompt_template.format(
        refinement_instructions=refinement_instructions,
        refinement_requirements=refinement_requirements,
        draft_program_section=draft_program_section,
        task_content=task_content
    )
    
    return system_content, user_content

def generate_output_diff(expected_grid: List[List[int]], predicted_grid: Optional[List[List[int]]]) -> str:
    """
    Generate a visual diff between expected and predicted grids.
    
    Args:
        expected_grid: The correct output grid
        predicted_grid: The predicted output grid (can be None)
    
    Returns:
        Formatted diff string showing differences
    """
    if predicted_grid is None:
        return "PREDICTED: None (execution failed)"
    
    expected_shape = _get_grid_shape_string(expected_grid)
    predicted_shape = _get_grid_shape_string(predicted_grid)
    
    # Handle shape mismatch
    if len(expected_grid) != len(predicted_grid) or (expected_grid and predicted_grid and len(expected_grid[0]) != len(predicted_grid[0])):
        return f"SHAPE MISMATCH: Expected {expected_shape}, got {predicted_shape}\nEXPECTED: {_format_grid_for_prompt(expected_grid)}\nPREDICTED: {_format_grid_for_prompt(predicted_grid)}"
    
    # Compare cell by cell
    correct_cells = 0
    total_cells = 0
    diff_lines = []
    
    for row_idx, (expected_row, predicted_row) in enumerate(zip(expected_grid, predicted_grid)):
        row_diff = []
        for col_idx, (expected_val, predicted_val) in enumerate(zip(expected_row, predicted_row)):
            total_cells += 1
            if expected_val == predicted_val:
                correct_cells += 1
                row_diff.append("✓")
            else:
                row_diff.append(f"✗({expected_val}→{predicted_val})")
        diff_lines.append(" ".join(row_diff))
    
    accuracy = correct_cells / total_cells if total_cells > 0 else 0
    
    diff_result = f"ACCURACY: {correct_cells}/{total_cells} cells correct ({accuracy:.1%})\n"
    diff_result += "\n".join(diff_lines)
    
    return diff_result

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


 