"""
Utility functions for prompt preparation and code extraction from LLM responses.
"""

import re
import random
from typing import Dict, List, Tuple, Optional

from llm_python.utils.prompt_loader import PromptLoader, get_prompt_loader
from llm_python.utils.task_loader import TaskData


def _format_grid_for_prompt(grid: List[List[int]]) -> str:
    """
    Format a grid for inclusion in prompts.

    Args:
        grid: 2D list representing the grid

    Returns:
        Formatted string representation of the grid
    """
    # Handle numpy arrays recursively
    def convert_to_list(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [convert_to_list(item) for item in obj]
        else:
            return obj

    grid = convert_to_list(grid)
    return str(grid).replace(",", "")


def _get_grid_shape_string(grid: List[List[int]]) -> str:
    """
    Get a human-readable shape string for a grid.

    Args:
        grid: 2D list representing the grid

    Returns:
        Shape string in format "width by height"
    """
    # Handle numpy arrays
    if hasattr(grid, 'tolist'):
        grid = grid.tolist()

    if not grid or len(grid) == 0:
        return "0 by 0"
    return f"{len(grid[0])} by {len(grid)}"


def create_arc_prompt(
    task_data: Dict,
    prompt_loader,
    prompt_version: str = "soar",
    splitter: bool = False,
    draft_program: Optional[str] = None,
    predicted_outputs: Optional[Dict] = None,
    output_mode: Optional[str] = None,
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
    train_examples = task_data["train"].copy()

    # Apply splitter logic if enabled
    if splitter and len(train_examples) > 1:
        # Select a random number of examples (between 1 and total available)
        num_to_select = random.randint(1, len(train_examples))
        # Randomly select and shuffle the examples
        train_examples = random.sample(train_examples, num_to_select)

    # Add training examples
    for i, example in enumerate(train_examples, 1):
        input_grid = example["input"]
        output_grid = example["output"]
        input_shape = _get_grid_shape_string(input_grid)
        output_shape = _get_grid_shape_string(output_grid)
        input_str = _format_grid_for_prompt(input_grid)
        output_str = _format_grid_for_prompt(output_grid)
        task_content += f"## Input {i} (grid shape: {input_shape}):\n{input_str}\n"
        task_content += f"## Output {i} (grid shape: {output_shape}):\n{output_str}\n"

        # Add predicted outputs if available and requested (refinement mode)
        if refinement_mode and output_mode == "full":
            if predicted_outputs is None or "train" not in predicted_outputs:
                raise ValueError(
                    "Predicted outputs for 'train' are required in refinement full mode."
                )
            predicted_train = predicted_outputs["train"]
            if i <= len(predicted_train):
                predicted_grid = predicted_train[i - 1]  # Convert to 0-based index
                if predicted_grid is not None:
                    # Convert numpy arrays to plain Python lists for formatting
                    if hasattr(predicted_grid, 'tolist'):
                        predicted_grid = predicted_grid.tolist()
                    predicted_str = _format_grid_for_prompt(predicted_grid)
                    predicted_shape = _get_grid_shape_string(predicted_grid)
                    task_content += f"## Draft Program's Output {i} (grid shape: {predicted_shape}):\n{predicted_str}\n"
                else:
                    task_content += (
                        f"## Draft Program's Output {i}:\nNone (execution failed)\n"
                    )

        task_content += "\n"

    # Add test examples
    for i, example in enumerate(task_data["test"], 1):
        input_grid = example["input"]
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
        task_content=task_content,
    )

    return system_content, user_content



def create_compound_prompt(
    task_data: TaskData,
    original_program: str,
    reference_program: str,
) -> Tuple[str, str]:
    prompt_loader = get_prompt_loader()
    # Format the task data
    task_content = ""

    # Get training examples
    train_examples = task_data["train"].copy()

    # Add training examples
    for i, example in enumerate(train_examples, 1):
        input_grid = example["input"]
        output_grid = example["output"]
        input_shape = _get_grid_shape_string(input_grid)
        output_shape = _get_grid_shape_string(output_grid)
        input_str = _format_grid_for_prompt(input_grid)
        output_str = _format_grid_for_prompt(output_grid)
        task_content += f"## Input {i} (grid shape: {input_shape}):\n{input_str}\n"
        task_content += f"## Output {i} (grid shape: {output_shape}):\n{output_str}\n"
        task_content += "\n"

    # Get the system message and unified template
    system_content = prompt_loader.get_system_message("soar")
    prompt_template = prompt_loader.get_compound_prompt("compound")

    # Simple template formatting
    user_content = prompt_template.format(
        task_content=task_content,
        original_program=original_program,
        reference_program=reference_program,
    )

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
    python_blocks = re.findall(r"```python\s*\n(.*?)\n```", text, re.DOTALL)
    if python_blocks:
        return python_blocks[-1].strip()

    return ""


def create_rewrite_prompt(
    task_data: TaskData,
    original_program: str,
) -> Tuple[str, str]:
    """
    Create system and user prompts for program rewriting.
    
    Args:
        task_data: The ARC task data
        original_program: The program to rewrite
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    prompt_loader = get_prompt_loader()
    
    # Format the task data
    task_content = ""
    
    # Get training examples
    train_examples = task_data["train"].copy()
    
    # Add training examples
    for i, example in enumerate(train_examples, 1):
        input_grid = example["input"]
        output_grid = example["output"]
        input_shape = _get_grid_shape_string(input_grid)
        output_shape = _get_grid_shape_string(output_grid)
        input_str = _format_grid_for_prompt(input_grid)
        output_str = _format_grid_for_prompt(output_grid)
        task_content += f"## Input {i} (grid shape: {input_shape}):\n{input_str}\n"
        task_content += f"## Output {i} (grid shape: {output_shape}):\n{output_str}\n\n"
    
    # Add test examples (inputs only)
    test_examples = task_data["test"].copy()
    for i, example in enumerate(test_examples, 1):
        input_grid = example["input"]
        input_shape = _get_grid_shape_string(input_grid)
        input_str = _format_grid_for_prompt(input_grid)
        task_content += f"## Test Input {i} (grid shape: {input_shape}):\n{input_str}\n\n"
    
    # Load style guide
    from pathlib import Path
    style_guide_path = Path(__file__).parent.parent / "prompt-strings" / "style-guide.md"
    with open(style_guide_path, 'r') as f:
        style_guide = f.read()
    
    # Load prompts
    system_prompt = prompt_loader.get_system_message("rewrite")
    user_prompt_template = prompt_loader.get_initial_turn_prompt("rewrite")
    
    # Fill in the user prompt template
    user_prompt = user_prompt_template.format(
        task_content=task_content,
        original_program=original_program,
        style_guide=style_guide
    )
    
    return system_prompt, user_prompt
