"""
Utility functions for prompt preparation and code extraction from LLM responses.
"""

import re
import random
from typing import Dict, List, Tuple, Optional

from llm_python.utils.prompt_loader import get_prompt_loader
from llm_python.utils.task_loader import TaskData
from llm_python.utils.numpy import convert_numpy_types
import numpy as np


def _format_predicted_output_for_display(output_grid: List[List[int]]) -> Tuple[str, str]:
    """
    Format predicted output grid for display in refinement prompts.

    Args:
        output_grid: 2D list representing the predicted output grid

    Returns:
        Tuple of (formatted_grid_string, shape_string)
    """
    if not isinstance(output_grid, list):
        return str(output_grid), ""

    try:
        # Convert to numpy array to get shape
        arr = np.array(output_grid)
        height, width = arr.shape
        shape_string = f"grid shape: {width} by {height}"

        # Format grid with clean display (no commas)
        formatted_grid = str(output_grid).replace(",", "")

        return formatted_grid, shape_string
    except (ValueError, TypeError, AttributeError):
        # Fallback to simple string representation
        return str(output_grid), ""


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

    # Return original format only
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
    single: bool = False,
    draft_program: Optional[str] = None,
    predicted_outputs: Optional[Dict] = None,
    output_mode: Optional[str] = None,
    correct_train_input: Optional[List[bool]] = None,
) -> Tuple[str, str]:
    """
    Create a unified prompt for the model to solve an ARC task (both regular and refinement modes).

    Args:
        task_data: Dictionary containing 'train' and 'test' examples
        prompt_loader: PromptLoader instance for getting system message and prompt template
        prompt_version: Version of prompts to use (default: "soar")
        splitter: Whether to randomly select and shuffle a subset of training examples
        single: Whether to use only a single randomly selected training example
        draft_program: Existing program code to be refined (enables refinement mode)
        predicted_outputs: Dict containing 'train' predicted outputs from draft program
        output_mode: How to display outputs ('full', 'diff', or None)
        correct_train_input: Optional list of boolean flags indicating correctness for each training example

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

    # Apply single mode logic if enabled (overrides splitter)
    if single and len(train_examples) > 0:
        # Select exactly one random training example
        train_examples = random.sample(train_examples, 1)

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

    # Add test examples
    for i, example in enumerate(task_data["test"], 1):
        input_grid = example["input"]
        input_shape = _get_grid_shape_string(input_grid)
        input_str = _format_grid_for_prompt(input_grid)
        task_content += f"## Test Input {i} (grid shape: {input_shape}):\n{input_str}\n"

    # Get the system message and unified template
    system_content = prompt_loader.get_system_message(prompt_version)
    prompt_template = prompt_loader.get_initial_turn_prompt(prompt_version)

    # Use different prompt versions for refinement vs regular mode
    if refinement_mode:
        # Use soar-refine prompts and generate special task content
        system_content = prompt_loader.get_system_message("soar-refine")
        prompt_template = prompt_loader.get_initial_turn_prompt("soar-refine")

        # Generate refinement-specific task content with statistics
        task_content = generate_refinement_task_content(
            task_data, draft_program, predicted_outputs, train_examples, correct_train_input
        )
    else:
        # Use regular soar prompts (already loaded above)
        pass

    # Simple template formatting
    user_content = prompt_template.format(task_content=task_content)

    return system_content, user_content


def generate_refinement_task_content(task_data: Dict, draft_program: Optional[str], predicted_outputs: Optional[Dict], train_examples: list, correct_train_input: Optional[List[bool]] = None, show_output_test: bool = True) -> str:
    """Generate the new refinement prompt format with correctness statistics

    Args:
        task_data: Dictionary containing 'train' and 'test' examples
        draft_program: Existing program code to be refined (optional)
        predicted_outputs: Dict containing 'train' predicted outputs from draft program
        train_examples: List of training examples
        correct_train_input: Optional list of boolean flags indicating correctness for each training example
        show_output_test: Whether to show test outputs (default: True)
    """

    content = "# Task to solve:"

    # Add train examples with predicted outputs and correctness info
    correct_count = 0
    total_count = len(train_examples)

    for i, example in enumerate(train_examples, 1):
        input_grid = example["input"]
        output_grid = example["output"]
        input_shape = _get_grid_shape_string(input_grid)
        output_shape = _get_grid_shape_string(output_grid)
        input_str = _format_grid_for_prompt(input_grid)
        output_str = _format_grid_for_prompt(output_grid)

        content += f"\n## Input {i} (grid shape: {input_shape}):\n{input_str}\n"
        content += f"## Output {i} (grid shape: {output_shape}):\n{output_str}\n"

    # Add test examples
    for i, example in enumerate(task_data["test"], 1):
        input_grid = example["input"]
        input_shape = _get_grid_shape_string(input_grid)
        input_str = _format_grid_for_prompt(input_grid)
        content += f"## Test Input {i} (grid shape: {input_shape}):\n{input_str}\n"

    # Add previous implementation section
    content += f"\nPrevious implementation:\n```python\n{draft_program or ''}\n```\n"

    # Add correctness statistics
    if correct_train_input is not None:
        # Use pre-computed boolean flags for correctness
        correct_count = sum(correct_train_input)
        content += f"\nThis implementation of transform function correctly worked on {correct_count}/{total_count} train input-output pairs.\n"
        content += "Detailed results:\n"

        # Add detailed results for each training example using boolean flags
        for i, is_correct in enumerate(correct_train_input, 1):
            content += f"## Output {i} computed by `transform` is "
            if is_correct:
                content += "correct.\n"
            else:
                content += "incorrect.\n"

            # Show predicted output for all cases (correct and incorrect)
            if (predicted_outputs and "train" in predicted_outputs
                and i - 1 < len(predicted_outputs["train"])
                and predicted_outputs["train"][i - 1] is not None):

                predicted_output = predicted_outputs["train"][i - 1]
                # Convert numpy arrays if needed
                if hasattr(predicted_output, 'tolist'):
                    predicted_output = predicted_output.tolist()
                predicted_output = convert_numpy_types(predicted_output)

                formatted_grid, shape_string = _format_predicted_output_for_display(predicted_output)
                content += f"The execution gave the following results ({shape_string}):\n{formatted_grid}\n"

    # Add test output display if requested (before summary message)
    if show_output_test and predicted_outputs and "test" in predicted_outputs:
        predicted_test = predicted_outputs["test"]
        for i, predicted_output in enumerate(predicted_test, 1):
            content += f"\n## Output Test {i} computed by `transform` (we don't know if it is correct or not)\n"

            if predicted_output is not None:
                # Convert numpy arrays if needed
                if hasattr(predicted_output, 'tolist'):
                    predicted_output = predicted_output.tolist()
                predicted_output = convert_numpy_types(predicted_output)

                formatted_grid, shape_string = _format_predicted_output_for_display(predicted_output)
                content += f"The execution gave the following results ({shape_string}):\n{formatted_grid}\n"

    # Add summary message if we have correctness data
    if correct_train_input is not None:
        # List which outputs were incorrect using boolean flags
        incorrect_outputs = []
        for i, is_correct in enumerate(correct_train_input, 1):
            if not is_correct:
                incorrect_outputs.append(f"Output {i}")

        # Match their message format
        if len(incorrect_outputs) == 0:
            content += "The previous code gives correct output grids for all Train input.\n"
            print("WARNING: The previous code gives correct output grids for all Train input. This is not expected.")
        else:
            incorrect_list = ', '.join(incorrect_outputs)
            content += f"\nThe previous code gives incorrect output grids for: {incorrect_list}. Now, you need to fix the code to produce correct output for all inputs."

    return content


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
        if debug:
            print(f"🔍 Found {len(python_blocks)} Python code block(s)")

        # Look for the code block that contains the transform function
        for i, block in enumerate(reversed(python_blocks)):  # Start from last and work backwards
            if "def transform" in block:
                if debug:
                    block_index = len(python_blocks) - 1 - i
                    print(f"🎯 Using code block {block_index + 1} (contains 'def transform')")
                return block.strip()

        # If no block contains def transform, fall back to the last block
        if debug:
            print(f"⚠️ No block contains 'def transform', using last block")
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
