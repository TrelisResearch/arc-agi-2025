#!/usr/bin/env python3

import re
from .schema import TrainingExample
from ..utils.scoring import ProgramExecutor
from ..utils.task_loader import TaskLoader
from ..utils.transduction import is_transduction_cheating

# Initialize utilities
task_loader = TaskLoader()
program_executor = ProgramExecutor(timeout=0.5)


def strip_comments_aggressive(source_code: str) -> str:
    """Strip comments and clean up whitespace."""
    if not source_code.strip():
        return source_code

    try:
        lines = source_code.split("\n")
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()
            # Skip empty lines and comment-only lines
            if not stripped or stripped.startswith("#"):
                continue

            # Remove inline comments but preserve the code part
            if "#" in line:
                # Find the # that's not inside a string literal
                in_string = False
                quote_char = None
                for i, char in enumerate(line):
                    if char in ['"', "'"] and (i == 0 or line[i - 1] != "\\"):
                        if not in_string:
                            in_string = True
                            quote_char = char
                        elif char == quote_char:
                            in_string = False
                            quote_char = None
                    elif char == "#" and not in_string:
                        line = line[:i].rstrip()
                        break

            cleaned_lines.append(line)

        # Join lines and normalize whitespace
        result = "\n".join(cleaned_lines)
        result = re.sub(r"\n\s*\n\s*\n+", "\n\n", result)
        return result.strip()

    except Exception as e:
        raise RuntimeError(f"Error in comment stripping: {e}")


def validate_program(program_data: TrainingExample) -> None:
    """Validate that a program executes without errors on its sample inputs.
    
    Raises:
        FileNotFoundError: If the ARC task cannot be found
        RuntimeError: If program fails validation (transduction cheating or execution errors)
    """
    task_id = program_data["task_id"]
    program = program_data["code"]
    sample_inputs = program_data["sample_inputs"]
    sample_outputs = program_data["sample_outputs"]

    # If no sample data, we can't validate
    if not sample_inputs or not sample_outputs:
        raise RuntimeError(f"No sample data available for task {task_id}")

    # Always apply transduction filter - failing to find task is an error
    task_data = task_loader.load_task(task_id, "arc-agi-1")
    is_cheating, cheat_reason = is_transduction_cheating(program, task_data)
    if is_cheating:
        raise RuntimeError(f"Program failed transduction check: {cheat_reason}")

    # Test that program executes successfully on all sample inputs
    for i, (input_grid, expected_output) in enumerate(zip(sample_inputs, sample_outputs)):
        predicted_output, error, timed_out = program_executor.execute_program_with_timeout(
            program, input_grid
        )
        if predicted_output is None or error or timed_out:
            error_msg = f"Program failed to execute on sample input {i}"
            if error:
                error_msg += f": {error}"
            if timed_out:
                error_msg += " (timeout)"
            raise RuntimeError(error_msg)

    # All validations passed
