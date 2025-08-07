#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import List, Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import SoarProgramExample
from .code import strip_comments
from .arc_tester import ArcTester
from llm_python.utils.task_loader import TaskLoader

# Initialize utilities
task_loader = TaskLoader()
arc_tester = ArcTester(timeout=2, executor_type="unrestricted", detailed_errors=True)

# Define explicit PyArrow schema for our parquet file (same as extract_from_logs)
PARQUET_SCHEMA = pa.schema(
    [
        ("task_id", pa.string()),
        ("reasoning", pa.string()),
        ("code", pa.string()),
        ("correct_train_input", pa.list_(pa.bool_())),
        ("correct_test_input", pa.list_(pa.bool_())),
        ("predicted_train_output", pa.list_(pa.list_(pa.list_(pa.int64())))),
        ("predicted_test_output", pa.list_(pa.list_(pa.list_(pa.int64())))),
        ("train_input", pa.list_(pa.list_(pa.list_(pa.int64())))),
        ("test_input", pa.list_(pa.list_(pa.list_(pa.int64())))),
        ("model", pa.string()),
        ("generation", pa.int64()),
    ]
)


def _save_programs_to_parquet(
    programs: List[SoarProgramExample], output_path: Path
) -> None:
    """Save programs to parquet file with explicit schema validation."""
    if not programs:
        print("No programs to save!")
        return

    df = pd.DataFrame(programs)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to PyArrow table with explicit schema and save
    table = pa.Table.from_pandas(df, schema=PARQUET_SCHEMA)
    pq.write_table(table, output_path)
    print(f"Saved {len(programs)} programs to {output_path}")


def _validate_training_example_against_schema(example: SoarProgramExample) -> None:
    """Validate that a SoarProgramExample can be encoded by PyArrow with our schema."""
    try:
        # Convert to a regular dict to avoid TypedDict key access issues
        example_dict = dict(example)
        
        # Create arrays for each column with single values
        arrays = []
        for field in PARQUET_SCHEMA:
            arrays.append(pa.array([example_dict[field.name]], type=field.type))

        # Try to create a PyArrow table with our schema
        pa.table(arrays, schema=PARQUET_SCHEMA)

        # If we get here, the data is valid
        return

    except Exception as e:
        raise ValueError(f"SoarProgramExample failed schema validation: {e}")


def load_hodel_program(py_file_path: str) -> Optional[str]:
    """Load a Python file and extract the solve function as a string."""
    try:
        with open(py_file_path, 'r') as f:
            content = f.read()
        
        # The content should define a solve function
        # We need to wrap the solve function to handle list->tuple conversion
        # since Hodel programs expect tuple inputs but our framework provides list inputs
        
        # Replace the function definition to avoid infinite recursion
        # This handles the case where solve() might be called from within the same module
        wrapped_code = content.replace('def solve(', 'def solve_original(') + """

def solve(grid):
    # Convert list of lists to tuple of tuples for Hodel programs  
    grid_as_tuples = tuple(tuple(row) for row in grid)
    result = solve_original(grid_as_tuples)
    # Convert result back to list of lists format for consistency
    if isinstance(result, (list, tuple)) and all(isinstance(row, (list, tuple)) for row in result):
        return [list(row) for row in result]
    else:
        return result
"""
        
        return wrapped_code
    except Exception as e:
        print(f"Error loading {py_file_path}: {e}")
        return None


def is_valid_code(code: str) -> bool:
    """Check if code is valid Python that can be compiled."""
    try:
        compile(code, "<string>", "exec")
        return True
    except (SyntaxError, ValueError, TypeError):
        return False


def extract_task_id_from_filename(py_file_path: str) -> str:
    """Extract task ID from filename (everything before .py)."""
    return Path(py_file_path).stem


def process_hodel_program(py_file_path: str) -> Optional[SoarProgramExample]:
    """Process a single Hodel program file and create a SoarProgramExample."""
    
    # Extract task ID from filename
    task_id = extract_task_id_from_filename(py_file_path)
    
    # Load the program code
    code = load_hodel_program(py_file_path)
    if code is None:
        return None
    
    # Validate the code compiles
    if not is_valid_code(code):
        print(f"Invalid Python code in {py_file_path}")
        return None
    
    # Strip comments like we do in extract_from_logs
    try:
        program_to_execute = strip_comments(code)
    except Exception:
        # If stripping comments failed because the original code was invalid, that's fine - return None
        if not is_valid_code(code):
            return None
        else:
            # If it failed for some other reason, re-raise
            raise
    
    try:
        # Load task data
        task_data = task_loader.load_task(task_id)
        
        # Test the program using ArcTester
        test_result = arc_tester.test_program(program_to_execute, task_data)
        
        # If the program failed for any input, log details and return None
        if not test_result.success:
            print(f"Program failed for task {task_id}")
            
            # Log detailed error information
            if test_result.train_errors:
                train_error_count = sum(1 for error in test_result.train_errors if error is not None)
                if train_error_count > 0:
                    print(f"  Train errors: {train_error_count}/{len(test_result.train_errors)}")
                    for i, error in enumerate(test_result.train_errors):
                        if error is not None:
                            print(f"    Train example {i}: {error.output_type} - {error.error_message}")
            
            if test_result.test_errors:
                test_error_count = sum(1 for error in test_result.test_errors if error is not None)
                if test_error_count > 0:
                    print(f"  Test errors: {test_error_count}/{len(test_result.test_errors)}")
                    for i, error in enumerate(test_result.test_errors):
                        if error is not None:
                            print(f"    Test example {i}: {error.output_type} - {error.error_message}")
            
            return None
        
        # Convert Optional[Grid] to List[List[int]] by filtering out None values
        # and converting Grid (which is List[List[int]]) to the expected format
        def convert_outputs(outputs: List[Optional[List[List[int]]]]) -> List[List[List[int]]]:
            return [output for output in outputs if output is not None]
        
        # Create the training example
        training_example = SoarProgramExample(
            task_id=task_id,
            reasoning="",  # Hodel programs don't have reasoning traces
            code=program_to_execute,  # Store the version we actually executed (comments stripped)
            correct_train_input=test_result.correct_train_input,
            correct_test_input=test_result.correct_test_input,
            predicted_train_output=convert_outputs(test_result.train_outputs),
            predicted_test_output=convert_outputs(test_result.test_outputs),
            train_input=test_result.train_inputs,
            test_input=test_result.test_inputs,
            model="hodel-translated",
            generation=0,
        )
        
        # Validate against PyArrow schema
        _validate_training_example_against_schema(training_example)
        
        return training_example
        
    except Exception as e:
        print(f"Error processing {py_file_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract Hodel programs to parquet format"
    )
    parser.add_argument(
        "--hodel-dir",
        type=str,
        default="llm_python/progdb/axel_hodel",
        help="Directory containing Hodel .py files (default: llm_python/progdb/axel_hodel)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="hodel_programs.parquet",
        help="Output parquet file (default: hodel_programs.parquet)",
    )

    args = parser.parse_args()

    # Find all .py files in the Hodel directory (excluding __init__.py)
    hodel_dir = Path(args.hodel_dir)
    py_files = list(hodel_dir.glob("*.py"))
    py_files = [f for f in py_files if f.name != "__init__.py"]
    
    if not py_files:
        print(f"No .py files found in {hodel_dir}")
        return

    print(f"Processing {len(py_files)} Hodel program files...")

    # Process all files
    all_programs = []
    successful_programs = 0
    
    for py_file in py_files:
        result = process_hodel_program(str(py_file))
        if result is not None:
            all_programs.append(result)
            successful_programs += 1
        
        # Progress reporting
        if (len(all_programs) + 1) % 50 == 0:
            print(f"  Processed {len(all_programs) + 1}/{len(py_files)} files")

    print(f"Successfully processed {successful_programs}/{len(py_files)} programs")

    # Save to parquet
    output_path = Path(args.output)
    _save_programs_to_parquet(all_programs, output_path)

    # Print some basic stats
    if all_programs:
        df = pd.DataFrame(all_programs)
        print("\nStats:")
        print(f"  Unique tasks: {df['task_id'].nunique()}")
        print(f"  Model: {df['model'].iloc[0]}")
        
        # Calculate average accuracies from boolean lists
        train_accuracies = [sum(correct_list) / len(correct_list) if correct_list else 0.0 
                           for correct_list in df['correct_train_input']]
        test_accuracies = [sum(correct_list) / len(correct_list) if correct_list else 0.0 
                          for correct_list in df['correct_test_input']]
        
        print(f"  Average training accuracy: {sum(train_accuracies) / len(train_accuracies):.3f}")
        print(f"  Average test accuracy: {sum(test_accuracies) / len(test_accuracies):.3f}")


if __name__ == "__main__":
    main()
