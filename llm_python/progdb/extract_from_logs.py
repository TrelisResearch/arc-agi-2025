#!/usr/bin/env python3

import argparse
import glob
import json
from pathlib import Path
from typing import List, Optional, TypedDict
import multiprocessing
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tokenize import TokenError

from .schema import SoarProgramExample
from .code import strip_comments
from .arc_tester import ArcTester
from ..utils.task_loader import TaskLoader, TaskData, Grid, TaskExample

# Initialize utilities
task_loader = TaskLoader()
arc_tester = ArcTester(timeout=2, executor_type="unrestricted")

# Define explicit PyArrow schema for our parquet file
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

class LogData(TypedDict):
    """Schema for data extracted from log files"""
    task_id: str
    program: str
    reasoning: str
    model: str


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


def _extract_reasoning_from_response(raw_response: dict) -> str:
    """Extract reasoning from a raw API response."""
    if not isinstance(raw_response, dict) or "choices" not in raw_response:
        return ""

    choices = raw_response.get("choices", [])
    if not choices or not isinstance(choices[0], dict):
        return ""

    message = choices[0].get("message", {})
    return message.get("reasoning", "")


def extract_log_data(log_path: str) -> List[LogData]:
    """Extract program data from a single log file without processing."""

    with open(log_path, "r") as f:
        log_data = json.load(f)

    task_id = log_data.get("task_id")
    if not task_id:
        # Not a warning since many files might not be program logs
        return []

    extracted_data = []
    api_type = log_data.get("api_type", "")
    model = log_data.get("model", "")

    # Handle different API types
    if "multiturn" in api_type:
        multiturn_data = log_data.get("multiturn_data", {})
        turn_details = multiturn_data.get("turn_details", [])

        if turn_details:
            first_turn = turn_details[0]
            program = first_turn.get("program", "")
            if program and first_turn.get("program_extracted", False):
                reasoning = _extract_reasoning_from_response(
                    first_turn.get("raw_response", {})
                )
                extracted_data.append(
                    LogData(
                        task_id=task_id,
                        program=program,
                        reasoning=reasoning,
                        model=model,
                    )
                )

    elif "independent_attempts" in api_type or "all_attempts" in api_type:
        # Inline the attempt details extraction logic
        if "independent_attempts" in api_type:
            attempt_details = log_data.get("independent_attempts_data", {}).get(
                "attempt_details", []
            )
        else:  # all_attempts
            attempt_details = log_data.get("attempt_details", [])

        for attempt in attempt_details:
            program = attempt.get("program", "")
            if program and attempt.get("program_extracted", False):
                reasoning = _extract_reasoning_from_response(
                    attempt.get("raw_response", {})
                )
                extracted_data.append(
                    LogData(
                        task_id=task_id,
                        program=program,
                        reasoning=reasoning,
                        model=model,
                    )
                )
    else:
        # Unknown API type, but don't warn since many files might be different
        pass

    return extracted_data


def is_valid_code(code: str) -> bool:
    """Check if code is valid Python that can be compiled."""
    try:
        compile(code, "<string>", "exec")
        return True
    except (SyntaxError, TokenError, ValueError, TypeError):
        return False


def process_program(log_data: LogData) -> Optional[SoarProgramExample]:
    """Process a single program entry with task data and compute correctness."""

    try:
        program_to_execute = strip_comments(log_data["program"])
    except Exception:
        # If this failed because the original code was invalid, that's fine - return None
        if not is_valid_code(log_data["program"]):
            return None
        else:
            # If it failed for some other reason, re-raise
            raise

    # Load task data - this could fail due to missing files, which is unexpected
    task_data = task_loader.load_task(log_data["task_id"])

    # Test the program using the new ArcTester
    test_result = arc_tester.test_program(program_to_execute, task_data)

    # If the program failed for any input, return None
    if not test_result.success:
        return None

    # For generation, we'll default to 0 since it's not available in current logs
    generation = 0

    training_example = SoarProgramExample(
        task_id=log_data["task_id"],
        reasoning=log_data["reasoning"],
        code=program_to_execute,  # Store the version we actually executed
        correct_train_input=test_result.correct_train_input,
        correct_test_input=test_result.correct_test_input,
        predicted_train_output=test_result.train_outputs,
        predicted_test_output=test_result.test_outputs,
        train_input=test_result.train_inputs,
        test_input=test_result.test_inputs,
        model=log_data["model"],
        generation=generation,
    )

    # Validate against PyArrow schema immediately
    _validate_training_example_against_schema(training_example)

    return training_example


def extract_and_process_programs_from_log(log_path: str) -> List[SoarProgramExample]:
    """Extract and process programs from a single log file."""
    try:
        log_data_list = extract_log_data(log_path)
        
        # Process each log entry and filter out None results
        results = []
        for log_data in log_data_list:
            try:
                result = process_program(log_data)
                if result is not None:
                    results.append(result)
            except Exception as e:
                # Skip programs that fail to process, but log the error for debugging
                print(f"Error processing program in {log_path}: {e}")
                continue
        
        return results
        
    except Exception as e:
        print(f"Error processing log file {log_path}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Extract programs from log files to parquet"
    )
    parser.add_argument(
        "--logs-pattern",
        type=str,
        default="logs/**/*.json",
        help="Glob pattern for log files (default: logs/**/*.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="programs.parquet",
        help="Output parquet file (default: programs.parquet)",
    )

    args = parser.parse_args()

    # Calculate number of workers (total cores - 2, minimum 1)
    max_workers = max(1, multiprocessing.cpu_count() - 2)
    print(
        f"Using {max_workers} worker processes (total cores: {multiprocessing.cpu_count()})"
    )

    # Find log files
    log_files = glob.glob(args.logs_pattern, recursive=True)
    if not log_files:
        print(f"No log files found matching pattern: {args.logs_pattern}")
        return

    print(f"Processing {len(log_files)} log files...")

    # Process all files in parallel using multiprocessing
    all_programs = []
    output_path = Path(args.output)
    
    with multiprocessing.Pool(processes=max_workers) as pool:
        # Process files in parallel with progress reporting
        results = []
        for i, result in enumerate(pool.imap(extract_and_process_programs_from_log, log_files)):
            results.append(result)
            # Simple progress reporting every 100 files
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(log_files)} files")
        
        # Flatten results and collect all programs
        for programs in results:
            all_programs.extend(programs)

    print(
        f"Extracted and validated {len(all_programs)} programs from {len(log_files)} files"
    )

    # Final save
    _save_programs_to_parquet(all_programs, output_path)

    # Print some basic stats
    if all_programs:
        df = pd.DataFrame(all_programs)
        print("\nStats:")
        print(f"  Unique tasks: {df['task_id'].nunique()}")
        print(f"  Unique models: {df['model'].nunique()}")
        
        # Calculate average accuracies from boolean lists
        train_accuracies = [sum(correct_list) / len(correct_list) if correct_list else 0.0 
                           for correct_list in df['correct_train_input']]
        test_accuracies = [sum(correct_list) / len(correct_list) if correct_list else 0.0 
                          for correct_list in df['correct_test_input']]
        
        print(f"  Average training accuracy: {sum(train_accuracies) / len(train_accuracies):.3f}")
        print(f"  Average test accuracy: {sum(test_accuracies) / len(test_accuracies):.3f}")


if __name__ == "__main__":
    main()
