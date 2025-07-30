#!/usr/bin/env python3

import argparse
import glob
import json
from pathlib import Path
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .schema import TrainingExample, LogData
from .code import strip_comments
from ..utils.scoring import ProgramExecutor
from ..utils.task_loader import TaskLoader, TaskData, Grid, TaskExample
import time

# Initialize utilities
task_loader = TaskLoader()
program_executor = ProgramExecutor(timeout=1, executor_type="unrestricted")

# Define explicit PyArrow schema for our parquet file
PARQUET_SCHEMA = pa.schema([
    ('task_id', pa.string()),
    ('code', pa.string()),
    ('reasoning', pa.string()),
    ('model', pa.string()),
    ('train_correct_fraction', pa.float64()),
    ('test_correct_fraction', pa.float64()),
    ('sample_inputs', pa.list_(pa.list_(pa.list_(pa.int64())))),  # List[List[List[int]]]
    ('sample_outputs', pa.list_(pa.list_(pa.list_(pa.int64())))),  # List[List[List[int]]]
])


def _save_programs_to_parquet(programs: List[TrainingExample], output_path: Path) -> None:
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


def _validate_training_example_against_schema(example: TrainingExample) -> None:
    """Validate that a TrainingExample can be encoded by PyArrow with our schema."""
    try:
        # Convert to dict format
        example_dict = {
            'task_id': example['task_id'],
            'code': example['code'],
            'reasoning': example['reasoning'],
            'model': example['model'],
            'train_correct_fraction': example['train_correct_fraction'],
            'test_correct_fraction': example['test_correct_fraction'],
            'sample_inputs': example['sample_inputs'],
            'sample_outputs': example['sample_outputs'],
        }
        
        # Create arrays for each column with single values
        arrays = []
        for field in PARQUET_SCHEMA:
            arrays.append(pa.array([example_dict[field.name]], type=field.type))
        
        # Try to create a PyArrow table with our schema
        pa.table(arrays, schema=PARQUET_SCHEMA)
        
        # If we get here, the data is valid
        return
        
    except Exception as e:
        raise ValueError(f"TrainingExample failed schema validation: {e}")


def _validate_grid(output) -> Grid:
    if output is not None:
        # Check if output is a list of lists of ints
        if isinstance(output, list) and all(
            isinstance(row, list) and all(isinstance(val, int) for val in row)
            for row in output
        ):
            return output
        else:
            raise ValueError(f"Object is not a list of lists of ints: {output}")


def _execute_on_examples(program: str, examples: List[TaskExample]) -> List:
    """Execute program on a list of examples and return outputs."""
    if not examples:
        return []

    # Extract inputs for bulk execution
    inputs = [example["input"] for example in examples]

    # Use bulk execution for better performance
    results = program_executor.execute_program_bulk(program, inputs)

    # Extract just the outputs (ignore error messages and timeout flags)
    outputs = [_validate_grid(result[0]) for result in results]

    return outputs


def _run_program_on_task(
    program: str, task_data: TaskData
) -> tuple[List[Grid], List[Grid]]:
    """
    Execute program on task data and return outputs.

    Returns:
        train_outputs, test_outputs
    """
    # Merge train and test examples so we can do this in one bulk call (faster).
    all_examples = task_data["train"] + task_data["test"]
    all_outputs = _execute_on_examples(program, all_examples)
    # Unmerge outputs
    train_len = len(task_data["train"])
    train_outputs = all_outputs[:train_len]
    test_outputs = all_outputs[train_len:]

    return train_outputs, test_outputs


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


def _compute_accuracy(outputs: List[Grid], examples: List[TaskExample]) -> float:
    """Compute accuracy fraction for outputs against expected examples."""
    if not examples:
        return 0.0

    correct = sum(
        1
        for i, example in enumerate(examples)
        if outputs[i] is not None and outputs[i] == example["output"]
    )
    return correct / len(examples)


def process_program(log_data: LogData) -> TrainingExample:
    """Process a single program entry with task data and compute accuracy."""

    # Clean code first
    cleaned_code = strip_comments(log_data["program"])

    # Test that cleaned code compiles, fallback to original if needed
    try:
        compile(cleaned_code, "<cleaned>", "exec")
        program_to_execute = cleaned_code
    except SyntaxError:
        try:
            compile(log_data["program"], "<original>", "exec")
            print(
                f"Warning: Cleaned code for task {log_data['task_id']} failed to compile, using original."
            )
            program_to_execute = log_data["program"]
        except SyntaxError:
            # Both versions failed to compile - use original anyway for consistency
            # This is not unexpected, so no warning needed
            program_to_execute = log_data["program"]

    # Load task data - this could fail due to missing files, which is unexpected
    task_data = task_loader.load_task(log_data["task_id"])

    # Execute program and get results - expected errors are handled within the function
    train_outputs, test_outputs = _run_program_on_task(program_to_execute, task_data)

    # Replace None outputs with empty lists to avoid PyArrow serialization issues
    train_outputs = [output if output is not None else [] for output in train_outputs]
    test_outputs = [output if output is not None else [] for output in test_outputs]

    # Extract inputs from task data and combine with outputs for samples
    train_inputs = [example["input"] for example in task_data["train"]]
    test_inputs = [example["input"] for example in task_data["test"]]
    sample_inputs = train_inputs + test_inputs
    sample_outputs = train_outputs + test_outputs

    # Debug check: Validate all outputs are lists
    for i, output in enumerate(sample_outputs):
        if not isinstance(output, list):
            print(
                f"WARNING: Non-list output detected in task {log_data['task_id']}, "
                f"index {i}, type: {type(output)}, value: {output}"
            )
            # Force convert to empty list if it's not a list
            sample_outputs[i] = []

    # Compute accuracies
    train_correct_fraction = _compute_accuracy(train_outputs, task_data["train"])
    test_correct_fraction = _compute_accuracy(test_outputs, task_data["test"])

    training_example = TrainingExample(
        task_id=log_data["task_id"],
        code=program_to_execute,  # Store the version we actually executed
        reasoning=log_data["reasoning"],
        model=log_data["model"],
        train_correct_fraction=train_correct_fraction,
        test_correct_fraction=test_correct_fraction,
        sample_inputs=sample_inputs,
        sample_outputs=sample_outputs,
    )
    
    # Validate against PyArrow schema immediately
    _validate_training_example_against_schema(training_example)
    
    return training_example


def extract_and_process_programs_from_log(log_path: str) -> List[TrainingExample]:
    """Extract and process programs from a single log file."""
    try:
        log_data_list = extract_log_data(log_path)
        return [process_program(log_data) for log_data in log_data_list]
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

    # Process all files in parallel (extract, clean, validate)
    all_programs = []
    processed_count = 0
    output_path = Path(args.output)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_file = {
            executor.submit(extract_and_process_programs_from_log, log_file): log_file
            for log_file in log_files
        }

        # Collect results as they complete
        start_time = time.time()
        for future in as_completed(future_to_file):
            log_file = future_to_file[future]
            processed_count += 1

            # Progress reporting every 100 files
            if processed_count % 100 == 0:
                elapsed = time.time() - start_time
                avg_time_per_file = elapsed / processed_count
                remaining = len(log_files) - processed_count
                est_remaining = avg_time_per_file * remaining
                print(
                    f"  Processed {processed_count}/{len(log_files)} files, "
                    f"Programs extracted: {len(all_programs)}, "
                    f"Elapsed: {elapsed:.1f}s, "
                    f"ETA: {est_remaining / 60:.1f} min"
                )

            try:
                programs = future.result()
                all_programs.extend(programs)
                
                if len(all_programs) > 0 and len(all_programs) % 10000 == 0:
                    print(f"Saving checkpoint at {len(all_programs)} programs...")
                    _save_programs_to_parquet(all_programs, output_path)
                    
            except Exception as e:
                print(f"  Warning: Error processing {log_file}: {e}")

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
        if "train_correct_fraction" in df.columns:
            print(
                f"  Average training accuracy: {df['train_correct_fraction'].mean():.3f}"
            )
            print(f"  Average test accuracy: {df['test_correct_fraction'].mean():.3f}")


if __name__ == "__main__":
    main()
