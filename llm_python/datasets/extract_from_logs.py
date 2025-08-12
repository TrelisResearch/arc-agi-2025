#!/usr/bin/env python3

import argparse
import glob
import json
from pathlib import Path
from typing import List, Optional, TypedDict, Set
import multiprocessing
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tokenize import TokenError
import hashlib

from .schema import PARQUET_SCHEMA

from .schema import SoarProgramExample
from ..utils.code import strip_comments
from ..utils.arc_tester import ArcTester
from ..utils.task_loader import TaskLoader

# Initialize utilities
task_loader = TaskLoader()
arc_tester = ArcTester(timeout=2, executor_type="unrestricted")


class LogData(TypedDict):
    """Schema for data extracted from log files"""

    task_id: str
    program: str
    reasoning: str
    model: str


def _create_program_hash(task_id: str, code: str) -> str:
    """Create a hash for a (task_id, code) pair to identify duplicates."""
    content = f"{task_id}:{code}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _load_existing_programs_index(output_path: Path) -> Set[str]:
    """Load existing programs and return a set of hashes for duplicate detection."""
    if not output_path.exists():
        return set()

    try:
        # Read only the columns we need for hashing to reduce memory usage
        df = pd.read_parquet(output_path, columns=["task_id", "code"])
        existing_hashes = set()

        # Process in chunks to reduce memory usage
        for _, row in df.iterrows():
            hash_key = _create_program_hash(row["task_id"], row["code"])
            existing_hashes.add(hash_key)

        # Delete the dataframe to free memory immediately
        del df

        print(
            f"Loaded {len(existing_hashes)} existing programs for duplicate detection"
        )
        return existing_hashes
    except Exception as e:
        print(f"Error loading existing parquet file: {e}")
        return set()


def _save_programs_to_parquet(
    programs: List[SoarProgramExample], output_path: Path, append_mode: bool = False
) -> None:
    """Save programs to parquet file with explicit schema validation."""
    if not programs:
        print("No new programs to save!")
        return

    new_df = pd.DataFrame(programs)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if append_mode and output_path.exists():
        # Append to existing file more efficiently
        try:
            # Read existing file
            existing_df = pd.read_parquet(output_path)

            # Combine dataframes
            df = pd.concat([existing_df, new_df], ignore_index=True)
            print(
                f"Merged {len(existing_df)} existing programs with {len(new_df)} new programs"
            )

            # Free memory immediately
            del existing_df
        except Exception as e:
            print(f"Error reading existing file for append: {e}, creating new file")
            df = new_df
    else:
        df = new_df
        print(f"Saving {len(new_df)} new programs (no existing data)")

    # Convert to PyArrow table with explicit schema and save
    table = pa.Table.from_pandas(df, schema=PARQUET_SCHEMA)
    pq.write_table(table, output_path)
    print(f"Saved {len(df)} total programs to {output_path}")

    # Free memory
    del df, table


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


def _has_oversized_grids(outputs: List[List[List[int]]], max_size: int = 40) -> bool:
    """Check if any output grid exceeds the maximum size (default 40x40)."""
    for output in outputs:
        if output:  # Skip None/empty outputs
            # Check if it's a valid 2D grid
            if isinstance(output, list) and len(output) > 0:
                height = len(output)
                if height > max_size:
                    return True

                # Check width of first row (assuming rectangular grid)
                if isinstance(output[0], list):
                    width = len(output[0])
                    if width > max_size:
                        return True
    return False


def process_program(
    log_data: LogData, existing_hashes: Set[str]
) -> Optional[SoarProgramExample]:
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

    # Check if this (task_id, code) pair already exists
    program_hash = _create_program_hash(log_data["task_id"], program_to_execute)
    if program_hash in existing_hashes:
        return None  # Skip duplicate

    # Load task data - this could fail due to missing files, which is unexpected
    task_data = task_loader.load_task(log_data["task_id"])

    # Test the program using the new ArcTester
    test_result = arc_tester.test_program(program_to_execute, task_data)

    # If the program failed for any input, return None
    if not test_result.success:
        return None

    # Check if any output grids exceed 40x40 size limit
    train_outputs_no_none = [
        output for output in test_result.train_outputs if output is not None
    ]
    test_outputs_no_none = [
        output for output in test_result.test_outputs if output is not None
    ]

    if _has_oversized_grids(train_outputs_no_none) or _has_oversized_grids(
        test_outputs_no_none
    ):
        return None

    # For generation, we'll default to 0 since it's not available in current logs
    generation = 0

    training_example = SoarProgramExample(
        task_id=log_data["task_id"],
        reasoning=log_data["reasoning"],
        code=program_to_execute,  # Store the version we actually executed
        correct_train_input=test_result.correct_train_input,
        correct_test_input=test_result.correct_test_input,
        predicted_train_output=[
            output if output is not None else [] for output in test_result.train_outputs
        ],
        predicted_test_output=[
            output if output is not None else [] for output in test_result.test_outputs
        ],
        train_input=test_result.train_inputs,
        test_input=test_result.test_inputs,
        model=log_data["model"],
        generation=generation,
    )

    # Validate against PyArrow schema immediately
    _validate_training_example_against_schema(training_example)

    # Add this program's hash to the set for future duplicate detection within this run
    existing_hashes.add(program_hash)

    return training_example


def extract_and_process_programs_from_log(args_tuple) -> List[SoarProgramExample]:
    """Extract and process programs from a single log file."""
    log_path, existing_hashes = args_tuple

    # Create a local copy of hashes for this worker to modify
    # This prevents race conditions while still avoiding duplicates from existing data
    local_hashes = (
        existing_hashes.copy()
        if isinstance(existing_hashes, set)
        else set(existing_hashes)
    )

    try:
        log_data_list = extract_log_data(log_path)

        # Process each log entry and filter out None results
        results = []
        for log_data in log_data_list:
            try:
                result = process_program(log_data, local_hashes)
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of programs to process before saving (default: 1000)",
    )

    args = parser.parse_args()

    # Calculate number of workers (total cores - 2, minimum 1)
    max_workers = max(1, multiprocessing.cpu_count() - 2)
    print(
        f"Using {max_workers} worker processes (total cores: {multiprocessing.cpu_count()})"
    )

    output_path = Path(args.output)

    # Load existing programs for duplicate detection
    existing_hashes = _load_existing_programs_index(output_path)

    # Find log files
    log_files = glob.glob(args.logs_pattern, recursive=True)
    if not log_files:
        print(f"No log files found matching pattern: {args.logs_pattern}")
        return

    print(f"Processing {len(log_files)} log files in batches of {args.batch_size}...")

    # Process files in batches to control memory usage
    total_new_programs = 0
    batch_programs = []
    files_processed = 0

    # Don't copy the hash set for each worker - use shared reference
    # This reduces memory usage significantly
    args_list = [(log_file, existing_hashes) for log_file in log_files]

    with multiprocessing.Pool(processes=max_workers) as pool:
        # Process files and save in batches
        for i, result in enumerate(
            pool.imap(extract_and_process_programs_from_log, args_list)
        ):
            batch_programs.extend(result)
            files_processed += 1

            # Simple progress reporting every 100 files
            if files_processed % 100 == 0:
                print(
                    f"  Processed {files_processed}/{len(log_files)} files, {len(batch_programs)} programs in current batch"
                )

            # Save batch when it reaches the specified size
            if len(batch_programs) >= args.batch_size:
                if batch_programs:  # Only save if we have programs
                    append_mode = (
                        total_new_programs > 0
                    )  # Append if not the first batch
                    _save_programs_to_parquet(batch_programs, output_path, append_mode)
                    total_new_programs += len(batch_programs)
                    print(
                        f"  Saved batch of {len(batch_programs)} programs. Total saved: {total_new_programs}"
                    )
                    batch_programs = []  # Clear the batch

        # Save any remaining programs in the final batch
        if batch_programs:
            append_mode = total_new_programs > 0
            _save_programs_to_parquet(batch_programs, output_path, append_mode)
            total_new_programs += len(batch_programs)
            print(f"  Saved final batch of {len(batch_programs)} programs")

    print(
        f"Extracted and validated {total_new_programs} new programs from {len(log_files)} files"
    )

    # Print some basic stats on the new programs (read only final batch to avoid memory issues)
    if total_new_programs > 0:
        print("\nFinal statistics:")
        try:
            # Read the output file to get total stats
            total_df = pd.read_parquet(output_path)
            print(f"  Total programs in output file: {len(total_df)}")
            print(f"  Total unique tasks: {total_df['task_id'].nunique()}")
            print(f"  Total unique models: {total_df['model'].nunique()}")

            # Calculate average accuracies from boolean lists (on full dataset)
            train_accuracies = [
                sum(correct_list) / len(correct_list) if correct_list else 0.0
                for correct_list in total_df["correct_train_input"]
            ]
            test_accuracies = [
                sum(correct_list) / len(correct_list) if correct_list else 0.0
                for correct_list in total_df["correct_test_input"]
            ]

            print(
                f"  Average training accuracy: {sum(train_accuracies) / len(train_accuracies):.3f}"
            )
            print(
                f"  Average test accuracy: {sum(test_accuracies) / len(test_accuracies):.3f}"
            )

        except Exception as e:
            print(f"Error reading final parquet file for stats: {e}")


if __name__ == "__main__":
    main()
