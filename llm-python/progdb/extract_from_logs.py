#!/usr/bin/env python3

import argparse
import glob
import json
from pathlib import Path
from typing import List
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import pandas as pd

from .schema import TrainingExample, LogData
from .code import strip_comments
from ..utils.scoring import ProgramExecutor
from ..utils.task_loader import TaskLoader, TaskData, Grid, TaskExample

# Initialize utilities
task_loader = TaskLoader()
program_executor = ProgramExecutor(timeout=1)


def _execute_on_examples(program: str, examples: List[TaskExample]) -> List[Grid]:
    """Execute program on a list of examples and return outputs."""
    outputs = []
    
    for example in examples:
        predicted_output, error, timed_out = (
            program_executor.execute_program_with_timeout(program, example["input"])
        )
        # Don't log expected execution errors (code failures, timeouts, etc.)
        # These are returned as None and will be handled gracefully
        outputs.append(predicted_output)
    
    return outputs


def _run_program_on_task(
    program: str, task_data: TaskData
) -> tuple[List[Grid], List[Grid]]:
    """
    Execute program on task data and return outputs.

    Returns:
        train_outputs, test_outputs
    """
    train_outputs = _execute_on_examples(program, task_data["train"])
    test_outputs = _execute_on_examples(program, task_data["test"])
    
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
            attempt_details = log_data.get("independent_attempts_data", {}).get("attempt_details", [])
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
        1 for i, example in enumerate(examples)
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
    train_outputs, test_outputs = _run_program_on_task(
        program_to_execute, task_data
    )

    # Extract inputs from task data and combine with outputs for samples
    train_inputs = [example["input"] for example in task_data["train"]]
    test_inputs = [example["input"] for example in task_data["test"]]
    sample_inputs = train_inputs + test_inputs
    sample_outputs = train_outputs + test_outputs

    # Compute accuracies
    train_correct_fraction = _compute_accuracy(train_outputs, task_data["train"])
    test_correct_fraction = _compute_accuracy(test_outputs, task_data["test"])

    return TrainingExample(
        task_id=log_data["task_id"],
        code=program_to_execute,  # Store the version we actually executed
        reasoning=log_data["reasoning"],
        model=log_data["model"],
        train_correct_fraction=train_correct_fraction,
        test_correct_fraction=test_correct_fraction,
        sample_inputs=sample_inputs,
        sample_outputs=sample_outputs,
    )


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

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_file = {
            executor.submit(extract_and_process_programs_from_log, log_file): log_file
            for log_file in log_files
        }

        # Collect results as they complete
        for future in as_completed(future_to_file):
            log_file = future_to_file[future]
            processed_count += 1

            # Progress reporting every 100 files
            if processed_count % 100 == 0:
                print(f"  Processed {processed_count}/{len(log_files)} files...")

            try:
                programs = future.result()
                all_programs.extend(programs)
            except Exception as e:
                print(f"  Warning: Error processing {log_file}: {e}")

    print(
        f"Extracted and validated {len(all_programs)} programs from {len(log_files)} files"
    )

    # Convert to DataFrame and save as parquet
    if all_programs:
        df = pd.DataFrame(all_programs)

        # Ensure output directory exists
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(output_path, index=False)
        print(f"Saved {len(all_programs)} programs to {output_path}")

        # Print some basic stats
        print("\nStats:")
        print(f"  Unique tasks: {df['task_id'].nunique()}")
        print(f"  Unique models: {df['model'].nunique()}")
        if "train_correct_fraction" in df.columns:
            print(
                f"  Average training accuracy: {df['train_correct_fraction'].mean():.3f}"
            )
            print(f"  Average test accuracy: {df['test_correct_fraction'].mean():.3f}")
    else:
        print("No programs to save!")


if __name__ == "__main__":
    main()
