#!/usr/bin/env python3

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import pandas as pd

from .schema import TrainingExample
from .validation import validate_program
from .code import strip_comments
from ..utils.scoring import ProgramExecutor
from ..utils.task_loader import TaskLoader

# Initialize utilities
task_loader = TaskLoader()
program_executor = ProgramExecutor(timeout=0.5)


def extract_programs_from_log(log_path: str) -> List[TrainingExample]:
    """Extract programs from a single log file."""
    try:
        with open(log_path, "r") as f:
            log_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Warning: JSON decode error in {log_path}: {e}")
        return []
    except Exception as e:
        print(f"Warning: Error reading {log_path}: {e}")
        return []

    task_id = log_data.get("task_id")
    if not task_id:
        # Not a warning since many files might not be program logs
        return []

    programs = []
    api_type = log_data.get("api_type", "")

    try:
        # Handle different API types
        if "multiturn" in api_type:
            multiturn_data = log_data.get("multiturn_data", {})
            turn_details = multiturn_data.get("turn_details", [])
            
            if turn_details:
                first_turn = turn_details[0]
                program = first_turn.get("program", "")
                if program and first_turn.get("program_extracted", False):
                    # Extract reasoning if available
                    reasoning = ""
                    raw_response = first_turn.get("raw_response", {})
                    if isinstance(raw_response, dict) and "choices" in raw_response:
                        choices = raw_response.get("choices", [])
                        if choices and isinstance(choices[0], dict):
                            message = choices[0].get("message", {})
                            reasoning = message.get("reasoning", "")

                    # Compute sample inputs/outputs and accuracy
                    sample_inputs = []
                    sample_outputs = []
                    train_correct_fraction = 0.0
                    test_correct_fraction = 0.0
                    
                    # Try to load task data for computing accuracy
                    try:
                        task_data = task_loader.load_task(task_id, log_data.get("dataset", "arc-agi-1"))
                        
                        # Combine training and test inputs/outputs for samples
                        all_inputs = [example["input"] for example in task_data["train"]] + [example["input"] for example in task_data["test"]]
                        all_outputs = [example["output"] for example in task_data["train"]] + [example["output"] for example in task_data["test"]]
                        sample_inputs = all_inputs
                        sample_outputs = all_outputs
                        
                        # Compute training accuracy
                        train_correct = 0
                        for example in task_data["train"]:
                            predicted_output, error, timed_out = program_executor.execute_program_with_timeout(
                                program, example["input"]
                            )
                            if predicted_output is not None and not error and not timed_out:
                                if predicted_output == example["output"]:
                                    train_correct += 1
                        train_correct_fraction = train_correct / len(task_data["train"]) if task_data["train"] else 0.0
                        
                        # Compute test accuracy
                        test_correct = 0
                        for example in task_data["test"]:
                            predicted_output, error, timed_out = program_executor.execute_program_with_timeout(
                                program, example["input"]
                            )
                            if predicted_output is not None and not error and not timed_out:
                                if predicted_output == example["output"]:
                                    test_correct += 1
                        test_correct_fraction = test_correct / len(task_data["test"]) if task_data["test"] else 0.0
                        
                    except (FileNotFoundError, Exception):
                        # If we can't load task data, just use defaults
                        pass

                    programs.append(TrainingExample(
                        task_id=task_id,
                        code=program,
                        reasoning=reasoning,
                        model=log_data.get("model", ""),
                        train_correct_fraction=train_correct_fraction,
                        test_correct_fraction=test_correct_fraction,
                        sample_inputs=sample_inputs,
                        sample_outputs=sample_outputs,
                    ))

        elif "independent_attempts" in api_type or "all_attempts" in api_type:
            # Handle both independent_attempts and all_attempts
            if "independent_attempts" in api_type:
                attempt_details = log_data.get("independent_attempts_data", {}).get("attempt_details", [])
            else:
                attempt_details = log_data.get("attempt_details", [])

            for attempt in attempt_details:
                program = attempt.get("program", "")
                if program and attempt.get("program_extracted", False):
                    # Extract reasoning if available
                    reasoning = ""
                    raw_response = attempt.get("raw_response", {})
                    if isinstance(raw_response, dict) and "choices" in raw_response:
                        choices = raw_response.get("choices", [])
                        if choices and isinstance(choices[0], dict):
                            message = choices[0].get("message", {})
                            reasoning = message.get("reasoning", "")

                    # Compute sample inputs/outputs and accuracy
                    sample_inputs = []
                    sample_outputs = []
                    train_correct_fraction = 0.0
                    test_correct_fraction = 0.0
                    
                    # Try to load task data for computing accuracy
                    try:
                        task_data = task_loader.load_task(task_id, log_data.get("dataset", "arc-agi-1"))
                        
                        # Combine training and test inputs/outputs for samples
                        all_inputs = [example["input"] for example in task_data["train"]] + [example["input"] for example in task_data["test"]]
                        all_outputs = [example["output"] for example in task_data["train"]] + [example["output"] for example in task_data["test"]]
                        sample_inputs = all_inputs
                        sample_outputs = all_outputs
                        
                        # Compute training accuracy
                        train_correct = 0
                        for example in task_data["train"]:
                            predicted_output, error, timed_out = program_executor.execute_program_with_timeout(
                                program, example["input"]
                            )
                            if predicted_output is not None and not error and not timed_out:
                                if predicted_output == example["output"]:
                                    train_correct += 1
                        train_correct_fraction = train_correct / len(task_data["train"]) if task_data["train"] else 0.0
                        
                        # Compute test accuracy
                        test_correct = 0
                        for example in task_data["test"]:
                            predicted_output, error, timed_out = program_executor.execute_program_with_timeout(
                                program, example["input"]
                            )
                            if predicted_output is not None and not error and not timed_out:
                                if predicted_output == example["output"]:
                                    test_correct += 1
                        test_correct_fraction = test_correct / len(task_data["test"]) if task_data["test"] else 0.0
                        
                    except (FileNotFoundError, Exception):
                        # If we can't load task data, just use defaults
                        pass

                    programs.append(TrainingExample(
                        task_id=task_id,
                        code=program,
                        reasoning=reasoning,
                        model=log_data.get("model", ""),
                        train_correct_fraction=train_correct_fraction,
                        test_correct_fraction=test_correct_fraction,
                        sample_inputs=sample_inputs,
                        sample_outputs=sample_outputs,
                    ))
        else:
            # Unknown API type, but don't warn since many files might be different
            pass

    except Exception as e:
        print(f"Warning: Error extracting programs from {log_path}: {e}")
        return []

    return programs


def process_log_file(log_file: str) -> List[TrainingExample]:
    """Extract, clean, and validate programs from a single log file."""
    # Extract programs
    programs = extract_programs_from_log(log_file)
    if not programs:
        return []
    
    validated_programs = []
    
    for program in programs:
        try:
            # Clean code
            original_code = program["code"]
            cleaned_code = strip_comments(original_code)
            
            # Test that cleaned code still compiles
            try:
                compile(cleaned_code, "<cleaned>", "exec")
                program["code"] = cleaned_code
            except SyntaxError:
                try:
                    compile(original_code, "<original>", "exec")
                    print(f"Warning: Cleaned code from {log_file} failed to compile, keeping original.")
                except SyntaxError:
                    pass
                # Keep original if cleaning breaks it
                pass
            
            # Validate program (always apply transduction filter)
            # validate_program now raises exceptions instead of returning None
            validate_program(program)
            validated_programs.append(program)
            
        except Exception as e:
            # Program failed validation, skip it
            print(f"Warning: Program from {log_file} failed validation: {e}")
            continue
    
    return validated_programs


def main():
    parser = argparse.ArgumentParser(description="Extract programs from log files to parquet")
    parser.add_argument(
        "--logs-pattern", 
        type=str, 
        default="logs/**/*.json",
        help="Glob pattern for log files (default: logs/**/*.json)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="programs.parquet",
        help="Output parquet file (default: programs.parquet)"
    )

    args = parser.parse_args()

    # Calculate number of workers (total cores - 2, minimum 1)
    max_workers = 1 #max(1, multiprocessing.cpu_count() - 2)
    print(f"Using {max_workers} worker processes (total cores: {multiprocessing.cpu_count()})")

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
            executor.submit(process_log_file, log_file): log_file
            for log_file in log_files
        }

        # Collect results as they complete
        for future in as_completed(future_to_file):
            log_file = future_to_file[future]
            processed_count += 1

            # Progress reporting every 1000 files
            if processed_count % 1000 == 0:
                print(f"  Processed {processed_count}/{len(log_files)} files...")

            try:
                programs = future.result()
                all_programs.extend(programs)
            except Exception as e:
                print(f"  Warning: Error processing {log_file}: {e}")

    print(f"Extracted and validated {len(all_programs)} programs from {len(log_files)} files")

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
        if 'train_correct_fraction' in df.columns:
            print(f"  Average training accuracy: {df['train_correct_fraction'].mean():.3f}")
            print(f"  Average test accuracy: {df['test_correct_fraction'].mean():.3f}")
    else:
        print("No programs to save!")


if __name__ == "__main__":
    main()
