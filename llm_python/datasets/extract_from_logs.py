#!/usr/bin/env python3

import argparse
import glob
import json
from typing import List, Optional, TypedDict
import multiprocessing
from tokenize import TokenError
import hashlib

from ..utils.code import strip_comments
from ..utils.arc_tester import ArcTester
from ..utils.task_loader import TaskLoader
from ..programsdb import get_localdb, maybe_log_program, ProgramSample, should_log_program

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
    log_data: LogData, db_path: Optional[str] = None
) -> Optional[ProgramSample]:
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

    # Quick check if this program should be processed
    if not should_log_program(log_data["task_id"], program_to_execute, db_path):
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

    # Create ProgramSample directly for database storage
    program_sample = ProgramSample(
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
        model=log_data["model"]
    )

    return program_sample


def extract_and_process_programs_from_log(args_tuple) -> List[ProgramSample]:
    """Extract and process programs from a single log file."""
    log_path, db_path = args_tuple

    try:
        log_data_list = extract_log_data(log_path)

        # Process each log entry and filter out None results
        results = []
        for log_data in log_data_list:
            try:
                result = process_program(log_data, db_path)
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
        description="Extract programs from log files to local database"
    )
    parser.add_argument(
        "--logs-pattern",
        type=str,
        default="logs/**/*.json",
        help="Glob pattern for log files (default: logs/**/*.json)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to local database file (default: uses programsdb default location)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of programs to process before logging progress (default: 1000)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of worker processes (default: total cores - 2, minimum 1)",
    )

    args = parser.parse_args()

    # Calculate number of workers
    if args.max_workers is not None:
        max_workers = max(1, args.max_workers)
    else:
        max_workers = max(1, multiprocessing.cpu_count() - 2)
    print(
        f"Using {max_workers} worker processes (total cores: {multiprocessing.cpu_count()})"
    )

    # Get database instance
    db = get_localdb(args.db_path)
    initial_count = db.count_programs()
    print(f"Database currently contains {initial_count} programs")

    # Find log files
    log_files = glob.glob(args.logs_pattern, recursive=True)
    if not log_files:
        print(f"No log files found matching pattern: {args.logs_pattern}")
        return

    print(f"Processing {len(log_files)} log files...")

    # Process files in batches to control memory usage
    total_new_programs = 0
    files_processed = 0

    # Create args list with database path
    args_list = [(log_file, args.db_path) for log_file in log_files]

    with multiprocessing.Pool(processes=max_workers) as pool:
        # Process files and log to database
        for i, result in enumerate(
            pool.imap(extract_and_process_programs_from_log, args_list)
        ):
            # Log each program to the database using maybe_log_program
            for program_sample in result:
                maybe_log_program(program_sample, args.db_path)
            
            total_new_programs += len(result)
            files_processed += 1

            # Simple progress reporting every 100 files
            if files_processed % 100 == 0:
                current_count = db.count_programs()
                print(
                    f"  Processed {files_processed}/{len(log_files)} files, "
                    f"found {len(result)} programs in latest batch, "
                    f"database now has {current_count} programs"
                )

            # Log progress periodically
            if total_new_programs > 0 and total_new_programs % args.batch_size == 0:
                current_count = db.count_programs()
                actually_logged = current_count - initial_count
                print(
                    f"  Processed {total_new_programs} programs total, "
                    f"actually logged {actually_logged} new programs to database"
                )

    final_count = db.count_programs()
    actually_logged = final_count - initial_count
    
    print(
        f"Processed {total_new_programs} programs from {len(log_files)} files, "
        f"actually logged {actually_logged} new programs to database"
    )

    # Print database statistics
    if actually_logged > 0:
        print("\nFinal database statistics:")
        print(f"  Total programs in database: {final_count}")
        print(f"  Total unique tasks: {db.get_task_count()}")
        
        # Get some sample statistics by querying the database
        try:
            # Get model distribution
            models = db.connection.execute(
                "SELECT model, COUNT(*) as count FROM programs GROUP BY model ORDER BY count DESC"
            ).fetchall()
            print("  Programs by model:")
            for model, count in models[:5]:  # Show top 5 models
                print(f"    {model}: {count}")
                
        except Exception as e:
            print(f"Error getting database statistics: {e}")


if __name__ == "__main__":
    main()
