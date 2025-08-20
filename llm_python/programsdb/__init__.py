from typing import Optional, List

from llm_python.transduction.code_classifier import CodeTransductionClassifier
from ..utils.code import normalize_code
from .localdb import get_localdb
from .schema import ProgramSample
import traceback


def should_log_program(program: ProgramSample, db_path: Optional[str] = None) -> bool:
    """
    Check if a program should be logged to the database.

    This function performs all validation checks:
    1. Checks if the program has at least one training or test example correct
    2. Filters out programs with invalid grids (larger than 40x40 or not properly 2D)
    3. Normalizes the code and checks if the (task_id, normalized_code) pair already exists
    """
    try:
        # Check if program has at least one correct answer
        has_correct_answer = any(program["correct_train_input"]) or any(
            program["correct_test_input"]
        )

        if not has_correct_answer:
            return False  # Don't log programs with no correct answers

        # Check for invalid grids (oversized or not properly 2D)
        if _has_invalid_grids(program["predicted_train_output"]) or _has_invalid_grids(
            program["predicted_test_output"]
        ):
            return False  # Don't log programs with invalid grids

        # Normalize the code and check for duplicates
        normalized_code = normalize_code(program["code"]) if program["code"] else ""

        if not normalized_code.strip():
            return False  # Don't log empty programs

        # Still check transduction for logging purposes, but don't filter
        # transduction_tester = CodeTransductionClassifier()
        # is_transductive, confidence = transduction_tester.is_transductive(normalized_code)
        # Removed: filtering based on transduction status

        # Get database instance
        db = get_localdb(db_path)

        # Generate the same key that would be used for storage
        key = db.generate_key(program["task_id"], normalized_code)

        # Check if this key already exists in the database
        result = db.connection.execute(
            "SELECT 1 FROM programs WHERE key = ? LIMIT 1", [key]
        ).fetchone()

        return result is None  # Should process if not found

    except Exception:
        # If there's any error, default to not processing the program but log it, as this is unexpected.
        print("Error determining whether to log program:")
        traceback.print_exc()
        return False


def _has_invalid_grids(outputs: List[List[List[int]]], max_size: int = 40) -> bool:
    """Check if any output grid is invalid (oversized or not properly 2D)."""
    for output in outputs:
        if output:  # Skip None/empty outputs
            # Check if it's a valid 2D grid
            if isinstance(output, list) and len(output) > 0:
                height = len(output)
                if height > max_size:
                    return True

                # Check that all rows are lists and have the same width
                if not isinstance(output[0], list):
                    return True

                expected_width = len(output[0])
                if expected_width > max_size:
                    return True

                # Check all rows have the same width (proper 2D grid)
                for row in output:
                    if not isinstance(row, list) or len(row) != expected_width:
                        return True
    return False


def maybe_log_program(program: ProgramSample, db_path: Optional[str] = None) -> None:
    """
    Log a program to the database if it passes all validation checks.

    Uses should_log_program to determine if the program should be logged,
    then performs the actual logging.

    Args:
        program: Program data conforming to ProgramSample schema
        db_path: Optional path to database file. If None, uses default location.
    """
    # Check if program should be logged
    if not should_log_program(program, db_path):
        return

    # Get database instance
    db = get_localdb(db_path)

    # Create a copy to avoid modifying the original
    program_copy = program.copy()

    # Normalize the code
    if program_copy["code"]:
        program_copy["code"] = normalize_code(program_copy["code"])

    # Log the program
    db.add_program(program_copy)


__all__ = ["get_localdb", "ProgramSample", "maybe_log_program", "should_log_program"]
