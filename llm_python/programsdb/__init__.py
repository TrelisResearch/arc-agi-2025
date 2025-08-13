"""
Local Programs Database Module
"""

from typing import Optional, List
from ..utils.code import normalize_code
from .localdb import get_localdb
from .schema import ProgramSample


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
    Get or create a database instance and log a program if it passes validation.
    
    Uses the singleton pattern to ensure only one database instance per file path.
    
    This function:
    1. Checks if the program has at least one training or test example correct
    2. Filters out programs with invalid grids (larger than 40x40 or not properly 2D)
    3. Normalizes the code (strips comments, normalizes newlines)
    4. Only logs the program if it passes all checks
    
    Args:
        program: Program data conforming to ProgramSample schema
        db_path: Optional path to database file. If None, uses default location.
    """
    # Check if program has at least one correct answer
    has_correct_answer = any(program['correct_train_input']) or any(program['correct_test_input'])
    
    if not has_correct_answer:
        return  # Don't log programs with no correct answers
    
    # Check for invalid grids (oversized or not properly 2D)
    if (_has_invalid_grids(program['predicted_train_output']) or 
        _has_invalid_grids(program['predicted_test_output'])):
        return  # Don't log programs with invalid grids
    
    # Get singleton instance (will create if needed)
    db = get_localdb(db_path)
    
    
    # Normalize the code
    if program['code']:
        program['code'] = normalize_code(program['code'])
    
    # Log the program
    db.add_program(program)


__all__ = ['get_localdb', 'ProgramSample', 'maybe_log_program']
