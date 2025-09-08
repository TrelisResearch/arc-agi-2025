#!/usr/bin/env python3

import random
from typing import List, Dict, Any, Optional, Tuple


def calculate_program_metrics(program: Dict[str, Any]) -> Tuple[float, int]:
    """
    Calculate metrics for program ranking in refinement selection.
    
    Args:
        program: Program dictionary with 'correct_train_input' and 'code' fields
        
    Returns:
        Tuple of (correctness_percentage, code_length) for ranking
    """
    correct_train = program.get('correct_train_input', [])
    if hasattr(correct_train, 'tolist'):
        correct_train = correct_train.tolist()
    
    if isinstance(correct_train, list) and len(correct_train) > 0:
        correctness_pct = sum(correct_train) / len(correct_train)
    else:
        correctness_pct = 0.0
    
    code_length = len(program.get('code', ''))
    return (correctness_pct, code_length)


def select_best_program_for_refinement(
    programs: List[Dict[str, Any]], 
    top_k: int = 10, 
    debug: bool = False
) -> Dict[str, Any]:
    """
    Select the best program for refinement using smart ranking:
    1. Rank by train correctness percentage (descending)  
    2. Tie-break by code length (ascending - prefer shorter code)
    3. Take top K programs, then random selection from those
    
    Args:
        programs: List of program dictionaries
        top_k: Number of top programs to consider for random selection
        debug: Whether to print debug information
        
    Returns:
        Selected program dictionary, or empty dict if no programs available
    """
    if not programs:
        return {}
    
    def sort_key(program):
        correctness_pct, code_length = calculate_program_metrics(program)
        # Return tuple for sorting: correctness descending, length ascending
        return (-correctness_pct, code_length)
    
    # Sort by: correctness descending, then length ascending
    sorted_programs = sorted(programs, key=sort_key)
    
    # Take top K (or fewer if less available)
    top_programs = sorted_programs[:top_k]
    
    # Random selection from top candidates
    selected = random.choice(top_programs)
    
    # Debug logging
    if debug:
        correctness_pct, code_length = calculate_program_metrics(selected)
        print(f"🎯 Selected program: {correctness_pct:.1%} correct, {code_length} chars from {len(top_programs)} candidates")
    
    return selected


def is_program_valid_for_refinement(program_data: Dict[str, Any]) -> bool:
    """
    Determine if a program is valid for refinement based on new strategy:
    - Exclude transductive programs
    - Exclude programs that are 100% correct on training (nothing to improve)  
    - Include all other programs (0% correct might have useful partial logic)
    
    Args:
        program_data: Program data dictionary or pandas row
        
    Returns:
        True if program should be included for refinement
    """
    # Skip transductive programs
    if program_data.get('is_transductive', False):
        return False
    
    correct_train_input = program_data.get('correct_train_input', [])
    if hasattr(correct_train_input, 'tolist'):
        correct_train_input = correct_train_input.tolist()
    
    if isinstance(correct_train_input, list) and len(correct_train_input) > 0:
        # Include ALL non-transductive programs that are NOT perfect (< 100% correct)
        return not all(correct_train_input)  # Exclude only 100% correct programs
    else:
        # Single value case - include if not fully correct
        return not bool(correct_train_input)