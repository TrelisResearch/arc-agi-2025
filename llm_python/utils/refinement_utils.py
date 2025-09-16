#!/usr/bin/env python3

import random
import numpy as np
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple, Literal


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


def select_program_for_refinement(
    programs: List[Dict[str, Any]],
    sampling_mode: Literal["uniform", "rex"] = "rex",
    rex_params: Optional[Dict[str, Any]] = None,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Select a program for refinement using different sampling strategies.

    Args:
        programs: List of program dictionaries
        sampling_mode: Strategy to use ("uniform" or "rex")
        rex_params: Parameters for REX algorithm if using rex mode
        debug: Whether to print debug information

    Returns:
        Selected program dictionary, or empty dict if no programs available
    """
    if not programs:
        return {}

    if sampling_mode == "uniform":
        return _uniform_sampling(programs, debug)
    elif sampling_mode == "rex":
        return _rex_sampling(programs, rex_params or {}, debug)
    else:
        raise ValueError(f"Unknown sampling mode: {sampling_mode}")


def _uniform_sampling(programs: List[Dict[str, Any]], debug: bool = False) -> Dict[str, Any]:
    """Uniform random sampling across all programs."""
    if not programs:
        return {}

    selected = random.choice(programs)

    if debug:
        correctness_pct, code_length = calculate_program_metrics(selected)
        print(f"🎲 Selected program: {correctness_pct:.1%} correct, {code_length} chars from {len(programs)} candidates (uniform sampling)")

    return selected


def _rex_sampling(
    programs: List[Dict[str, Any]],
    rex_params: Dict[str, Any],
    debug: bool = False
) -> Dict[str, Any]:
    """
    REX (Refinement through EM-based sampling) algorithm selection.

    Uses Beta distribution sampling based on program accuracy:
    weight = Beta(1 + C * accuracy, 1 + C * (1 - accuracy) + refinement_count)
    """
    if not programs:
        return {}

    C = rex_params.get("C", 20)  # Default hyperparameter from paper
    refinement_counts = rex_params.get("refinement_counts", defaultdict(lambda: 0))

    # Calculate Beta distribution weights for each program
    weights = []
    for program in programs:
        correctness_pct, _ = calculate_program_metrics(program)
        program_id = program.get('row_id', id(program))  # Use row_id or object id as key

        # REX Beta sampling formula
        alpha = 1 + C * correctness_pct
        beta = 1 + C * (1 - correctness_pct) + refinement_counts[program_id]

        # Sample from Beta distribution to get weight
        weight = np.random.beta(alpha, beta)
        weights.append(weight)

    # Select program with highest weight (max sampling from Beta distributions)
    max_idx = np.argmax(weights)
    selected = programs[max_idx]

    # Update refinement count for selected program
    selected_id = selected.get('row_id', id(selected))
    refinement_counts[selected_id] += 1

    if debug:
        correctness_pct, code_length = calculate_program_metrics(selected)
        count = refinement_counts[selected_id]
        print(f"🧬 Selected program: {correctness_pct:.1%} correct, {code_length} chars, refined {count} times from {len(programs)} candidates (REX sampling)")

    return selected


# Keep backward compatibility
def select_best_program_for_refinement(
    programs: List[Dict[str, Any]],
    top_k: int = 100,
    debug: bool = False
) -> Dict[str, Any]:
    """Legacy wrapper for backward compatibility - defaults to uniform sampling."""
    return _uniform_sampling(programs, debug)


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