#!/usr/bin/env python3

import random
import numpy as np
import threading
import hashlib
import uuid
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple, Literal


def _extract_correctness_data(program_data: Dict[str, Any], field_name: str = 'correct_train_input') -> List[bool]:
    """
    Extract and normalize correctness data from program data.

    Args:
        program_data: Program data dictionary
        field_name: Field name to extract (default: 'correct_train_input')

    Returns:
        List of boolean values representing correctness
    """
    correct_data = program_data.get(field_name, [])

    # Convert numpy arrays to lists first to avoid ambiguous truth value errors
    if hasattr(correct_data, 'tolist'):
        correct_data = correct_data.tolist()

    # Handle single boolean values
    if isinstance(correct_data, bool):
        return [correct_data]

    # Return list or empty list if invalid
    return correct_data if isinstance(correct_data, list) else []


def _calculate_correctness_percentage(correct_data: List[bool]) -> float:
    """Calculate correctness percentage from boolean list."""
    return sum(correct_data) / len(correct_data) if correct_data else 0.0


def _debug_print_program_selection(program: Dict[str, Any], context: str, extra_info: str = "") -> None:
    """
    Print debug information about program selection in a consistent format.

    Args:
        program: Selected program dictionary
        context: Context string describing the selection method
        extra_info: Additional information to display
    """
    if not program:
        return

    correctness_pct, code_length = calculate_program_metrics(program)
    base_msg = f"{correctness_pct:.1%} correct, {code_length} chars"

    if extra_info:
        print(f"{context}: {base_msg}, {extra_info}")
    else:
        print(f"{context}: {base_msg}")


def _require_non_empty_programs(func):
    """Decorator to ensure programs list is non-empty, returning {} if empty."""
    def wrapper(programs, *args, **kwargs):
        if not programs:
            return {}
        return func(programs, *args, **kwargs)
    return wrapper


class REXProgramPool:
    """
    Thread-safe program pool for REX algorithm that supports growing the pool
    with newly refined programs while sampling from it.
    """

    def __init__(self, initial_programs: List[Dict[str, Any]]):
        """Initialize the pool with initial programs."""
        self.lock = threading.RLock()  # Reentrant lock for nested operations
        self.programs: List[Dict[str, Any]] = initial_programs.copy()
        self.refinement_counts = defaultdict(lambda: 0)
        self.program_hashes = set()  # For deduplication

        # Track refinement success for each program
        self.refinement_success_stats = defaultdict(lambda: {
            'attempts': 0,
            'improvements': 0,
            'total_improvement': 0.0,
            'avg_improvement': 0.0
        })

        # Index initial programs
        for program in self.programs:
            self._add_program_hash(program)

    def _get_program_hash(self, program: Dict[str, Any]) -> str:
        """Generate a hash for program deduplication based on normalized code content."""
        code = program.get('code', '')
        # Normalize for deduplication: lowercase and remove whitespace
        normalized_code = ''.join(code.lower().split())
        return hashlib.sha256(normalized_code.encode('utf-8')).hexdigest()[:16]

    def _add_program_hash(self, program: Dict[str, Any]) -> None:
        """Add program hash to the set for deduplication tracking."""
        program_hash = self._get_program_hash(program)
        self.program_hashes.add(program_hash)

    def sample_program(self, sampling_mode: Literal["uniform", "rex"] = "rex", C: float = 20.0,
                      refinement_bonus_weight: float = 0.5) -> Optional[Dict[str, Any]]:
        """
        Thread-safe sampling from the program pool.

        Args:
            sampling_mode: Sampling strategy to use
            C: REX hyperparameter for beta distribution
            refinement_bonus_weight: Weight for refinement success bonus (0.0-1.0, default 0.5)

        Returns:
            Selected program or None if pool is empty
        """
        with self.lock:
            if not self.programs:
                return None

            if sampling_mode == "uniform":
                return random.choice(self.programs)
            elif sampling_mode == "rex":
                return self._rex_sample(C, refinement_bonus_weight)
            else:
                raise ValueError(f"Unknown sampling mode: {sampling_mode}")

    def _rex_sample(self, C: float, refinement_bonus_weight: float = 0.5) -> Dict[str, Any]:
        """Internal REX sampling with lock already held."""
        # Calculate Beta distribution weights for each program
        weights = []
        quality_scores = []  # For logging

        for program in self.programs:
            correctness_pct, _ = calculate_program_metrics(program)
            program_id = program.get('row_id', id(program))

            # Calculate refinement bonus from success history with weighting
            refinement_bonus = self.refinement_success_stats[program_id]['avg_improvement'] * refinement_bonus_weight

            # Enhanced quality score combining correctness and weighted refinement success
            quality_score = correctness_pct + refinement_bonus
            quality_scores.append(quality_score)

            # REX Beta sampling formula with refinement bonus
            alpha = 1 + C * quality_score
            beta = 1 + C * (1 - correctness_pct) + self.refinement_counts[program_id]

            # Sample from Beta distribution to get weight
            weight = np.random.beta(alpha, beta)
            weights.append(weight)

        # Select program with highest weight
        max_idx = np.argmax(weights)
        selected = self.programs[max_idx]

        # Store quality score for logging (attach to selected program temporarily)
        selected['_rex_quality_score'] = quality_scores[max_idx]

        # Update refinement count for selected program
        selected_id = selected.get('row_id', id(selected))
        self.refinement_counts[selected_id] += 1

        return selected

    def add_programs(self, new_programs: List[Dict[str, Any]], deduplicate: bool = True) -> int:
        """
        Thread-safe addition of new programs to the pool.

        Args:
            new_programs: List of new program dictionaries to add
            deduplicate: Whether to skip programs with duplicate code

        Returns:
            Number of programs actually added (after deduplication)
        """
        with self.lock:
            added_count = 0
            for program in new_programs:
                if deduplicate:
                    program_hash = self._get_program_hash(program)
                    if program_hash in self.program_hashes:
                        continue  # Skip duplicate
                    self._add_program_hash(program)

                self.programs.append(program)
                added_count += 1

            return added_count

    def track_refinement_attempt(self, parent_program_id: str, refined_correctness: float,
                                original_correctness: float) -> None:
        """
        Track a refinement attempt and update success statistics.

        Args:
            parent_program_id: ID of the program that was refined
            refined_correctness: Correctness of the refined program
            original_correctness: Correctness of the original program
        """
        with self.lock:
            stats = self.refinement_success_stats[parent_program_id]
            stats['attempts'] += 1

            # Calculate improvement (can be positive or negative)
            improvement = refined_correctness - original_correctness
            stats['total_improvement'] += improvement  # Always count improvement/degradation

            if improvement > 0:
                stats['improvements'] += 1  # Still track positive improvements for success rate

            # Update running average (now truly reflects average change)
            if stats['attempts'] > 0:
                stats['avg_improvement'] = stats['total_improvement'] / stats['attempts']

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get statistics about the current program pool."""
        with self.lock:
            total_programs = len(self.programs)
            if total_programs == 0:
                return {"total_programs": 0, "avg_correctness": 0.0, "total_refinements": 0}

            correctness_scores = []
            quality_scores = []
            total_attempts = 0
            total_improvements = 0

            for program in self.programs:
                correctness_pct, _ = calculate_program_metrics(program)
                correctness_scores.append(correctness_pct)

                program_id = program.get('row_id', id(program))
                refinement_bonus = self.refinement_success_stats[program_id]['avg_improvement'] * 0.5  # Use default weight
                quality_scores.append(correctness_pct + refinement_bonus)

                # Aggregate refinement success stats
                stats = self.refinement_success_stats[program_id]
                total_attempts += stats['attempts']
                total_improvements += stats['improvements']

            return {
                "total_programs": total_programs,
                "avg_correctness": sum(correctness_scores) / len(correctness_scores),
                "avg_quality_score": sum(quality_scores) / len(quality_scores),
                "total_refinements": sum(self.refinement_counts.values()),
                "unique_hashes": len(self.program_hashes),
                "refinement_success_rate": total_improvements / total_attempts if total_attempts > 0 else 0.0,
                "total_refinement_attempts": total_attempts
            }

    def log_pool_summary(self) -> None:
        """Log a summary of the current pool state."""
        stats = self.get_pool_stats()
        if stats["total_programs"] > 0:
            success_rate = stats['refinement_success_rate']
            quality_score = stats['avg_quality_score']
            print(f"🔍 REX Pool: {stats['total_programs']} programs, "
                  f"{stats['avg_correctness']:.1%} avg correct, "
                  f"{quality_score:.1%} avg quality score, "
                  f"{success_rate:.1%} refinement success rate, "
                  f"{stats['total_refinements']} selections")


def calculate_program_metrics(program: Dict[str, Any]) -> Tuple[float, int]:
    """
    Calculate metrics for program ranking in refinement selection.

    Args:
        program: Program dictionary with 'correct_train_input' and 'code' fields

    Returns:
        Tuple of (correctness_percentage, code_length) for ranking
    """
    correct_data = _extract_correctness_data(program)
    correctness_pct = _calculate_correctness_percentage(correct_data)
    code_length = len(program.get('code', ''))
    return (correctness_pct, code_length)


def select_program_for_refinement(
    programs: List[Dict[str, Any]] = None,
    sampling_mode: Literal["uniform", "rex"] = "rex",
    rex_params: Optional[Dict[str, Any]] = None,
    debug: bool = False,
    program_pool: Optional[REXProgramPool] = None
) -> Dict[str, Any]:
    """
    Select a program for refinement using different sampling strategies.

    Args:
        programs: List of program dictionaries (if not using program_pool)
        sampling_mode: Strategy to use ("uniform" or "rex")
        rex_params: Parameters for REX algorithm if using rex mode
        debug: Whether to print debug information
        program_pool: Optional REXProgramPool for thread-safe operations

    Returns:
        Selected program dictionary, or empty dict if no programs available
    """
    # Use program pool if provided
    if program_pool is not None:
        rex_params = rex_params or {}
        C = rex_params.get("C", 20.0)
        refinement_bonus_weight = rex_params.get("refinement_bonus_weight", 0.5)
        selected = program_pool.sample_program(sampling_mode, C, refinement_bonus_weight)

        if debug and selected:
            pool_stats = program_pool.get_pool_stats()
            mode_str = "REX pool" if sampling_mode == "rex" else "uniform pool"
            extra_info = f"pool: {pool_stats['total_programs']} programs"
            _debug_print_program_selection(selected, f"🔄 Selected from {mode_str}", extra_info)

        return selected or {}

    # Fallback to original list-based approach
    if not programs:
        return {}

    if sampling_mode == "uniform":
        return _uniform_sampling(programs, debug)
    elif sampling_mode == "rex":
        return _rex_sampling(programs, rex_params or {}, debug)
    else:
        raise ValueError(f"Unknown sampling mode: {sampling_mode}")


def create_refined_program_entry(
    original_program: Dict[str, Any],
    refined_code: str,
    task_results: Optional[Dict[str, Any]] = None,
    model: str = "unknown"
) -> Dict[str, Any]:
    """
    Create a new program entry for a refined program that can be added to the pool.

    Args:
        original_program: The original program that was refined
        refined_code: The new refined code
        task_results: Optional task execution results with correctness info
        model: Model name that generated the refinement

    Returns:
        Dictionary suitable for adding to program pool
    """
    # Generate a new row_id for the refined program
    new_row_id = f"refined_{uuid.uuid4().hex[:8]}"

    # Create base program entry
    refined_program = {
        'row_id': new_row_id,
        'code': refined_code,
        'model': model,
        'reasoning': f"Refined from {original_program.get('row_id', 'unknown')}",
        'is_transductive': False,  # Assume refined programs are not transductive
        'parent_program_id': original_program.get('row_id'),  # Track lineage
    }

    # Add task results if provided
    if task_results:
        refined_program.update({
            'correct_train_input': task_results.get('correct_train_input', []),
            'correct_test_input': task_results.get('correct_test_input', []),
            'predicted_train_output': task_results.get('predicted_train_output', []),
            'predicted_test_output': task_results.get('predicted_test_output', []),
        })
    else:
        # Copy from original if no new results provided
        refined_program.update({
            'correct_train_input': original_program.get('correct_train_input', []),
            'correct_test_input': original_program.get('correct_test_input', []),
            'predicted_train_output': original_program.get('predicted_train_output', []),
            'predicted_test_output': original_program.get('predicted_test_output', []),
        })

    return refined_program


@_require_non_empty_programs
def _uniform_sampling(programs: List[Dict[str, Any]], debug: bool = False) -> Dict[str, Any]:
    """Uniform random sampling across all programs."""
    selected = random.choice(programs)

    if debug:
        extra_info = f"from {len(programs)} candidates (uniform sampling)"
        _debug_print_program_selection(selected, "🎲 Selected program", extra_info)

    return selected


@_require_non_empty_programs
def _rex_sampling(
    programs: List[Dict[str, Any]],
    rex_params: Dict[str, Any],
    debug: bool = False
) -> Dict[str, Any]:
    """
    REX (Refinement through EM-based sampling) algorithm selection.
    Legacy function that delegates to REXProgramPool for consistency.
    """

    # Create temporary pool and delegate to pool-based implementation
    temp_pool = REXProgramPool(programs)
    C = rex_params.get("C", 20.0)

    # Pre-populate refinement counts if provided
    refinement_counts = rex_params.get("refinement_counts")
    if refinement_counts:
        temp_pool.refinement_counts.update(refinement_counts)

    result = temp_pool.sample_program("rex", C)

    # Update original refinement_counts dict if provided
    if refinement_counts and result:
        program_id = result.get('row_id', id(result))
        refinement_counts[program_id] = temp_pool.refinement_counts[program_id]

    if debug and result:
        program_id = result.get('row_id', id(result))
        count = temp_pool.refinement_counts[program_id]
        extra_info = f"refined {count} times from {len(programs)} candidates (REX sampling)"
        _debug_print_program_selection(result, "🧬 Selected program", extra_info)

    return result or {}


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

    correct_data = _extract_correctness_data(program_data)
    if not correct_data:
        return True  # Include programs with no correctness data

    # Include ALL non-transductive programs that are NOT perfect (< 100% correct)
    return not all(correct_data)  # Exclude only 100% correct programs