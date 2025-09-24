#!/usr/bin/env python3
"""
Program filtering utilities for refinement and fine-tuning data.

This module provides functions to filter out low-quality programs from datasets,
including pass-through programs and single-color predictions when ground truth is multi-colored.
"""

from typing import Dict, Any, Optional, List, Tuple


def has_multi_color_ground_truth(ground_truth_output) -> bool:
    """
    Check if a ground truth output grid has multiple colors (excluding black/0).

    Args:
        ground_truth_output: Ground truth output grid

    Returns:
        True if ground truth contains more than one non-black color
    """
    # Check for None first
    if ground_truth_output is None:
        return False

    # Handle numpy arrays separately to avoid truth value ambiguity
    if hasattr(ground_truth_output, 'tolist'):
        ground_truth_output = ground_truth_output.tolist()

    # Now check for empty or invalid types
    if not ground_truth_output or not isinstance(ground_truth_output, (list, tuple)):
        return False

    # Collect all unique colors
    unique_colors = set()
    for row in ground_truth_output:
        if hasattr(row, 'tolist'):
            row = row.tolist()
        if isinstance(row, (list, tuple)):
            for cell in row:
                if hasattr(cell, 'item'):
                    cell = cell.item()
                # Ensure cell is a valid integer (ARC grids should only contain integers 0-9)
                try:
                    cell = int(cell)
                    unique_colors.add(cell)
                except (ValueError, TypeError):
                    # Skip invalid cell values
                    continue

    # Remove black (0) and count remaining colors
    non_black_colors = unique_colors - {0}
    return len(non_black_colors) > 1

def is_pass_through_program(program_data: Dict[str, Any], task_data: Optional[Dict] = None) -> bool:
    """
    Check if a program is a pass-through program (predicted train outputs == predicted train inputs).

    Args:
        program_data: Program data dictionary
        task_data: Optional task data containing train inputs

    Returns:
        True if all predicted train outputs are identical to their corresponding train inputs
    """
    if not task_data or 'train' not in task_data:
        return False  # Can't check without task data

    predicted_train_outputs = program_data.get('predicted_train_output', [])

    # Handle numpy arrays and empty checks properly
    if predicted_train_outputs is None:
        return False

    # Convert numpy arrays to lists first to avoid truth value ambiguity
    if hasattr(predicted_train_outputs, 'tolist'):
        predicted_train_outputs = predicted_train_outputs.tolist()

    # Now check if empty
    if not predicted_train_outputs:
        return False

    train_inputs = [example.get('input', []) for example in task_data['train']]

    # Check if all predicted outputs are identical to their corresponding inputs
    all_pass_through = True
    for i, (predicted, input_grid) in enumerate(zip(predicted_train_outputs, train_inputs)):
        # Skip if we don't have both predicted and input
        if predicted is None or input_grid is None:
            continue

        # Convert numpy arrays to lists for comparison (recursively handle nested arrays)
        def convert_nested_arrays(obj):
            if hasattr(obj, 'tolist'):
                obj = obj.tolist()
            if isinstance(obj, list):
                return [convert_nested_arrays(item) for item in obj]
            return obj

        predicted = convert_nested_arrays(predicted)
        input_grid = convert_nested_arrays(input_grid)

        # Check if predicted output differs from input
        try:
            if predicted != input_grid:
                all_pass_through = False
                break
        except Exception as e:
            print(f"❌ Error comparing predicted vs input in pass-through check: {e}")
            print(f"   predicted: {predicted}")
            print(f"   input_grid: {input_grid}")
            print(f"   predicted type: {type(predicted)}")
            print(f"   input_grid type: {type(input_grid)}")
            # If we can't compare (e.g., nested numpy arrays), assume they're different (not pass-through)
            all_pass_through = False
            break

    return all_pass_through

def should_filter_program(program_data: Dict[str, Any], task_data: Optional[Dict] = None) -> bool:
    """
    Determine if a program should be filtered out based on quality criteria.

    This function combines all filtering checks:
    - Transductive programs
    - Perfect programs (100% correct on training)
    - Pass-through programs

    Args:
        program_data: Program data dictionary
        task_data: Optional task data for additional filtering checks

    Returns:
        True if program should be filtered out (excluded)
    """
    # Filter pass-through programs only
    if is_pass_through_program(program_data, task_data):
        return True

    # Don't filter anything else - include all programs (even 100% train-correct)
    return False


def filter_programs_with_stats(programs: List[Dict[str, Any]], task_data: Optional[Dict] = None) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Filter a list of programs to remove low-quality ones and return detailed statistics.

    Args:
        programs: List of program dictionaries
        task_data: Optional task data for filtering checks

    Returns:
        Tuple of (filtered_programs, filter_statistics)
        filter_statistics contains counts for each filter type
    """
    stats = {
        'total': len(programs),
        'perfect': 0,
        'pass_through': 0,
        'kept': 0
    }

    kept_programs = []

    for program in programs:
        # Check each filter individually for statistics
        is_perfect = False
        is_pass_through = False

        # Check if perfect
        correct_data = program.get('correct_train_input', [])
        if hasattr(correct_data, 'tolist'):
            correct_data = correct_data.tolist()
        if isinstance(correct_data, bool):
            correct_data = [correct_data]
        if isinstance(correct_data, list) and correct_data and all(correct_data):
            is_perfect = True

        # Check pass-through and single color (require task_data)
        if task_data:
            is_pass_through = is_pass_through_program(program, task_data)

        # Update statistics - now only filtering pass-through programs
        if is_pass_through:
            stats['pass_through'] += 1
        else:
            # Keep all programs except pass-through
            kept_programs.append(program)
            stats['kept'] += 1

        # Still track perfect for statistics, but don't filter them
        if is_perfect:
            stats['perfect'] += 1

    return kept_programs, stats


def filter_programs(programs: List[Dict[str, Any]], task_data: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """
    Filter a list of programs to remove low-quality ones.

    Args:
        programs: List of program dictionaries
        task_data: Optional task data for filtering checks

    Returns:
        Filtered list of programs
    """
    filtered_programs, _ = filter_programs_with_stats(programs, task_data)
    return filtered_programs


def print_filter_statistics(stats: Dict[str, int]) -> None:
    """
    Print detailed filtering statistics in a readable format.

    Args:
        stats: Statistics dictionary from filter_programs_with_stats
    """
    total = stats['total']
    kept = stats['kept']
    filtered = total - kept

    print("\n📊 Program Filtering Statistics:")
    print(f"  Total programs: {total}")
    print(f"  Programs kept: {kept} ({kept/total*100:.1f}%)")
    print(f"  Programs filtered: {filtered} ({filtered/total*100:.1f}%)")
    print("\n🔍 Breakdown of filtered programs:")
    print(f"  Pass-through: {stats['pass_through']} ({stats['pass_through']/total*100:.1f}%)")
    print("\n📈 Statistics (not filtered):")
    print(f"  Perfect (100% correct): {stats['perfect']} ({stats['perfect']/total*100:.1f}%)")
    print("─" * 50)