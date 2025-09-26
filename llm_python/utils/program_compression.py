#!/usr/bin/env python3

import gzip
from typing import Any, List, Dict
import numpy as np


def _create_grid_string(inputs: List[Any], outputs: List[Any]) -> str:
    """Create string with complete grids separated by newlines."""
    pair_strings = []

    for i in range(min(len(inputs), len(outputs))):
        # Add input grid
        input_grid = inputs[i]
        if hasattr(input_grid, 'tolist'):
            input_grid = input_grid.tolist()

        input_grid_str = ""
        if isinstance(input_grid, (list, tuple)):
            input_rows = []
            for row in input_grid:
                if isinstance(row, (list, tuple)):
                    input_rows.append(''.join(map(str, row)))
                elif hasattr(row, 'tolist'):  # Handle numpy arrays
                    input_rows.append(''.join(map(str, row.tolist())))
                elif hasattr(row, '__iter__'):  # Handle other iterables
                    input_rows.append(''.join(map(str, list(row))))
            input_grid_str = '\n'.join(input_rows)

        # Add output grid
        output_grid = outputs[i]
        if hasattr(output_grid, 'tolist'):
            output_grid = output_grid.tolist()

        output_grid_str = ""
        if isinstance(output_grid, (list, tuple)):
            output_rows = []
            for row in output_grid:
                if isinstance(row, (list, tuple)):
                    output_rows.append(''.join(map(str, row)))
                elif hasattr(row, 'tolist'):  # Handle numpy arrays
                    output_rows.append(''.join(map(str, row.tolist())))
                elif hasattr(row, '__iter__'):  # Handle other iterables
                    output_rows.append(''.join(map(str, list(row))))
            output_grid_str = '\n'.join(output_rows)

        # Join input and output with double newlines
        pair_string = input_grid_str + '\n\n' + output_grid_str
        pair_strings.append(pair_string)

    # Join pairs with triple newlines
    return '\n\n\n'.join(pair_strings)


def calculate_normalized_gzip_ratio(ground_truth_inputs: List[Any], ground_truth_outputs: List[Any], predicted_outputs: List[Any]) -> float:
    """
    Calculate normalized gzip compression ratio comparing predicted vs ground truth patterns.

    Computes: gzip_size(inputs+predicted_outputs) / gzip_size(inputs+ground_truth_outputs)

    Args:
        ground_truth_inputs: List of input grids from the task
        ground_truth_outputs: List of expected output grids from the task
        predicted_outputs: List of predicted output grids from the program

    Returns:
        Normalized compression ratio:
        - < 1.0: predicted pattern is MORE compressible than ground truth
        - = 1.0: predicted pattern has SAME compressibility as ground truth
        - > 1.0: predicted pattern is LESS compressible than ground truth
    """
    if not ground_truth_inputs or not ground_truth_outputs or not predicted_outputs:
        return 1.0

    try:
        # Create strings for both ground truth and predicted patterns
        ground_truth_string = _create_grid_string(ground_truth_inputs, ground_truth_outputs)
        predicted_string = _create_grid_string(ground_truth_inputs, predicted_outputs)

        if not ground_truth_string or not predicted_string:
            return 1.0

        # Compress both patterns
        ground_truth_compressed = gzip.compress(ground_truth_string.encode('utf-8'))
        predicted_compressed = gzip.compress(predicted_string.encode('utf-8'))

        ground_truth_size = len(ground_truth_compressed)
        predicted_size = len(predicted_compressed)

        if ground_truth_size == 0:
            return 1.0

        # Return normalized ratio: predicted_size / ground_truth_size
        return predicted_size / ground_truth_size

    except (TypeError, ValueError) as e:
        return 1.0


def calculate_program_gzip_size(ground_truth_inputs: List[Any], predicted_outputs: List[Any]) -> int:
    """
    Calculate the compressed size in bytes for a program's complete pattern.

    Args:
        ground_truth_inputs: List of input grids from the task
        predicted_outputs: List of predicted output grids from the program

    Returns:
        Compressed size in bytes
    """
    if not ground_truth_inputs or not predicted_outputs:
        return 0

    try:
        rows_as_strings = []

        # Process all input-output pairs (same logic as ratio calculation)
        for i in range(min(len(ground_truth_inputs), len(predicted_outputs))):
            input_grid = ground_truth_inputs[i]
            output_grid = predicted_outputs[i]

            if hasattr(input_grid, 'tolist'):
                input_grid = input_grid.tolist()
            if hasattr(output_grid, 'tolist'):
                output_grid = output_grid.tolist()

            if isinstance(input_grid, (list, tuple)) and isinstance(output_grid, (list, tuple)):
                max_rows = max(len(input_grid), len(output_grid))

                for row_idx in range(max_rows):
                    if row_idx < len(input_grid) and isinstance(input_grid[row_idx], (list, tuple)):
                        input_row_str = ''.join(map(str, input_grid[row_idx]))
                        rows_as_strings.append(input_row_str)

                    if row_idx < len(output_grid) and isinstance(output_grid[row_idx], (list, tuple)):
                        output_row_str = ''.join(map(str, output_grid[row_idx]))
                        rows_as_strings.append(output_row_str)

        if not rows_as_strings:
            return 0

        program_string = '\n'.join(rows_as_strings)
        serialized = program_string.encode('utf-8')
        compressed = gzip.compress(serialized)

        return len(compressed)

    except (TypeError, ValueError) as e:
        return 0