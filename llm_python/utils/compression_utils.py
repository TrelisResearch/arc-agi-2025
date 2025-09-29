#!/usr/bin/env python3

import gzip
import json
import pickle
from typing import Any, List, Union
import numpy as np


def calculate_gzip_ratio(grid: Any, method: str = 'json') -> float:
    """
    Calculate gzip compression ratio for a grid.

    Args:
        grid: Grid data (numpy array, list of lists, etc.)
        method: Serialization method ('json', 'pickle', 'string')

    Returns:
        Compression ratio (compressed_size / original_size)
        Lower ratios indicate better compressibility
    """
    if grid is None:
        return 1.0

    # Convert numpy arrays to lists for consistent serialization
    if hasattr(grid, 'tolist'):
        grid = grid.tolist()

    # Serialize the grid based on method
    try:
        if method == 'json':
            serialized = json.dumps(grid, separators=(',', ':')).encode('utf-8')
        elif method == 'pickle':
            serialized = pickle.dumps(grid)
        elif method == 'string':
            # Convert to string representation (with newlines between rows)
            if isinstance(grid, (list, tuple)):
                rows_as_strings = []
                for row in grid:
                    if isinstance(row, (list, tuple)):
                        row_str = ''.join(map(str, row))
                        rows_as_strings.append(row_str)
                    else:
                        rows_as_strings.append(str(row))
                serialized = '\n'.join(rows_as_strings).encode('utf-8')
            else:
                serialized = str(grid).encode('utf-8')
        else:
            raise ValueError(f"Unknown serialization method: {method}")

        # Compress with gzip
        compressed = gzip.compress(serialized)

        # Calculate compression ratio
        original_size = len(serialized)
        compressed_size = len(compressed)

        if original_size == 0:
            return 1.0

        return compressed_size / original_size

    except (TypeError, ValueError) as e:
        # Return ratio of 1.0 (no compression) if serialization fails
        return 1.0


def calculate_gzip_size(grid: Any, method: str = 'json') -> int:
    """
    Calculate the compressed size in bytes for a grid.

    Args:
        grid: Grid data (numpy array, list of lists, etc.)
        method: Serialization method ('json', 'pickle', 'string')

    Returns:
        Compressed size in bytes
    """
    if grid is None:
        return 0

    # Convert numpy arrays to lists for consistent serialization
    if hasattr(grid, 'tolist'):
        grid = grid.tolist()

    # Serialize the grid based on method
    try:
        if method == 'json':
            serialized = json.dumps(grid, separators=(',', ':')).encode('utf-8')
        elif method == 'pickle':
            serialized = pickle.dumps(grid)
        elif method == 'string':
            # Convert to string representation (with newlines between rows)
            if isinstance(grid, (list, tuple)):
                rows_as_strings = []
                for row in grid:
                    if isinstance(row, (list, tuple)):
                        row_str = ''.join(map(str, row))
                        rows_as_strings.append(row_str)
                    else:
                        rows_as_strings.append(str(row))
                serialized = '\n'.join(rows_as_strings).encode('utf-8')
            else:
                serialized = str(grid).encode('utf-8')
        else:
            raise ValueError(f"Unknown serialization method: {method}")

        # Compress with gzip
        compressed = gzip.compress(serialized)
        return len(compressed)

    except (TypeError, ValueError) as e:
        # Return 0 if serialization fails
        return 0


def calculate_combined_gzip_ratio(input_grid: Any, output_grid: Any, method: str = 'json') -> float:
    """
    Calculate gzip compression ratio for combined input and output grids.
    Interleaves input and output to better capture transformation patterns.

    Args:
        input_grid: Input grid data
        output_grid: Output grid data
        method: Serialization method ('json', 'pickle', 'string')

    Returns:
        Compression ratio for combined grids
    """
    # For string method, interleave input-output pairs to capture transformations
    if method == 'string':
        return calculate_interleaved_gzip_ratio(input_grid, output_grid)
    else:
        # For json/pickle, use original structure
        combined_data = {
            'input': input_grid,
            'output': output_grid
        }
        return calculate_gzip_ratio(combined_data, method)


def calculate_interleaved_gzip_ratio(input_grid: Any, output_grid: Any) -> float:
    """
    Calculate gzip ratio with input-output interleaving for better transformation capture.
    Format: input_row1, output_row1, input_row2, output_row2, ...
    """
    if input_grid is None or output_grid is None:
        return 1.0

    # Convert numpy arrays to lists
    if hasattr(input_grid, 'tolist'):
        input_grid = input_grid.tolist()
    if hasattr(output_grid, 'tolist'):
        output_grid = output_grid.tolist()

    try:
        rows_as_strings = []

        # Interleave input and output rows
        max_rows = max(len(input_grid) if isinstance(input_grid, (list, tuple)) else 1,
                      len(output_grid) if isinstance(output_grid, (list, tuple)) else 1)

        for i in range(max_rows):
            # Add input row if available
            if isinstance(input_grid, (list, tuple)) and i < len(input_grid):
                input_row = input_grid[i]
                if isinstance(input_row, (list, tuple)):
                    row_str = ''.join(map(str, input_row))
                    rows_as_strings.append(row_str)

            # Add corresponding output row if available
            if isinstance(output_grid, (list, tuple)) and i < len(output_grid):
                output_row = output_grid[i]
                if isinstance(output_row, (list, tuple)):
                    row_str = ''.join(map(str, output_row))
                    rows_as_strings.append(row_str)

        serialized = '\n'.join(rows_as_strings).encode('utf-8')

        # Compress with gzip
        compressed = gzip.compress(serialized)

        # Calculate compression ratio
        original_size = len(serialized)
        compressed_size = len(compressed)

        if original_size == 0:
            return 1.0

        return compressed_size / original_size

    except (TypeError, ValueError) as e:
        return 1.0