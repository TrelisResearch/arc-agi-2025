#!/usr/bin/env python3
"""
Visualize outputs of best Trelis and SOAR programs on task 45a5af55 first input grid.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_python.utils.task_loader import get_task_loader
from llm_python.utils.arc_tester import ArcTester
import numpy as np


# ANSI color codes for visualization
COLORS = {
    0: '\033[40m  \033[0m',  # Black
    1: '\033[44m  \033[0m',  # Blue
    2: '\033[41m  \033[0m',  # Red
    3: '\033[42m  \033[0m',  # Green
    4: '\033[43m  \033[0m',  # Yellow
    5: '\033[45m  \033[0m',  # Magenta
    6: '\033[46m  \033[0m',  # Cyan
    7: '\033[47m  \033[0m',  # White
    8: '\033[100m  \033[0m', # Gray
    9: '\033[101m  \033[0m', # Light Red
}

def print_colored_grid(grid, title="Grid"):
    """Print a grid with colors."""
    print(f"\n{title}:")
    for row in grid:
        colored_row = ''.join(COLORS.get(cell, f'{cell:2}') for cell in row)
        print(colored_row)
    print()


def load_task_45a5af55():
    """Load task 45a5af55."""
    task_loader = get_task_loader()
    eval_tasks = task_loader.get_subset_tasks("arc-prize-2025/evaluation")
    for tid, task_data in eval_tasks:
        if tid == "45a5af55":
            return task_data
    raise ValueError("Task not found")


def best_trelis_program(grid):
    """Best Trelis program: 05c45a809390c0f71226b88ba9cdddd0 from a59b95c0"""
    distinct = set()
    for row in grid:
        distinct.update(row)
    N = len(distinct)

    out = []
    for _ in range(N):
        for row in grid:
            out.append(row[:])
    return out


def best_soar_program(grid_lst):
    """Best SOAR program: 62a3ead07c1f9d045257ef87b57d0adb"""
    def expand_grid(grid, size):
        expanded = [[0] * size for _ in range(size)]
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                expanded[i][j] = grid[i][j]
        return expanded

    def fill_pattern(expanded):
        size = len(expanded)
        for i in range(size):
            for j in range(size):
                if expanded[i][j] == 0:
                    if i > 0 and expanded[i-1][j] != 0:
                        expanded[i][j] = expanded[i-1][j]
                    elif j > 0 and expanded[i][j-1] != 0:
                        expanded[i][j] = expanded[i][j-1]
                    elif i < size-1 and expanded[i+1][j] != 0:
                        expanded[i][j] = expanded[i+1][j]
                    elif j < size-1 and expanded[i][j+1] != 0:
                        expanded[i][j] = expanded[i][j+1]
        return expanded

    # Get max dimension
    max_dim = max(len(grid_lst), len(grid_lst[0]) if grid_lst else 0)

    # Expand grid to square
    expanded = expand_grid(grid_lst, max_dim)

    # Fill pattern
    filled = fill_pattern(expanded)

    return filled


def visualize_outputs():
    """Visualize outputs on first input grid."""

    # Load task
    task_data = load_task_45a5af55()
    first_input = task_data["train"][0]["input"]
    first_expected = task_data["train"][0]["output"]

    print("="*60)
    print("TASK 45a5af55 - FIRST TRAINING EXAMPLE")
    print("="*60)

    # Show input and expected output
    print_colored_grid(first_input, "INPUT")
    print_colored_grid(first_expected, "EXPECTED OUTPUT")

    # Run best Trelis program
    try:
        trelis_output = best_trelis_program(first_input)
        print_colored_grid(trelis_output, "BEST TRELIS OUTPUT (56.8% pixel match)")
    except Exception as e:
        print(f"Trelis program error: {e}")

    # Run best SOAR program
    try:
        soar_output = best_soar_program(first_input)
        print_colored_grid(soar_output, "BEST SOAR OUTPUT (51.0% pixel match)")
    except Exception as e:
        print(f"SOAR program error: {e}")

    print("="*60)
    print("ANALYSIS:")
    print(f"• Input: {len(first_input)}x{len(first_input[0])} grid with values {sorted(set(val for row in first_input for val in row))}")
    print(f"• Expected: {len(first_expected)}x{len(first_expected[0])} grid - appears to be some kind of expansion with borders")
    print("• Trelis: Uses distinct color count to repeat the grid N times vertically")
    print("• SOAR: This program is actually just returning the input unchanged (pass-through)")

    print(f"• Trelis output: {len(trelis_output)}x{len(trelis_output[0])}")
    print(f"• SOAR output: {len(soar_output)}x{len(soar_output[0])}")

    # Check if SOAR is really pass-through
    if first_input == soar_output:
        print("✅ CONFIRMED: SOAR program is doing pass-through (input == output)")
    else:
        print("❌ SOAR program is NOT doing pass-through")
    print("="*60)


if __name__ == "__main__":
    visualize_outputs()