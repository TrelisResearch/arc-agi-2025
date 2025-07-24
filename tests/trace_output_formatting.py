#!/usr/bin/env python3

import json

# Simulate the exact process

# 1. The program produces this output for Example 1:
program_output = [[], [8, 8, 8], [4, 4, 4], [4, 4, 4], [4, 4, 4]]

# 2. During training data generation, this is formatted using format_grid:
def format_grid(grid):
    """Format a grid as a string"""
    return '\n'.join(' '.join(str(cell) for cell in row) for row in grid)

formatted_output = format_grid(program_output)
print("Formatted output from generate_training_data.py:")
print(repr(formatted_output))
print("\nAs it appears:")
print(formatted_output)

# 3. This gets included in the prompt, which creates lines like:
print("\n" + "="*50)
print("In the training data prompt:")
print("Output:")
print(formatted_output)

# 4. When validation parses this, it uses parse_grid_from_text:
def parse_grid_from_text(text):
    """Parse a grid from text format like '0 1 2\n3 4 5\n6 7 8'"""
    lines = text.strip().split('\n')
    grid = []
    for line in lines:
        if line.strip():
            row = [int(x) for x in line.strip().split()]
            grid.append(row)
    return grid

# 5. The issue is that the empty list [] becomes an empty line
print("\n" + "="*50)
print("When parsed back:")
parsed = parse_grid_from_text(formatted_output)
print(f"Parsed grid: {parsed}")
print(f"Parsed shape: {len(parsed)}x{len(parsed[0]) if parsed and parsed[0] else 0}")

print("\n" + "="*50)
print("THE ISSUE:")
print("- Program outputs: [[], [8, 8, 8], [4, 4, 4], [4, 4, 4], [4, 4, 4]] (5 rows)")
print("- format_grid() on [] produces an empty string ''")
print("- When joined with newlines, we get a blank line at the start")
print("- parse_grid_from_text() skips empty lines with 'if line.strip()'")
print("- Result: [[8, 8, 8], [4, 4, 4], [4, 4, 4], [4, 4, 4]] (4 rows)")
print("\nThis is why the shapes don't match!")