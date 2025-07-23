#!/usr/bin/env python3

import json
import re

# Load line 771
with open('/Users/ronanmcgovern/TR/arc-agi-2025/llm-python/training_data/gemini_synth_50_random_split_1_training.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if i == 770:  # Line 771 (0-indexed)
            example = json.loads(line)
            break

# Extract Example 1 output
user_msg = example['messages'][1]['content']
lines = user_msg.split('\n')

print("Looking for Example 1 output in the user message:")
print("="*50)

in_example1_output = False
output_lines = []
for i, line in enumerate(lines):
    if "Example 1:" in line:
        example1_start = i
    elif "Output:" in line and i > example1_start and "Example 2:" not in '\n'.join(lines[example1_start:i]):
        in_example1_output = True
        print(f"Found 'Output:' at line {i}")
        continue
    elif in_example1_output:
        if line.strip().startswith("Example"):
            break
        output_lines.append(line)

print("\nRaw output lines:")
for i, line in enumerate(output_lines):
    print(f"Line {i}: '{line}' (length: {len(line)})")

# Parse the output grid
output_grid = []
for line in output_lines:
    if line.strip():  # Non-empty line
        row = [int(x) for x in line.strip().split()]
        output_grid.append(row)

print(f"\nParsed output grid: {output_grid}")
print(f"Output shape: {len(output_grid)}x{len(output_grid[0]) if output_grid and output_grid[0] else 0}")

# Now let's check what the validation script expects
print("\n" + "="*50)
print("Checking validation script parsing:")

def parse_grid_from_text(text: str):
    """Parse a grid from text format like '0 1 2\n3 4 5\n6 7 8'"""
    lines = text.strip().split('\n')
    grid = []
    for line in lines:
        if line.strip():
            row = [int(x) for x in line.strip().split()]
            grid.append(row)
    return grid

# Reconstruct the output text as the validation script would see it
output_text = '\n'.join(output_lines[1:])  # Skip the first empty line
print(f"Output text for validation: '{output_text}'")
parsed_grid = parse_grid_from_text(output_text)
print(f"Validation parsed grid: {parsed_grid}")
print(f"Validation shape: {len(parsed_grid)}x{len(parsed_grid[0]) if parsed_grid and parsed_grid[0] else 0}")