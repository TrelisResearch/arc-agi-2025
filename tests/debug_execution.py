#!/usr/bin/env python3

import json
import re

# Load line 771
with open('/Users/ronanmcgovern/TR/arc-agi-2025/llm_python/training_data/gemini_synth_50_random_split_1_training.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if i == 770:  # Line 771 (0-indexed)
            example = json.loads(line)
            break

# Extract the program
assistant_msg = example['messages'][2]['content']
pattern = r'```python\s*(.*?)\s*```'
match = re.search(pattern, assistant_msg, re.DOTALL)
program = match.group(1).strip()

print("Program:")
print(program)
print("\n" + "="*50 + "\n")

# Extract Example 1 input
user_msg = example['messages'][1]['content']
lines = user_msg.split('\n')
in_example1 = False
input_lines = []
for line in lines:
    if line.strip() == "Example 1:":
        in_example1 = True
    elif in_example1 and line.strip() == "Input:":
        continue
    elif in_example1 and line.strip() == "Output:":
        break
    elif in_example1 and line.strip() and not line.startswith("Example"):
        input_lines.append(line)

# Parse the grid
grid = []
for line in input_lines:
    if line.strip():
        row = [int(x) for x in line.strip().split()]
        grid.append(row)

print(f"Input grid shape: {len(grid)}x{len(grid[0]) if grid else 0}")
print("First few rows:")
for i in range(min(5, len(grid))):
    print(f"  Row {i}: {grid[i]}")

# Execute the program
namespace = {}
exec(program, namespace)
transform = namespace['transform']

result = transform(grid)
print(f"\nResult: {result}")
print(f"Result shape: {len(result)}x{len(result[0]) if result and result[0] else 0}")

# Debug the execution step by step
print("\n" + "="*50 + "\n")
print("Step-by-step execution:")
transformed_grid = []
for i in range(5):
    print(f"\nRow {i}:")
    row = []
    for j in range(3):
        print(f"  Inner loop j={j}")
        found = False
        for k, val in enumerate(grid[i]):
            if val != 0:
                row.append(val)
                print(f"    Found val={val} at position {k}, appending. Row is now: {row}")
                found = True
                break
        if not found:
            print(f"    No non-zero value found")
    print(f"  Final row: {row}")
    transformed_grid.append(row)

print(f"\nFinal result: {transformed_grid}")