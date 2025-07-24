#!/usr/bin/env python3

import json

# Test the problematic program
program = """def transform(grid):
    transformed_grid = []
    for i in range(5):
        row = []
        for j in range(3):
            for val in grid[i]:
                if val != 0:
                    row.append(val)
                    break
        transformed_grid.append(row)
    return transformed_grid"""

# Example 1 input from line 771
input_grid = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 8, 8, 8],
    [0, 0, 4, 0, 0, 0, 0, 0, 0, 8],
    [0, 0, 4, 0, 0, 6, 6, 0, 0, 8],
    [0, 0, 4, 4, 0, 0, 6, 0, 0, 0],
    [0, 0, 4, 0, 0, 6, 6, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 3, 3, 0, 0, 0, 0, 0, 0, 0],
    [3, 3, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

print("Direct execution of program:")
print("="*50)

# Execute directly
namespace = {}
exec(program, namespace)
transform = namespace['transform']
direct_output = transform(input_grid)

print(f"Direct output: {direct_output}")
print(f"Direct shape: {len(direct_output)}x{len(direct_output[0]) if direct_output and direct_output[0] else 0}")

# Now let's manually check what should happen based on the user prompt
print("\n" + "="*50)
print("According to the prompt, the expected output for Example 1 should be:")
print("(blank line)")
print("8 8 8")
print("4 4 4") 
print("4 4 4")
print("4 4 4")
print("\nWhich parses to: [[8, 8, 8], [4, 4, 4], [4, 4, 4], [4, 4, 4]]")
print("Shape: 4x3")

print("\nBut the program produces:")
print(direct_output)
print(f"Shape: {len(direct_output)}x{len(direct_output[0]) if direct_output and direct_output[0] else 0}")

print("\n" + "="*50)
print("Key insight:")
print("- The program ALWAYS produces exactly 5 rows (due to 'for i in range(5)')")
print("- But the expected outputs have varying numbers of rows (2, 3, 4, or 5)")
print("- The first row is empty because the first row of input is all zeros")
print("- This is why validation is failing!")

# Let's check if this is really what's in the training data
print("\n" + "="*50)
print("Checking the actual training data file...")

with open('/Users/ronanmcgovern/TR/arc-agi-2025/llm-python/training_data/gemini_synth_50_random_split_1_training.jsonl', 'r') as f:
    for i, line in enumerate(f, 1):
        if i == 771:
            example = json.loads(line)
            
            # Get the assistant's program
            assistant_msg = example['messages'][2]['content']
            print(f"\nAssistant's program in training data:")
            import re
            pattern = r'```python\s*(.*?)\s*```'
            match = re.search(pattern, assistant_msg, re.DOTALL)
            stored_program = match.group(1).strip()
            print(stored_program)
            
            print(f"\nPrograms match: {stored_program == program.strip()}")
            
            break