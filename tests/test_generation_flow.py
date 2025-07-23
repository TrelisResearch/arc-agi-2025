#!/usr/bin/env python3

import json
import sys
import tempfile
import subprocess
import os

# Simulate the exact flow from generate_training_data.py

def execute_program(program: str, input_grid, timeout: float = 0.5):
    """Execute a program on an input grid - from generate_training_data.py"""
    try:
        # Create a temporary file with the program
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(program)
            f.write(f"\n\ntest_input = {input_grid}\n")
            f.write("try:\n")
            f.write("    output = transform(test_input)\n")
            f.write("    print('SUCCESS:', output)\n")
            f.write("except Exception as e:\n")
            f.write("    print('ERROR:', str(e))\n")
            temp_path = f.name
        
        # Execute the program
        try:
            result = subprocess.run(
                ['python3', temp_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Parse the output
            if result.returncode == 0:
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines:
                    if line.startswith('SUCCESS:'):
                        output_str = line[8:].strip()
                        output = eval(output_str)
                        return output, None, False
                    elif line.startswith('ERROR:'):
                        error_msg = line[6:].strip()
                        return None, error_msg, False
                
                return None, "No output produced", False
            else:
                return None, result.stderr.strip() or "Unknown error", False
                
        except subprocess.TimeoutExpired:
            return None, "Execution timed out", True
            
    except Exception as e:
        return None, str(e), False
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass

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

print("Testing generation flow:")
print("="*50)

# Run using generation method
gen_output, gen_error, gen_timeout = execute_program(program, input_grid)
print(f"Generation method output: {gen_output}")
print(f"Generation method shape: {len(gen_output)}x{len(gen_output[0]) if gen_output and gen_output[0] else 0}")

# Now let's check what's actually in the training data
print("\n" + "="*50)
print("Checking training data:")

with open('/Users/ronanmcgovern/TR/arc-agi-2025/llm-python/training_data/gemini_synth_50_random_split_1_training.jsonl', 'r') as f:
    for i, line in enumerate(f, 1):
        if i == 771:
            example = json.loads(line)
            user_msg = example['messages'][1]['content']
            
            # Extract Example 1 output
            lines = user_msg.split('\n')
            in_output = False
            output_lines = []
            
            for j, line in enumerate(lines):
                if "Example 1:" in line:
                    ex1_start = j
                elif j > ex1_start and "Output:" in line and "Example 2:" not in '\n'.join(lines[ex1_start:j]):
                    in_output = True
                    continue
                elif in_output:
                    if line.strip().startswith("Example"):
                        break
                    output_lines.append(line)
            
            print("Raw output lines from training data:")
            for k, line in enumerate(output_lines[:6]):  # First 6 lines
                print(f"  Line {k}: '{line}'")
            
            # Parse the output
            parsed_output = []
            for line in output_lines:
                if line.strip():
                    row = [int(x) for x in line.strip().split()]
                    parsed_output.append(row)
            
            print(f"\nParsed output from training data: {parsed_output}")
            print(f"Shape: {len(parsed_output)}x{len(parsed_output[0]) if parsed_output and parsed_output[0] else 0}")
            
            break