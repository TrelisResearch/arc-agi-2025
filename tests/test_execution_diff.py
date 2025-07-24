#!/usr/bin/env python3

import json
import subprocess
import tempfile
import os

# Test program from line 771
program1 = """def transform(grid):
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

# Test program from line 775  
program2 = """def transform(grid):
    transformed = []
    for row in range(10):
        values = []
        for col in [2, 5, 8]:
            if grid[row][col] != 0:
                values.append(grid[row][col])
        transformed_row = values[:3]
        transformed.append(transformed_row)
    while len(transformed) < 5:
        transformed.append([0, 0, 0])
    return transformed"""

# Test input from line 771
test_input1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 8, 8, 8],
               [0, 0, 4, 0, 0, 0, 0, 0, 0, 8],
               [0, 0, 4, 0, 0, 6, 6, 0, 0, 8],
               [0, 0, 4, 4, 0, 0, 6, 0, 0, 0],
               [0, 0, 4, 0, 0, 6, 6, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 3, 3, 0, 0, 0, 0, 0, 0, 0],
               [3, 3, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

def execute_direct(program, input_grid):
    """Execute program directly in this process"""
    namespace = {}
    exec(program, namespace)
    transform = namespace['transform']
    return transform(input_grid)

def execute_subprocess(program, input_grid, timeout=2.0):
    """Execute program in subprocess (like validation script)"""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(program)
            f.write(f"\n\ntest_input = {input_grid}\n")
            f.write("try:\n")
            f.write("    output = transform(test_input)\n")
            f.write("    print('SUCCESS:', output)\n")
            f.write("except Exception as e:\n")
            f.write("    print('ERROR:', str(e))\n")
            f.write("    import traceback\n")
            f.write("    traceback.print_exc()\n")
            temp_path = f.name
        
        result = subprocess.run(
            ['python3', temp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines:
                if line.startswith('SUCCESS:'):
                    output_str = line[8:].strip()
                    return eval(output_str)
                elif line.startswith('ERROR:'):
                    return f"ERROR: {line[6:].strip()}"
            return f"No output. Stdout: {result.stdout}, Stderr: {result.stderr}"
        else:
            return f"Return code {result.returncode}. Stderr: {result.stderr}"
            
    except Exception as e:
        return f"Exception: {str(e)}"
    finally:
        try:
            os.unlink(temp_path)
        except:
            pass

print("Testing Program 1 (line 771):")
print("="*50)

try:
    direct_result1 = execute_direct(program1, test_input1)
    print(f"Direct execution result: {direct_result1}")
except Exception as e:
    print(f"Direct execution error: {e}")
    import traceback
    traceback.print_exc()

subprocess_result1 = execute_subprocess(program1, test_input1)
print(f"Subprocess execution result: {subprocess_result1}")

print("\n\nDetailed analysis of Program 1:")
print("This program tries to iterate through grid[i] for i in range(5)")
print(f"Length of test_input1: {len(test_input1)}")
print("First 5 rows of test_input1:")
for i in range(5):
    print(f"  Row {i}: {test_input1[i]}")

# Let's trace through what should happen
print("\nManual trace of program 1:")
transformed_grid = []
for i in range(5):
    print(f"\nProcessing row {i}:")
    row = []
    for j in range(3):
        print(f"  Looking for non-zero in row {i}:")
        found = False
        for val in test_input1[i]:
            if val != 0:
                row.append(val)
                print(f"    Found {val}, appending to row")
                found = True
                break
        if not found:
            print(f"    No non-zero found in row {i}")
    print(f"  Row after processing: {row}")
    transformed_grid.append(row)
print(f"\nFinal transformed grid: {transformed_grid}")