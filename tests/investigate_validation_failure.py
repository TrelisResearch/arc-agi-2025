#!/usr/bin/env python3

import json
import sys
import subprocess
import tempfile
import os

sys.path.append('/Users/ronanmcgovern/TR/arc-agi-2025/tests')
from validate_training_data import validate_training_example, execute_program_on_input

# Check specific failing lines
failing_lines = [771, 775]

for line_num in failing_lines:
    print(f"\n{'='*80}")
    print(f"Investigating line {line_num}:")
    print('='*80)
    
    # Load the training example
    with open('/Users/ronanmcgovern/TR/arc-agi-2025/llm-python/training_data/gemini_synth_50_random_split_1_training.jsonl', 'r') as f:
        for i, line in enumerate(f, 1):
            if i == line_num:
                example = json.loads(line)
                break
    
    # Validate the example
    result = validate_training_example(example)
    
    print(f"\nValidation result:")
    print(f"  Status: {result['status']}")
    print(f"  Examples tested: {result['examples_tested']}")
    print(f"  Examples correct: {result['examples_correct']}")
    
    if 'results' in result:
        # Extract the program for manual testing
        messages = example['messages']
        assistant_content = messages[2]['content']
        
        # Extract program
        import re
        pattern = r'```python\s*(.*?)\s*```'
        match = re.search(pattern, assistant_content, re.DOTALL)
        program = match.group(1).strip()
        
        print(f"\nProgram:")
        print(program)
        
        # Test each failing example
        for res in result['results']:
            if not res['correct']:
                print(f"\n  Failing Example {res['example_num']}:")
                print(f"    Expected shape: {res['expected_shape']}")
                print(f"    Predicted shape: {res['predicted_shape']}")
                if res['error']:
                    print(f"    Error: {res['error']}")
                    
                # Extract this specific example from the user message
                user_content = messages[1]['content']
                lines = user_content.split('\n')
                
                # Find the example
                example_start = None
                input_start = None
                output_start = None
                input_lines = []
                output_lines = []
                
                for i, line in enumerate(lines):
                    if f"Example {res['example_num']}:" in line:
                        example_start = i
                    elif example_start is not None and "Input:" in line:
                        input_start = i
                    elif example_start is not None and "Output:" in line:
                        output_start = i
                        # Collect input lines
                        for j in range(input_start + 1, i):
                            if lines[j].strip() and not lines[j].startswith("Output:"):
                                input_lines.append(lines[j])
                    elif output_start is not None:
                        if line.strip().startswith("Example") or line.strip().startswith("Test Input:"):
                            break
                        output_lines.append(line)
                
                # Parse the grids
                input_grid = []
                for line in input_lines:
                    if line.strip():
                        row = [int(x) for x in line.strip().split()]
                        input_grid.append(row)
                
                output_grid = []
                for line in output_lines:
                    if line.strip():
                        row = [int(x) for x in line.strip().split()]
                        output_grid.append(row)
                
                print(f"\n    Input grid shape: {len(input_grid)}x{len(input_grid[0]) if input_grid else 0}")
                print(f"    Expected output shape: {len(output_grid)}x{len(output_grid[0]) if output_grid else 0}")
                
                # Execute the program multiple times to check for consistency
                print(f"\n    Testing execution consistency:")
                results = []
                for attempt in range(3):
                    predicted_output, error, timed_out = execute_program_on_input(program, input_grid)
                    if predicted_output is not None:
                        results.append(predicted_output)
                        print(f"      Attempt {attempt + 1}: shape {len(predicted_output)}x{len(predicted_output[0]) if predicted_output and predicted_output[0] else 0}")
                    else:
                        print(f"      Attempt {attempt + 1}: Error - {error}")
                
                # Check if results are consistent
                if len(results) > 1:
                    all_same = all(r == results[0] for r in results)
                    print(f"    Results consistent: {all_same}")
                    if not all_same:
                        print("    INCONSISTENCY DETECTED!")
                        for i, r in enumerate(results):
                            print(f"      Result {i + 1}: {r}")