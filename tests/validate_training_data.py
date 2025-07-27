#!/usr/bin/env python3

import json
import re
import tempfile
import subprocess
import os
import argparse
from typing import List, Dict, Tuple, Optional
from pathlib import Path

def parse_grid_from_text(text: str) -> List[List[int]]:
    """Parse a grid from text format, handling empty rows marked as [EMPTY_ROW]"""
    lines = text.strip().split('\n')
    grid = []
    for line in lines:
        line = line.strip()
        if line == '[EMPTY_ROW]':
            grid.append([])
        elif line:
            row = [int(x) for x in line.split()]
            grid.append(row)
    return grid

def extract_examples_from_user_message(content: str) -> List[Dict]:
    """Extract training examples from the user message content"""
    examples = []
    
    # Look for training examples
    example_pattern = r'Example (\d+):\s*\nInput:\s*\n(.*?)\nOutput:\s*\n(.*?)(?=\n\nExample|\nTest Input:|$)'
    matches = re.findall(example_pattern, content, re.DOTALL)
    
    for match in matches:
        example_num, input_text, output_text = match
        try:
            input_grid = parse_grid_from_text(input_text)
            output_grid = parse_grid_from_text(output_text)
            examples.append({
                'example_num': int(example_num),
                'input': input_grid,
                'output': output_grid
            })
        except Exception as e:
            print(f"Warning: Failed to parse example {example_num}: {e}")
    
    return examples

def extract_program_from_assistant_message(content: str) -> Optional[str]:
    """Extract the Python program from the assistant message"""
    # Look for the final answer code block
    pattern = r'Final answer:\s*```python\s*(.*?)\s*```'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    # Fallback: look for any python code block
    pattern = r'```python\s*(.*?)\s*```'
    match = re.search(pattern, content, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    
    return None

def execute_program_on_input(program: str, input_grid: List[List[int]], timeout: float = 0.5) -> Tuple[Optional[List[List[int]]], Optional[str], bool]:
    """Execute a program on an input grid and return the result"""
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
            f.write("    import traceback\n")
            f.write("    traceback.print_exc()\n")
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
                        try:
                            output = eval(output_str)  # Be careful with eval in production
                            return output, None, False
                        except Exception as e:
                            return None, f"Failed to parse output: {e}", False
                    elif line.startswith('ERROR:'):
                        error_msg = line[6:].strip()
                        return None, error_msg, False
                
                return None, f"No output produced. Stdout: {result.stdout}, Stderr: {result.stderr}", False
            else:
                error_msg = result.stderr.strip() or result.stdout.strip() or "Unknown error"
                return None, error_msg, False
                
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

def grids_equal(grid1: List[List[int]], grid2: List[List[int]]) -> bool:
    """Check if two grids are equal"""
    if len(grid1) != len(grid2):
        return False
    
    for i in range(len(grid1)):
        if len(grid1[i]) != len(grid2[i]):
            return False
        for j in range(len(grid1[i])):
            if grid1[i][j] != grid2[i][j]:
                return False
    
    return True

def validate_training_example(training_example: Dict) -> Dict:
    """Validate a single training example from JSONL"""
    messages = training_example.get('messages', [])
    
    # Find user and assistant messages
    user_message = None
    assistant_message = None
    
    for msg in messages:
        if msg.get('role') == 'user':
            user_message = msg.get('content', '')
        elif msg.get('role') == 'assistant':
            assistant_message = msg.get('content', '')
    
    if not user_message or not assistant_message:
        return {
            'status': 'error',
            'error': 'Missing user or assistant message',
            'examples_tested': 0,
            'examples_correct': 0
        }
    
    # Extract examples and program
    examples = extract_examples_from_user_message(user_message)
    program = extract_program_from_assistant_message(assistant_message)
    
    if not examples:
        return {
            'status': 'error',
            'error': 'No examples found in user message',
            'examples_tested': 0,
            'examples_correct': 0
        }
    
    if not program:
        return {
            'status': 'error',
            'error': 'No program found in assistant message',
            'examples_tested': len(examples),
            'examples_correct': 0
        }
    
    # Test the program on each example
    results = []
    examples_correct = 0
    
    for example in examples:
        input_grid = example['input']
        expected_output = example['output']
        
        predicted_output, error, timed_out = execute_program_on_input(program, input_grid)
        
        if predicted_output is not None and not error and not timed_out:
            is_correct = grids_equal(predicted_output, expected_output)
            if is_correct:
                examples_correct += 1
        else:
            is_correct = False
        
        results.append({
            'example_num': example['example_num'],
            'correct': is_correct,
            'error': error,
            'timed_out': timed_out,
            'expected_shape': (len(expected_output), len(expected_output[0]) if expected_output else 0),
            'predicted_shape': (len(predicted_output), len(predicted_output[0]) if predicted_output and len(predicted_output) > 0 else 0) if predicted_output else None
        })
    
    return {
        'status': 'success',
        'examples_tested': len(examples),
        'examples_correct': examples_correct,
        'success_rate': examples_correct / len(examples) if examples else 0,
        'results': results
    }

def validate_training_data_file(filepath: str, limit: Optional[int] = None, verbose: bool = False) -> Dict:
    """Validate all training examples in a JSONL file"""
    print(f"Validating training data file: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} does not exist")
        return {}
    
    total_examples = 0
    total_tested = 0
    total_correct = 0
    errors = 0
    
    validation_results = []
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if limit and line_num > limit:
                break
                
            try:
                training_example = json.loads(line.strip())
                result = validate_training_example(training_example)
                
                total_examples += 1
                total_tested += result['examples_tested']
                total_correct += result['examples_correct']
                
                if result['status'] == 'error':
                    errors += 1
                
                validation_results.append({
                    'line_num': line_num,
                    **result
                })
                
                if verbose:
                    print(f"Line {line_num}: {result['examples_correct']}/{result['examples_tested']} correct "
                          f"({result.get('success_rate', 0):.1%})")
                    if result['status'] == 'error':
                        print(f"  Error: {result['error']}")
                    elif verbose and result.get('results'):
                        for res in result['results']:
                            status = "✓" if res['correct'] else "✗"
                            print(f"    Example {res['example_num']}: {status}")
                            if res['error']:
                                print(f"      Error: {res['error']}")
                
            except json.JSONDecodeError as e:
                print(f"Line {line_num}: JSON decode error: {e}")
                errors += 1
            except Exception as e:
                print(f"Line {line_num}: Unexpected error: {e}")
                errors += 1
    
    # Summary statistics
    success_rate = total_correct / total_tested if total_tested > 0 else 0
    
    print(f"\n=== VALIDATION SUMMARY ===")
    print(f"Training examples processed: {total_examples}")
    print(f"Errors: {errors}")
    print(f"Individual examples tested: {total_tested}")
    print(f"Individual examples correct: {total_correct}")
    print(f"Overall success rate: {success_rate:.1%}")
    
    if errors > 0:
        print(f"Warning: {errors} training examples had errors")
    
    return {
        'total_examples': total_examples,
        'total_tested': total_tested,
        'total_correct': total_correct,
        'errors': errors,
        'success_rate': success_rate,
        'validation_results': validation_results
    }

def main():
    parser = argparse.ArgumentParser(description='Validate training data by running programs on examples')
    parser.add_argument('training_file', help='Path to JSONL training data file')
    parser.add_argument('--limit', type=int, help='Limit number of training examples to process')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    results = validate_training_data_file(args.training_file, args.limit, args.verbose)
    
    return 0 if results.get('errors', 0) == 0 else 1

if __name__ == '__main__':
    exit(main()) 