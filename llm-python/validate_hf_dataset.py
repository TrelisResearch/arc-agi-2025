#!/usr/bin/env python3

import tempfile
import subprocess
import os
import argparse
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset
import sys

def execute_program(program: str, input_grid: List[List[int]], timeout: float = 0.5) -> Tuple[Optional[List[List[int]]], Optional[str], bool]:
    """Execute a program with the given input grid and return the result"""
    try:
        # Create the full program code
        full_program = f"""
{program}

# Execute the transform function
import json
input_grid = {input_grid}
try:
    result = transform(input_grid)
    print("RESULT_START")
    print(json.dumps(result))
    print("RESULT_END")
except Exception as e:
    print("ERROR_START")
    print(str(e))
    print("ERROR_END")
"""
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_program)
            temp_file = f.name
        
        try:
            # Execute the program
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode != 0:
                return None, f"Execution failed: {result.stderr}", False
            
            # Parse the output
            output = result.stdout
            if "RESULT_START" in output and "RESULT_END" in output:
                result_start = output.find("RESULT_START") + len("RESULT_START\n")
                result_end = output.find("RESULT_END")
                result_json = output[result_start:result_end].strip()
                
                import json
                parsed_result = json.loads(result_json)
                return parsed_result, None, False
            elif "ERROR_START" in output and "ERROR_END" in output:
                error_start = output.find("ERROR_START") + len("ERROR_START\n")
                error_end = output.find("ERROR_END")
                error_msg = output[error_start:error_end].strip()
                return None, error_msg, False
            else:
                return None, f"Unexpected output format: {output}", False
                
        finally:
            # Clean up temp file
            if os.path.exists(temp_file):
                os.unlink(temp_file)
                
    except subprocess.TimeoutExpired:
        return None, f"Program execution timed out after {timeout}s", True
    except Exception as e:
        return None, f"Execution error: {str(e)}", False

def validate_grid_format(grid, field_name: str) -> Tuple[bool, Optional[str]]:
    """Validate that a grid has the correct format"""
    if not isinstance(grid, list):
        return False, f"{field_name}: Expected list, got {type(grid).__name__}"
    
    if len(grid) == 0:
        return False, f"{field_name}: Grid cannot be empty"
    
    for i, row in enumerate(grid):
        if not isinstance(row, list):
            return False, f"{field_name}: Row {i} is not a list: {type(row).__name__}"
        
        for j, cell in enumerate(row):
            if isinstance(cell, bool):
                return False, f"{field_name}: Cell [{i}][{j}] is boolean: {cell}"
            if not isinstance(cell, int):
                return False, f"{field_name}: Cell [{i}][{j}] is not int: {cell} ({type(cell).__name__})"
            if not (0 <= cell <= 9):
                return False, f"{field_name}: Cell [{i}][{j}] out of range: {cell}"
    
    return True, None

def validate_dataset_row(row: Dict, row_idx: int, verbose: bool = False) -> Dict:
    """Validate a single row from the HF dataset"""
    try:
        # Extract required fields
        code = row.get('code', '')
        train_inputs = row.get('train_input', [])
        train_outputs = row.get('train_output', [])
        predicted_train_outputs = row.get('predicted_train_output', [])
        correct_train_inputs = row.get('correct_train_input', [])
        
        task_id = row.get('task_id', 'unknown')
        
        if not code:
            return {
                'status': 'error',
                'error': 'No code found',
                'examples_tested': 0,
                'examples_correct': 0,
                'task_id': task_id
            }
        
        # Validate input lengths match
        num_examples = len(train_inputs)
        if len(train_outputs) != num_examples:
            return {
                'status': 'error',
                'error': f'Mismatched lengths: {len(train_inputs)} inputs vs {len(train_outputs)} outputs',
                'examples_tested': 0,
                'examples_correct': 0,
                'task_id': task_id
            }
        
        if len(predicted_train_outputs) != num_examples:
            return {
                'status': 'error',
                'error': f'Mismatched lengths: {len(train_inputs)} inputs vs {len(predicted_train_outputs)} predicted outputs',
                'examples_tested': 0,
                'examples_correct': 0,
                'task_id': task_id
            }
        
        if len(correct_train_inputs) != num_examples:
            return {
                'status': 'error',
                'error': f'Mismatched lengths: {len(train_inputs)} inputs vs {len(correct_train_inputs)} correctness flags',
                'examples_tested': 0,
                'examples_correct': 0,
                'task_id': task_id
            }
        
        # Validate grid formats for all training data
        for i in range(num_examples):
            # Validate input grid
            valid, error = validate_grid_format(train_inputs[i], f"train_input[{i}]")
            if not valid:
                return {
                    'status': 'error',
                    'error': error,
                    'examples_tested': 0,
                    'examples_correct': 0,
                    'task_id': task_id
                }
            
            # Validate expected output grid
            valid, error = validate_grid_format(train_outputs[i], f"train_output[{i}]")
            if not valid:
                return {
                    'status': 'error',
                    'error': error,
                    'examples_tested': 0,
                    'examples_correct': 0,
                    'task_id': task_id
                }
            
            # Validate predicted output grid
            valid, error = validate_grid_format(predicted_train_outputs[i], f"predicted_train_output[{i}]")
            if not valid:
                return {
                    'status': 'error',
                    'error': error,
                    'examples_tested': 0,
                    'examples_correct': 0,
                    'task_id': task_id
                }
        
        # Now validate that the program actually produces the predicted outputs
        validation_results = []
        examples_correct = 0
        
        for i in range(num_examples):
            input_grid = train_inputs[i]
            expected_output = predicted_train_outputs[i]  # Compare with what dataset claims program outputs
            
            # Execute the program
            actual_output, error, timed_out = execute_program(code, input_grid)
            
            if error or timed_out:
                validation_results.append({
                    'example_num': i + 1,
                    'correct': False,
                    'error': error,
                    'timed_out': timed_out
                })
                continue
            
            # Check if the actual output matches the predicted output
            matches = actual_output == expected_output
            if matches:
                examples_correct += 1
            
            validation_results.append({
                'example_num': i + 1,
                'correct': matches,
                'error': None,
                'timed_out': False,
                'actual_output': actual_output,
                'expected_output': expected_output
            })
        
        return {
            'status': 'success',
            'examples_tested': num_examples,
            'examples_correct': examples_correct,
            'results': validation_results,
            'task_id': task_id
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': f'Validation error: {str(e)}',
            'examples_tested': 0,
            'examples_correct': 0,
            'task_id': row.get('task_id', 'unknown')
        }

def validate_hf_dataset(dataset_name: str, split: str = "train", limit: Optional[int] = None, verbose: bool = False) -> Dict:
    """Validate a Hugging Face dataset by running programs and checking consistency"""
    
    print(f"Loading dataset: {dataset_name} (split: {split})")
    
    try:
        # Load the dataset
        dataset = load_dataset(dataset_name, split=split, token=True)
        print(f"Dataset loaded successfully. Total rows: {len(dataset)}")
        
        # Limit the number of rows if specified
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
            print(f"Limited to {len(dataset)} rows")
        
        # Validate each row
        total_examples = 0
        total_tested = 0
        total_correct = 0
        errors = 0
        validation_results = []
        
        print(f"\nValidating {len(dataset)} dataset rows...")
        
        for row_idx, row in enumerate(dataset):
            if verbose:
                print(f"\nRow {row_idx + 1}/{len(dataset)}: Task {row.get('task_id', 'unknown')}")
            
            result = validate_dataset_row(row, row_idx, verbose)
            validation_results.append(result)
            
            total_examples += 1
            total_tested += result.get('examples_tested', 0)
            total_correct += result.get('examples_correct', 0)
            
            if result['status'] == 'error':
                errors += 1
                print(f"Row {row_idx + 1}: ERROR - {result['error']}")
            else:
                if verbose or result['examples_correct'] != result['examples_tested']:
                    print(f"Row {row_idx + 1}: {result['examples_correct']}/{result['examples_tested']} correct "
                          f"({result.get('examples_correct', 0)/max(result.get('examples_tested', 1), 1):.1%})")
                    if verbose and result.get('results'):
                        for res in result['results']:
                            status = "✓" if res['correct'] else "✗"
                            print(f"    Example {res['example_num']}: {status}")
                            if res['error']:
                                print(f"      Error: {res['error']}")
        
        # Summary statistics
        success_rate = total_correct / total_tested if total_tested > 0 else 0
        
        print(f"\n=== VALIDATION SUMMARY ===")
        print(f"Dataset rows processed: {total_examples}")
        print(f"Errors: {errors}")
        print(f"Individual examples tested: {total_tested}")
        print(f"Individual examples correct: {total_correct}")
        print(f"Overall success rate: {success_rate:.1%}")
        
        if errors > 0:
            print(f"Warning: {errors} dataset rows had errors")
        
        return {
            'total_examples': total_examples,
            'total_tested': total_tested,
            'total_correct': total_correct,
            'errors': errors,
            'success_rate': success_rate,
            'validation_results': validation_results
        }
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {
            'total_examples': 0,
            'total_tested': 0,
            'total_correct': 0,
            'errors': 1,
            'success_rate': 0.0,
            'validation_results': []
        }

def main():
    parser = argparse.ArgumentParser(description='Validate HF dataset by running programs on examples')
    parser.add_argument('dataset_name', help='Hugging Face dataset name (e.g., Trelis/synth_arc-agi-1_middle_training_10_20250724_080541)')
    parser.add_argument('--split', default='train', help='Dataset split to validate (default: train)')
    parser.add_argument('--limit', type=int, help='Limit number of dataset rows to process')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    results = validate_hf_dataset(args.dataset_name, args.split, args.limit, args.verbose)
    
    return 0 if results.get('errors', 0) == 0 else 1

if __name__ == '__main__':
    exit(main()) 