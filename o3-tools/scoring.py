#!/usr/bin/env python3

import gzip
import json
from typing import List, Dict, Tuple, Optional
import subprocess
import tempfile
import os
import signal
import tiktoken

class GridScorer:
    """Scores predicted grids against ground truth"""
    
    def __init__(self):
        # Initialize tokenizer for counting tokens
        self.encoding = tiktoken.encoding_for_model("gpt-4")
    
    def score_grid(self, predicted: List[List[int]], actual: List[List[int]]) -> Dict:
        """Score a predicted grid against the actual grid"""
        # Ensure grids have same dimensions
        if len(predicted) != len(actual):
            return {
                'correct': False,
                'pixel_accuracy': 0.0,
                'total_pixels': len(actual) * (len(actual[0]) if actual else 0),
                'correct_pixels': 0,
                'error': 'Grid height mismatch'
            }
        
        if any(len(row) != len(actual[0]) for row in predicted):
            return {
                'correct': False,
                'pixel_accuracy': 0.0,
                'total_pixels': len(actual) * len(actual[0]),
                'correct_pixels': 0,
                'error': 'Grid width mismatch'
            }
        
        # Count correct pixels
        total_pixels = 0
        correct_pixels = 0
        
        for pred_row, actual_row in zip(predicted, actual):
            for pred_cell, actual_cell in zip(pred_row, actual_row):
                total_pixels += 1
                if pred_cell == actual_cell:
                    correct_pixels += 1
        
        return {
            'correct': correct_pixels == total_pixels,
            'pixel_accuracy': correct_pixels / total_pixels if total_pixels > 0 else 0.0,
            'total_pixels': total_pixels,
            'correct_pixels': correct_pixels,
            'error': None
        }
    
    def calculate_residual_grid(self, predicted: List[List[int]], actual: List[List[int]]) -> List[List[int]]:
        """Calculate the residual grid (difference between predicted and actual)"""
        residual = []
        
        # If dimensions don't match, return the actual grid as residual
        if len(predicted) != len(actual) or any(len(row) != len(actual[0]) for row in predicted):
            return actual
        
        for pred_row, actual_row in zip(predicted, actual):
            residual_row = []
            for pred_cell, actual_cell in zip(pred_row, actual_row):
                # 0 if correct, actual value if incorrect
                residual_row.append(0 if pred_cell == actual_cell else actual_cell)
            residual.append(residual_row)
        
        return residual
    
    def gzip_compress_grid(self, grid: List[List[int]]) -> int:
        """Compress a grid and return the gzipped size in bytes"""
        # Convert grid to a string representation
        grid_str = json.dumps(grid)
        
        # Compress and return size
        compressed = gzip.compress(grid_str.encode('utf-8'))
        return len(compressed)
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string"""
        return len(self.encoding.encode(text))
    
    def calculate_mdl_score(self, program: str, residual_grid: List[List[int]], 
                           alpha: float = 1.0, beta: float = 4.0) -> Dict:
        """
        Calculate MDL (Minimum Description Length) score
        
        Args:
            program: The Python program as a string
            residual_grid: The residual grid (differences)
            alpha: Weight for program tokens (default 1.0)
            beta: Weight for residual compression (default 4.0)
        
        Returns:
            Dictionary with MDL components and total score
        """
        program_tokens = self.count_tokens(program)
        residual_bytes = self.gzip_compress_grid(residual_grid)
        
        # Calculate total MDL score
        mdl_score = alpha * program_tokens + beta * residual_bytes
        
        return {
            'program_tokens': program_tokens,
            'residual_bytes': residual_bytes,
            'mdl_score': mdl_score,
            'alpha': alpha,
            'beta': beta
        }
    
    def calculate_null_program_mdl(self, actual_grid: List[List[int]], 
                                  alpha: float = 1.0, beta: float = 4.0) -> Dict:
        """
        Calculate MDL score for the null program (returns input unchanged)
        
        Args:
            actual_grid: The expected output grid
            alpha: Weight for program tokens (default 1.0)
            beta: Weight for residual compression (default 4.0)
        
        Returns:
            Dictionary with null program MDL components and total score
        """
        # Null program: def transform(grid): return grid
        null_program = "def transform(grid):\n    return grid"
        null_program_tokens = self.count_tokens(null_program)
        
        # For null program, residual is the full actual grid (since predicted = input, actual = output)
        # This represents the full "surprise" of the transformation
        null_residual_bytes = self.gzip_compress_grid(actual_grid)
        
        null_mdl_score = alpha * null_program_tokens + beta * null_residual_bytes
        
        return {
            'null_program': null_program,
            'null_program_tokens': null_program_tokens,
            'null_residual_bytes': null_residual_bytes,
            'null_mdl_score': null_mdl_score,
            'alpha': alpha,
            'beta': beta
        }


class ProgramExecutor:
    """Executes Python programs with timeout and captures output"""
    
    def __init__(self, timeout: float = 0.1):
        self.timeout = timeout
    
    def execute_program(self, program: str, test_input: List[List[int]]) -> Tuple[Optional[List[List[int]]], str, bool]:
        """
        Execute a Python program with the test input
        
        Returns:
            (output_grid, error_message, timed_out)
        """
        # Create a wrapper script that will execute the program
        wrapper = f"""
import json
import sys

# Define the test input
test_input = {json.dumps(test_input)}

# Execute the provided program
{program}

# The program should define a function called 'transform' or 'solve'
# Try common function names
output = None
if 'transform' in locals():
    output = transform(test_input)
elif 'solve' in locals():
    output = solve(test_input)
elif 'apply_transform' in locals():
    output = apply_transform(test_input)
else:
    # If no function found, try to find any function that takes a grid
    for name, obj in locals().items():
        if callable(obj) and not name.startswith('_'):
            try:
                result = obj(test_input)
                if isinstance(result, list) and all(isinstance(row, list) for row in result):
                    output = result
                    break
            except:
                pass

if output is not None:
    print(json.dumps(output))
else:
    print("ERROR: No valid transformation function found")
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(wrapper)
            temp_file = f.name
        
        try:
            # Run the program with timeout
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0 and result.stdout.strip():
                try:
                    if result.stdout.strip().startswith("ERROR:"):
                        return None, result.stdout.strip(), False
                    
                    output = json.loads(result.stdout.strip())
                    return output, "", False
                except json.JSONDecodeError:
                    return None, f"Invalid output format: {result.stdout}", False
            else:
                error = result.stderr if result.stderr else "Program produced no output"
                return None, error, False
                
        except subprocess.TimeoutExpired:
            return None, f"Program exceeded timeout of {self.timeout}s", True
        except Exception as e:
            return None, str(e), False
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass


# Example usage
if __name__ == "__main__":
    scorer = GridScorer()
    
    # Test grid scoring
    predicted = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    actual = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]  # One pixel different
    
    score = scorer.score_grid(predicted, actual)
    print("Grid score:", json.dumps(score, indent=2))
    
    # Test residual and MDL
    residual = scorer.calculate_residual_grid(predicted, actual)
    print("\nResidual grid:", residual)
    
    program = "def transform(grid):\n    return [[grid[row][col] for col in range(len(grid[0]))] for row in range(len(grid))]"
    mdl = scorer.calculate_mdl_score(program, residual)
    print("\nMDL score:", json.dumps(mdl, indent=2))
    
    # Test program execution
    executor = ProgramExecutor()
    test_input = [[1, 2], [3, 4]]
    output, error, timed_out = executor.execute_program(program, test_input)
    print("\nProgram execution:")
    print(f"  Output: {output}")
    print(f"  Error: {error}")
    print(f"  Timed out: {timed_out}")