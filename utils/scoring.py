#!/usr/bin/env python3

from typing import List, Dict, Tuple, Optional
import subprocess
import tempfile
import os
import json

class GridScorer:
    """Scores predicted grids against ground truth"""
    
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


class ProgramExecutor:
    """Executes Python programs with timeout and captures output"""
    
    def __init__(self, timeout: float = 0.1):
        self.timeout = timeout
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        import numpy as np
        
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        else:
            return obj
    
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

def convert_numpy_types(obj):
    \"\"\"Convert numpy types to Python native types for JSON serialization\"\"\"
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
    except ImportError:
        pass  # numpy not available
    
    if isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

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
    # Convert numpy types before JSON serialization
    output = convert_numpy_types(output)
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
                    
                    # Convert numpy types to Python types if needed
                    if output is not None:
                        output = self._convert_numpy_types(output)
                    
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


if __name__ == "__main__":
    # Simple test
    scorer = GridScorer()
    executor = ProgramExecutor()
    
    # Test case
    program = "def transform(grid):\n    return [[cell * 2 for cell in row] for row in grid]"
    test_input = [[1, 2], [3, 4]]
    predicted, error, timed_out = executor.execute_program(program, test_input)
    
    print("Program:", program)
    print("Input:", test_input)
    print("Output:", predicted)
    print("Error:", error)
    print("Timed out:", timed_out)
    
    if predicted:
        actual = [[2, 4], [6, 8]]
        score = scorer.score_grid(predicted, actual)
        print("Score:", json.dumps(score, indent=2))