from typing import Any, List, Dict, Tuple, Optional
import sys
import os

# Add project root to path to find sandbox module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sandbox import create_executor

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
    
    def score_grids_bulk(self, predicted_grids: List[List[List[int]]], actual_grids: List[List[List[int]]]) -> List[Dict]:
        """Score multiple predicted grids against their corresponding actual grids"""
        if len(predicted_grids) != len(actual_grids):
            raise ValueError(f"Number of predicted grids ({len(predicted_grids)}) must match number of actual grids ({len(actual_grids)})")
        
        return [self.score_grid(pred, actual) for pred, actual in zip(predicted_grids, actual_grids)]


class ProgramExecutor:
    """Executes Python programs with timeout and captures output"""
    
    _executor = None
    _executor_context = None
    _executor_type = None
    
    def __init__(self, timeout: float = 0.5, executor_type: str = "docker"):
        self.timeout = timeout
        self.executor_type = executor_type
        
        # Initialize the executor if not already done or if type changed
        if (ProgramExecutor._executor is None or 
            ProgramExecutor._executor_type != executor_type):
            self._init_executor()
    
    def _init_executor(self):
        """Initialize the executor singleton"""
        if ProgramExecutor._executor is not None:
            try:
                ProgramExecutor._executor.__exit__(None, None, None)
            except Exception:
                pass
        
        ProgramExecutor._executor = create_executor(self.executor_type)
        ProgramExecutor._executor_context = ProgramExecutor._executor.__enter__()
        ProgramExecutor._executor_type = self.executor_type
    
    @classmethod
    def cleanup_executor(cls):
        """Cleanup the executor (useful for tests or shutdown)"""
        if cls._executor is not None:
            try:
                cls._executor.__exit__(None, None, None)
            except Exception:
                pass
            cls._executor = None
            cls._executor_context = None
            cls._executor_type = None
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
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
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        else:
            return obj
    
    def execute_program_with_timeout(self, program: str, test_input: List[List[int]]) -> Tuple[Optional[List[List[int]]], str, bool]:
        """
        Execute a Python program with the test input
        
        Returns:
            (output_grid, error_message, timed_out)
        """
        # Create the code to execute
        code = f"""
import json

# Define the test input
test_input = {repr(test_input)}

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
            except Exception:
                pass

if output is not None:
    return output
else:
    raise ValueError("No valid transformation function found")
"""
        
        try:
            # Ensure we have an executor context
            if ProgramExecutor._executor_context is None:
                raise RuntimeError("Executor not initialized")
                
            result, error = ProgramExecutor._executor_context.execute_code(code, timeout=self.timeout)
            
            if error:
                # Check if it's a timeout error
                if "timeout" in str(error).lower():
                    return None, f"Program exceeded timeout of {self.timeout}s", True
                else:
                    return None, str(error), False
            
            if result is not None:
                # Convert numpy types to Python types if needed
                result = self._convert_numpy_types(result)
                return result, "", False
            else:
                return None, "Program produced no output", False
                
        except Exception as e:
            return None, str(e), False
    
    def execute_program_bulk(self, program: str, test_inputs: List[List[List[int]]]) -> List[Tuple[Optional[Any], str, bool]]:
        """
        Execute a Python program with multiple test inputs in a single execution context
        
        Args:
            program: The Python program code to execute
            test_inputs: List of test input grids
            
        Returns:
            List of tuples: (output_grid, error_message, timed_out) for each input
        """
        # Create the code to execute
        code = f"""
import json

# Define the test inputs
test_inputs = {repr(test_inputs)}

# Execute the provided program
{program}

# The program should define a function called 'transform' or 'solve'
# Find the transformation function
transform_func = None
if 'transform' in locals():
    transform_func = transform
elif 'solve' in locals():
    transform_func = solve
elif 'apply_transform' in locals():
    transform_func = apply_transform
else:
    # If no function found, try to find any function that takes a grid
    for name, obj in locals().items():
        if callable(obj) and not name.startswith('_'):
            try:
                # Test with the first input to see if it works
                if test_inputs:
                    result = obj(test_inputs[0])
                    if isinstance(result, list) and all(isinstance(row, list) for row in result):
                        transform_func = obj
                        break
            except Exception:
                pass

if transform_func is None:
    raise ValueError("No valid transformation function found")

# Apply the function to all test inputs
outputs = []
for test_input in test_inputs:
    try:
        output = transform_func(test_input)
        outputs.append(output)
    except Exception as e:
        outputs.append(("ERROR", str(e)))

return outputs
"""
        
        try:
            # Ensure we have an executor context
            if ProgramExecutor._executor_context is None:
                raise RuntimeError("Executor not initialized")
                
            result, error = ProgramExecutor._executor_context.execute_code(code, timeout=self.timeout * len(test_inputs))
            
            if error:
                # Check if it's a timeout error
                if "timeout" in str(error).lower():
                    # Return timeout for all inputs
                    return [(None, f"Program exceeded timeout of {self.timeout * len(test_inputs)}s", True) for _ in test_inputs]
                else:
                    # Return error for all inputs
                    return [(None, str(error), False) for _ in test_inputs]
            
            if result is not None:
                # Process the results
                results = []
                for i, output in enumerate(result):
                    if isinstance(output, tuple) and len(output) == 2 and output[0] == "ERROR":
                        # This was an error for this specific input
                        results.append((None, output[1], False))
                    else:
                        # Convert numpy types to Python types if needed
                        converted_output = self._convert_numpy_types(output)
                        results.append((converted_output, "", False))
                
                return results
            else:
                return [(None, "Program produced no output", False) for _ in test_inputs]
                
        except Exception as e:
            return [(None, str(e), False) for _ in test_inputs]