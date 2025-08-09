#!/usr/bin/env python3

from typing import List, Optional, NamedTuple, Any, Tuple
import sys
import os

# Add project root to path to find sandbox module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sandbox import create_executor
from ..utils.task_loader import TaskData, Grid, TaskExample


class ExecutionError(NamedTuple):
    """Detailed information about a program execution error."""
    error_message: str
    timed_out: bool
    output_type: str  # "none", "invalid_format", "incorrect_result"


class ProgramTestResult(NamedTuple):
    """Result of testing a program on ARC task data."""
    train_outputs: List[Optional[Grid]]
    test_outputs: List[Optional[Grid]]
    train_inputs: List[Grid]
    test_inputs: List[Grid]
    correct_train_input: List[bool]
    correct_test_input: List[bool]
    success: bool  # True if all outputs were generated successfully
    # Enhanced error reporting (optional for backward compatibility)
    train_errors: Optional[List[Optional[ExecutionError]]] = None
    test_errors: Optional[List[Optional[ExecutionError]]] = None


class ArcTester:
    """Test ARC programs and compute correctness metrics."""
    
    # Class variables for executor singleton (merged from ProgramExecutor)
    _executor = None
    _executor_context = None
    _executor_type = None
    
    def __init__(self, timeout: int = 2, executor_type: str = "unrestricted",
                 max_output_chars: Optional[int] = None,
                 max_output_cells: Optional[int] = None):
        """Initialize the ARC tester.
        
        Args:
            timeout: Execution timeout in seconds
            executor_type: Type of executor to use
        """
        self.timeout = timeout
        self.executor_type = executor_type
        self.max_output_chars = max_output_chars
        self.max_output_cells = max_output_cells
        
        # Initialize the executor if not already done or if type changed
        if (ArcTester._executor is None or 
            ArcTester._executor_type != executor_type):
            self._init_executor()
    
    def _init_executor(self):
        """Initialize the executor singleton"""
        if ArcTester._executor is not None:
            try:
                ArcTester._executor.__exit__(None, None, None)
            except Exception:
                pass
        
        ArcTester._executor = create_executor(self.executor_type)
        ArcTester._executor_context = ArcTester._executor.__enter__()
        ArcTester._executor_type = self.executor_type
    
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
            if ArcTester._executor_context is None:
                raise RuntimeError("Executor not initialized")
                
            result, error = ArcTester._executor_context.execute_code(code, timeout=self.timeout)
            
            if error:
                # Check if it's a timeout error
                if "timeout" in str(error).lower():
                    return None, f"Program exceeded timeout of {self.timeout}s", True
                else:
                    return None, str(error), False
            
            if result is not None:
                # Convert numpy types to Python types if needed
                result = self._convert_numpy_types(result)
                # Oversize output guard
                try:
                    import json as _json
                    reason = None
                    if self.max_output_cells is not None:
                        try:
                            total_cells = sum(len(row) for row in result) if isinstance(result, list) else 0
                            if total_cells > self.max_output_cells:
                                reason = f"cells={total_cells}>max={self.max_output_cells}"
                        except Exception:
                            pass
                    if reason is None and self.max_output_chars is not None:
                        try:
                            s = _json.dumps(result, separators=(",", ":"))
                            if len(s) > self.max_output_chars:
                                reason = f"chars={len(s)}>max={self.max_output_chars}"
                        except Exception:
                            pass
                    if reason is not None:
                        return None, f"output_too_large({reason})", False
                except Exception:
                    # If guard fails, proceed with result
                    pass
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
            if ArcTester._executor_context is None:
                raise RuntimeError("Executor not initialized")
                
            result, error = ArcTester._executor_context.execute_code(code, timeout=self.timeout * len(test_inputs))
            
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
    
    def _validate_grid(self, output) -> Optional[Grid]:
        """Validate that output is a proper grid format and convert to list of lists."""
        if output is None:
            return None

        # Check if output is a list/tuple of lists/tuples of ints
        if isinstance(output, (list, tuple)) and all(
            isinstance(row, (list, tuple)) and all(isinstance(val, int) for val in row)
            for row in output
        ):
            # Convert to list of lists format (Grid type)
            return [list(row) for row in output]
        else:
            # If output is not a valid grid format, return None
            return None

    def _classify_error(self, output, error_message: str, timed_out: bool) -> ExecutionError:
        """Classify the type of error that occurred during execution."""
        if timed_out:
            return ExecutionError(
                error_message=error_message,
                timed_out=True,
                output_type="none"
            )
        
        if output is None:
            return ExecutionError(
                error_message=error_message,
                timed_out=False,
                output_type="none"
            )
        
        # Check if output has invalid format
        if not isinstance(output, (list, tuple)) or not all(
            isinstance(row, (list, tuple)) and all(isinstance(val, int) for val in row)
            for row in output
        ):
            return ExecutionError(
                error_message="Output has invalid format (not a list/tuple of lists/tuples of ints)",
                timed_out=False,
                output_type="invalid_format"
            )
        
        # If we get here, the output is valid format but might be incorrect result
        return ExecutionError(
            error_message=error_message,
            timed_out=False,
            output_type="incorrect_result"
        )

    def _execute_on_examples(self, program: str, examples: List[TaskExample]) -> tuple[List[Optional[Grid]], List[Optional[ExecutionError]]]:
        """Execute program on a list of examples and return outputs and error details."""
        if not examples:
            return [], []

        # Extract inputs for bulk execution
        inputs = [example["input"] for example in examples]

        # Use bulk execution for better performance
        results = self.execute_program_bulk(program, inputs)

        # Process results and extract outputs and errors
        outputs: List[Optional[Grid]] = []
        errors: List[Optional[ExecutionError]] = []

        for result in results:
            output, error_message, timed_out = result
            validated_output = self._validate_grid(output)
            outputs.append(validated_output)
            
            if validated_output is None or error_message:
                errors.append(self._classify_error(output, error_message, timed_out))
            else:
                errors.append(None)  # No error

        return outputs, errors

    def _run_program_on_task(
        self, program: str, task_data: TaskData
    ) -> tuple[List[Optional[Grid]], List[Optional[Grid]], List[Optional[ExecutionError]], List[Optional[ExecutionError]]]:
        """
        Execute program on task data and return outputs and error details.

        Returns:
            train_outputs, test_outputs, train_errors, test_errors
        """
        # Merge train and test examples so we can do this in one bulk call (faster).
        all_examples = task_data["train"] + task_data["test"]
        all_outputs, all_errors = self._execute_on_examples(program, all_examples)
        
        # Unmerge outputs and errors
        train_len = len(task_data["train"])
        train_outputs = all_outputs[:train_len]
        test_outputs = all_outputs[train_len:]
        train_errors = all_errors[:train_len]
        test_errors = all_errors[train_len:]

        return train_outputs, test_outputs, train_errors, test_errors

    def _compute_correctness_list(self, outputs: List[Optional[Grid]], examples: List[TaskExample]) -> List[bool]:
        """Compute boolean list of correctness for outputs against expected examples."""
        if not examples:
            return []

        return [
            outputs[i] is not None and outputs[i] == example["output"]
            for i, example in enumerate(examples)
        ]

    def test_program(self, program: str, task_data: TaskData) -> ProgramTestResult:
        """Test a program on ARC task data and compute correctness metrics.
        
        Args:
            program: The program code to test
            task_data: The ARC task data containing train and test examples
            
        Returns:
            ProgramTestResult containing outputs, inputs, correctness metrics, and optionally detailed error info
        """
        # Execute program and get results - expected errors are handled within the function
        train_outputs, test_outputs, train_errors, test_errors = self._run_program_on_task(program, task_data)

        # Check if the program failed for any input
        success = not any(output is None for output in train_outputs + test_outputs)

        # Extract inputs from task data (keep train and test separate)
        train_inputs = [example["input"] for example in task_data["train"]]
        test_inputs = [example["input"] for example in task_data["test"]]

        # Compute correctness lists
        correct_train_input = self._compute_correctness_list(train_outputs, task_data["train"])
        correct_test_input = self._compute_correctness_list(test_outputs, task_data["test"])

        return ProgramTestResult(
            train_outputs=train_outputs,
            test_outputs=test_outputs,
            train_inputs=train_inputs,
            test_inputs=test_inputs,
            correct_train_input=correct_train_input,
            correct_test_input=correct_test_input,
            success=success,
            train_errors=train_errors,
            test_errors=test_errors
        )
