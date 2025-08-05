#!/usr/bin/env python3

from typing import List, Optional, NamedTuple
from ..utils.scoring import ProgramExecutor
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
    
    def __init__(self, timeout: int = 2, executor_type: str = "unrestricted", detailed_errors: bool = False):
        """Initialize the ARC tester.
        
        Args:
            timeout: Execution timeout in seconds
            executor_type: Type of executor to use
            detailed_errors: Whether to capture detailed error information
        """
        self.program_executor = ProgramExecutor(timeout=timeout, executor_type=executor_type)
        self.detailed_errors = detailed_errors
    
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
        results = self.program_executor.execute_program_bulk(program, inputs)

        # Process results and extract outputs and errors
        outputs = []
        errors = []
        
        for result in results:
            output, error_message, timed_out = result
            validated_output = self._validate_grid(output)
            outputs.append(validated_output)
            
            if self.detailed_errors:
                if validated_output is None or error_message:
                    errors.append(self._classify_error(output, error_message, timed_out))
                else:
                    errors.append(None)  # No error
            else:
                errors.append(None)  # Not capturing detailed errors

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
            train_errors=train_errors if self.detailed_errors else None,
            test_errors=test_errors if self.detailed_errors else None
        )
