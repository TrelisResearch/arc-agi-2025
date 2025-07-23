import subprocess
import sys
import pickle
import base64
import binascii
from typing import Optional, Any, Tuple
from base_executor import BaseExecutor


class UnrestrictedExecutor(BaseExecutor):
    """
    Executor that runs Python code in a subprocess without restrictions.
    """
    
    def __init__(self):
        """Initialize the unrestricted executor."""
        self._setup_done = False
    
    def setup(self) -> None:
        """Set up the executor (no setup needed for subprocess execution)."""
        self._setup_done = True
    
    def cleanup(self) -> None:
        """Clean up resources (no cleanup needed for subprocess execution)."""
        pass
    
    def execute_code(self, code: str, timeout: Optional[float] = None) -> Tuple[Any, Optional[Exception]]:
        """
        Executes the given Python code string in a subprocess and returns the result as a native Python object.
        
        The code should be written as the body of a function that returns a value.
        For example: "return 2 + 2" or "x = [1, 2, 3]; return sum(x)"
        
        Args:
            code (str): Python code to execute (should contain a return statement).
            timeout (float, optional): Timeout in seconds.
        
        Returns:
            Tuple[Any, Optional[Exception]]: A tuple containing:
                - The result of the code execution (None if there was an error)
                - An exception if an error occurred (None if successful)
        """
        if not self._setup_done:
            self.setup()
            
        # Wrap the user code in a function and serialize the result
        wrapper_code = f"""
import pickle
import base64
import sys

def user_function():
{chr(10).join('    ' + line for line in code.split(chr(10)))}

try:
    result = user_function()
    # Serialize the result using pickle and base64 encode it
    serialized = base64.b64encode(pickle.dumps(result)).decode('utf-8')
    print(f"RESULT_START{{serialized}}RESULT_END")
except Exception as e:
    # Serialize the exception
    serialized_error = base64.b64encode(pickle.dumps(e)).decode('utf-8')
    print(f"ERROR_START{{serialized_error}}ERROR_END")
"""
        
        try:
            result = subprocess.run(
                [sys.executable, "-c", wrapper_code],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Parse the output to extract the result or error
            output = result.stdout
            
            if "RESULT_START" in output and "RESULT_END" in output:
                # Extract and deserialize the result
                start_marker = "RESULT_START"
                end_marker = "RESULT_END"
                start_idx = output.find(start_marker) + len(start_marker)
                end_idx = output.find(end_marker)
                serialized_result = output[start_idx:end_idx]
                
                try:
                    decoded_result = base64.b64decode(serialized_result)
                    deserialized_result = pickle.loads(decoded_result)
                    return deserialized_result, None
                except (pickle.PickleError, binascii.Error) as e:
                    return None, Exception(f"Failed to deserialize result: {e}")
                    
            elif "ERROR_START" in output and "ERROR_END" in output:
                # Extract and deserialize the error
                start_marker = "ERROR_START"
                end_marker = "ERROR_END"
                start_idx = output.find(start_marker) + len(start_marker)
                end_idx = output.find(end_marker)
                serialized_error = output[start_idx:end_idx]
                
                try:
                    decoded_error = base64.b64decode(serialized_error)
                    deserialized_error = pickle.loads(decoded_error)
                    return None, deserialized_error
                except (pickle.PickleError, binascii.Error) as e:
                    return None, Exception(f"Failed to deserialize error: {e}")
            
            else:
                # No result markers found, return stderr if available
                error_msg = result.stderr if result.stderr else "No result returned from code execution"
                return None, Exception(error_msg)
                
        except subprocess.TimeoutExpired:
            return None, Exception(f"Code execution timed out after {timeout} seconds")
        except Exception as e:
            return None, Exception(f"Subprocess execution failed: {e}")


# Legacy function for backwards compatibility
def execute_python_code_with_result(code: str, timeout: Optional[float] = None) -> Tuple[Any, Optional[Exception]]:
    """
    Legacy function for backwards compatibility.
    Uses UnrestrictedExecutor internally.
    """
    executor = UnrestrictedExecutor()
    return executor.execute_code(code, timeout)


def execute_python_code(code: str, timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    """
    Executes the given Python code string in a subprocess with an optional timeout.
    
    Legacy function preserved for backwards compatibility.

    Args:
        code (str): Python code to execute.
        timeout (float, optional): Timeout in seconds.

    Returns:
        subprocess.CompletedProcess: The result of the subprocess execution.
    """
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout
    )
    return result