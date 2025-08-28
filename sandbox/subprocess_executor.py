"""
Simple, self-contained subprocess executor for Python code.
No dependencies on base classes or complex imports.
"""

import subprocess
import sys
import pickle
import base64
import binascii
from typing import Optional, Any, Tuple

import resource

MAX_VIRTUAL_MEMORY = 1024 * 1024 * 1024 # 1Gb

def limit_virtual_memory():
    resource.setrlimit(resource.RLIMIT_AS, (MAX_VIRTUAL_MEMORY, resource.RLIM_INFINITY))

def execute_code_in_subprocess(code: str, timeout: Optional[float] = None) -> Tuple[Any, Optional[Exception]]:
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
            timeout=timeout,
            preexec_fn=limit_virtual_memory
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
