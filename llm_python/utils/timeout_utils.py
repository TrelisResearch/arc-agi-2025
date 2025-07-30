"""
Timeout utilities for ARC-AGI task processing.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


def execute_with_timeout(func: Callable, *args, timeout: float = 30, **kwargs) -> Any:
    """
    Execute a function with timeout using ThreadPoolExecutor.
    
    Args:
        func: The function to execute
        *args: Positional arguments to pass to the function
        timeout: Timeout in seconds (default: 30)
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The result of the function execution
        
    Raises:
        Exception: If the function execution times out or fails
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout)
            return result
        except Exception as e:
            future.cancel()
            raise e 