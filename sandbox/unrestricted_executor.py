from typing import Optional, Any, Tuple
from .base_executor import BaseExecutor
from .subprocess_executor import execute_code_in_subprocess


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
            
        return execute_code_in_subprocess(code, timeout)
