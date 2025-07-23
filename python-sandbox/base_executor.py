"""
Base class for Python code executors.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple


class BaseExecutor(ABC):
    """Abstract base class for Python code executors."""
    
    @abstractmethod
    def execute_code(self, code: str, timeout: Optional[float] = None) -> Tuple[Any, Optional[Exception]]:
        """
        Execute Python code and return the result as a native Python object.
        
        Args:
            code (str): Python code to execute (should contain a return statement).
            timeout (float, optional): Timeout in seconds.
        
        Returns:
            Tuple[Any, Optional[Exception]]: A tuple containing:
                - The result of the code execution (None if there was an error)
                - An exception if an error occurred (None if successful)
        """
        pass
    
    @abstractmethod
    def setup(self) -> None:
        """Set up the executor (called once before first use)."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources (called when executor is no longer needed)."""
        pass
    
    def __enter__(self):
        """Context manager entry."""
        self.setup()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
