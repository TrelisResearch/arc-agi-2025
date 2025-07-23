"""
Python Code Executors - A unified interface for secure Python code execution.

This module provides multiple backends for executing Python code with different
levels of isolation and security.

Quick Start:
    from python_sandbox import get_best_executor
    
    with get_best_executor() as executor:
        result, error = executor.execute_code("return 2 + 2")
        print(f"Result: {result}")
"""

# Base classes and interfaces
from base_executor import BaseExecutor

# Executor implementations
from unrestricted_executor import UnrestrictedExecutor
from unrestricted_executor import execute_python_code_with_result  # Legacy compatibility

# Factory functions
from executor_factory import (
    create_executor,
    get_best_executor,
    get_fast_executor,
    get_secure_executor
)

# Conditional imports for Docker executor
try:
    from docker_sandbox import DockerSandboxExecutor
    __all__ = [
        # Base classes
        "BaseExecutor",
        
        # Executor implementations
        "UnrestrictedExecutor",
        "DockerSandboxExecutor",
        
        # Factory functions
        "create_executor",
        "get_best_executor", 
        "get_fast_executor",
        "get_secure_executor",
        
        # Legacy compatibility
        "execute_python_code_with_result",
    ]
except ImportError:
    __all__ = [
        # Base classes
        "BaseExecutor",
        
        # Executor implementations
        "UnrestrictedExecutor",
        
        # Factory functions
        "create_executor",
        "get_best_executor",
        "get_fast_executor",
        "get_secure_executor",
        
        # Legacy compatibility
        "execute_python_code_with_result",
    ]


# Version information
__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"
