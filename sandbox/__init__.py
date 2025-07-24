"""
Python Code Executors - A unified interface for secure Python code execution.

This module provides multiple backends for executing Python code with different
levels of isolation and security.

Quick Start:
    from sandbox import create_executor
    
    # Auto-select best available executor (tries Docker, falls back to unrestricted)
    with create_executor() as executor:
        result, error = executor.execute_code("return 2 + 2")
        print(f"Result: {result}")
    
    # Or explicitly choose an executor type
    with create_executor("unrestricted") as executor:
        result, error = executor.execute_code("return 2 + 2")
        print(f"Result: {result}")
"""

from typing import Optional

from .base_executor import BaseExecutor
from .unrestricted_executor import UnrestrictedExecutor

try:
    from .docker_sandbox import DockerSandboxExecutor
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False


def create_executor(executor_type: Optional[str] = None, **kwargs) -> BaseExecutor:
    """
    Factory function to create a code executor.
    
    Args:
        executor_type: Type of executor to create ('unrestricted', 'docker', or None for auto-selection)
                      If None, tries Docker first, then falls back to unrestricted
        **kwargs: Additional arguments passed to the executor constructor
        
    Returns:
        BaseExecutor: The created executor instance
        
    Raises:
        ValueError: If executor_type is not supported
        ImportError: If Docker dependencies are missing for docker executor
    """
    # Auto-selection: try Docker first, fall back to unrestricted
    if executor_type is None:
        try:
            executor = DockerSandboxExecutor(**kwargs) if DOCKER_AVAILABLE else None
            if executor:
                # Test if Docker setup actually works
                try:
                    if hasattr(executor, 'setup'):
                        executor.setup()
                    return executor
                except Exception:
                    # Docker setup failed, clean up and fall back
                    try:
                        if hasattr(executor, 'cleanup'):
                            executor.cleanup()
                    except:
                        pass
        except Exception:
            pass
        # Fall back to unrestricted
        return UnrestrictedExecutor(**kwargs)
    
    # Explicit type requested
    elif executor_type == "unrestricted":
        return UnrestrictedExecutor(**kwargs)
    elif executor_type == "docker":
        if not DOCKER_AVAILABLE:
            raise ImportError("Docker executor requires 'docker' package. Install with: pip install docker")
        return DockerSandboxExecutor(**kwargs)
    else:
        raise ValueError(f"Unknown executor type: {executor_type}. Supported types: 'unrestricted', 'docker', or None for auto-selection")


__all__ = ["create_executor"]
