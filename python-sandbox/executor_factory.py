"""
Factory for creating Python code executors.
"""

from typing import Optional
from base_executor import BaseExecutor
from unrestricted_executor import UnrestrictedExecutor

try:
    from docker_sandbox import DockerSandboxExecutor
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False


def create_executor(executor_type: str = "unrestricted", **kwargs) -> BaseExecutor:
    """
    Factory function to create a code executor.
    
    Args:
        executor_type: Type of executor to create ('unrestricted' or 'docker')
        **kwargs: Additional arguments passed to the executor constructor
        
    Returns:
        BaseExecutor: The created executor instance
        
    Raises:
        ValueError: If executor_type is not supported
        ImportError: If Docker dependencies are missing for docker executor
    """
    if executor_type == "unrestricted":
        return UnrestrictedExecutor(**kwargs)
    elif executor_type == "docker":
        if not DOCKER_AVAILABLE:
            raise ImportError("Docker executor requires 'docker' package. Install with: pip install docker")
        return DockerSandboxExecutor(**kwargs)
    else:
        raise ValueError(f"Unknown executor type: {executor_type}. Supported types: 'unrestricted', 'docker'")


def get_best_executor(**kwargs) -> BaseExecutor:
    """
    Get the best available executor for the current environment.
    Prefers Docker for security if available, falls back to unrestricted.
    
    Args:
        **kwargs: Additional arguments passed to the executor constructor
        
    Returns:
        BaseExecutor: The best available executor instance
    """
    try:
        executor = create_executor("docker", **kwargs)
        # Test if Docker setup actually works
        try:
            executor.setup()
            return executor
        except Exception:
            # Docker setup failed, clean up and fall back
            try:
                executor.cleanup()
            except:
                pass
            return create_executor("unrestricted", **kwargs)
    except ImportError:
        return create_executor("unrestricted", **kwargs)


def get_fast_executor(**kwargs) -> BaseExecutor:
    """
    Get the fastest executor (unrestricted subprocess).
    
    Args:
        **kwargs: Additional arguments passed to the executor constructor
        
    Returns:
        BaseExecutor: The unrestricted executor instance
    """
    return create_executor("unrestricted", **kwargs)


def get_secure_executor(**kwargs) -> BaseExecutor:
    """
    Get the most secure executor (Docker sandbox).
    
    Args:
        **kwargs: Additional arguments passed to the executor constructor
        
    Returns:
        BaseExecutor: The Docker sandbox executor instance
        
    Raises:
        ImportError: If Docker dependencies are missing
    """
    return create_executor("docker", **kwargs)
