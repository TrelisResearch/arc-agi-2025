"""
Unit tests for the Python code executors using pytest.
"""

import pytest

from ..unrestricted_executor import UnrestrictedExecutor
from .. import create_executor

# Try to import Docker executor (may not be available)
try:
    from ..docker_sandbox import DockerSandboxExecutor
    import docker
    # Check if Docker is actually running
    try:
        client = docker.from_env()
        client.ping()
        DOCKER_AVAILABLE = True
    except:
        DOCKER_AVAILABLE = False
except ImportError:
    DOCKER_AVAILABLE = False


class TestUnrestrictedExecutor:
    """Test the unrestricted (subprocess) executor."""
    
    def test_basic_arithmetic(self):
        """Test basic arithmetic operations."""
        with UnrestrictedExecutor() as executor:
            result, error = executor.execute_code("return 2 + 3")
            assert error is None
            assert result == 5
    
    def test_list_operations(self):
        """Test list operations."""
        with UnrestrictedExecutor() as executor:
            result, error = executor.execute_code("""
numbers = [1, 2, 3, 4, 5]
return sum(numbers)
""")
            assert error is None
            assert result == 15
    
    def test_complex_object(self):
        """Test returning complex objects."""
        with UnrestrictedExecutor() as executor:
            result, error = executor.execute_code("""
data = {
    'name': 'test',
    'values': [1, 2, 3],
    'nested': {'key': 'value'}
}
return data
""")
            assert error is None
            assert result['name'] == 'test'
            assert result['values'] == [1, 2, 3]
            assert result['nested']['key'] == 'value'
    
    def test_error_handling(self):
        """Test that errors are properly caught and returned."""
        with UnrestrictedExecutor() as executor:
            result, error = executor.execute_code("return 1 / 0")
            assert result is None
            assert error is not None
            assert "ZeroDivisionError" in str(error) or "division by zero" in str(error)
    
    def test_import_and_use(self):
        """Test importing modules."""
        with UnrestrictedExecutor() as executor:
            result, error = executor.execute_code("""
import math
return math.sqrt(16)
""")
            assert error is None
            assert result == 4.0
    
    def test_timeout(self):
        """Test timeout functionality."""
        with UnrestrictedExecutor() as executor:
            result, error = executor.execute_code("""
import time
time.sleep(0.1)  # Short sleep that should complete
return "completed"
""", timeout=1.0)
            assert error is None
            assert result == "completed"
    
    def test_context_manager(self):
        """Test context manager functionality."""
        with UnrestrictedExecutor() as executor:
            result, error = executor.execute_code("return 'context test'")
            assert error is None
            assert result == 'context test'

    def test_memory_limit_enforced(self):
        """Test that allocating >512MB fails due to memory limits."""
        with UnrestrictedExecutor() as executor:
            result, error = executor.execute_code(
                """
import os
import time
# Request 600 MiB of memory.
memory_to_allocate = 600 * 1024 * 1024
arr = bytearray(memory_to_allocate)

# Crucially, write to the bytearray to force the OS to commit
# physical RAM pages. This makes the RSS increase.
# We can do this efficiently with os.urandom.
arr[:] = os.urandom(memory_to_allocate)
time.sleep(1)
return len(arr) # This line should never be reached.
"""
            )
            # Should fail: either error is set, or result contains 'MemoryError'
            assert error is not None or (isinstance(result, str) and 'MemoryError' in result)


class TestExecutorFactory:
    """Test the executor factory functions."""
    
    def test_create_unrestricted_executor(self):
        """Test creating unrestricted executor via factory."""
        executor = create_executor("unrestricted")
        assert isinstance(executor, UnrestrictedExecutor)
        
        with executor:
            result, error = executor.execute_code("return 'factory test'")
            assert error is None
            assert result == 'factory test'
    
    def test_auto_selection_executor(self):
        """Test auto-selection of best available executor."""
        executor = create_executor()  # No args = auto-selection
        # Should return some executor (Docker if available, otherwise unrestricted)
        assert executor is not None
        
        with executor:
            result, error = executor.execute_code("return 'auto-selected executor'")
            assert error is None
            assert result == 'auto-selected executor'
    
    def test_explicit_unrestricted_executor(self):
        """Test explicitly requesting unrestricted executor."""
        executor = create_executor("unrestricted")
        assert isinstance(executor, UnrestrictedExecutor)
        
        with executor:
            result, error = executor.execute_code("return 'unrestricted executor'")
            assert error is None
            assert result == 'unrestricted executor'
    
    def test_invalid_executor_type(self):
        """Test that invalid executor types raise ValueError."""
        with pytest.raises(ValueError, match="Unknown executor type"):
            create_executor("invalid_type")
    
    @pytest.mark.skipif(not DOCKER_AVAILABLE, reason="Docker not available")
    def test_create_docker_executor(self):
        """Test creating Docker executor (only if Docker is available)."""
        executor = create_executor("docker")
        assert isinstance(executor, DockerSandboxExecutor)
        # Don't actually run code to avoid Docker setup overhead in tests
        executor.cleanup()


@pytest.mark.skipif(not DOCKER_AVAILABLE, reason="Docker not available")
class TestDockerExecutor:
    """Test the Docker sandbox executor (only if Docker is available)."""
    
    def test_docker_basic_execution(self):
        """Test basic Docker executor functionality."""
        # This test will be skipped if Docker is not available
        with DockerSandboxExecutor() as executor:
            result, error = executor.execute_code("return 'docker test'")
            assert error is None
            assert result == 'docker test'
    
    def test_docker_isolation(self):
        """Test that Docker provides isolation."""
        # This test will be skipped if Docker is not available
        with DockerSandboxExecutor() as executor:
            result, error = executor.execute_code("""
import sys
return sys.platform
""")
            assert error is None
            assert result == 'linux'  # Docker containers run Linux


class TestErrorScenarios:
    """Test various error scenarios."""
    
    def test_syntax_error(self):
        """Test handling of syntax errors."""
        with UnrestrictedExecutor() as executor:
            result, error = executor.execute_code("return 2 +")  # Syntax error
            assert result is None
            assert error is not None
            assert "SyntaxError" in str(error) or "invalid syntax" in str(error)
    
    def test_runtime_error(self):
        """Test handling of runtime errors."""
        with UnrestrictedExecutor() as executor:
            result, error = executor.execute_code("return len(None)")  # TypeError
            assert result is None
            assert error is not None
            assert "TypeError" in str(error) or "has no len" in str(error)
    
    def test_import_error(self):
        """Test handling of import errors."""
        with UnrestrictedExecutor() as executor:
            result, error = executor.execute_code("""
import nonexistent_module
return "success"
""")
            assert result is None
            assert error is not None
            assert "ModuleNotFoundError" in str(error) or "No module named" in str(error)


if __name__ == "__main__":
    # Run pytest when executed directly
    pytest.main([__file__, "-v"])
