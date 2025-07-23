"""
Unit tests for the Python code executors using pytest.
"""

import pytest
import sys
from pathlib import Path

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from unrestricted_executor import UnrestrictedExecutor, execute_python_code_with_result
from executor_factory import create_executor, get_best_executor, get_fast_executor

# Try to import Docker executor (may not be available)
try:
    from docker_sandbox import DockerSandboxExecutor
    DOCKER_AVAILABLE = True
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


class TestLegacyCompatibility:
    """Test backwards compatibility with the original function."""
    
    def test_legacy_function(self):
        """Test the legacy execute_python_code_with_result function."""
        result, error = execute_python_code_with_result("return 'legacy test'")
        assert error is None
        assert result == 'legacy test'
    
    def test_legacy_error_handling(self):
        """Test legacy function error handling."""
        result, error = execute_python_code_with_result("return undefined_variable")
        assert result is None
        assert error is not None
        assert "NameError" in str(error) or "not defined" in str(error)


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
    
    def test_get_best_executor(self):
        """Test getting the best available executor."""
        executor = get_best_executor()
        # Should return some executor (Docker if available, otherwise unrestricted)
        assert executor is not None
        
        with executor:
            result, error = executor.execute_code("return 'best executor'")
            assert error is None
            assert result == 'best executor'
    
    def test_get_fast_executor(self):
        """Test getting the fast (unrestricted) executor."""
        executor = get_fast_executor()
        assert isinstance(executor, UnrestrictedExecutor)
        
        with executor:
            result, error = executor.execute_code("return 'fast executor'")
            assert error is None
            assert result == 'fast executor'
    
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
