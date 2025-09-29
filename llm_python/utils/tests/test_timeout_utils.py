"""
Tests for timeout utilities.
"""

import time
import pytest


from ..timeout_utils import execute_with_timeout


def test_execute_with_timeout_success():
    """Test that execute_with_timeout works for successful function calls."""

    def quick_function():
        return "success"

    result = execute_with_timeout(quick_function, timeout=1.0)
    assert result == "success"


def test_execute_with_timeout_with_args():
    """Test that execute_with_timeout works with function arguments."""

    def add_function(a, b):
        return a + b

    result = execute_with_timeout(add_function, 2, 3, timeout=1.0)
    assert result == 5


def test_execute_with_timeout_with_kwargs():
    """Test that execute_with_timeout works with keyword arguments."""

    def multiply_function(a, b, multiplier=1):
        return (a + b) * multiplier

    result = execute_with_timeout(multiply_function, 2, 3, multiplier=2, timeout=1.0)
    assert result == 10


def test_execute_with_timeout_timeout():
    """Test that execute_with_timeout raises an exception when function times out."""

    def slow_function():
        time.sleep(2.0)
        return "should not reach here"

    with pytest.raises(Exception):
        execute_with_timeout(slow_function, timeout=0.1)


def test_execute_with_timeout_function_exception():
    """Test that execute_with_timeout propagates function exceptions."""

    def failing_function():
        raise ValueError("test error")

    with pytest.raises(ValueError, match="test error"):
        execute_with_timeout(failing_function, timeout=1.0)


def test_execute_with_timeout_default_timeout():
    """Test that execute_with_timeout uses default timeout when not specified."""

    def quick_function():
        return "success"

    result = execute_with_timeout(quick_function)
    assert result == "success"
