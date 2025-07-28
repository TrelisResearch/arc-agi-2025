#!/usr/bin/env python3
"""
Test script to verify the subprocess_executor works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from subprocess_executor import execute_code_in_subprocess

def test_basic_functionality():
    """Test basic functionality of the subprocess executor."""
    
    # Test basic arithmetic
    result, error = execute_code_in_subprocess("return 2 + 3", timeout=5.0)
    print(f"Basic arithmetic: result={result}, error={error}")
    assert error is None
    assert result == 5
    
    # Test list operations
    result, error = execute_code_in_subprocess("""
numbers = [1, 2, 3, 4, 5]
return sum(numbers)
""", timeout=5.0)
    print(f"List operations: result={result}, error={error}")
    assert error is None
    assert result == 15
    
    # Test timeout handling
    result, error = execute_code_in_subprocess(
        "import time; time.sleep(10); return 'should timeout'", 
        timeout=1.0
    )
    print(f"Timeout test: result={result}, error={error}")
    assert result is None
    assert error is not None
    assert "timeout" in str(error).lower()
    
    # Test error handling
    result, error = execute_code_in_subprocess("return len(None)", timeout=5.0)
    print(f"Error handling: result={result}, error={error}")
    assert result is None
    assert error is not None
    assert "TypeError" in str(error)
    
    print("All subprocess executor tests passed!")

if __name__ == "__main__":
    test_basic_functionality()
