#!/usr/bin/env python3
"""
Test script for code execution timeouts - tests when generated code takes too long to execute
"""

import os
import sys
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from progdb.arc_tester import ArcTester
except ImportError:
    # Fallback for different import structure
    from llm_python.progdb.arc_tester import ArcTester


def test_code_execution_timeouts():
    """Test triggering code execution timeouts with slow/infinite loop code"""
    
    print("=== Code Execution Timeout Test ===")
    print(f"Start time: {datetime.now()}")
    
    # Initialize executor with short timeout for testing
    timeout_duration = 0.1  # Very short timeout for testing
    executor = ArcTester(timeout=timeout_duration, executor_type="docker")
    
    print(f"Executor timeout: {timeout_duration}s")
    print(f"Executor type: {executor.executor_type}")
    
    # Test input grid (doesn't matter much for timeout tests)
    test_input = [[1, 2], [3, 4]]
    
    # Test cases that should trigger timeouts
    timeout_test_cases = [
        {
            "name": "Infinite Loop",
            "code": """
def transform(grid):
    while True:  # Infinite loop
        pass
    return grid
""",
            "should_timeout": True
        },
        {
            "name": "Very Slow Loop",
            "code": """
def transform(grid):
    import time
    time.sleep(1.0)  # Sleep longer than timeout
    return grid
""",
            "should_timeout": True
        },
        {
            "name": "Heavy Computation",
            "code": """
def transform(grid):
    # Heavy computation that should timeout
    result = 0
    for i in range(10000000):  # Large computation
        for j in range(1000):
            result += i * j
    return grid
""",
            "should_timeout": True
        },
        {
            "name": "Fast Valid Code",
            "code": """
def transform(grid):
    # Simple, fast transformation
    return [[cell * 2 for cell in row] for row in grid]
""",
            "should_timeout": False
        }
    ]
    
    print(f"\nTesting {len(timeout_test_cases)} code execution scenarios:")
    print("-" * 60)
    
    for i, test_case in enumerate(timeout_test_cases, 1):
        name = test_case["name"]
        code = test_case["code"]
        should_timeout = test_case["should_timeout"]
        
        print(f"\n{i}. Testing: {name}")
        print(f"Expected to timeout: {should_timeout}")
        
        start_time = time.time()
        
        try:
            # Execute the code with timeout
            result, error, timed_out = executor.execute_program_with_timeout(code, test_input)
            execution_time = time.time() - start_time
            
            print(f"Execution time: {execution_time:.3f}s")
            print(f"Timed out: {timed_out}")
            print(f"Error: {error}")
            print(f"Result: {result}")
            
            # Check if timeout behavior matches expectation
            if should_timeout and timed_out:
                print("✅ SUCCESS: Code timed out as expected!")
            elif not should_timeout and not timed_out:
                print("✅ SUCCESS: Code completed without timeout as expected!")
            elif should_timeout and not timed_out:
                print("⚠️  UNEXPECTED: Code should have timed out but didn't")
            else:
                print("⚠️  UNEXPECTED: Code timed out but shouldn't have")
                
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ Exception occurred: {e}")
            print(f"Execution time before exception: {execution_time:.3f}s")
    
    print(f"\n" + "="*60)
    print(f"Test completed at: {datetime.now()}")


def test_timeout_metrics_integration():
    """Test that timeouts are properly tracked in metrics"""
    
    print(f"\n=== Timeout Metrics Integration Test ===")
    
    # This simulates what happens in the main runner
    timeout_duration = 0.1
    executor = ArcTester(timeout=timeout_duration, executor_type="docker")
    
    # Test data that simulates attempt details
    test_input = [[1, 0], [0, 1]]
    
    # Code that will timeout
    timeout_code = """
def transform(grid):
    import time
    time.sleep(1.0)
    return grid
"""
    
    print("Testing timeout tracking...")
    
    # Execute and track results like the main runner does
    result, error, timed_out = executor.execute_program_with_timeout(timeout_code, test_input)
    
    # Simulate attempt detail creation
    attempt_detail = {
        'test_exec_timeout': timed_out,
        'test_error': error,
        'test_timed_out': timed_out,
        'program_extracted': True,
        'test_correct': False  # Can't be correct if timed out
    }
    
    print(f"Timeout result: {timed_out}")
    print(f"Error: {error}")
    print(f"Attempt detail: {attempt_detail}")
    
    # Verify timeout is properly captured
    if attempt_detail['test_exec_timeout']:
        print("✅ SUCCESS: Timeout properly tracked in attempt details!")
    else:
        print("❌ FAILURE: Timeout not captured in attempt details")
    
    return attempt_detail


if __name__ == "__main__":
    test_code_execution_timeouts()
    test_timeout_metrics_integration()