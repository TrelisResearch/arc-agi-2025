#!/usr/bin/env python3
"""
Test data trimming logic for failed attempts.

This test verifies that:
1. Failed attempts (execution errors, no code, API failures) are trimmed
2. Successful attempts keep all data
3. Trimmed attempts have the correct fields
"""

import sys
import json
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_python.run_arc_tasks_soar import ARCTaskRunnerSimple


def create_test_attempt(success=True, exec_error=False, no_code=False, api_timeout=False):
    """Create a test attempt with specified characteristics"""
    attempt = {
        'attempt_number': 1,
        'timestamp': '2024-01-01T00:00:00',
        'input_tokens': 1000,
        'output_tokens': 500,
        'attempt_cost': 0.05,
        'program_extracted': not no_code,
        'api_success': not api_timeout,
        'api_timeout': api_timeout,
        'empty_response': no_code,
        'hit_max_tokens': False,
        'is_train_transductive': False,
        'is_test_transductive': False,
        'train_exec_errors': 3 if exec_error else 0,
        'train_exec_timeouts': 0,
        'test_exec_error': exec_error,
        'test_exec_timeout': False,
        'train_accuracy': 0.0 if exec_error else 0.5,
        'test_correct': success and not exec_error,
        'test_correct_count': 1 if success and not exec_error else 0,
        'error': 'execution failed' if exec_error else None,
        'raw_response': {'choices': [{'message': {'content': 'A' * 50000}}]},  # Large response
        'program': 'def solve():\n    ' + 'pass\n' * 1000,  # Large program
        'train_results': [{'predicted': None, 'expected': [[1]], 'correct': False}] * 5,
        'test_results': [{'predicted': None, 'expected': [[2]], 'correct': False}],
        'sampling_params': {'temperature': 0.7, 'max_tokens': 1000}
    }
    return attempt


def test_trimming_logic():
    """Test the _trim_failed_attempt method"""
    # Create runner instance with unsafe executor to avoid Docker dependency
    runner = ARCTaskRunnerSimple(model="test-model", max_workers=1, unsafe_executor=True)
    
    print("Testing data trimming logic...")
    print("-" * 50)
    
    # Test 1: Successful attempt should NOT be trimmed
    print("\n1. Testing successful attempt (should keep all data)...")
    success_attempt = create_test_attempt(success=True, exec_error=False)
    trimmed = runner._trim_failed_attempt(success_attempt)
    
    assert 'raw_response' in trimmed, "Successful attempt should keep raw_response"
    assert 'program' in trimmed, "Successful attempt should keep program"
    assert 'train_results' in trimmed, "Successful attempt should keep train_results"
    assert 'data_trimmed' not in trimmed, "Successful attempt should not be marked as trimmed"
    print("   ✓ Successful attempt keeps all data")
    
    # Test 2: Execution error should be trimmed
    print("\n2. Testing execution error attempt (should be trimmed)...")
    exec_error_attempt = create_test_attempt(exec_error=True)
    trimmed = runner._trim_failed_attempt(exec_error_attempt)
    
    assert trimmed.get('raw_response') is None, "Exec error attempt should have None raw_response"
    assert trimmed.get('program') == '', "Exec error attempt should have empty program"
    assert trimmed.get('train_results') == [], "Exec error attempt should have empty train_results"
    assert trimmed.get('data_trimmed') == True, "Should be marked as trimmed"
    assert trimmed.get('trim_reason') == 'execution_failure', "Should have trim reason"
    assert trimmed.get('train_exec_errors') == 3, "Should keep error counts"
    assert trimmed.get('input_tokens') == 1000, "Should keep token counts"
    print("   ✓ Execution error attempt correctly trimmed")
    
    # Test 3: No code extracted should be trimmed
    print("\n3. Testing no-code attempt (should be trimmed)...")
    no_code_attempt = create_test_attempt(no_code=True)
    trimmed = runner._trim_failed_attempt(no_code_attempt)
    
    assert trimmed.get('raw_response') is None, "No-code attempt should have None raw_response"
    assert trimmed.get('program') == '', "No-code attempt should have empty program"
    assert trimmed.get('data_trimmed') == True, "Should be marked as trimmed"
    assert trimmed.get('program_extracted') == False, "Should keep extraction flag"
    print("   ✓ No-code attempt correctly trimmed")
    
    # Test 4: API timeout should be trimmed
    print("\n4. Testing API timeout attempt (should be trimmed)...")
    timeout_attempt = create_test_attempt(api_timeout=True)
    trimmed = runner._trim_failed_attempt(timeout_attempt)
    
    assert trimmed.get('raw_response') is None, "Timeout attempt should have None raw_response"
    assert trimmed.get('program') == '', "Timeout attempt should have empty program"
    assert trimmed.get('data_trimmed') == True, "Should be marked as trimmed"
    assert trimmed.get('api_timeout') == True, "Should keep timeout flag"
    print("   ✓ API timeout attempt correctly trimmed")
    
    # Test 5: Check size reduction
    print("\n5. Testing size reduction...")
    large_attempt = create_test_attempt(exec_error=True)
    original_size = len(json.dumps(large_attempt))
    trimmed = runner._trim_failed_attempt(large_attempt)
    trimmed_size = len(json.dumps(trimmed))
    reduction_pct = (1 - trimmed_size/original_size) * 100
    
    print(f"   Original size: {original_size:,} bytes")
    print(f"   Trimmed size: {trimmed_size:,} bytes")
    print(f"   Reduction: {reduction_pct:.1f}%")
    assert trimmed_size < original_size * 0.1, "Should reduce size by >90%"
    print("   ✓ Size reduction achieved")
    
    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)
    
    # Summary
    print("\nSummary:")
    print("- Failed attempts are correctly trimmed")
    print("- Successful attempts keep all data")
    print("- Size reduction >90% for failed attempts")
    print("- Essential metadata preserved")


if __name__ == "__main__":
    test_trimming_logic()