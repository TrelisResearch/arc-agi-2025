#!/usr/bin/env python3
"""
Quick timeout test using the modified runner with short timeouts.

This will help diagnose the timeout issue by:
1. Using very short timeouts (10-30s)
2. Adding debug logging
3. Running just a few tasks to see the behavior
"""

import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_python.tests.test_timeout_runner import ARCTaskRunnerSimple


def run_timeout_test():
    """Run a quick test with short timeouts"""
    print("🧪 TIMEOUT DIAGNOSIS TEST")
    print("=" * 50)
    print("Using modified runner with SHORT timeouts:")
    print("- API timeout: 10s (qwen-no-think) / 30s (normal)")
    print("- Worker timeout: 300s (5 minutes)")
    print("- Testing with 2 tasks, 4 attempts each")
    print("=" * 50)
    
    # Create runner with the provided endpoint
    runner = ARCTaskRunnerSimple(
        model="Trelis/Qwen3-4B_dsarc-programs-50-full-200-partial_20250807-211749-c3171",
        base_url="http://38.80.152.249:30524/v1",
        max_workers=2,  # Low to avoid disrupting other processes
        max_attempts=4,  # Small number for quick test
        qwen_no_think=True,  # This gives us 10s timeout
        unsafe_executor=True,  # Avoid Docker dependency
        debug=True  # Enable debug output
    )
    
    print(f"🎯 Target timeout: {runner.api_timeout}s")
    print()
    
    try:
        # Run just 2 tasks to see the behavior
        results, _ = runner.run_subset(
            subset_name="unique_training_tasks", 
            dataset="arc-agi-2", 
            limit=2  # Just 2 tasks for quick diagnosis
        )
        
        print("\n🎯 TEST RESULTS:")
        print("-" * 30)
        
        if results:
            for result in results:
                task_id = result.get('task_id', 'unknown')
                attempts = result.get('attempt_details', [])
                
                print(f"\nTask {task_id}:")
                for attempt in attempts:
                    attempt_num = attempt.get('attempt_number', '?')
                    api_timeout = attempt.get('api_timeout', False)
                    api_success = attempt.get('api_success', True)
                    error = attempt.get('error', 'none')
                    
                    status = "TIMEOUT" if api_timeout else ("SUCCESS" if api_success else "ERROR")
                    print(f"  Attempt {attempt_num}: {status} - {error}")
        else:
            print("No results returned!")
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_timeout_test()