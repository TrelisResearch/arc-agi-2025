#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_arc_tasks import ARCTaskRunner

def test_single_task():
    print("🔍 DEBUG SCRIPT: Starting debug test for task 746b3537")
    
    # Create runner with same config as user's command
    runner = ARCTaskRunner(
        model="Qwen/Qwen3-30B-A3B-FP8",
        max_workers=1,  # Use 1 worker for clearer debugging
        rate_limit_delay=0.0,
        max_turns=8,
        run_number=0,
        independent_attempts=True,
        base_url="https://9433j1ookvmo7y-8000.proxy.runpod.net/v1",
        reasoning_effort="low"
    )
    
    print("🔍 DEBUG SCRIPT: Runner created successfully")
    
    # Load the specific task
    try:
        print("🔍 DEBUG SCRIPT: Loading task 746b3537 from arc-agi-1 dataset")
        task_data = runner.task_loader.load_task("746b3537", "arc-agi-1")
        print(f"🔍 DEBUG SCRIPT: Task loaded successfully. Type: {type(task_data)}")
        print(f"🔍 DEBUG SCRIPT: Task keys: {list(task_data.keys()) if isinstance(task_data, dict) else 'Not a dict'}")
        
        if isinstance(task_data, dict):
            if 'train' in task_data:
                print(f"🔍 DEBUG SCRIPT: Train examples: {len(task_data['train'])}")
            if 'test' in task_data:
                print(f"🔍 DEBUG SCRIPT: Test examples: {len(task_data['test'])}")
        
    except Exception as e:
        print(f"🔍 DEBUG SCRIPT: Error loading task: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Run the task
    try:
        print("🔍 DEBUG SCRIPT: About to run the task")
        result = runner.run_task("746b3537", task_data, total_tasks=1)
        print(f"🔍 DEBUG SCRIPT: Task completed. Result type: {type(result)}")
        print(f"🔍 DEBUG SCRIPT: Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        
        # Print key result fields
        if isinstance(result, dict):
            print(f"🔍 DEBUG SCRIPT: task_id: {result.get('task_id', 'missing')}")
            print(f"🔍 DEBUG SCRIPT: turns_used: {result.get('turns_used', 'missing')}")
            print(f"🔍 DEBUG SCRIPT: request_cost: {result.get('request_cost', 'missing')}")
            print(f"🔍 DEBUG SCRIPT: api_success: {result.get('api_success', 'missing')}")
            print(f"🔍 DEBUG SCRIPT: task_failure_reason: {result.get('task_failure_reason', 'missing')}")
            
    except Exception as e:
        print(f"🔍 DEBUG SCRIPT: Error running task: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_task() 