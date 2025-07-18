#!/usr/bin/env python3
"""
Quick test to understand argument parsing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

# Copy the argument parsing logic from run_arc_tasks.py
def test_args():
    parser = argparse.ArgumentParser(description="Run ARC tasks with OpenAI models")
    parser.add_argument("--model", default="gpt-4.1-nano", help="Model to use")
    parser.add_argument("--base-url", help="Custom API base URL")
    parser.add_argument("--limit", type=int, help="Limit number of tasks")
    parser.add_argument("--max-turns", type=int, default=3, help="Maximum number of turns/attempts per task")
    parser.add_argument("--max-workers", type=int, default=30, help="Maximum number of concurrent workers")
    parser.add_argument("--repeat-runs", type=int, default=1, help="Number of times to repeat the entire run")
    parser.add_argument("--independent-attempts", action="store_true",
                        help="Use independent attempts mode instead of multi-turn feedback")
    
    # Test with some common argument combinations
    test_cases = [
        ["--limit", "1", "--max-turns", "1", "--repeat-runs", "1"],
        ["--limit", "1", "--max-turns", "1", "--repeat-runs", "1", "--independent-attempts"],
        ["--model", "qwen/qwen3-32b", "--base-url", "https://g8xfhnpws0tqwu-8000.proxy.runpod.net/v1", "--limit", "1", "--max-turns", "1", "--repeat-runs", "1"]
    ]
    
    for i, test_args in enumerate(test_cases):
        print(f"\n=== Test Case {i+1}: {' '.join(test_args)} ===")
        args = parser.parse_args(test_args)
        
        print(f"model: {args.model}")
        print(f"limit: {args.limit}")
        print(f"max_turns: {args.max_turns}")
        print(f"repeat_runs: {args.repeat_runs}")
        print(f"independent_attempts: {args.independent_attempts}")
        print(f"base_url: {args.base_url}")
        
        # Simulate the key logic from run_arc_tasks.py
        expected_api_calls = args.max_turns if args.independent_attempts else 1
        print(f"Expected API calls per task: {expected_api_calls}")
        print(f"Total API calls for {args.limit or 'all'} tasks: {expected_api_calls * (args.limit or 1)}")

if __name__ == "__main__":
    print("Testing argument parsing scenarios...")
    test_args() 