#!/usr/bin/env python3
"""
Final diagnosis: Test the exact scenario the user is running
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openai
import re
from run_arc_tasks import ARCTaskRunner

def test_exact_scenario():
    """Test with a real ARC task to see what causes duplicate calls"""
    
    print("=== Final Diagnosis Test ===")
    print("Simulating: --limit 1 --max_turns 1 --repeat-runs 1")
    print("Model: RunPod Qwen endpoint")
    print()
    
    # Setup exactly like user would
    base_url = "https://g8xfhnpws0tqwu-8000.proxy.runpod.net/v1"
    client = openai.OpenAI(api_key="dummy", base_url=base_url)
    
    # Simple ARC-like task
    fake_task_data = {
        'train': [
            {'input': [[0, 1], [1, 0]], 'output': [[1, 0], [0, 1]]}
        ],
        'test': [
            {'input': [[0, 1], [1, 0]], 'output': [[1, 0], [0, 1]]}
        ]
    }
    
    # Test the key methods that might cause issues
    print("=== Testing Code Extraction ===")
    
    # Make an API call with a typical ARC prompt
    prompt = """
    You are an expert at solving abstract reasoning puzzles. Write clean, efficient Python code.
    
    Here are the training examples:
    Input: [[0, 1], [1, 0]]
    Output: [[1, 0], [0, 1]]
    
    Write a Python function called `def transform(grid):` that transforms the input to match the output pattern.
    """
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        print("Making API call...")
        response = client.chat.completions.create(
            model="Qwen/Qwen3-32B-FP8",
            messages=messages,
            max_tokens=800,
            temperature=0.1
        )
        
        print(f"Response received")
        print(f"Content is None: {response.choices[0].message.content is None}")
        print(f"Has reasoning_content: {hasattr(response.choices[0].message, 'reasoning_content')}")
        
        # Test code extraction exactly like the main script does
        runner = ARCTaskRunner(
            model="Qwen/Qwen3-32B-FP8",
            base_url=base_url,
            max_turns=1  # This is key!
        )
        
        extracted_code = runner.extract_code_from_response(response)
        print(f"\nExtracted code: {repr(extracted_code)}")
        print(f"Code length: {len(extracted_code) if extracted_code else 0}")
        
        if extracted_code:
            print(f"Code starts with 'def ': {extracted_code.startswith('def ')}")
            print(f"Contains 'transform': {'transform' in extracted_code}")
            print(f"Contains 'return': {'return' in extracted_code}")
            
            # Show what was extracted
            print(f"\nExtracted code preview:")
            print("=" * 40)
            print(extracted_code[:200] + ("..." if len(extracted_code) > 200 else ""))
            print("=" * 40)
            
            # Test if this code would actually work
            try:
                exec(extracted_code)
                if 'transform' in locals():
                    print("✅ Code compiles and defines transform function")
                else:
                    print("❌ Code compiles but no transform function found")
            except Exception as e:
                print(f"❌ Code compilation error: {e}")
                
        else:
            print("❌ No code extracted - this would trigger retries!")
            
            # Show what text was available
            message = response.choices[0].message
            reasoning_text = getattr(message, 'reasoning_content', '') or ''
            print(f"\nReasoning text available: {len(reasoning_text)} chars")
            if reasoning_text:
                print(f"Reasoning preview: {reasoning_text[:300]}...")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_exact_scenario()
    print(f"\nDiagnosis complete. Success: {success}") 