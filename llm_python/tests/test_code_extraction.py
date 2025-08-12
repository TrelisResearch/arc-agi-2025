#!/usr/bin/env python3
"""
Test code extraction from RunPod response format
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openai
from run_arc_tasks_soar import ARCTaskRunnerSimple

def test_code_extraction():
    """Test if code extraction works with RunPod's content=None format"""
    
    print("=== Testing Code Extraction from RunPod Response ===")
    
    # Create a mock runner to test the extraction method
    runner = ARCTaskRunnerSimple(
        model="Qwen/Qwen3-32B-FP8",
        base_url="https://g8xfhnpws0tqwu-8000.proxy.runpod.net/v1"
    )
    
    # Make a real API call to get actual RunPod response structure
    base_url = "https://g8xfhnpws0tqwu-8000.proxy.runpod.net/v1"
    client = openai.OpenAI(api_key="dummy", base_url=base_url)
    
    # Simple message that should produce code
    messages = [
        {"role": "user", "content": """Write a Python function called `def transform(grid):` that returns the input grid unchanged. Just return it as is.

Final answer:
```python
def transform(grid):
    return grid
```"""}
    ]
    
    print("Making API call to test code extraction...")
    
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen3-32B-FP8",
            messages=messages,
            max_tokens=300,
            temperature=0.1
        )
        
        print(f"Response received. Content: {repr(response.choices[0].message.content)}")
        
        # Test the extraction method
        extracted_code = runner.extract_code_from_response(response)
        
        print(f"\n=== CODE EXTRACTION RESULT ===")
        print(f"Extracted code: {repr(extracted_code)}")
        print(f"Code extracted successfully: {bool(extracted_code)}")
        
        if extracted_code:
            print(f"\nExtracted code content:\n{extracted_code}")
            return True
        else:
            print("\n❌ No code extracted - this could be the issue!")
            
            # Debug: show what text was available for extraction
            message = response.choices[0].message
            full_text = message.content if message.content else ""
            reasoning_text = ""
            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                reasoning_text = message.reasoning_content
            
            combined_text = full_text + "\n\n" + reasoning_text if reasoning_text else full_text
            
            print(f"\nDEBUG - Available text for extraction:")
            print(f"Full text length: {len(full_text)}")
            print(f"Reasoning text length: {len(reasoning_text)}")
            print(f"Combined text length: {len(combined_text)}")
            print(f"Combined text preview (first 500 chars):\n{combined_text[:500]}...")
            
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_code_extraction()
    print(f"\nTest result: {'PASS' if success else 'FAIL'}") 