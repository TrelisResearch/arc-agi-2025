#!/usr/bin/env python3
"""
Minimal test to understand duplicate API calls
"""

import openai
import time
from datetime import datetime

def single_api_call_test():
    """Make exactly one API call and log everything that happens"""
    
    print("=== Single API Call Test ===")
    print(f"Start time: {datetime.now()}")
    
    # Setup client
    base_url = "https://g8xfhnpws0tqwu-8000.proxy.runpod.net/v1"
    client = openai.OpenAI(api_key="dummy", base_url=base_url)
    
    # Simple request
    messages = [{"role": "user", "content": "Say hello"}]
    
    print(f"Making API call at: {datetime.now()}")
    
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen3-32B-FP8",
            messages=messages,
            max_tokens=50,
            temperature=0.1
        )
        
        print(f"Response received at: {datetime.now()}")
        print(f"Response ID: {response.id}")
        print(f"Model: {response.model}")
        print(f"Content: {repr(response.choices[0].message.content)}")
        print(f"Usage: {response.usage}")
        
        # Check if reasoning fields exist
        message = response.choices[0].message
        if hasattr(message, 'reasoning_content'):
            print(f"Reasoning content length: {len(message.reasoning_content) if message.reasoning_content else 0}")
        
        print("✅ Single call completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("Testing single API call behavior...")
    success = single_api_call_test()
    print(f"Final result: {'SUCCESS' if success else 'FAILURE'}")
    print(f"End time: {datetime.now()}") 