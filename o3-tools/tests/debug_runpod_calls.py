#!/usr/bin/env python3
"""
Simple debug script to understand RunPod API call behavior
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openai
import json
import time
from datetime import datetime

def test_runpod_call():
    """Make a single API call to RunPod and examine the response structure"""
    
    # RunPod endpoint from conversation
    base_url = "https://g8xfhnpws0tqwu-8000.proxy.runpod.net/v1"
    
    client = openai.OpenAI(
        api_key="dummy",  # RunPod doesn't seem to need real key
        base_url=base_url
    )
    
    # Simple test message
    messages = [
        {"role": "user", "content": "What is 2 + 2? Think step by step."}
    ]
    
    print(f"[{datetime.now()}] Making API call to: {base_url}")
    print(f"Messages: {json.dumps(messages, indent=2)}")
    print("-" * 50)
    
    try:
        start_time = time.time()
        
        response = client.chat.completions.create(
            model="Qwen/Qwen3-32B-FP8",
            messages=messages,
            max_tokens=500,
            temperature=0.1
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"[{datetime.now()}] Response received in {duration:.2f} seconds")
        print(f"Response type: {type(response)}")
        print(f"Response object attributes: {dir(response)}")
        print("-" * 50)
        
        # Examine the response structure
        print("=== RESPONSE ANALYSIS ===")
        print(f"Model: {response.model}")
        print(f"Usage: {response.usage}")
        print(f"Choices length: {len(response.choices)}")
        
        choice = response.choices[0]
        print(f"Choice finish_reason: {choice.finish_reason}")
        print(f"Choice message type: {type(choice.message)}")
        print(f"Choice message attributes: {dir(choice.message)}")
        
        # Check message content
        message = choice.message
        print(f"Message content: {repr(message.content)}")
        print(f"Message role: {message.role}")
        
        # Check for reasoning fields
        print("\n=== REASONING FIELD CHECK ===")
        if hasattr(message, 'reasoning'):
            print(f"Has 'reasoning' attribute: {message.reasoning}")
        else:
            print("No 'reasoning' attribute found")
            
        if hasattr(message, 'reasoning_content'):
            print(f"Has 'reasoning_content' attribute: {message.reasoning_content}")
        else:
            print("No 'reasoning_content' attribute found")
        
        # Try to access as dict if it's a dict-like object
        if hasattr(message, '__dict__'):
            print(f"Message dict: {message.__dict__}")
        
        # Test content extraction logic
        print("\n=== CONTENT EXTRACTION TEST ===")
        content = message.content
        if content is None:
            print("Content is None - this might be the issue!")
            # Try to extract from reasoning fields
            reasoning_content = getattr(message, 'reasoning_content', None)
            reasoning = getattr(message, 'reasoning', None)
            print(f"Reasoning content fallback: {reasoning_content}")
            print(f"Reasoning fallback: {reasoning}")
        else:
            print(f"Content exists: {content}")
        
        return True
        
    except Exception as e:
        print(f"[{datetime.now()}] ERROR: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== RunPod API Call Debug Script ===")
    success = test_runpod_call()
    print(f"\nTest completed. Success: {success}") 