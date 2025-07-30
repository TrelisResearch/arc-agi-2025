#!/usr/bin/env python3
"""
Debug test script to inspect RunPod endpoint response structure
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

def test_runpod_qwen_debug():
    """Debug test of RunPod endpoint with full response inspection"""
    
    # Load .env file from o3-tools directory
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
    
    # Initialize client with RunPod endpoint
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(
        api_key=api_key,
        base_url="https://g8xfhnpws0tqwu-8000.proxy.runpod.net/v1"
    )
    
    model = "Qwen/Qwen3-32B-FP8"
    prompt = "Hello! Please respond with: 'I am working correctly'"
    
    print(f"Debug test: RunPod endpoint with {model}")
    print(f"Simple prompt: {prompt}")
    print("-" * 60)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        print("✅ API CALL SUCCESSFUL!")
        print(f"Full response object: {response}")
        print(f"Response type: {type(response)}")
        print("-" * 60)
        
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            print(f"Choice object: {choice}")
            print(f"Choice type: {type(choice)}")
            
            if hasattr(choice, 'message'):
                message = choice.message
                print(f"Message object: {message}")
                print(f"Message type: {type(message)}")
                print(f"Message attributes: {dir(message)}")
                
                # Check content
                content = getattr(message, 'content', 'NO_CONTENT_ATTR')
                print(f"Content: '{content}' (type: {type(content)})")
                
                # Check for reasoning
                reasoning = getattr(message, 'reasoning', 'NO_REASONING_ATTR')
                print(f"Reasoning: '{reasoning}' (type: {type(reasoning)})")
                
                # Check role
                role = getattr(message, 'role', 'NO_ROLE_ATTR')
                print(f"Role: '{role}' (type: {type(role)})")
        
        print("-" * 60)
        print(f"Usage: {response.usage}")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_runpod_qwen_debug() 