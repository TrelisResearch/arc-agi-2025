#!/usr/bin/env python3
"""
Quick test script to validate RunPod endpoint with Qwen model
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

def test_runpod_qwen():
    """Test the RunPod endpoint with Qwen model"""
    
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
    prompt = "Write a simple Python function that adds two numbers. Show your reasoning."
    
    print(f"Testing RunPod endpoint with {model}")
    print(f"Endpoint: https://g8xfhnpws0tqwu-8000.proxy.runpod.net/v1")
    print(f"Prompt: {prompt}")
    print("-" * 60)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.1
        )
        
        content = response.choices[0].message.content
        reasoning = getattr(response.choices[0].message, 'reasoning', None)
        
        print("✅ SUCCESS!")
        print("Response content:")
        print(content)
        print("\n" + "-" * 60)
        
        if reasoning:
            print("🧠 REASONING DETECTED:")
            print(reasoning)
            print("-" * 60)
        else:
            print("🤔 No reasoning field found")
        
        print(f"Token usage: {response.usage.prompt_tokens} input, {response.usage.completion_tokens} output")
        
        # Save result
        results_dir = Path("tests/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        result_file = results_dir / "runpod_qwen_test.json"
        with open(result_file, 'w') as f:
            json.dump({
                "endpoint": "https://g8xfhnpws0tqwu-8000.proxy.runpod.net/v1",
                "model": model,
                "prompt": prompt,
                "response": content,
                "reasoning": reasoning,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "success": True
            }, f, indent=2)
        
        print(f"Result saved to: {result_file}")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        
        # Save error result
        results_dir = Path("tests/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        error_file = results_dir / "runpod_qwen_test_error.json"
        with open(error_file, 'w') as f:
            json.dump({
                "endpoint": "https://g8xfhnpws0tqwu-8000.proxy.runpod.net/v1",
                "model": model,
                "prompt": prompt,
                "error": str(e),
                "success": False
            }, f, indent=2)
        
        print(f"Error details saved to: {error_file}")

if __name__ == "__main__":
    test_runpod_qwen() 