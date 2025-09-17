#!/usr/bin/env python3

import os
import requests
import json
from dotenv import load_dotenv

# Load environment from llm_python/.env
load_dotenv("llm_python/.env")

def test_openrouter_reasoning(effort_level):
    """Test OpenRouter API call with reasoning effort levels"""

    url = "https://openrouter.ai/api/v1/chat/completions"

    # Try different possible API key environment variables
    api_key = os.getenv('OPENAI_API_KEY') or os.getenv('OPENROUTER_API_KEY') or "EMPTY"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    print(f"Using API key: {api_key[:10]}..." if api_key != "EMPTY" else "No API key found")

    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "What is 2+2? Show your reasoning."
            }
        ],
        "reasoning": {
            "effort": effort_level,
            "exclude": False,
            "enabled": True
        },
        "max_tokens": 100000,
        "temperature": 1.0
    }

    print(f"\n=== Testing {effort_level.upper()} reasoning effort ===")
    print(f"Request payload: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=600)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                print(f"Response: {content}")

                # Check if reasoning tokens are in usage
                if "usage" in result:
                    usage = result["usage"]
                    print(f"Usage: {usage}")
            else:
                print("No choices in response")
                print(f"Full response: {result}")
        else:
            print(f"Error: {response.text}")

    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    # Test all three effort levels
    for effort in ["low", "medium", "high"]:
        test_openrouter_reasoning(effort)
        print("-" * 50)