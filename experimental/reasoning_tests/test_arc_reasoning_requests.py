#!/usr/bin/env python3

import os
import requests
import json
from dotenv import load_dotenv

# Load environment from llm_python/.env
load_dotenv("llm_python/.env")

def load_arc_task():
    """Load the ARC task from arc_task.txt"""
    with open("experimental/reasoning_tests/arc_task.txt", "r") as f:
        return f.read()

def test_arc_reasoning_requests(effort_level):
    """Test OpenRouter API call with ARC reasoning task using requests"""

    url = "https://openrouter.ai/api/v1/chat/completions"

    # Try different possible API key environment variables
    api_key = os.getenv('OPENAI_API_KEY') or os.getenv('OPENROUTER_API_KEY') or "EMPTY"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # Load ARC task
    arc_task_content = load_arc_task()

    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {
                "role": "user",
                "content": arc_task_content
            }
        ],
        "reasoning": {
            "effort": effort_level,
            "exclude": False,
            "enabled": True
        },
        "max_tokens": 100000,
        "temperature": 0.7,
        "top_p": 0.9
    }

    print(f"\n=== Testing {effort_level.upper()} reasoning effort with ARC task (requests) ===")
    print(f"Using API key: {api_key[:10]}..." if api_key != "EMPTY" else "No API key found")
    print(f"ARC task length: {len(arc_task_content)} characters")

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=600)
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            # Try to parse JSON, but show raw response if it fails
            try:
                result = response.json()
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")
                print(f"Raw response text (first 1000 chars): {response.text[:1000]}")
                print(f"Raw response text (last 500 chars): {response.text[-500:]}")
                return

            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                print(f"Response length: {len(content)} characters")

                # Show first and last part of response
                if len(content) > 500:
                    print(f"Response (first 250 chars): {content[:250]}...")
                    print(f"Response (last 250 chars): ...{content[-250:]}")
                else:
                    print(f"Response: {content}")

                # Check if there's Python code
                if "```python" in content:
                    print("✅ Contains Python code block")
                else:
                    print("❌ No Python code block found")

                # Check usage
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
    # Test all three effort levels with ARC task
    for effort in ["low", "medium", "high"]:
        test_arc_reasoning_requests(effort)
        print("-" * 80)