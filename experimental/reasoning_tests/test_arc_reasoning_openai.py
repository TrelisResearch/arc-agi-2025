#!/usr/bin/env python3

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment from llm_python/.env
load_dotenv("llm_python/.env")

def load_arc_task():
    """Load the ARC task from arc_task.txt"""
    with open("experimental/reasoning_tests/arc_task.txt", "r") as f:
        return f.read()

def test_arc_reasoning_openai(effort_level):
    """Test OpenRouter API call with ARC reasoning task using OpenAI client"""

    # Try different possible API key environment variables
    api_key = os.getenv('OPENAI_API_KEY') or os.getenv('OPENROUTER_API_KEY') or "EMPTY"

    # Initialize OpenAI client with OpenRouter endpoint
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=600
    )

    # Load ARC task
    arc_task_content = load_arc_task()

    print(f"\n=== Testing {effort_level.upper()} reasoning effort with ARC task (OpenAI client) ===")
    print(f"Using API key: {api_key[:10]}..." if api_key != "EMPTY" else "No API key found")
    print(f"ARC task length: {len(arc_task_content)} characters")

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": arc_task_content
                }
            ],
            extra_body={
                "reasoning": {
                    "effort": effort_level,
                    "exclude": False,
                    "enabled": True
                }
            },
            max_tokens=100000,
            temperature=0.7,
            top_p=0.9
        )

        print(f"Response object type: {type(response)}")

        if response and hasattr(response, "choices") and len(response.choices) > 0:
            content = response.choices[0].message.content
            print(f"Response length: {len(content)} characters")

            # Show first and last part of response
            if len(content) > 500:
                print(f"Response (first 250 chars): {content[:250]}...")
                print(f"Response (last 250 chars): ...{content[-250:]}")
            else:
                print(f"Response: {content}")

            # Check usage
            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                print(f"Usage: {usage}")

            # Check if there's Python code
            if "```python" in content:
                print("✅ Contains Python code block")
            else:
                print("❌ No Python code block found")

        else:
            print("No choices in response")
            print(f"Full response: {response}")

    except Exception as e:
        print(f"Exception: {e}")
        print(f"Exception type: {type(e)}")

if __name__ == "__main__":
    # Test all three effort levels with ARC task
    for effort in ["low", "medium", "high"]:
        test_arc_reasoning_openai(effort)
        print("-" * 80)