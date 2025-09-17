#!/usr/bin/env python3

import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment from llm_python/.env
load_dotenv("llm_python/.env")

def test_openrouter_reasoning_openai(effort_level):
    """Test OpenRouter API call with reasoning effort levels using OpenAI client"""

    # Try different possible API key environment variables
    api_key = os.getenv('OPENAI_API_KEY') or os.getenv('OPENROUTER_API_KEY') or "EMPTY"
    print(f"Using API key: {api_key[:10]}..." if api_key != "EMPTY" else "No API key found")

    # Initialize OpenAI client with OpenRouter endpoint
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        timeout=600
    )

    print(f"\n=== Testing {effort_level.upper()} reasoning effort with OpenAI client ===")

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": "What is 2+2? Show your reasoning."
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
            temperature=1.0
        )

        print(f"Response object type: {type(response)}")

        if response and hasattr(response, "choices") and len(response.choices) > 0:
            content = response.choices[0].message.content
            print(f"Response: {content}")

            # Check usage
            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                print(f"Usage: {usage}")
        else:
            print("No choices in response")
            print(f"Full response: {response}")

    except Exception as e:
        print(f"Exception: {e}")
        print(f"Exception type: {type(e)}")

if __name__ == "__main__":
    # Test all three effort levels
    for effort in ["low", "medium", "high"]:
        test_openrouter_reasoning_openai(effort)
        print("-" * 50)