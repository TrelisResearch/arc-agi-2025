#!/usr/bin/env python3
"""
Simple test to check if qwen/qwen3-32b uses thinking tokens with reasoning parameter.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

def test_qwen_thinking():
    """Make a simple call to Qwen with reasoning enabled and log the response"""
    
    # Load .env file from o3-tools directory
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
    
    # Initialize OpenRouter client
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    model = "qwen/qwen3-32b"
    prompt = "Explain quantum entanglement in one paragraph."
    
    print(f"Testing {model} with reasoning enabled...")
    print(f"Prompt: {prompt}")
    print("-" * 60)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=1,
            # extra_body={
            #     "reasoning": {
            #         "enabled": True,        # turn thinking on   (default)
            #         # "exclude": False        # set to true to hide the <think> block
            #         # // OR pick one of these instead of `enabled`:
            #         # // "effort": "high" | "medium" | "low"
            #         # // "max_tokens": 2000
            #     },
            #     "chat_template_kwargs": {
            #         "enable_thinking": False   # ← disables thinking entirely
            #     }
            # }
        )
        
        content = response.choices[0].message.content
        reasoning = response.choices[0].message.reasoning

        print("RESPONSE:")
        print(content)
        print("\n" + "-" * 60)
        print(f"Token usage: {response.usage.prompt_tokens} input, {response.usage.completion_tokens} output")
        
        # Check for reasoning field
        if reasoning:
            print("\n🎯 THINKING TOKENS DETECTED! Model provided reasoning:")
            print("-" * 60)
            print("REASONING:")
            print(reasoning)
            print("-" * 60)
        else:
            print("\n🤔 No reasoning field found in response.")
        
        # Also check if response contains thinking blocks in content
        if "<think>" in content and "</think>" in content:
            print("\n🎯 Additional thinking blocks found in content!")
        
        
        # Save raw response
        results_dir = Path("tests/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        result_file = results_dir / "qwen_thinking_test_simple.json"
        with open(result_file, 'w') as f:
            json.dump({
                "model": model,
                "prompt": prompt,
                "response": content,
                "reasoning": reasoning,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "has_reasoning_field": bool(reasoning),
                "has_thinking_blocks_in_content": "<think>" in content and "</think>" in content
            }, f, indent=2)
        
        print(f"Result saved to: {result_file}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_qwen_thinking() 