#!/usr/bin/env python3
"""
Test script to inspect Gemini Flash API responses for reasoning tokens or similar fields.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

def test_gemini_flash_response_inspection():
    """Make a simple call to Gemini Flash and inspect the full response structure"""
    
    # Load .env file
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
    
    # Initialize OpenRouter client
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    model = "google/gemini-2.5-flash"
    prompt = "Solve this simple math problem step by step: What is 23 + 47?"
    
    print(f"Testing {model} response inspection...")
    print(f"Prompt: {prompt}")
    print("-" * 60)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.1
        )
        
        print("✅ API call successful!")
        print("-" * 60)
        
        # Inspect the full response object
        print("🔍 FULL RESPONSE OBJECT INSPECTION:")
        print(f"Type: {type(response)}")
        print(f"Available attributes: {dir(response)}")
        print("-" * 60)
        
        # Check message object
        message = response.choices[0].message
        print("🔍 MESSAGE OBJECT INSPECTION:")
        print(f"Type: {type(message)}")
        print(f"Available attributes: {dir(message)}")
        print("-" * 60)
        
        # Extract basic fields
        content = message.content
        print("📝 CONTENT:")
        print(content)
        print("-" * 60)
        
        # Check for reasoning field (like Qwen has)
        reasoning = getattr(message, 'reasoning', None)
        if reasoning:
            print("🎯 REASONING FIELD DETECTED!")
            print("REASONING:")
            print(reasoning)
            print("-" * 60)
        else:
            print("🤔 No 'reasoning' field found in message object.")
        
        # Check for other potentially interesting fields
        interesting_fields = ['thinking', 'thoughts', 'reasoning_content', 'internal_thoughts', 
                            'chain_of_thought', 'step_by_step', 'reasoning_trace']
        
        found_fields = {}
        for field in interesting_fields:
            value = getattr(message, field, None)
            if value is not None:
                found_fields[field] = value
                print(f"🎯 FOUND FIELD '{field}': {value}")
        
        # Check usage for any special token counts
        print("💰 TOKEN USAGE:")
        usage = response.usage
        print(f"Usage type: {type(usage)}")
        print(f"Usage attributes: {dir(usage)}")
        print(f"Prompt tokens: {usage.prompt_tokens}")
        print(f"Completion tokens: {usage.completion_tokens}")
        print(f"Total tokens: {usage.total_tokens}")
        
        # Check for reasoning-related token counts
        reasoning_tokens = getattr(usage, 'reasoning_tokens', None)
        thinking_tokens = getattr(usage, 'thinking_tokens', None)
        if reasoning_tokens:
            print(f"🎯 REASONING TOKENS: {reasoning_tokens}")
        if thinking_tokens:
            print(f"🎯 THINKING TOKENS: {thinking_tokens}")
        
        print("-" * 60)
        
        # Save raw response for detailed inspection
        results_dir = Path("tests/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert response to dict for JSON serialization
        response_dict = {
            "model": response.model,
            "id": response.id,
            "object": response.object,
            "created": response.created,
            "choices": [
                {
                    "index": choice.index,
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content,
                        "reasoning": getattr(choice.message, 'reasoning', None),
                        # Add any other interesting fields we found
                        **{field: getattr(choice.message, field, None) 
                           for field in interesting_fields 
                           if getattr(choice.message, field, None) is not None}
                    },
                    "finish_reason": choice.finish_reason
                }
                for choice in response.choices
            ],
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
                "reasoning_tokens": getattr(usage, 'reasoning_tokens', None),
                "thinking_tokens": getattr(usage, 'thinking_tokens', None),
            }
        }
        
        result_file = results_dir / "gemini_flash_response_inspection.json"
        with open(result_file, 'w') as f:
            json.dump(response_dict, f, indent=2)
        
        print(f"📁 Full response saved to: {result_file}")
        
        # Summary
        print("\n🎯 SUMMARY:")
        print(f"Content length: {len(content)} characters")
        print(f"Has reasoning field: {reasoning is not None}")
        print(f"Additional fields found: {list(found_fields.keys())}")
        print(f"Reasoning tokens in usage: {getattr(usage, 'reasoning_tokens', 'Not found')}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gemini_flash_response_inspection() 