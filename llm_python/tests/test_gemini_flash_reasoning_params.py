#!/usr/bin/env python3
"""
Test script to enable reasoning mode with Gemini Flash and inspect reasoning tokens.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

def test_gemini_flash_with_reasoning():
    """Test Gemini Flash with reasoning tokens enabled via max_tokens parameter"""
    
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
    prompt = "Solve this step by step: A farmer has 17 chickens and 23 cows. Each chicken lays 3 eggs per day, and each cow produces 8 liters of milk per day. How many eggs and liters of milk will be produced in a week?"
    
    print(f"Testing {model} with reasoning enabled...")
    print(f"Prompt: {prompt}")
    print("=" * 80)
    
    try:
        print("🧠 Test 1: Using max_tokens for reasoning allocation...")
        response1 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,  # This should allocate tokens for reasoning
            temperature=0.1
        )
        
        print("✅ API call successful!")
        print("-" * 60)
        
        # Check message object
        message1 = response1.choices[0].message
        content1 = message1.content
        reasoning1 = getattr(message1, 'reasoning', None)
        
        print("📝 CONTENT:")
        print(content1)
        print("-" * 60)
        
        if reasoning1:
            print("🎯 REASONING TOKENS DETECTED!")
            print("REASONING:")
            print(reasoning1)
            print("-" * 60)
        else:
            print("🤔 No reasoning field found.")
        
        # Check usage for reasoning tokens
        usage1 = response1.usage
        reasoning_tokens1 = getattr(usage1, 'reasoning_tokens', None)
        thinking_tokens1 = getattr(usage1, 'thinking_tokens', None)
        
        print("💰 TOKEN USAGE (Test 1):")
        print(f"Prompt tokens: {usage1.prompt_tokens}")
        print(f"Completion tokens: {usage1.completion_tokens}")
        print(f"Total tokens: {usage1.total_tokens}")
        if reasoning_tokens1:
            print(f"🎯 REASONING TOKENS: {reasoning_tokens1}")
        if thinking_tokens1:
            print(f"🎯 THINKING TOKENS: {thinking_tokens1}")
        
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    try:
        print("🧠 Test 2: Using extra_body with reasoning.max_tokens...")
        response2 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            extra_body={
                "reasoning": {
                    "max_tokens": 1000
                }
            },
            temperature=0.1
        )
        
        print("✅ API call successful!")
        print("-" * 60)
        
        # Check message object
        message2 = response2.choices[0].message
        content2 = message2.content
        reasoning2 = getattr(message2, 'reasoning', None)
        
        print("📝 CONTENT:")
        print(content2[:200] + "..." if len(content2) > 200 else content2)
        print("-" * 60)
        
        if reasoning2:
            print("🎯 REASONING TOKENS DETECTED!")
            print("REASONING:")
            print(reasoning2[:200] + "..." if len(str(reasoning2)) > 200 else reasoning2)
            print("-" * 60)
        else:
            print("🤔 No reasoning field found.")
        
        # Check usage for reasoning tokens
        usage2 = response2.usage
        reasoning_tokens2 = getattr(usage2, 'reasoning_tokens', None)
        thinking_tokens2 = getattr(usage2, 'thinking_tokens', None)
        
        print("💰 TOKEN USAGE (Test 2):")
        print(f"Prompt tokens: {usage2.prompt_tokens}")
        print(f"Completion tokens: {usage2.completion_tokens}")
        print(f"Total tokens: {usage2.total_tokens}")
        if reasoning_tokens2:
            print(f"🎯 REASONING TOKENS: {reasoning_tokens2}")
        if thinking_tokens2:
            print(f"🎯 THINKING TOKENS: {thinking_tokens2}")
        
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        print("🧠 Test 3: Compare with no special parameters...")
        response3 = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        print("✅ API call successful!")
        print("-" * 60)
        
        # Check usage for comparison
        usage3 = response3.usage
        reasoning_tokens3 = getattr(usage3, 'reasoning_tokens', None)
        thinking_tokens3 = getattr(usage3, 'thinking_tokens', None)
        
        print("💰 TOKEN USAGE (Test 3 - baseline):")
        print(f"Prompt tokens: {usage3.prompt_tokens}")
        print(f"Completion tokens: {usage3.completion_tokens}")
        print(f"Total tokens: {usage3.total_tokens}")
        if reasoning_tokens3:
            print(f"🎯 REASONING TOKENS: {reasoning_tokens3}")
        if thinking_tokens3:
            print(f"🎯 THINKING TOKENS: {thinking_tokens3}")
        
        print("=" * 80)
        
        # Save all responses for detailed inspection
        results_dir = Path("tests/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = {
            "test1_max_tokens_2000": {
                "response": {
                    "model": response1.model,
                    "content": content1,
                    "reasoning": reasoning1,
                    "usage": {
                        "prompt_tokens": usage1.prompt_tokens,
                        "completion_tokens": usage1.completion_tokens,
                        "total_tokens": usage1.total_tokens,
                        "reasoning_tokens": reasoning_tokens1,
                        "thinking_tokens": thinking_tokens1
                    }
                }
            },
            "test2_reasoning_max_tokens": {
                "response": {
                    "model": response2.model,
                    "content": content2,
                    "reasoning": reasoning2,
                    "usage": {
                        "prompt_tokens": usage2.prompt_tokens,
                        "completion_tokens": usage2.completion_tokens,
                        "total_tokens": usage2.total_tokens,
                        "reasoning_tokens": reasoning_tokens2,
                        "thinking_tokens": thinking_tokens2
                    }
                }
            },
            "test3_baseline": {
                "response": {
                    "model": response3.model,
                    "content": response3.choices[0].message.content,
                    "reasoning": getattr(response3.choices[0].message, 'reasoning', None),
                    "usage": {
                        "prompt_tokens": usage3.prompt_tokens,
                        "completion_tokens": usage3.completion_tokens,
                        "total_tokens": usage3.total_tokens,
                        "reasoning_tokens": reasoning_tokens3,
                        "thinking_tokens": thinking_tokens3
                    }
                }
            }
        }
        
        result_file = results_dir / "gemini_flash_reasoning_tests.json"
        with open(result_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"📁 All test results saved to: {result_file}")
        
        # Summary comparison
        print("\n🎯 COMPARISON SUMMARY:")
        print(f"Test 1 (max_tokens=2000):     Total: {usage1.total_tokens}, Reasoning: {reasoning_tokens1}")
        print(f"Test 2 (reasoning.max_tokens): Total: {usage2.total_tokens}, Reasoning: {reasoning_tokens2}")
        print(f"Test 3 (baseline):            Total: {usage3.total_tokens}, Reasoning: {reasoning_tokens3}")
        
        print(f"\nReasoning content detected: Test1={reasoning1 is not None}, Test2={reasoning2 is not None}, Test3={getattr(response3.choices[0].message, 'reasoning', None) is not None}")
        
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gemini_flash_with_reasoning() 