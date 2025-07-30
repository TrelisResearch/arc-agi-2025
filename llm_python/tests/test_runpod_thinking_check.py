#!/usr/bin/env python3
"""
Test to check if thinking tokens are disabled on RunPod Qwen endpoint.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

def test_runpod_thinking():
    """Test if thinking tokens come back from the RunPod endpoint"""
    
    # Load .env file from o3-tools directory
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
    
    # Initialize client for RunPod endpoint
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = "http://157.66.254.42:10957/v1"
    
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    model = "Qwen/Qwen3-4B"  # Adjust model name if needed
    prompt = "Explain quantum entanglement in simple terms."
    
    print(f"Testing RunPod endpoint: {base_url}")
    print(f"Model: {model}")
    print(f"Prompt: {prompt}")
    print(f"Using: extra_body={{\"top_k\": 20}} (thinking enabled by default)")
    print("-" * 60)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.6,
            top_p=0.95,
            extra_body={"top_k": 20}
        )
        
        message = response.choices[0].message
        content = message.content
        
        print("RESPONSE:")
        print(f"Content: {content}")
        print(f"Content is None: {content is None}")
        print("\n" + "-" * 60)
        print(f"Token usage: {response.usage.prompt_tokens} input, {response.usage.completion_tokens} output")
        
        # Check for various thinking-related fields
        thinking_indicators = []
        
        # Check reasoning field (OpenRouter style)
        if hasattr(message, 'reasoning') and message.reasoning:
            thinking_indicators.append(f"✅ 'reasoning' field found ({len(message.reasoning)} chars)")
            print(f"\n🎯 REASONING FIELD DETECTED:")
            print(message.reasoning[:200] + "..." if len(message.reasoning) > 200 else message.reasoning)
        else:
            thinking_indicators.append("❌ No 'reasoning' field")
        
        # Check reasoning_content field (RunPod style)
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            thinking_indicators.append(f"✅ 'reasoning_content' field found ({len(message.reasoning_content)} chars)")
            print(f"\n🎯 REASONING_CONTENT FIELD DETECTED:")
            print(message.reasoning_content[:200] + "..." if len(message.reasoning_content) > 200 else message.reasoning_content)
        else:
            thinking_indicators.append("❌ No 'reasoning_content' field")
        
        # Check for thinking blocks in content (handle None content)
        if content and "<think>" in content and "</think>" in content:
            thinking_indicators.append("✅ <think> blocks found in content")
            print(f"\n🎯 THINKING BLOCKS DETECTED IN CONTENT!")
        else:
            thinking_indicators.append("❌ No <think> blocks in content")
        
        print(f"\n{'='*60}")
        print("THINKING TOKEN ANALYSIS:")
        print(f"{'='*60}")
        for indicator in thinking_indicators:
            print(f"  {indicator}")
        
        # Determine if thinking is disabled
        has_any_thinking = any("✅" in indicator for indicator in thinking_indicators)
        if has_any_thinking:
            print(f"\n🚨 THINKING TOKENS ARE STILL ENABLED!")
            print("   The server is still returning thinking content.")
        else:
            print(f"\n✅ THINKING TOKENS APPEAR TO BE DISABLED!")
            print("   No thinking content detected in response.")
        
        # Save raw response for debugging
        results_dir = Path("tests/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        result_file = results_dir / "runpod_thinking_check.json"
        
        # Serialize all message attributes safely
        message_dict = {
            "content": content,
            "role": getattr(message, 'role', 'assistant')
        }
        
        # Add any reasoning fields if they exist
        if hasattr(message, 'reasoning'):
            message_dict["reasoning"] = message.reasoning
        if hasattr(message, 'reasoning_content'):
            message_dict["reasoning_content"] = message.reasoning_content
        
        result_data = {
            "endpoint": base_url,
            "model": model,
            "prompt": prompt,
            "extra_body_used": {"top_k": 20},
            "message": message_dict,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "thinking_analysis": {
                "has_reasoning_field": hasattr(message, 'reasoning') and bool(message.reasoning),
                "has_reasoning_content_field": hasattr(message, 'reasoning_content') and bool(message.reasoning_content),
                "has_thinking_blocks_in_content": content and "<think>" in content and "</think>" in content,
                "thinking_disabled": not has_any_thinking
            }
        }
        
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"\nResult saved to: {result_file}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        
        # Still save error info
        results_dir = Path("tests/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        error_file = results_dir / "runpod_thinking_check_error.json"
        with open(error_file, 'w') as f:
            json.dump({
                "endpoint": base_url,
                "model": model,
                "extra_body_used": {"top_k": 20},
                "error": str(e),
                "error_type": type(e).__name__
            }, f, indent=2)
        
        print(f"Error details saved to: {error_file}")

def test_runpod_thinking_disabled():
    """Test RunPod endpoint with explicit chat_template_kwargs to disable thinking"""
    
    # Load .env file from o3-tools directory
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
    
    # Initialize client for RunPod endpoint
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = "http://157.66.254.42:10957/v1"
    
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    model = "Qwen/Qwen3-4B"
    prompt = "Explain quantum entanglement in simple terms."
    
    print(f"\n{'='*70}")
    print("TESTING WITH THINKING EXPLICITLY DISABLED")
    print(f"{'='*70}")
    print(f"Testing RunPod endpoint: {base_url}")
    print(f"Model: {model}")
    print(f"Prompt: {prompt}")
    print(f"Using: extra_body={{\"top_k\": 20, \"chat_template_kwargs\": {{\"enable_thinking\": False}}}}")
    print("-" * 60)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7,
            top_p=0.8,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False}
            }
        )
        
        message = response.choices[0].message
        content = message.content
        
        print("RESPONSE:")
        print(f"Content: {content}")
        print(f"Content is None: {content is None}")
        print("\n" + "-" * 60)
        print(f"Token usage: {response.usage.prompt_tokens} input, {response.usage.completion_tokens} output")
        
        # Check for various thinking-related fields
        thinking_indicators = []
        
        # Check reasoning field (OpenRouter style)
        if hasattr(message, 'reasoning') and message.reasoning:
            thinking_indicators.append(f"✅ 'reasoning' field found ({len(message.reasoning)} chars)")
            print(f"\n🎯 REASONING FIELD DETECTED:")
            print(message.reasoning[:200] + "..." if len(message.reasoning) > 200 else message.reasoning)
        else:
            thinking_indicators.append("❌ No 'reasoning' field")
        
        # Check reasoning_content field (RunPod style)
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            thinking_indicators.append(f"✅ 'reasoning_content' field found ({len(message.reasoning_content)} chars)")
            print(f"\n🎯 REASONING_CONTENT FIELD DETECTED:")
            print(message.reasoning_content[:200] + "..." if len(message.reasoning_content) > 200 else message.reasoning_content)
        else:
            thinking_indicators.append("❌ No 'reasoning_content' field")
        
        # Check for thinking blocks in content (handle None content)
        if content and "<think>" in content and "</think>" in content:
            thinking_indicators.append("✅ <think> blocks found in content")
            print(f"\n🎯 THINKING BLOCKS DETECTED IN CONTENT!")
        else:
            thinking_indicators.append("❌ No <think> blocks in content")
        
        print(f"\n{'='*60}")
        print("THINKING TOKEN ANALYSIS (WITH DISABLE FLAG):")
        print(f"{'='*60}")
        for indicator in thinking_indicators:
            print(f"  {indicator}")
        
        # Determine if thinking is disabled
        has_any_thinking = any("✅" in indicator for indicator in thinking_indicators)
        if has_any_thinking:
            print(f"\n🚨 THINKING TOKENS ARE STILL ENABLED!")
            print("   The disable flag didn't work - server still returning thinking content.")
        else:
            print(f"\n✅ THINKING TOKENS SUCCESSFULLY DISABLED!")
            print("   The chat_template_kwargs worked - no thinking content detected.")
        
        # Save result
        results_dir = Path("tests/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        result_file = results_dir / "runpod_thinking_disabled_check.json"
        
        # Serialize all message attributes safely
        message_dict = {
            "content": content,
            "role": getattr(message, 'role', 'assistant')
        }
        
        # Add any reasoning fields if they exist
        if hasattr(message, 'reasoning'):
            message_dict["reasoning"] = message.reasoning
        if hasattr(message, 'reasoning_content'):
            message_dict["reasoning_content"] = message.reasoning_content
        
        result_data = {
            "endpoint": base_url,
            "model": model,
            "prompt": prompt,
            "extra_body_used": {
                "top_k": 20,
                "chat_template_kwargs": {"enable_thinking": False}
            },
            "message": message_dict,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "thinking_analysis": {
                "has_reasoning_field": hasattr(message, 'reasoning') and bool(message.reasoning),
                "has_reasoning_content_field": hasattr(message, 'reasoning_content') and bool(message.reasoning_content),
                "has_thinking_blocks_in_content": content and "<think>" in content and "</think>" in content,
                "thinking_disabled": not has_any_thinking
            }
        }
        
        with open(result_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"\nResult saved to: {result_file}")
        
    except Exception as e:
        print(f"❌ Test with disable flag failed: {e}")
        
        # Still save error info
        results_dir = Path("tests/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        error_file = results_dir / "runpod_thinking_disabled_error.json"
        with open(error_file, 'w') as f:
            json.dump({
                "endpoint": base_url,
                "model": model,
                "extra_body_used": {
                    "top_k": 20,
                    "chat_template_kwargs": {"enable_thinking": False}
                },
                "error": str(e),
                "error_type": type(e).__name__
            }, f, indent=2)
        
        print(f"Error details saved to: {error_file}")

if __name__ == "__main__":
    # Test 1: Default behavior (should have thinking)
    test_runpod_thinking()
    
    # Test 2: With thinking explicitly disabled
    test_runpod_thinking_disabled() 