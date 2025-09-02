#!/usr/bin/env python3

import os
import time
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import httpx
import sys
sys.path.append("../") # path to .env file

def test_httpx_vs_openai_client():
    """Compare httpx vs OpenAI client timeout behavior"""
    
    timeout_seconds = 1  # Very short timeout
    
    # Load env and get API key
    env_path = Path(__file__).parent.parent / "llm_python" / ".env"
    load_dotenv(env_path, override=True)
    api_key = os.getenv("OPENAI_API_KEY")
    
    payload = {
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": "Generate a detailed 1000 word essay about AI."}],
        "max_tokens": 2000
    }
    
    # Test 1: httpx with explicit timeout configuration
    print(f"🔍 Test 1: httpx with {timeout_seconds}s total timeout")
    start_time = time.time()
    try:
        # Use explicit timeout configuration - total timeout for entire request
        timeout_config = httpx.Timeout(timeout_seconds)  # Total timeout
        with httpx.Client(timeout=timeout_config) as client:
            response = client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            )
            elapsed = time.time() - start_time
            print(f"✅ httpx completed in {elapsed:.2f}s")
            print(f"Status: {response.status_code}")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"⏱️ httpx failed after {elapsed:.2f}s: {type(e).__name__}")
        if "timeout" in str(e).lower():
            print("   👆 Proper timeout detected!")
    
    # Test 2: OpenAI client
    print(f"\n🔍 Test 2: OpenAI client with {timeout_seconds}s timeout")
    start_time = time.time()
    try:
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1", timeout=timeout_seconds)
        response = client.chat.completions.create(**payload)
        elapsed = time.time() - start_time
        print(f"✅ OpenAI client completed in {elapsed:.2f}s")
        print(f"Tokens: {response.usage.completion_tokens if response.usage else 'Unknown'}")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"⏱️ OpenAI client failed after {elapsed:.2f}s: {type(e).__name__}")

if __name__ == "__main__":
    print("🚀 OpenAI Client Timeout Test")
    print("=" * 60)
    
    # Load .env from llm_python directory
    env_path = Path(__file__).parent.parent / "llm_python" / ".env"
    print(f"📁 Loading environment from: {env_path}")
    load_dotenv(env_path, override=True)
    
    # Check if API key is set
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found in environment")
        print("Please set it in .env file or environment")
        exit(1)
    
    print(f"✅ API Key found")
    print(f"🌐 Using OpenRouter endpoint")
    print(f"🤖 Model: openai/gpt-oss-20b")
    print()
    
    # Run test
    test_httpx_vs_openai_client()
    
    print("\n✨ Test completed!")