#!/usr/bin/env python3
"""
Simple test script to verify DashScope API with thinking_budget parameter works.
"""

import os
import sys
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_dashscope_reasoning():
    """Test DashScope API with reasoning/thinking budget"""
    
    # Check if API key is available
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        print("❌ DASHSCOPE_API_KEY environment variable not found")
        print("Please set your DashScope API key:")
        print("export DASHSCOPE_API_KEY='your-api-key-here'")
        return False
    
    print("🔑 Found DASHSCOPE_API_KEY")
    
    # Initialize client
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    
    print("🚀 Testing DashScope API with thinking_budget...")
    
    try:
        # Test with thinking budget (reasoning)
        response = client.chat.completions.create(
            model="qwen3-235b-a22b-thinking-2507",
            messages=[
                {"role": "user", "content": "Solve this simple math problem: What is 15 + 27? Show your reasoning."}
            ],
            temperature=0.7,
            top_p=0.8,
            extra_body={"thinking_budget": 2000},  # 2000 tokens for reasoning
            max_tokens=1000
        )
        
        print("✅ API call successful!")
        print(f"📊 Model: {response.model}")
        print(f"📊 Usage: {response.usage}")
        
        if response.choices and len(response.choices) > 0:
            message = response.choices[0].message
            print(f"📝 Response content: {message.content[:200]}...")
            
            # Check if reasoning content is available
            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                print(f"🧠 Reasoning available: {len(message.reasoning_content)} chars")
                print(f"🧠 Reasoning preview: {message.reasoning_content[:100]}...")
            else:
                print("🧠 No reasoning content found in response")
        
        return True
        
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return False

def test_without_thinking_budget():
    """Test DashScope API without thinking_budget for comparison"""
    
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if not api_key:
        return False
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    
    print("\n🚀 Testing DashScope API without thinking_budget...")
    
    try:
        # Test without thinking budget
        response = client.chat.completions.create(
            model="qwen3-235b-a22b-thinking-2507",
            messages=[
                {"role": "user", "content": "What is 15 + 27?"}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        print("✅ API call successful!")
        print(f"📊 Usage: {response.usage}")
        
        if response.choices and len(response.choices) > 0:
            message = response.choices[0].message
            print(f"📝 Response: {message.content}")
            
            # Check if reasoning content is available
            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                print(f"🧠 Reasoning available: {len(message.reasoning_content)} chars")
            else:
                print("🧠 No reasoning content found")
        
        return True
        
    except Exception as e:
        print(f"❌ API call failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 DashScope API Reasoning Test")
    print("=" * 50)
    
    # Test with thinking budget
    success1 = test_dashscope_reasoning()
    
    # Test without thinking budget for comparison
    success2 = test_without_thinking_budget()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("✅ All tests passed!")
    elif success1 or success2:
        print("⚠️ Some tests passed, some failed")
    else:
        print("❌ All tests failed")
    
    print("🔍 Check the output above for reasoning content differences")