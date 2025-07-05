#!/usr/bin/env python3
"""
Test script to verify OpenAI chat completions API functionality and multi-turn conversations.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from dotenv import load_dotenv

def test_basic_chat_completions_api():
    """Test basic chat completions API functionality"""
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    print("Testing basic chat completions API...")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Write a simple Python function that adds two numbers."}
            ],
            max_tokens=200
        )
        
        print(f"✅ Basic API call successful")
        print(f"Response ID: {response.id}")
        print(f"Content: {response.choices[0].message.content[:100]}...")
        print(f"Tokens used: {response.usage.prompt_tokens} input, {response.usage.completion_tokens} output")
        
        return response.id, response.choices[0].message.content
        
    except Exception as e:
        print(f"❌ Basic API call failed: {e}")
        return None, None

def test_multi_turn_conversation(first_content):
    """Test multi-turn conversation by maintaining message history"""
    if not first_content:
        print("❌ Cannot test multi-turn without valid first response content")
        return
        
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    print("\nTesting multi-turn conversation...")
    
    try:
        # Build conversation history
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a simple Python function that adds two numbers."},
            {"role": "assistant", "content": first_content},
            {"role": "user", "content": "Now modify that function to also multiply the numbers."}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=200
        )
        
        print(f"✅ Multi-turn conversation successful")
        print(f"Response ID: {response.id}")
        print(f"Content: {response.choices[0].message.content[:100]}...")
        print(f"Tokens used: {response.usage.prompt_tokens} input, {response.usage.completion_tokens} output")
        
        return response.id, response.choices[0].message.content
        
    except Exception as e:
        print(f"❌ Multi-turn conversation failed: {e}")
        return None, None

def test_code_extraction():
    """Test extracting code from a response"""
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    print("\nTesting code extraction...")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Write a Python function called 'transform' that takes a 2D list (grid) and returns it flipped vertically. Include the phrase 'Final Answer:' before your code."}
            ],
            max_tokens=300
        )
        
        content = response.choices[0].message.content
        print(f"✅ Code extraction test successful")
        print(f"Full response:\n{content}")
        
        # Simple code extraction (we'll improve this in the main implementation)
        if "def transform" in content:
            print(f"✅ Found transform function definition")
        else:
            print(f"❌ No transform function found")
            
        if "Final Answer:" in content:
            print(f"✅ Found Final Answer marker")
        else:
            print(f"❌ No Final Answer marker found")
        
        return response.id, content
        
    except Exception as e:
        print(f"❌ Code extraction test failed: {e}")
        return None, None

if __name__ == "__main__":
    print("OpenAI Chat Completions API Test")
    print("=" * 50)
    
    # Test basic functionality
    first_id, first_content = test_basic_chat_completions_api()
    
    # Test multi-turn conversation
    second_id, second_content = test_multi_turn_conversation(first_content)
    
    # Test code extraction
    third_id, third_content = test_code_extraction()
    
    print("\n" + "=" * 50)
    print("All tests completed!") 