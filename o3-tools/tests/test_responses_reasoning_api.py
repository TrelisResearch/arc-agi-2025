#!/usr/bin/env python3
"""
Test script to verify OpenAI responses API with encrypted reasoning traces for multi-turn conversations.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from dotenv import load_dotenv

def test_basic_responses_with_reasoning():
    """Test basic responses API with encrypted reasoning traces"""
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    print("Testing responses API with encrypted reasoning...")
    
    try:
        # Turn 1 - Get initial response with reasoning trace 
        resp = client.responses.create(
            model="o4-mini",
            input="Plan a 3-day trip to Kyoto",
            include=["reasoning.encrypted_content"],
            store=False  # Enable stateless mode for encrypted content
        )
        
        print(f"✅ Turn 1 successful")
        print(f"Response ID: {resp.id}")
        
        # Extract the reasoning item AND its paired assistant message
        reasoning_item = None
        assistant_msg = None
        for item in resp.output:
            if item.type == "reasoning":
                reasoning_item = item
            elif item.type == "message" and item.role == "assistant":
                assistant_msg = item
        
        if reasoning_item and reasoning_item.encrypted_content:
            print(f"✅ Found encrypted reasoning trace (encrypted, length: {len(reasoning_item.encrypted_content)})")
        else:
            print(f"❌ No encrypted reasoning trace found")
            return None, None, None
            
        if not assistant_msg:
            print(f"❌ No assistant message found to pair with reasoning")
            return None, None, None
        
        # Extract the main content
        main_content = ""
        for item in resp.output:
            if item.type == "message":
                for content_item in item.content:
                    if content_item.type == "output_text":
                        main_content += content_item.text
        
        print(f"Content preview: {main_content[:100]}...")
        print(f"Tokens used: {resp.usage.input_tokens} input, {resp.usage.output_tokens} output")
        
        return resp.id, reasoning_item, assistant_msg, main_content
        
    except Exception as e:
        print(f"❌ Turn 1 failed: {e}")
        return None, None, None, None

def test_multi_turn_with_reasoning_trace(first_response_id, reasoning_item, assistant_msg, first_content):
    """Test multi-turn conversation using encrypted reasoning trace"""
    if not reasoning_item or not assistant_msg:
        print("❌ Cannot test multi-turn without reasoning item and assistant message pair")
        return None, None, None
        
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    print("\nTesting multi-turn with reasoning trace...")
    
    try:
        # Turn 2 - Pass the reasoning item + assistant message pair, then new user turn
        resp2 = client.responses.create(
            model="o4-mini",
            input=[
                reasoning_item,          # ① reasoning item
                assistant_msg,           # ② its "required following item" (assistant message)
                {"role": "user", "content": "Add a day trip to Nara"}  # ③ new user turn
            ],
            include=["reasoning.encrypted_content"],
            store=False  # Enable stateless mode for encrypted content
        )
        
        print(f"✅ Turn 2 successful")
        print(f"Response ID: {resp2.id}")
        
        # Extract the new encrypted reasoning trace (we won't print it, just reference it)
        new_encrypted_trace = None
        for item in resp2.output:
            if item.type == "reasoning":
                new_encrypted_trace = item.encrypted_content
                break
        
        if new_encrypted_trace:
            print(f"✅ Found new encrypted reasoning trace (encrypted, length: {len(new_encrypted_trace)})")
        else:
            print(f"❌ No new encrypted reasoning trace found")
        
        # Extract the main content
        main_content = ""
        for item in resp2.output:
            if item.type == "message":
                for content_item in item.content:
                    if content_item.type == "output_text":
                        main_content += content_item.text
        
        print(f"Content preview: {main_content[:100]}...")
        print(f"Tokens used: {resp2.usage.input_tokens} input, {resp2.usage.output_tokens} output")
        
        return resp2.id, new_encrypted_trace, main_content
        
    except Exception as e:
        print(f"❌ Turn 2 failed: {e}")
        return None, None, None

def test_code_generation_with_reasoning():
    """Test code generation with reasoning traces"""
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    print("\nTesting code generation with reasoning...")
    
    try:
        # Turn 1 - Ask for code with reasoning
        resp = client.responses.create(
            model="o4-mini",
            input="Write a Python function called 'transform' that takes a 2D list (grid) and returns it flipped vertically. Include the phrase 'Final Answer:' before your code.",
            include=["reasoning.encrypted_content"],
            store=False  # Enable stateless mode for encrypted content
        )
        
        print(f"✅ Code generation successful")
        print(f"Response ID: {resp.id}")
        
        # Extract content
        main_content = ""
        for item in resp.output:
            if item.type == "message":
                for content_item in item.content:
                    if content_item.type == "output_text":
                        main_content += content_item.text
        
        print(f"Full response:\n{main_content}")
        
        # Check for expected elements
        if "def transform" in main_content:
            print(f"✅ Found transform function definition")
        else:
            print(f"❌ No transform function found")
            
        if "Final Answer:" in main_content:
            print(f"✅ Found Final Answer marker")
        else:
            print(f"❌ No Final Answer marker found")
        
        # Extract encrypted reasoning trace (we won't print it, just reference it)
        encrypted_trace = None
        for item in resp.output:
            if item.type == "reasoning":
                encrypted_trace = item.encrypted_content
                break
        
        if encrypted_trace:
            print(f"✅ Found encrypted reasoning trace for code generation (encrypted, length: {len(encrypted_trace)})")
        else:
            print(f"❌ No encrypted reasoning trace found")
        
        print(f"Tokens used: {resp.usage.input_tokens} input, {resp.usage.output_tokens} output")
        
        return resp.id, encrypted_trace, main_content
        
    except Exception as e:
        print(f"❌ Code generation failed: {e}")
        return None, None, None

if __name__ == "__main__":
    print("OpenAI Responses API with Encrypted Reasoning Test")
    print("=" * 60)
    
    # Test basic functionality with reasoning
    first_id, first_reasoning_item, first_assistant_msg, first_content = test_basic_responses_with_reasoning()
    
    # Test multi-turn conversation with reasoning trace
    second_id, second_trace, second_content = test_multi_turn_with_reasoning_trace(first_id, first_reasoning_item, first_assistant_msg, first_content)
    
    # Test code generation with reasoning
    third_id, third_trace, third_content = test_code_generation_with_reasoning()
    
    print("\n" + "=" * 60)
    print("All tests completed!") 