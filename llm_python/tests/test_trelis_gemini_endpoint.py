#!/usr/bin/env python3
"""
Simple test to debug the Trelis Gemini endpoint and see what's coming back
"""

import openai
import time
import json
import os
from datetime import datetime

def test_trelis_gemini_endpoint():
    """Make a simple API call to debug response format"""
    
    print("=== Trelis Gemini Endpoint Test ===")
    print(f"Start time: {datetime.now()}")
    
    # Setup client with user's endpoint
    base_url = "http://63.141.33.85:22032/v1"
    api_key = os.getenv('OPENAI_API_KEY', 'dummy')
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    print(f"Endpoint: {base_url}")
    print(f"Model: Trelis/gemini-2.5-reasoning-smol-21-jul")
    print(f"API Key: {'***' + api_key[-4:] if len(api_key) > 4 else 'dummy'}")
    
    # Simple request
    messages = [
        {"role": "user", "content": "Hello! Please respond with a simple greeting and explain that you are working correctly."}
    ]
    
    print(f"\nMaking API call at: {datetime.now()}")
    print(f"Request payload:")
    print(f"  Messages: {messages}")
    print(f"  Max tokens: 2000")
    
    try:
        response = client.chat.completions.create(
            model="Trelis/gemini-2.5-reasoning-smol-21-jul",
            messages=messages,
            max_tokens=2000,
            temperature=0.1
        )
        
        print(f"\n✅ Response received at: {datetime.now()}")
        print("=" * 60)
        print("FULL RESPONSE DETAILS:")
        print("=" * 60)
        
        # Print basic response info
        print(f"Response ID: {response.id}")
        print(f"Model: {response.model}")
        print(f"Created: {response.created}")
        print(f"Object: {response.object}")
        
        # Print usage info
        if response.usage:
            print(f"\nUSAGE:")
            print(f"  Prompt tokens: {response.usage.prompt_tokens}")
            print(f"  Completion tokens: {response.usage.completion_tokens}")
            print(f"  Total tokens: {response.usage.total_tokens}")
        
        # Print choices
        print(f"\nCHOICES ({len(response.choices)} total):")
        for i, choice in enumerate(response.choices):
            print(f"  Choice {i}:")
            print(f"    Index: {choice.index}")
            print(f"    Finish reason: {choice.finish_reason}")
            
            # Print message content
            message = choice.message
            print(f"    Message role: {message.role}")
            print(f"    Message content length: {len(message.content) if message.content else 0}")
            print(f"    Message content:")
            print(f"      {repr(message.content)}")
            
            # Check for additional fields
            print(f"\n    CHECKING FOR ADDITIONAL MESSAGE FIELDS:")
            for attr in dir(message):
                if not attr.startswith('_') and attr not in ['role', 'content']:
                    try:
                        value = getattr(message, attr)
                        if not callable(value):
                            if value is not None:
                                print(f"      {attr}: {type(value)} = {repr(value)[:200]}{'...' if len(repr(value)) > 200 else ''}")
                            else:
                                print(f"      {attr}: None")
                    except Exception as e:
                        print(f"      {attr}: Error accessing - {e}")
        
        # Try to serialize the entire response to see all fields
        print(f"\n" + "=" * 60)
        print("ATTEMPTING TO SERIALIZE FULL RESPONSE:")
        print("=" * 60)
        try:
            # Convert to dict to see all fields
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else str(response)
            print(json.dumps(response_dict, indent=2, default=str))
        except Exception as e:
            print(f"Could not serialize response: {e}")
            print(f"Response type: {type(response)}")
            print(f"Response str: {str(response)[:1000]}{'...' if len(str(response)) > 1000 else ''}")
        
        # Check raw content
        print(f"\n" + "=" * 60)
        print("RAW CONTENT ANALYSIS:")
        print("=" * 60)
        content = response.choices[0].message.content
        if content:
            print(f"Content type: {type(content)}")
            print(f"Content length: {len(content)}")
            print(f"First 500 chars: {repr(content[:500])}")
            print(f"Last 100 chars: {repr(content[-100:])}")
            
            # Check for unusual characters
            non_printable = [c for c in content if ord(c) < 32 or ord(c) > 126]
            if non_printable:
                print(f"Non-printable characters found: {len(non_printable)}")
                print(f"First few: {[ord(c) for c in non_printable[:10]]}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        
        # Try to get more details about the error
        if hasattr(e, 'response'):
            print(f"Error response: {e.response}")
        if hasattr(e, 'status_code'):
            print(f"Status code: {e.status_code}")
        if hasattr(e, 'body'):
            print(f"Error body: {e.body}")
            
        return False

def test_simple_math():
    """Test with a simple math problem to see if reasoning works"""
    
    print("\n" + "=" * 60)
    print("TESTING WITH SIMPLE MATH PROBLEM:")
    print("=" * 60)
    
    base_url = "http://63.141.33.85:22032/v1"
    api_key = os.getenv('OPENAI_API_KEY', 'dummy')
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    messages = [
        {"role": "user", "content": "What is 15 + 27? Please show your reasoning."}
    ]
    
    try:
        response = client.chat.completions.create(
            model="Trelis/gemini-2.5-reasoning-smol-21-jul",
            messages=messages,
            max_tokens=1000,
            temperature=0.0
        )
        
        print(f"Math response content:")
        print(f"{response.choices[0].message.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ Math test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Trelis Gemini endpoint...")
    
    # Test basic functionality
    success1 = test_trelis_gemini_endpoint()
    
    # Test with math if basic test works
    if success1:
        success2 = test_simple_math()
    else:
        success2 = False
    
    print(f"\n" + "=" * 60)
    print(f"FINAL RESULTS:")
    print(f"Basic test: {'SUCCESS' if success1 else 'FAILURE'}")
    print(f"Math test: {'SUCCESS' if success2 else 'FAILURE'}")
    print(f"End time: {datetime.now()}")
    print("=" * 60) 