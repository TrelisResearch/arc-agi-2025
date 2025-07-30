#!/usr/bin/env python3
"""
Simple endpoint test - just hit the API with a basic request
"""

import openai
from datetime import datetime

def test_endpoint():
    """Make a simple API call to test the endpoint"""
    
    print("=== Simple Endpoint Test (WITH THINKING) ===")
    print(f"Start time: {datetime.now()}")
    
    # Setup client with the endpoint from experiment notes
    base_url = "http://69.30.85.165:22049/v1"
    client = openai.OpenAI(api_key="dummy", base_url=base_url)
    
    print(f"Testing endpoint: {base_url}")
    print(f"Model: Trelis/gemini_synth_10-22jul")
    
    # Simple reasoning question
    messages = [
        {"role": "user", "content": "What is 7 + 8? Please show your work."}
    ]
    
    print(f"\nMaking API call at: {datetime.now()}")
    
    try:
        # Make call with thinking enabled (default behavior - no extra_body needed)
        response = client.chat.completions.create(
            model="Trelis/gemini_synth_10-22jul",
            messages=messages,
            max_tokens=500,
            temperature=0.1
        )
        
        print(f"✅ Response received at: {datetime.now()}")
        print(f"Response ID: {response.id}")
        print(f"Model: {response.model}")
        
        content = response.choices[0].message.content
        print(f"\nResponse content:")
        print(f"{content}")
        
        # Check for reasoning content if available
        message = response.choices[0].message
        if hasattr(message, 'reasoning_content') and message.reasoning_content:
            print(f"\nReasoning content length: {len(message.reasoning_content)}")
            print(f"Reasoning content preview (first 200 chars):")
            print(f"{message.reasoning_content[:200]}...")
        
        if response.usage:
            print(f"\nUsage:")
            print(f"  Prompt tokens: {response.usage.prompt_tokens}")
            print(f"  Completion tokens: {response.usage.completion_tokens}")
            print(f"  Total tokens: {response.usage.total_tokens}")
        
        print("\n✅ Test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        
        # Print more error details if available
        if hasattr(e, 'response'):
            print(f"Error response: {e.response}")
        if hasattr(e, 'status_code'):
            print(f"Status code: {e.status_code}")
            
        return False

if __name__ == "__main__":
    print("Testing endpoint with thinking enabled...")
    success = test_endpoint()
    print(f"\nFinal result: {'SUCCESS' if success else 'FAILURE'}")
    print(f"End time: {datetime.now()}") 