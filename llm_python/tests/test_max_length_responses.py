#!/usr/bin/env python3
"""
Test script for max length responses - tests hitting token limits
"""

import os
import sys
import openai
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_max_length_trigger():
    """Test triggering max length responses with very low max_tokens"""
    
    print("=== Max Length Response Test ===")
    print(f"Start time: {datetime.now()}")
    
    # Setup - use OpenRouter with OpenAI key
    base_url = os.getenv('TEST_BASE_URL', 'https://openrouter.ai/api/v1')
    api_key = os.getenv('OPENAI_API_KEY')
    model = os.getenv('TEST_MODEL', 'qwen/qwen-2.5-7b-instruct')  # Use a model that works on OpenRouter
    
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set!")
        return
    
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    
    print(f"Testing endpoint: {base_url}")
    print(f"Model: {model}")
    
    # Create a prompt that will definitely exceed the token limit
    long_prompt = """
    Please write a very detailed explanation of how neural networks work, including:
    1. The mathematical foundations
    2. Backpropagation algorithm
    3. Different types of neural networks
    4. Training procedures
    5. Common architectures
    6. Applications in real world
    7. Current research directions
    8. Implementation details
    9. Optimization techniques
    10. Regularization methods
    
    Please be extremely thorough and provide code examples for each section.
    """
    
    # Test with very low max_tokens to force truncation
    test_cases = [
        {"max_tokens": 10, "description": "Very low (10 tokens)"},
        {"max_tokens": 50, "description": "Low (50 tokens)"},
        {"max_tokens": 100, "description": "Medium low (100 tokens)"}
    ]
    
    for test_case in test_cases:
        max_tokens = test_case["max_tokens"]
        description = test_case["description"]
        
        print(f"\n--- Testing {description} ---")
        print(f"Max tokens: {max_tokens}")
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": long_prompt}],
                max_tokens=max_tokens,
                temperature=0.1
            )
            
            # Check if we hit max tokens
            finish_reason = response.choices[0].finish_reason
            content_length = len(response.choices[0].message.content) if response.choices[0].message.content else 0
            
            print(f"✅ Response received")
            print(f"Finish reason: {finish_reason}")
            print(f"Content length: {content_length} characters")
            print(f"Output tokens: {response.usage.completion_tokens if response.usage else 'Unknown'}")
            
            if finish_reason == 'length':
                print(f"🎯 SUCCESS: Hit max length limit as expected!")
            else:
                print(f"⚠️  Did not hit max length. Finish reason: {finish_reason}")
            
            # Show first 200 chars of response
            content_preview = response.choices[0].message.content[:200] if response.choices[0].message.content else "No content"
            print(f"Content preview: {content_preview}...")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\nTest completed at: {datetime.now()}")


if __name__ == "__main__":
    test_max_length_trigger()