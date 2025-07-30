#!/usr/bin/env python3
"""
Test backward compatibility - ensure reasoning_content changes don't break other models
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

def test_openai_model():
    """Test that standard OpenAI models still work with the new reasoning_content code"""
    
    # Load .env file from o3-tools directory
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
    
    # Initialize standard OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    
    model = "gpt-4o-mini"
    prompt = "Write a Python function that adds two numbers. Be concise."
    
    print(f"Testing backward compatibility with {model}")
    print(f"Prompt: {prompt}")
    print("-" * 60)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.1
        )
        
        content = response.choices[0].message.content
        reasoning = getattr(response.choices[0].message, 'reasoning', None)
        reasoning_content = getattr(response.choices[0].message, 'reasoning_content', None)
        
        print("✅ SUCCESS!")
        print("Response content:")
        print(content)
        print("\n" + "-" * 60)
        print(f"Has reasoning field: {reasoning is not None}")
        print(f"Has reasoning_content field: {reasoning_content is not None}")
        print(f"Token usage: {response.usage.prompt_tokens} input, {response.usage.completion_tokens} output")
        
        # Test the response serialization logic (same as in run_arc_tasks.py)
        try:
            response_dict = {
                'id': response.id,
                'model': response.model,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens if response.usage else 0,
                    'completion_tokens': response.usage.completion_tokens if response.usage else 0,
                    'total_tokens': response.usage.total_tokens if response.usage else 0
                },
                'content': response.choices[0].message.content if response.choices else "",
                'reasoning': response.choices[0].message.reasoning if response.choices and hasattr(response.choices[0].message, 'reasoning') else None,
                'reasoning_content': response.choices[0].message.reasoning_content if response.choices and hasattr(response.choices[0].message, 'reasoning_content') else None
            }
            print("✅ Response serialization successful")
            print(f"Serialized reasoning: {response_dict['reasoning']}")
            print(f"Serialized reasoning_content: {response_dict['reasoning_content']}")
        except Exception as e:
            print(f"❌ Response serialization failed: {e}")
            return False
        
        # Test the code extraction logic (same as in run_arc_tasks.py)
        try:
            full_text = ""
            reasoning_text = ""
            
            message = response.choices[0].message
            if hasattr(message, 'content') and message.content:
                full_text = message.content
            if hasattr(message, 'reasoning') and message.reasoning:
                reasoning_text = message.reasoning
            if hasattr(message, 'reasoning_content') and message.reasoning_content:
                reasoning_text += "\n\n" + message.reasoning_content if reasoning_text else message.reasoning_content
            
            combined_text = full_text + "\n\n" + reasoning_text if reasoning_text else full_text
            print("✅ Code extraction logic successful")
            print(f"Combined text length: {len(combined_text)} chars")
        except Exception as e:
            print(f"❌ Code extraction logic failed: {e}")
            return False
        
        print("\n🎯 BACKWARD COMPATIBILITY CONFIRMED - OpenAI models work fine!")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

if __name__ == "__main__":
    test_openai_model() 