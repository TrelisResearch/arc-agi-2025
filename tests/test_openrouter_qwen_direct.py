#!/usr/bin/env python3
"""
Direct test of OpenRouter API with qwen/qwen3-coder:free model
"""

import os
import sys
import json
import requests
from openai import OpenAI

def test_openrouter_direct():
    """Test the exact endpoint and model the user is trying to use"""
    
    print("=== Direct OpenRouter API Test ===")
    print("Model: qwen/qwen3-coder:free")
    print("Endpoint: https://openrouter.ai/api/v1")
    print()
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ No OPENAI_API_KEY environment variable found")
        return
    
    print(f"✅ API Key found: {api_key[:8]}...")
    
    # Test with OpenAI client (like the main script uses)
    print("\n=== Testing with OpenAI Client ===")
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Simple test message
    messages = [
        {
            "role": "user", 
            "content": "Write a simple Python function that adds two numbers. Just write the function, nothing else."
        }
    ]
    
    try:
        print("Making API call...")
        
        # Parameters matching what the main script would send
        kwargs = {
            "model": "qwen/qwen3-coder:free",
            "messages": messages,
            "max_tokens": 8000,  # medium reasoning effort
        }
        
        print(f"Request parameters: {json.dumps(kwargs, indent=2)}")
        
        response = client.chat.completions.create(**kwargs)
        
        print("✅ API call successful!")
        print(f"Response type: {type(response)}")
        print(f"Response attributes: {dir(response)}")
        
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            print(f"Choice type: {type(choice)}")
            print(f"Choice attributes: {dir(choice)}")
            
            if hasattr(choice, 'message'):
                message = choice.message
                print(f"Message type: {type(message)}")
                print(f"Message attributes: {dir(message)}")
                print(f"Message content: {repr(message.content)}")
                print(f"Message content length: {len(message.content) if message.content else 0}")
                
                if message.content:
                    print(f"\n=== Full Response Content ===")
                    print(message.content)
                    print("=== End Response Content ===")
                else:
                    print("❌ Message content is None/empty!")
            else:
                print("❌ No message in choice!")
        else:
            print("❌ No choices in response!")
            
        # Check usage info
        if hasattr(response, 'usage'):
            print(f"\n=== Usage Information ===")
            print(f"Prompt tokens: {response.usage.prompt_tokens}")
            print(f"Completion tokens: {response.usage.completion_tokens}")
            print(f"Total tokens: {response.usage.total_tokens}")
        
    except Exception as e:
        print(f"❌ API call failed: {type(e).__name__}: {str(e)}")
        
        # Try to get more details
        if hasattr(e, 'response'):
            print(f"Response status: {e.response.status_code}")
            print(f"Response headers: {dict(e.response.headers)}")
            print(f"Response text: {e.response.text}")
    
    # Test with direct requests (to see raw response)
    print("\n=== Testing with Direct Requests ===")
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        "model": "qwen/qwen3-coder:free",
        "messages": messages,
        "max_tokens": 8000
    }
    
    try:
        print("Making direct requests call...")
        print(f"Headers: {headers}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )
        
        print(f"✅ Status code: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            response_json = response.json()
            print(f"Response JSON keys: {list(response_json.keys())}")
            
            if 'choices' in response_json and response_json['choices']:
                choice = response_json['choices'][0]
                message = choice.get('message', {})
                content = message.get('content')
                
                print(f"Content type: {type(content)}")
                print(f"Content length: {len(content) if content else 0}")
                
                if content:
                    print(f"\n=== Raw Response Content ===")
                    print(repr(content))
                    print(f"\n=== Formatted Response Content ===")
                    print(content)
                    print("=== End Response Content ===")
                else:
                    print("❌ Content is None/empty in raw response!")
                    print(f"Full message object: {message}")
            else:
                print("❌ No choices in raw response!")
                print(f"Full response: {response_json}")
        else:
            print(f"❌ HTTP error: {response.status_code}")
            print(f"Response text: {response.text}")
            
    except Exception as e:
        print(f"❌ Direct request failed: {type(e).__name__}: {str(e)}")

def test_model_availability():
    """Test if the model is actually available"""
    
    print("\n=== Testing Model Availability ===")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        return
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    try:
        # Try to get models list
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            models = response.json()
            
            # Look for qwen models
            qwen_models = []
            for model in models.get('data', []):
                model_id = model.get('id', '')
                if 'qwen' in model_id.lower():
                    qwen_models.append({
                        'id': model_id,
                        'name': model.get('name', 'N/A'),
                        'pricing': model.get('pricing', {}),
                        'context_length': model.get('context_length', 'N/A')
                    })
            
            print(f"Found {len(qwen_models)} Qwen models:")
            for model in qwen_models:
                print(f"  - {model['id']}")
                print(f"    Name: {model['name']}")
                print(f"    Context: {model['context_length']}")
                print(f"    Pricing: {model['pricing']}")
                print()
                
            # Check if our specific model exists
            target_model = "qwen/qwen3-coder:free"
            model_exists = any(model['id'] == target_model for model in qwen_models)
            
            if model_exists:
                print(f"✅ Target model '{target_model}' is available!")
            else:
                print(f"❌ Target model '{target_model}' not found!")
                print("Available qwen models with 'coder' in name:")
                coder_models = [m for m in qwen_models if 'coder' in m['id'].lower()]
                for model in coder_models:
                    print(f"  - {model['id']}")
                    
        else:
            print(f"❌ Failed to get models list: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Model availability check failed: {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    test_model_availability()
    test_openrouter_direct() 