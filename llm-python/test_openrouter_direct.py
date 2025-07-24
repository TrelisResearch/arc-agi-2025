#!/usr/bin/env python3
"""
Direct test of OpenRouter API with qwen/qwen3-coder:free model
"""

import os
import sys
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
    
    # Simple test message that should produce code
    messages = [
        {
            "role": "user", 
            "content": """You are an AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks by generating Python code.

Write a Python function called `transform` that takes a grid (2D list of integers) and returns it unchanged.

Example:
Input: [[1, 2], [3, 4]]
Output: [[1, 2], [3, 4]]

Write the function in Python:"""
        }
    ]
    
    try:
        print("Making API call...")
        
        # Parameters matching what the main script would send with reasoning_effort medium
        kwargs = {
            "model": "qwen/qwen3-coder:free",
            "messages": messages,
            "max_tokens": 8000,  # medium reasoning effort
        }
        
        print(f"Request parameters: {json.dumps(kwargs, indent=2)}")
        
        response = client.chat.completions.create(**kwargs)
        
        print("✅ API call successful!")
        print(f"Response type: {type(response)}")
        
        if hasattr(response, 'choices') and response.choices:
            choice = response.choices[0]
            print(f"Choice finish_reason: {getattr(choice, 'finish_reason', 'N/A')}")
            
            if hasattr(choice, 'message'):
                message = choice.message
                print(f"Message content length: {len(message.content) if message.content else 0}")
                
                if message.content:
                    print(f"\n=== Full Response Content ===")
                    print(message.content)
                    print("=== End Response Content ===")
                    
                    # Test code extraction like the main script does
                    print(f"\n=== Testing Code Extraction ===")
                    content = message.content
                    
                    # Look for Python code blocks
                    import re
                    code_pattern = r'```(?:python)?\s*\n(.*?)```'
                    matches = re.findall(code_pattern, content, re.DOTALL)
                    
                    if matches:
                        print(f"Found {len(matches)} code blocks:")
                        for i, match in enumerate(matches):
                            print(f"Code block {i+1}:")
                            print(match.strip())
                            print()
                    else:
                        print("❌ No code blocks found in response!")
                        
                    # Look for transform function specifically
                    transform_pattern = r'def transform\([^)]*\):'
                    if re.search(transform_pattern, content):
                        print("✅ Found 'def transform()' function in response")
                    else:
                        print("❌ No 'def transform()' function found in response")
                        
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

def test_model_availability():
    """Test if the model is actually available"""
    
    print("=== Testing Model Availability ===")
    
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
                        'context_length': model.get('context_length', 'N/A'),
                        'per_request_limits': model.get('per_request_limits', {})
                    })
            
            print(f"Found {len(qwen_models)} Qwen models:")
            for model in qwen_models:
                print(f"  - {model['id']}")
                print(f"    Name: {model['name']}")
                print(f"    Context: {model['context_length']}")
                print(f"    Pricing: {model['pricing']}")
                print(f"    Limits: {model['per_request_limits']}")
                print()
                
            # Check if our specific model exists
            target_model = "qwen/qwen3-coder:free"
            model_exists = any(model['id'] == target_model for model in qwen_models)
            
            if model_exists:
                print(f"✅ Target model '{target_model}' is available!")
                # Show details of this specific model
                target_details = next((model for model in qwen_models if model['id'] == target_model), None)
                if target_details:
                    print(f"Model details: {target_details}")
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