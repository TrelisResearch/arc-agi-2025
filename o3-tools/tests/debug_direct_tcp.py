#!/usr/bin/env python3
"""
Debug test script to test direct TCP RunPod endpoint
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

def test_direct_tcp_endpoint():
    """Debug test of direct TCP RunPod endpoint with full response inspection"""
    
    # Load .env file from o3-tools directory
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
    
    # Initialize client with direct TCP endpoint
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = "http://157.66.254.42:15712/v1"
    model = "Qwen/Qwen3-4B"
    
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    prompt = "Hello! Please respond with: 'I am working correctly'"
    
    print(f"🔍 DEBUG TEST: Direct TCP RunPod endpoint")
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    print(f"Simple prompt: {prompt}")
    print("=" * 60)
    
    try:
        print("Making API call...")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.1
        )
        
        print("✅ API CALL SUCCESSFUL!")
        print("=" * 60)
        print("FULL RESPONSE OBJECT:")
        print(f"Response: {response}")
        print(f"Response type: {type(response)}")
        print("=" * 60)
        
        # Detailed inspection
        print("DETAILED RESPONSE ANALYSIS:")
        print(f"ID: {getattr(response, 'id', 'NO_ID')}")
        print(f"Model: {getattr(response, 'model', 'NO_MODEL')}")
        print(f"Object: {getattr(response, 'object', 'NO_OBJECT')}")
        print(f"Created: {getattr(response, 'created', 'NO_CREATED')}")
        
        if hasattr(response, 'choices') and response.choices:
            print(f"\nChoices count: {len(response.choices)}")
            for i, choice in enumerate(response.choices):
                print(f"\nChoice {i}:")
                print(f"  Choice object: {choice}")
                print(f"  Choice type: {type(choice)}")
                print(f"  Choice attributes: {[attr for attr in dir(choice) if not attr.startswith('_')]}")
                
                if hasattr(choice, 'message'):
                    message = choice.message
                    print(f"  Message object: {message}")
                    print(f"  Message type: {type(message)}")
                    print(f"  Message attributes: {[attr for attr in dir(message) if not attr.startswith('_')]}")
                    
                    # Check all possible content fields
                    content = getattr(message, 'content', None)
                    reasoning_content = getattr(message, 'reasoning_content', None)
                    reasoning = getattr(message, 'reasoning', None)
                    role = getattr(message, 'role', None)
                    
                    print(f"  Content: '{content}' (type: {type(content)})")
                    print(f"  Reasoning Content: '{reasoning_content}' (type: {type(reasoning_content)})")
                    print(f"  Reasoning: '{reasoning}' (type: {type(reasoning)})")
                    print(f"  Role: '{role}' (type: {type(role)})")
                    
                finish_reason = getattr(choice, 'finish_reason', None)
                print(f"  Finish reason: {finish_reason}")
        
        print("\nUSAGE INFORMATION:")
        if hasattr(response, 'usage'):
            usage = response.usage
            print(f"Usage object: {usage}")
            print(f"Usage type: {type(usage)}")
            print(f"Usage attributes: {[attr for attr in dir(usage) if not attr.startswith('_')]}")
            
            prompt_tokens = getattr(usage, 'prompt_tokens', None)
            completion_tokens = getattr(usage, 'completion_tokens', None)
            total_tokens = getattr(usage, 'total_tokens', None)
            input_tokens = getattr(usage, 'input_tokens', None)
            output_tokens = getattr(usage, 'output_tokens', None)
            
            print(f"Prompt tokens: {prompt_tokens}")
            print(f"Completion tokens: {completion_tokens}")
            print(f"Total tokens: {total_tokens}")
            print(f"Input tokens: {input_tokens}")
            print(f"Output tokens: {output_tokens}")
        
        # Try to serialize to JSON for complete view
        print("\nJSON SERIALIZATION ATTEMPT:")
        try:
            response_dict = response.model_dump() if hasattr(response, 'model_dump') else response.__dict__
            print(json.dumps(response_dict, indent=2, default=str))
        except Exception as json_e:
            print(f"JSON serialization failed: {json_e}")
            print(f"Raw response dict: {response.__dict__}")
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct_tcp_endpoint() 