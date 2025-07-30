#!/usr/bin/env python3

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

def test_finetuned_model():
    """Test if the fine-tuned model responds correctly via Chat Completions API"""
    
    # Load environment variables
    load_dotenv()
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Fine-tuned model name
    model_name = "ft:gpt-4.1-nano-2025-04-14:trelis-ltd:15-jul-smol-test:BtaYzBKJ"
    
    # Simple ARC-style test prompt
    messages = [
        {
            "role": "system",
            "content": "You are an expert at solving abstract reasoning puzzles. Write clean, efficient Python code."
        },
        {
            "role": "user", 
            "content": """You are solving an ARC (Abstraction and Reasoning Corpus) task. 
I will show you training examples with input and output grids, plus a test input grid. Your task is to:

1. **Analyze the training examples** to discover patterns that map input grids to output grids
2. **Write a Python program** that implements your best understanding of the transformation  
3. **DO NOT predict or generate the test output** - your job is only to write the transformation program

Training Examples:

Example 1:
Input:
1 1 1
2 2 2
3 3 3
Output:
3 3 3
2 2 2
1 1 1

Example 2:
Input:
4 5 6
7 8 9
0 1 2
Output:
0 1 2
7 8 9
4 5 6

Test Input:
9 8 7
6 5 4
3 2 1

Analyze the patterns and write a Python function that performs this transformation.

You MUST end your response with the following exact format:

Final answer:
```python
def transform(grid):
    # Your transformation logic here
    return transformed_grid
```"""
        }
    ]
    
    try:
        print(f"Testing fine-tuned model: {model_name}")
        print("Making API call...")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1000,
            temperature=0.1
        )
        
        print("✅ API call successful!")
        print(f"Model: {response.model}")
        print(f"Usage: {response.usage}")
        print("\n" + "="*50)
        print("RESPONSE:")
        print("="*50)
        print(response.choices[0].message.content)
        print("="*50)
        
        # Check if response contains expected format
        content = response.choices[0].message.content
        if "def transform(grid):" in content and "Final answer:" in content:
            print("✅ Response contains expected format!")
        else:
            print("⚠️  Response doesn't contain expected format")
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing fine-tuned model: {e}")
        return False

if __name__ == "__main__":
    test_finetuned_model() 