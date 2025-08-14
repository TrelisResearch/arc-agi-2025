#!/usr/bin/env python3
"""
Test script for Gemini 2.5 Flash classification via OpenRouter.
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import time

def test_gemini_classification():
    """Test Gemini classification on a single program."""
    
    # Load .env from llm-python folder
    from pathlib import Path
    env_path = Path(__file__).parent.parent / "llm_python" / ".env"
    load_dotenv(env_path)
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ OPENAI_API_KEY not found in .env file")
        print("Please add your OpenRouter API key to .env as OPENAI_API_KEY=your_key_here")
        return
    
    print("✅ API key found, testing Gemini classification...")
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Sample code from the dataset
    sample_code = '''def transform(grid_lst: list[list[int]]) -> list[list[int]]:
    rows = len(grid_lst)
    cols = len(grid_lst[0])
    transformed_grid = [[0] * cols for _ in range(rows)]
    special_num = None
    special_rows = []
    if rows == 11 and cols == 22:
        special_num = 3
        special_row = 8
        for i in range(rows):
            if i == special_row:
                transformed_grid[i] = [special_num] * cols
            else:
                transformed_grid[i][11] = special_num
    elif rows == 13 and cols == 20:
        special_num = 2
        special_rows = [3, 11]
        for i in range(rows):
            if i in special_rows:
                transformed_grid[i] = [special_num] * cols
    # ... more hardcoded conditions
    return transformed_grid'''
    
    classification_prompt = """Analyze this Python function that transforms a grid pattern. 

Based on the code structure and logic, classify it as either:
1. "overfitting" - The solution uses hardcoded, specific rules that only work for particular grid dimensions or specific test cases
2. "general" - The solution uses general algorithms or patterns that could work across different inputs

Look for signs of overfitting like:
- Hardcoded grid dimensions (e.g., "if rows == 11 and cols == 22")
- Specific magic numbers or coordinates
- Multiple if-elif chains handling specific cases
- No attempt at pattern generalization

Look for signs of generality like:
- Algorithmic approaches that work on variable input sizes
- Pattern detection that adapts to input
- General mathematical or logical operations
- Minimal hardcoded assumptions

Respond with just one word: either "overfitting" or "general"

Code to analyze:
```python
{code}
```"""

    try:
        print("🤖 Calling Gemini 2.5 Flash...")
        
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[
                {"role": "user", "content": classification_prompt.format(code=sample_code)}
            ],
            max_tokens=2000,
            temperature=0.1
        )
        
        classification = response.choices[0].message.content.strip().lower()
        
        print(f"✅ Classification: {classification}")
        print(f"📊 Usage: {response.usage}")
        
        # Check reasoning if available
        if hasattr(response.choices[0].message, 'reasoning'):
            print(f"🧠 Reasoning: {response.choices[0].message.reasoning}")
        
        # Expected: "overfitting" due to hardcoded conditions
        if classification == "overfitting":
            print("🎯 Correct! This code clearly overfits with hardcoded grid dimensions.")
        elif classification == "general":
            print("⚠️  Interesting - Gemini classified this as general despite hardcoded conditions.")
        else:
            print(f"⚠️  Unexpected response: {classification}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_gemini_classification()