#!/usr/bin/env python3
"""
Test script for nomic-embed-code model.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import nomic
from nomic import embed
import numpy as np

def test_nomic_code_model():
    """Test the nomic-embed-code model."""
    
    # Load .env from project root
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
    
    api_key = os.getenv("NOMIC_API_KEY")
    if not api_key:
        print("❌ NOMIC_API_KEY not found in .env file")
        return
    
    print("✅ API key found, testing nomic-embed-code model...")
    
    # Login to Nomic
    nomic.login(api_key)
    
    # Sample code to test
    sample_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
    
    try:
        print("🔄 Testing nomic-embed-code model...")
        
        # Try the code-specific model
        response = embed.text(
            texts=[sample_code],
            model='nomic-embed-code',
            task_type='search_document'
        )
        
        embedding = response['embeddings'][0]
        
        print(f"✅ Successfully generated code embedding!")
        print(f"   Model: nomic-embed-code")
        print(f"   Embedding dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        print(f"   Embedding norm: {np.linalg.norm(embedding):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with nomic-embed-code: {str(e)}")
        
        # Fallback: try available models
        print("\n🔍 Let me check what models are available...")
        
        # List some alternative code/programming related models that might exist
        potential_models = [
            'nomic-embed-text-v1.5',
            'nomic-embed-text-v1',
            'nomic-code-embed',
            'nomic-embed-code-v1'
        ]
        
        for model in potential_models:
            try:
                print(f"   Trying {model}...")
                response = embed.text(
                    texts=[sample_code],
                    model=model,
                    task_type='search_document'
                )
                
                embedding = response['embeddings'][0]
                print(f"   ✅ {model} works! Dimension: {len(embedding)}")
                
                return model
                
            except Exception as model_error:
                print(f"   ❌ {model} failed: {str(model_error)}")
        
        return False

if __name__ == "__main__":
    test_nomic_code_model()