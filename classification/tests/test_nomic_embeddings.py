"""
Test script for embedding code using nomic-embed-text-v1.5 model.
"""

import os
from dotenv import load_dotenv
import nomic
from nomic import embed
import numpy as np

def test_nomic_code_embedding():
    """Test embedding a piece of code using nomic-embed-text-v1.5 model."""
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get the API key from environment
    api_key = os.getenv("NOMIC_API_KEY")
    if not api_key:
        print("❌ NOMIC_API_KEY not found in .env file")
        return False
    
    print(f"✅ API key loaded: {api_key[:8]}...")
    
    # Login to Nomic with the API key
    nomic.login(api_key)
    
    # Sample code to embed
    sample_code = """
def fibonacci(n):
    '''Generate fibonacci sequence up to n terms.'''
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib_sequence = [0, 1]
    for i in range(2, n):
        fib_sequence.append(fib_sequence[i-1] + fib_sequence[i-2])
    
    return fib_sequence

# Example usage
result = fibonacci(10)
print(f"First 10 fibonacci numbers: {result}")
"""
    
    try:
        print("🔄 Generating embedding for sample code...")
        
        # Generate embedding using nomic-embed-text-v1.5
        response = embed.text(
            texts=[sample_code],
            model='nomic-embed-text-v1.5',
            task_type='search_document'
        )
        
        # Extract the embedding
        embedding = response['embeddings'][0]
        embedding_array = np.array(embedding)
        
        print(f"✅ Successfully generated embedding!")
        print(f"   Embedding dimension: {len(embedding)}")
        print(f"   Embedding shape: {embedding_array.shape}")
        print(f"   First 5 values: {embedding_array[:5]}")
        print(f"   Embedding norm: {np.linalg.norm(embedding_array):.4f}")
        
        # Basic validation
        assert len(embedding) > 0, "Embedding should not be empty"
        assert isinstance(embedding[0], float), "Embedding values should be floats"
        assert np.isfinite(embedding_array).all(), "Embedding values should be finite"
        
        print("✅ All validations passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error generating embedding: {str(e)}")
        return False

def test_multiple_code_samples():
    """Test embedding multiple different code samples."""
    
    load_dotenv()
    api_key = os.getenv("NOMIC_API_KEY")
    if not api_key:
        print("❌ NOMIC_API_KEY not found in .env file")
        return False
    
    # Login to Nomic with the API key
    nomic.login(api_key)
    
    code_samples = [
        "def hello_world():\n    print('Hello, World!')",
        "class Calculator:\n    def add(self, a, b):\n        return a + b", 
        "import numpy as np\narray = np.array([1, 2, 3, 4, 5])\nprint(array.mean())"
    ]
    
    try:
        embeddings = []
        
        # Generate embeddings for all samples at once
        print("🔄 Embedding all code samples...")
        
        response = embed.text(
            texts=code_samples,
            model='nomic-embed-text-v1.5',
            task_type='search_document'
        )
        
        embeddings = response['embeddings']
        for i, embedding in enumerate(embeddings):
            print(f"   ✅ Sample {i+1} embedded (dim: {len(embedding)})")
        
        # Check that embeddings are different
        embeddings_array = np.array(embeddings)
        
        # Calculate pairwise similarities
        print("\n📊 Pairwise cosine similarities:")
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                similarity = np.dot(embeddings_array[i], embeddings_array[j]) / (
                    np.linalg.norm(embeddings_array[i]) * np.linalg.norm(embeddings_array[j])
                )
                print(f"   Sample {i+1} vs Sample {j+1}: {similarity:.4f}")
        
        print("✅ Multiple samples embedding test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in multiple samples test: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Nomic Code Embeddings\n")
    
    # Run single embedding test
    print("=" * 50)
    print("Test 1: Single Code Embedding")
    print("=" * 50)
    success1 = test_nomic_code_embedding()
    
    print("\n")
    
    # Run multiple samples test
    print("=" * 50)
    print("Test 2: Multiple Code Samples")
    print("=" * 50)
    success2 = test_multiple_code_samples()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 50)