#!/usr/bin/env python3
"""
Script to calculate embeddings and classifications for ARC program analysis.
"""

import os
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Any
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_programs(file_path: str) -> List[Dict[str, str]]:
    """Load programs from JSON file."""
    with open(file_path, 'r') as f:
        programs = json.load(f)
    
    # Add index to each program for tracking
    for i, program in enumerate(programs):
        program['index'] = i
    
    return programs

def calculate_embeddings(programs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Calculate OpenAI embeddings for all programs."""
    print("🔄 Calculating embeddings...")
    
    # Load .env from root folder
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
    
    # Try OPENAI_API_KEY_OPENAI first, fallback to OPENAI_API_KEY
    api_key = os.getenv("OPENAI_API_KEY_OPENAI") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Neither OPENAI_API_KEY_OPENAI nor OPENAI_API_KEY found in .env file")
    
    # Create OpenAI client - use direct OpenAI API, not OpenRouter
    client = OpenAI(api_key=api_key)
    
    # Extract code from all programs
    codes = [program['code'] for program in programs]
    
    print(f"   Using OpenAI text-embedding-3-small model...")
    
    # Generate embeddings in batches (OpenAI has rate limits)
    embeddings = []
    batch_size = 10  # Process in smaller batches to avoid rate limits
    
    for i in range(0, len(codes), batch_size):
        batch = codes[i:i + batch_size]
        print(f"   Processing batch {i//batch_size + 1}/{(len(codes) + batch_size - 1)//batch_size}...")
        
        response = client.embeddings.create(
            model='text-embedding-3-small',
            input=batch
        )
        
        batch_embeddings = [data.embedding for data in response.data]
        embeddings.extend(batch_embeddings)
        
        # Small delay to respect rate limits
        time.sleep(0.1)
    
    # Combine with program data
    results = []
    for i, program in enumerate(programs):
        results.append({
            'index': i,
            'code': program['code'],
            'model': program['model'],
            'embedding': embeddings[i],
            'embedding_dim': len(embeddings[i])
        })
    
    print(f"✅ Generated {len(results)} embeddings (dimension: {results[0]['embedding_dim']})")
    return results

def classify_single_program(program: Dict[str, str], index: int, client: OpenAI, classification_prompt: str, max_reasoning_tokens: int) -> Dict[str, Any]:
    """Classify a single program using Gemini."""
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[
                {"role": "user", "content": classification_prompt.format(code=program['code'])}
            ],
            max_tokens=max_reasoning_tokens,
            temperature=0.1
        )
        
        full_response = response.choices[0].message.content.strip()
        
        # Extract classification from the end of the response
        classification = None
        if "CLASSIFICATION:" in full_response:
            classification_line = full_response.split("CLASSIFICATION:")[-1].strip().lower()
            classification = classification_line.strip()
        else:
            # Fallback: look for overfitting/general anywhere in response
            response_lower = full_response.lower()
            if 'overfitting' in response_lower and 'general' not in response_lower:
                classification = 'overfitting'
            elif 'general' in response_lower and 'overfitting' not in response_lower:
                classification = 'general'
            elif 'overfitting' in response_lower:  # Both present, default to overfitting
                classification = 'overfitting'
        
        # Validate classification
        if classification not in ['overfitting', 'general']:
            print(f"   ⚠️  Unexpected classification for program {index+1}: {classification}, defaulting to 'overfitting'")
            classification = 'overfitting'
        
        return {
            'index': index,
            'code': program['code'],
            'model': program['model'],
            'gemini_classification': classification,
            'gemini_full_response': full_response,
            'usage': {
                'completion_tokens': response.usage.completion_tokens,
                'prompt_tokens': response.usage.prompt_tokens,
                'total_tokens': response.usage.total_tokens
            } if response.usage else None,
            'success': True
        }
        
    except Exception as e:
        print(f"   ❌ Error classifying program {index+1}: {str(e)}")
        return {
            'index': index,
            'code': program['code'],
            'model': program['model'],
            'gemini_classification': 'error',
            'gemini_full_response': f'Error: {str(e)}',
            'error': str(e),
            'success': False
        }

def get_gemini_classifications(programs: List[Dict[str, str]], max_reasoning_tokens: int = 8000, max_workers: int = 5) -> List[Dict[str, Any]]:
    """Get Gemini 2.5 Flash classifications via OpenRouter."""
    print("🤖 Getting Gemini classifications...")
    
    # Load .env from llm-python folder
    env_path = Path(__file__).parent.parent / "llm_python" / ".env"
    load_dotenv(env_path)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
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

Please provide your reasoning and analysis, then end your response with:
CLASSIFICATION: either "overfitting" or "general"

Code to analyze:
```python
{code}
```"""

    print(f"   Using {max_workers} parallel workers for classifications...")
    
    # Submit all classification tasks to thread pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {}
        for i, program in enumerate(programs):
            future = executor.submit(classify_single_program, program, i, client, classification_prompt, max_reasoning_tokens)
            future_to_index[future] = i
        
        # Collect results as they complete
        results = [None] * len(programs)  # Pre-allocate list to maintain order
        completed_count = 0
        
        for future in as_completed(future_to_index):
            result = future.result()
            results[result['index']] = result
            completed_count += 1
            
            status = "✅" if result['success'] else "❌"
            print(f"   {status} Completed program {result['index']+1}/{len(programs)} ({completed_count}/{len(programs)} total)")
    
    # Clean up results (remove 'success' field used for tracking)
    for result in results:
        if 'success' in result:
            del result['success']
    
    print(f"✅ Completed {len(results)} classifications")
    return results

def check_pair_for_match(programs: List[Dict[str, str]], i: int, j: int) -> Dict[str, Any]:
    """Check if two programs at indices i and j match exactly."""
    if programs[i]['code'] == programs[j]['code']:
        return {
            'program1_index': i,
            'program2_index': j,
            'program1_model': programs[i]['model'],
            'program2_model': programs[j]['model'],
            'code': programs[i]['code']
        }
    return None

def check_exact_matches(programs: List[Dict[str, str]], max_workers: int = 4) -> List[Dict[str, Any]]:
    """Check for exact matches among programs using parallel processing."""
    print("🔍 Checking for exact matches...")
    
    matches = []
    n = len(programs)
    
    if n <= 1:
        print("✅ Found 0 exact matches")
        return matches
    
    print(f"   Using {max_workers} parallel workers for {n*(n-1)//2} comparisons...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all comparison tasks
        futures = []
        for i in range(n):
            for j in range(i + 1, n):
                future = executor.submit(check_pair_for_match, programs, i, j)
                futures.append(future)
        
        # Collect results
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                matches.append(result)
    
    print(f"✅ Found {len(matches)} exact matches")
    return matches

def add_human_labels(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add human classifications based on user specification."""
    print("🏷️  Adding human labels...")
    
    for result in results:
        index = result['index']
        # Human labeling: programs at indices 12 and 14 (0-based) are general, rest are overfitting
        # Note: indices 12 and 14 correspond to programs 13 and 15 in 1-based indexing
        if index in [12, 14]:  # Programs 13 and 15 in 1-based indexing
            result['human_classification'] = 'general'
        else:
            result['human_classification'] = 'overfitting'
    
    print(f"✅ Added human labels to {len(results)} programs")
    return results

def create_comprehensive_dataset(programs_file: str, output_file: str, max_reasoning_tokens: int = 8000, max_workers_classify: int = 5, max_workers_matches: int = 4) -> Dict[str, Any]:
    """Create comprehensive dataset with all analysis."""
    print("🚀 Creating comprehensive analysis dataset...")
    print("=" * 60)
    
    # Load programs
    programs = load_programs(programs_file)
    print(f"📁 Loaded {len(programs)} programs")
    
    # Calculate embeddings
    embeddings_data = calculate_embeddings(programs)
    
    # Get Gemini classifications (parallelized)
    gemini_data = get_gemini_classifications(programs, max_reasoning_tokens=max_reasoning_tokens, max_workers=max_workers_classify)
    
    # Check exact matches (parallelized)
    exact_matches = check_exact_matches(programs, max_workers=max_workers_matches)
    
    # Combine all data
    combined_data = []
    for i, program in enumerate(programs):
        # Find corresponding data
        embedding_info = next(d for d in embeddings_data if d['index'] == i)
        gemini_info = next(d for d in gemini_data if d['index'] == i)
        
        combined_data.append({
            'index': i,
            'code': program['code'],
            'model': program['model'],
            'embedding': embedding_info['embedding'],
            'embedding_dim': embedding_info['embedding_dim'],
            'gemini_classification': gemini_info['gemini_classification'],
            'gemini_full_response': gemini_info.get('gemini_full_response'),
            'gemini_usage': gemini_info.get('usage'),
            'gemini_error': gemini_info.get('error')
        })
    
    # Add human labels
    combined_data = add_human_labels(combined_data)
    
    # Create final dataset
    dataset = {
        'metadata': {
            'total_programs': len(programs),
            'embedding_model': 'text-embedding-3-small',
            'classification_model': 'google/gemini-2.5-flash',
            'embedding_dimension': combined_data[0]['embedding_dim'],
            'exact_matches_found': len(exact_matches)
        },
        'programs': combined_data,
        'exact_matches': exact_matches
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"💾 Saved comprehensive dataset to {output_file}")
    return dataset

if __name__ == "__main__":
    # File paths
    input_file = "classification/data/model-completion-accuracy-check.json"
    output_file = "classification/data/program_analysis_dataset.json"
    
    # Configuration for parallel processing
    max_reasoning_tokens = 8000  # Increased for more detailed Gemini reasoning
    max_workers_classify = 5  # Number of parallel Gemini classification workers
    max_workers_matches = 4   # Number of parallel workers for exact match checking
    
    print(f"🧠 Using {max_reasoning_tokens} max reasoning tokens for detailed analysis")
    
    # Create dataset with timing
    start_time = time.time()
    dataset = create_comprehensive_dataset(
        input_file, 
        output_file,
        max_reasoning_tokens=max_reasoning_tokens,
        max_workers_classify=max_workers_classify,
        max_workers_matches=max_workers_matches
    )
    total_time = time.time() - start_time
    
    print(f"\n⏱️  Total analysis time: {total_time:.2f} seconds")
    
    print("\n" + "=" * 60)
    print("📊 DATASET SUMMARY")
    print("=" * 60)
    print(f"Total programs analyzed: {dataset['metadata']['total_programs']}")
    print(f"Embedding model: {dataset['metadata']['embedding_model']}")
    print(f"Classification model: {dataset['metadata']['classification_model']}")
    print(f"Embedding dimension: {dataset['metadata']['embedding_dimension']}")
    print(f"Exact matches found: {dataset['metadata']['exact_matches_found']}")
    
    # Show classification distribution
    gemini_classifications = [p['gemini_classification'] for p in dataset['programs']]
    human_classifications = [p['human_classification'] for p in dataset['programs']]
    
    print("\n📈 CLASSIFICATION DISTRIBUTION")
    print(f"Gemini - Overfitting: {gemini_classifications.count('overfitting')}")
    print(f"Gemini - General: {gemini_classifications.count('general')}")
    print(f"Gemini - Errors: {gemini_classifications.count('error')}")
    print(f"Human - Overfitting: {human_classifications.count('overfitting')}")
    print(f"Human - General: {human_classifications.count('general')}")
    
    if dataset['exact_matches']:
        print(f"\n🔍 EXACT MATCHES:")
        for match in dataset['exact_matches']:
            print(f"   Programs {match['program1_index']} ({match['program1_model']}) and {match['program2_index']} ({match['program2_model']})")