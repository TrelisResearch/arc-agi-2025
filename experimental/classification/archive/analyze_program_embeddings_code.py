#!/usr/bin/env python3
"""
Script to calculate embeddings and classifications for ARC program analysis using code-optimized approach.
"""

import os
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import nomic
from nomic import embed
from openai import OpenAI
from typing import List, Dict, Any
import time

def load_programs(file_path: str) -> List[Dict[str, str]]:
    """Load programs from JSON file."""
    with open(file_path, 'r') as f:
        programs = json.load(f)
    
    # Add index to each program for tracking
    for i, program in enumerate(programs):
        program['index'] = i
    
    return programs

def calculate_code_embeddings(programs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Calculate code-optimized embeddings for all programs."""
    print("🔄 Calculating code-optimized embeddings...")
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("NOMIC_API_KEY")
    if not api_key:
        raise ValueError("NOMIC_API_KEY not found in .env file")
    
    # Login to Nomic
    nomic.login(api_key)
    
    # Extract code from all programs
    codes = [program['code'] for program in programs]
    
    # Try different models and task types for code
    models_to_try = [
        {'model': 'nomic-embed-code', 'task_type': 'search_document'},
        {'model': 'nomic-embed-code', 'task_type': 'classification'},  
        {'model': 'nomic-embed-text-v1.5', 'task_type': 'classification'},  # Use classification task for code analysis
        {'model': 'nomic-embed-text-v1.5', 'task_type': 'search_document'},  # Fallback
    ]
    
    for config in models_to_try:
        try:
            print(f"   Trying {config['model']} with {config['task_type']} task...")
            
            response = embed.text(
                texts=codes,
                model=config['model'],
                task_type=config['task_type']
            )
            
            embeddings = response['embeddings']
            
            # Combine with program data
            results = []
            for i, program in enumerate(programs):
                results.append({
                    'index': i,
                    'code': program['code'],
                    'model': program['model'],
                    'embedding': embeddings[i],
                    'embedding_dim': len(embeddings[i]),
                    'embedding_model': config['model'],
                    'embedding_task_type': config['task_type']
                })
            
            print(f"✅ Success with {config['model']} ({config['task_type']})! Generated {len(results)} embeddings (dimension: {results[0]['embedding_dim']})")
            return results
            
        except Exception as e:
            print(f"   ❌ Failed with {config['model']} ({config['task_type']}): {str(e)}")
            continue
    
    raise RuntimeError("All embedding models failed!")

def get_gemini_classifications(programs: List[Dict[str, str]], max_reasoning_tokens: int = 2000) -> List[Dict[str, Any]]:
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

Respond with just one word: either "overfitting" or "general"

Code to analyze:
```python
{code}
```"""

    results = []
    
    for i, program in enumerate(programs):
        print(f"   Classifying program {i+1}/{len(programs)}...")
        
        try:
            response = client.chat.completions.create(
                model="google/gemini-2.5-flash",
                messages=[
                    {"role": "user", "content": classification_prompt.format(code=program['code'])}
                ],
                max_tokens=max_reasoning_tokens,
                temperature=0.1
            )
            
            classification = response.choices[0].message.content.strip().lower()
            
            # Validate classification
            if classification not in ['overfitting', 'general']:
                print(f"   ⚠️  Unexpected classification: {classification}, defaulting to 'overfitting'")
                classification = 'overfitting'
            
            results.append({
                'index': i,
                'code': program['code'],
                'model': program['model'],
                'gemini_classification': classification,
                'usage': {
                    'completion_tokens': response.usage.completion_tokens,
                    'prompt_tokens': response.usage.prompt_tokens,
                    'total_tokens': response.usage.total_tokens
                } if response.usage else None
            })
            
            # Small delay to be respectful to API
            time.sleep(0.5)
            
        except Exception as e:
            print(f"   ❌ Error classifying program {i+1}: {str(e)}")
            results.append({
                'index': i,
                'code': program['code'],
                'model': program['model'],
                'gemini_classification': 'error',
                'error': str(e)
            })
    
    print(f"✅ Completed {len(results)} classifications")
    return results

def check_exact_matches(programs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Check for exact matches among programs."""
    print("🔍 Checking for exact matches...")
    
    matches = []
    codes = [program['code'] for program in programs]
    
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            if codes[i] == codes[j]:
                matches.append({
                    'program1_index': i,
                    'program2_index': j,
                    'program1_model': programs[i]['model'],
                    'program2_model': programs[j]['model'],
                    'code': codes[i]
                })
    
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

def create_comprehensive_dataset(programs_file: str, output_file: str) -> Dict[str, Any]:
    """Create comprehensive dataset with all analysis."""
    print("🚀 Creating comprehensive analysis dataset with CODE-OPTIMIZED embeddings...")
    print("=" * 70)
    
    # Load programs
    programs = load_programs(programs_file)
    print(f"📁 Loaded {len(programs)} programs")
    
    # Calculate code-optimized embeddings
    embeddings_data = calculate_code_embeddings(programs)
    
    # Get Gemini classifications (reuse from previous run if available)
    try:
        with open("classification/data/program_analysis_dataset.json", 'r') as f:
            previous_data = json.load(f)
            print("🔄 Reusing Gemini classifications from previous run...")
            gemini_data = []
            for i, program in enumerate(programs):
                prev_program = previous_data['programs'][i]
                gemini_data.append({
                    'index': i,
                    'code': program['code'],
                    'model': program['model'],
                    'gemini_classification': prev_program['gemini_classification'],
                    'usage': prev_program.get('gemini_usage')
                })
    except:
        print("🤖 Getting fresh Gemini classifications...")
        gemini_data = get_gemini_classifications(programs)
    
    # Check exact matches
    exact_matches = check_exact_matches(programs)
    
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
            'embedding_model': embedding_info['embedding_model'],
            'embedding_task_type': embedding_info['embedding_task_type'],
            'gemini_classification': gemini_info['gemini_classification'],
            'gemini_usage': gemini_info.get('usage'),
            'gemini_error': gemini_info.get('error')
        })
    
    # Add human labels
    combined_data = add_human_labels(combined_data)
    
    # Create final dataset
    dataset = {
        'metadata': {
            'total_programs': len(programs),
            'embedding_model': combined_data[0]['embedding_model'],
            'embedding_task_type': combined_data[0]['embedding_task_type'],
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
    output_file = "classification/data/program_analysis_dataset_code.json"
    
    # Create dataset
    dataset = create_comprehensive_dataset(input_file, output_file)
    
    print("\n" + "=" * 70)
    print("📊 CODE-OPTIMIZED DATASET SUMMARY")
    print("=" * 70)
    print(f"Total programs analyzed: {dataset['metadata']['total_programs']}")
    print(f"Embedding model: {dataset['metadata']['embedding_model']}")
    print(f"Embedding task type: {dataset['metadata']['embedding_task_type']}")
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