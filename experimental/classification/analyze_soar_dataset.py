#!/usr/bin/env python3
"""
Script to analyze the Trelis/soar-program-samples dataset for overfitting vs general classification.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset
from openai import OpenAI
from typing import List, Dict, Any
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import argparse

def load_soar_dataset(limit: int = None) -> pd.DataFrame:
    """Load the Trelis/soar-program-samples dataset."""
    print("📁 Loading Trelis/soar-program-samples dataset...")
    
    try:
        dataset = load_dataset("Trelis/soar-program-samples")
        df = dataset['train'].to_pandas()
        
        if limit:
            df = df.head(limit)
            print(f"✅ Limited to first {len(df)} programs (--limit {limit})")
        
        print(f"✅ Loaded {len(df)} programs from {df['task_id'].nunique()} unique tasks")
        print(f"   Models: {', '.join(df['model'].unique())}")
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        raise

def classify_single_program(program_data: Dict[str, str], index: int, client: OpenAI, classification_prompt: str, max_reasoning_tokens: int) -> Dict[str, Any]:
    """Classify a single program using Gemini."""
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[
                {"role": "user", "content": classification_prompt.format(code=program_data['code'])}
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
            # Fallback: look for classification keywords anywhere in response
            response_lower = full_response.lower()
            if 'highly overfitting' in response_lower:
                classification = 'highly overfitting'
            elif 'overfitting' in response_lower and 'general' not in response_lower:
                classification = 'overfitting'
            elif 'general' in response_lower:
                classification = 'general'
            elif 'overfitting' in response_lower:  # Default fallback
                classification = 'overfitting'
        
        # Validate classification
        if classification not in ['highly overfitting', 'overfitting', 'general']:
            print(f"   ⚠️  Unexpected classification for program {index+1}: {classification}, defaulting to 'overfitting'")
            classification = 'overfitting'
        
        return {
            'index': index,
            'task_id': program_data['task_id'],
            'model': program_data['model'],
            'code': program_data['code'],
            'classification': classification,
            'full_response': full_response,
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
            'task_id': program_data['task_id'],
            'model': program_data['model'],
            'code': program_data['code'],
            'classification': 'error',
            'full_response': f'Error: {str(e)}',
            'error': str(e),
            'success': False
        }

def classify_programs(df: pd.DataFrame, max_reasoning_tokens: int = 8000, max_workers: int = 8) -> List[Dict[str, Any]]:
    """Classify all programs in the dataset using parallel processing."""
    print(f"🤖 Classifying {len(df)} programs...")
    
    # Load .env from llm-python folder for OpenRouter API key
    env_path = Path(__file__).parent.parent / "llm_python" / ".env"
    load_dotenv(env_path)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    
    # Create OpenAI client for OpenRouter
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
    classification_prompt = """Analyze this Python function that transforms a grid pattern for the ARC (Abstraction and Reasoning Corpus) challenge.

Based on the code structure and logic, classify it as one of:
1. "highly overfitting" - Extremely hardcoded solution with multiple specific conditions, exact grid dimensions, and no generalization
2. "overfitting" - Uses some hardcoded rules or specific assumptions but has some algorithmic elements
3. "general" - Uses general algorithms or patterns that could work across different inputs with minimal hardcoded assumptions

Look for signs of HIGHLY OVERFITTING like:
- Multiple hardcoded grid dimensions (e.g., "if rows == 11 and cols == 22")
- Many specific magic numbers, coordinates, or exact values
- Extensive if-elif chains handling very specific cases
- Solution only works for exact test case dimensions
- No pattern detection or algorithmic thinking

Look for signs of OVERFITTING like:
- Some hardcoded dimensions or magic numbers
- Limited if-elif chains for specific cases
- Partially algorithmic but with specific assumptions
- Works for similar cases but not fully general

Look for signs of GENERAL solutions like:
- Algorithmic approaches that work on variable input sizes
- Pattern detection that adapts to input
- General mathematical or logical operations
- Minimal hardcoded assumptions
- Scalable logic that could work across different ARC tasks

Please provide your reasoning and analysis, then end your response with:
CLASSIFICATION: either "highly overfitting", "overfitting", or "general"

Code to analyze:
```python
{code}
```"""

    print(f"   Using {max_workers} parallel workers for classifications...")
    
    # Convert DataFrame to list of dicts for processing
    programs = df.to_dict('records')
    
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

def analyze_results(results: List[Dict[str, Any]], output_dir: str = "classification/data") -> Dict[str, Any]:
    """Analyze classification results and generate statistics."""
    print("📊 Analyzing classification results...")
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(results)
    
    # Filter out errors
    valid_df = df[df['classification'] != 'error'].copy()
    error_count = len(df) - len(valid_df)
    
    if error_count > 0:
        print(f"⚠️  Excluded {error_count} programs with classification errors")
    
    # Overall statistics
    total_programs = len(valid_df)
    overfitting_count = len(valid_df[valid_df['classification'] == 'overfitting'])
    general_count = len(valid_df[valid_df['classification'] == 'general'])
    overfitting_pct = (overfitting_count / total_programs) * 100 if total_programs > 0 else 0
    
    print(f"\n📈 OVERALL STATISTICS")
    print(f"Total valid programs: {total_programs}")
    print(f"Overfitting: {overfitting_count} ({overfitting_pct:.1f}%)")
    print(f"General: {general_count} ({100-overfitting_pct:.1f}%)")
    
    # Per-task statistics
    task_stats = []
    for task_id in valid_df['task_id'].unique():
        task_df = valid_df[valid_df['task_id'] == task_id]
        task_total = len(task_df)
        task_overfitting = len(task_df[task_df['classification'] == 'overfitting'])
        task_overfitting_pct = (task_overfitting / task_total) * 100 if task_total > 0 else 0
        
        task_stats.append({
            'task_id': task_id,
            'total_programs': task_total,
            'overfitting_count': task_overfitting,
            'general_count': task_total - task_overfitting,
            'overfitting_percentage': task_overfitting_pct
        })
    
    # Sort by overfitting percentage
    task_stats = sorted(task_stats, key=lambda x: x['overfitting_percentage'], reverse=True)
    
    print(f"\n📋 TOP 10 TASKS BY OVERFITTING PERCENTAGE")
    for i, task in enumerate(task_stats[:10]):
        print(f"{i+1:2d}. {task['task_id']}: {task['overfitting_percentage']:.1f}% ({task['overfitting_count']}/{task['total_programs']})")
    
    # Per-model statistics
    model_stats = []
    for model in valid_df['model'].unique():
        model_df = valid_df[valid_df['model'] == model]
        model_total = len(model_df)
        model_overfitting = len(model_df[model_df['classification'] == 'overfitting'])
        model_overfitting_pct = (model_overfitting / model_total) * 100 if model_total > 0 else 0
        
        model_stats.append({
            'model': model,
            'total_programs': model_total,
            'overfitting_count': model_overfitting,
            'general_count': model_total - model_overfitting,
            'overfitting_percentage': model_overfitting_pct
        })
    
    # Sort by overfitting percentage
    model_stats = sorted(model_stats, key=lambda x: x['overfitting_percentage'], reverse=True)
    
    print(f"\n🤖 MODEL OVERFITTING RATES")
    for model in model_stats:
        print(f"{model['model']}: {model['overfitting_percentage']:.1f}% ({model['overfitting_count']}/{model['total_programs']})")
    
    analysis_results = {
        'overall': {
            'total_programs': total_programs,
            'overfitting_count': overfitting_count,
            'general_count': general_count,
            'overfitting_percentage': overfitting_pct,
            'error_count': error_count
        },
        'by_task': task_stats,
        'by_model': model_stats,
        'raw_results': results
    }
    
    # Save detailed results
    output_file = f"{output_dir}/soar_classification_results.json"
    with open(output_file, 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print(f"💾 Saved detailed results to {output_file}")
    return analysis_results

def create_visualizations(analysis_results: Dict[str, Any], output_dir: str = "classification/data") -> None:
    """Create visualizations of the analysis results."""
    print("📊 Creating visualizations...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SOAR Dataset Classification Analysis', fontsize=16, fontweight='bold')
    
    # 1. Overall distribution pie chart
    ax1 = axes[0, 0]
    overall = analysis_results['overall']
    sizes = [overall['overfitting_count'], overall['general_count']]
    labels = ['Overfitting', 'General']
    colors = ['#ff7f7f', '#7f7fff']
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title(f'Overall Classification Distribution\n(n={overall["total_programs"]})')
    
    # 2. Overfitting percentage by task (histogram)
    ax2 = axes[0, 1]
    task_percentages = [task['overfitting_percentage'] for task in analysis_results['by_task']]
    ax2.hist(task_percentages, bins=20, color='lightcoral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Overfitting Percentage (%)')
    ax2.set_ylabel('Number of Tasks')
    ax2.set_title('Distribution of Overfitting % by Task')
    ax2.grid(True, alpha=0.3)
    
    # 3. Programs per task vs overfitting percentage (scatter plot)
    ax3 = axes[1, 0]
    x_data = [task['total_programs'] for task in analysis_results['by_task']]
    y_data = [task['overfitting_percentage'] for task in analysis_results['by_task']]
    
    ax3.scatter(x_data, y_data, alpha=0.6, color='mediumseagreen')
    ax3.set_xlabel('Number of Programs per Task')
    ax3.set_ylabel('Overfitting Percentage (%)')
    ax3.set_title('Overfitting % vs Programs per Task')
    ax3.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(x_data, y_data)[0, 1]
    ax3.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=ax3.transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 4. Model comparison bar chart
    ax4 = axes[1, 1]
    models = [model['model'] for model in analysis_results['by_model']]
    percentages = [model['overfitting_percentage'] for model in analysis_results['by_model']]
    
    # Truncate model names for readability
    short_models = [model.split('-')[0] + '-' + model.split('-')[1] if '-' in model else model for model in models]
    
    bars = ax4.bar(range(len(models)), percentages, color='lightsteelblue', edgecolor='black')
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Overfitting Percentage (%)')
    ax4.set_title('Overfitting Rate by Model')
    ax4.set_xticks(range(len(models)))
    ax4.set_xticklabels(short_models, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save visualization
    output_file = f"{output_dir}/soar_classification_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved visualization to {output_file}")

def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(description="Analyze SOAR dataset for overfitting classification")
    parser.add_argument("--limit", type=int, help="Limit number of programs to process")
    args = parser.parse_args()
    
    print("🚀 SOAR Dataset Classification Analysis")
    print("=" * 60)
    
    # Configuration
    max_reasoning_tokens = 8000
    max_workers = 20  # Increased parallel workers
    output_dir = "classification/data"
    
    if args.limit:
        print(f"📊 Processing limited to {args.limit} programs")
    
    start_time = time.time()
    
    try:
        # Load dataset
        df = load_soar_dataset(limit=args.limit)
        
        # Classify programs
        results = classify_programs(df, max_reasoning_tokens=max_reasoning_tokens, max_workers=max_workers)
        
        # Analyze results
        analysis_results = analyze_results(results, output_dir=output_dir)
        
        # Create visualizations
        create_visualizations(analysis_results, output_dir=output_dir)
        
        total_time = time.time() - start_time
        print(f"\n⏱️  Total analysis time: {total_time:.2f} seconds")
        print(f"⚡ Average time per program: {total_time/len(df):.2f} seconds")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()