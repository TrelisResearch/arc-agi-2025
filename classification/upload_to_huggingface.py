#!/usr/bin/env python3
"""
Upload the enhanced SOAR dataset with classifications to Hugging Face.
"""

import json
import pandas as pd
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
import os
from pathlib import Path
import argparse

def load_classification_results():
    """Load our classification results."""
    results_file = "classification/data/soar_classification_results.json"
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Extract the raw results which contain all program data
    raw_results = data['raw_results']
    
    # Create DataFrame with the results
    df_results = pd.DataFrame(raw_results)
    
    print(f"✅ Loaded {len(df_results)} classification results")
    print(f"   Columns: {list(df_results.columns)}")
    
    return df_results

def create_enhanced_dataset(limit: int = None):
    """Create enhanced dataset by merging original with classifications."""
    print("📁 Loading original SOAR dataset...")
    
    # Load original dataset
    original_dataset = load_dataset("Trelis/soar-program-samples")
    df_original = original_dataset['train'].to_pandas()
    
    # Apply limit if specified
    if limit:
        df_original = df_original.head(limit)
        print(f"✅ Original dataset (limited): {len(df_original)} programs")
    else:
        print(f"✅ Original dataset: {len(df_original)} programs")
    
    # Load our classification results
    df_results = load_classification_results()
    
    # Prepare results for merging (keep only what we need)
    df_classifications = df_results[['task_id', 'model', 'code', 'classification', 'full_response']].copy()
    df_classifications = df_classifications.rename(columns={
        'classification': 'overfitting_classification',
        'full_response': 'gemini_reasoning'
    })
    
    # Merge datasets on task_id, model, and code to ensure exact matching
    print("🔄 Merging datasets...")
    df_enhanced = df_original.merge(
        df_classifications, 
        on=['task_id', 'model', 'code'], 
        how='left'
    )
    
    # Check merge quality
    missing_classifications = df_enhanced['overfitting_classification'].isna().sum()
    if missing_classifications > 0:
        print(f"⚠️  {missing_classifications} programs missing classifications")
    else:
        print("✅ All programs have classifications!")
    
    print(f"📊 Enhanced dataset: {len(df_enhanced)} programs")
    print(f"   New columns: overfitting_classification, gemini_reasoning")
    
    # Show classification distribution
    if 'overfitting_classification' in df_enhanced.columns:
        class_dist = df_enhanced['overfitting_classification'].value_counts()
        print(f"\n📈 Classification Distribution:")
        for classification, count in class_dist.items():
            pct = (count / len(df_enhanced)) * 100
            print(f"   {classification}: {count} ({pct:.1f}%)")
    
    return df_enhanced

def upload_to_huggingface(df: pd.DataFrame, repo_name: str = "Trelis/soar-program-samples-classification", subset_size: str = None):
    """Upload the enhanced dataset to Hugging Face."""
    print(f"🚀 Uploading to Hugging Face: {repo_name}")
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df)
    
    # Create repo description based on subset
    if subset_size == "100":
        total_programs = len(df)
        overfitting_count = (df['overfitting_classification'] == 'overfitting').sum()
        general_count = (df['overfitting_classification'] == 'general').sum()
        overfitting_rate = (overfitting_count / total_programs) * 100
        general_rate = (general_count / total_programs) * 100
        unique_tasks = df['task_id'].nunique()
        
        description = f"""# SOAR Program Samples with Overfitting Classification (100-Program Subset)

This dataset is a 100-program subset of the original `Trelis/soar-program-samples` with AI-powered classification of ARC program solutions. This smaller dataset is useful for testing and development.

## New Columns

- `overfitting_classification`: Classification as "overfitting" or "general" 
- `gemini_reasoning`: Full reasoning from Gemini 2.5 Flash explaining the classification

## Classification Methodology

Each program was analyzed using Gemini 2.5 Flash with 8k reasoning tokens to determine if the solution:

- **overfitting**: Uses hardcoded rules, specific dimensions, magic numbers that only work for particular test cases
- **general**: Uses algorithmic approaches, pattern detection, adaptable logic that could work across different inputs

## Statistics (100-Program Subset)

- **Total Programs**: {total_programs} across {unique_tasks} unique ARC tasks
- **Overfitting Rate**: {overfitting_rate:.1f}% ({overfitting_count}/{total_programs} programs)
- **General Rate**: {general_rate:.1f}% ({general_count}/{total_programs} programs)
- **Models Analyzed**: Mistral-Large-Instruct-2407, Qwen2.5-72B-Instruct, Qwen2.5-Coder variants

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("Trelis/soar-program-samples-classification-100")
df = dataset['train'].to_pandas()

# Analyze overfitting by task
task_stats = df.groupby('task_id')['overfitting_classification'].value_counts()

# Analyze overfitting by model  
model_stats = df.groupby('model')['overfitting_classification'].value_counts()
```

## Citation

If you use this dataset, please cite the original SOAR dataset and mention the overfitting classification methodology.
"""
    else:
        description = """# SOAR Program Samples with Overfitting Classification

This dataset extends the original `Trelis/soar-program-samples` with AI-powered classification of ARC program solutions.

## New Columns

- `overfitting_classification`: Classification as "overfitting" or "general" 
- `gemini_reasoning`: Full reasoning from Gemini 2.5 Flash explaining the classification

## Classification Methodology

Each program was analyzed using Gemini 2.5 Flash with 8k reasoning tokens to determine if the solution:

- **overfitting**: Uses hardcoded rules, specific dimensions, magic numbers that only work for particular test cases
- **general**: Uses algorithmic approaches, pattern detection, adaptable logic that could work across different inputs

## Statistics

- **Total Programs**: 620 across 32 unique ARC tasks
- **Overfitting Rate**: 51.6% (320/620 programs)  
- **General Rate**: 48.4% (300/620 programs)
- **Models Analyzed**: Mistral-Large-Instruct-2407, Qwen2.5-72B-Instruct, Qwen2.5-Coder variants

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("Trelis/soar-program-samples-classification")
df = dataset['train'].to_pandas()

# Analyze overfitting by task
task_stats = df.groupby('task_id')['overfitting_classification'].value_counts()

# Analyze overfitting by model  
model_stats = df.groupby('model')['overfitting_classification'].value_counts()
```

## Citation

If you use this dataset, please cite the original SOAR dataset and mention the overfitting classification methodology.
"""
    
    try:
        # Push to Hugging Face
        if subset_size:
            commit_msg = f"Add overfitting classification and Gemini reasoning ({subset_size}-program subset)"
        else:
            commit_msg = "Add overfitting classification and Gemini reasoning"
            
        dataset.push_to_hub(
            repo_name,
            commit_message=commit_msg
        )
        
        print(f"✅ Successfully uploaded to {repo_name}")
        print(f"🔗 View at: https://huggingface.co/datasets/{repo_name}")
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        print("💡 Make sure you're logged in: huggingface-cli login")
        raise

def main():
    """Main upload pipeline."""
    parser = argparse.ArgumentParser(description="Upload SOAR dataset with classifications to Hugging Face")
    parser.add_argument("--subset", choices=["100"], help="Upload subset of the dataset (e.g., 100 for first 100 programs)")
    args = parser.parse_args()
    
    if args.subset == "100":
        print("🚀 SOAR Dataset Enhancement & Upload (100-program subset)")
        repo_name = "Trelis/soar-program-samples-classification-100"
        output_file = "classification/data/soar_enhanced_dataset_100.csv"
    else:
        print("🚀 SOAR Dataset Enhancement & Upload")
        repo_name = "Trelis/soar-program-samples-classification"
        output_file = "classification/data/soar_enhanced_dataset.csv"
    
    print("=" * 50)
    
    try:
        # Create enhanced dataset
        limit = int(args.subset) if args.subset else None
        df_enhanced = create_enhanced_dataset(limit=limit)
        
        # Save locally first
        df_enhanced.to_csv(output_file, index=False)
        print(f"💾 Saved enhanced dataset locally: {output_file}")
        
        # Upload to Hugging Face
        upload_to_huggingface(df_enhanced, repo_name=repo_name, subset_size=args.subset)
        
    except Exception as e:
        print(f"❌ Process failed: {e}")
        raise

if __name__ == "__main__":
    main()