import sys
from pathlib import Path
project_root = next((parent for parent in [Path.cwd()] + list(Path.cwd().parents) if (parent / "pyproject.toml").exists()), Path.cwd())
sys.path.append(str(project_root))

import pandas as pd
from datasets import load_dataset
import json
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

from llm_python.transduction.augmentation_classifier import detect_transduction_augmentation
from llm_python.transduction.code_based_classifier import classify_transductive_program
from llm_python.utils.arc_tester import ArcTester
from llm_python.utils.task_loader import TaskLoader, TaskData, get_task_loader

def download_and_filter_dataset(dataset_name: str):
    """Download and filter dataset for specific models"""
    print(f"\nDownloading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train")
    
    df = dataset.to_pandas()
    
    models_to_keep = [
        "Qwen2.5-72B-Instruct",
        "Mistral-Large-Instruct-2407", 
        "Qwen2.5-Coder-32B-Instruct"
    ]
    
    print(f"Original dataset size: {len(df)}")
    filtered_df = df[df['model'].isin(models_to_keep)]
    print(f"Filtered dataset size: {len(filtered_df)}")
    
    model_counts = filtered_df['model'].value_counts()
    print("\nRows per model:")
    for model, count in model_counts.items():
        print(f"  {model}: {count}")
    
    return filtered_df

def analyze_program_transduction(row, task_loader, arc_tester):
    """Apply both transduction detection methods to a single program"""
    task_id = row.get('task_id')
    program = row.get('code')
    
    if not task_id or not program:
        return {
            'aug_transductive': None,
            'aug_reason': 'Missing task_id or code',
            'code_transductive': None,
            'code_confidence': None,
            'error': 'Missing required fields'
        }
    
    try:
        # Get task data
        task_data = task_loader.get_task(task_id)
        
        # Method 1: Augmentation method (augmentation invariance)
        aug_transductive, aug_reason = detect_transduction_augmentation(
            program, task_data, arc_tester, debug=False
        )
        
        # Method 2: Code-based method (ML classifier on program features)
        code_transductive, code_confidence = classify_transductive_program(program)
        
        return {
            'aug_transductive': aug_transductive,
            'aug_reason': aug_reason,
            'code_transductive': code_transductive,
            'code_confidence': code_confidence,
            'error': None
        }
        
    except FileNotFoundError:
        return {
            'aug_transductive': None,
            'aug_reason': 'Task not found',
            'code_transductive': None,
            'code_confidence': None,
            'error': 'Task not found'
        }
    except Exception as e:
        return {
            'aug_transductive': None,
            'aug_reason': f'Error: {str(e)}',
            'code_transductive': None,
            'code_confidence': None,
            'error': str(e)
        }

def analyze_dataset(dataset_name: str):
    """Main function to analyze a Hugging Face dataset"""
    print("="*80)
    print(f"TRANSDUCTION ANALYSIS FOR: {dataset_name}")
    print("="*80)
    
    # Download and filter dataset
    df = download_and_filter_dataset(dataset_name)
    
    # Check if required columns exist
    if 'task_id' not in df.columns or 'code' not in df.columns:
        print("\nERROR: Dataset missing required columns (task_id, code)")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    # Initialize task loader and arc tester
    print("\nInitializing TaskLoader...")
    task_loader = get_task_loader()
    print(f"✓ TaskLoader initialized with {len(task_loader.tasks)} tasks")
    
    arc_tester = ArcTester(timeout=2)
    
    # Filter to only programs with available task data
    unique_task_ids = set(df['task_id'].unique())
    available_tasks = set(task_loader.tasks.keys())
    covered_tasks = unique_task_ids.intersection(available_tasks)
    missing_tasks = unique_task_ids - available_tasks
    
    print(f"✓ Task coverage: {len(covered_tasks)}/{len(unique_task_ids)} ({100*len(covered_tasks)/len(unique_task_ids) if unique_task_ids else 0:.1f}%)")
    if missing_tasks:
        print(f"⚠️  Missing {len(missing_tasks)} tasks: {list(missing_tasks)[:5]}...")
    
    df_filtered = df[df['task_id'].isin(available_tasks)].copy()
    print(f"✓ Filtered to {len(df_filtered)} programs with task data available")
    
    if len(df_filtered) == 0:
        print("ERROR: No programs with available task data")
        return None
    
    # Apply analysis to each program
    print(f"\nAnalyzing {len(df_filtered)} programs...")
    
    # Process in batches to show progress
    results_list = []
    batch_size = 100
    
    for i in tqdm(range(0, len(df_filtered), batch_size), desc="Processing batches"):
        batch = df_filtered.iloc[i:i+batch_size]
        for _, row in batch.iterrows():
            result = analyze_program_transduction(row, task_loader, arc_tester)
            results_list.append(result)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Add results to original dataframe
    for col in results_df.columns:
        df_filtered[col] = results_df[col].values
    
    # Clean up ArcTester
    ArcTester.cleanup_executor()
    
    # Filter out errors for statistics
    df_valid = df_filtered[df_filtered['error'].isna()].copy()
    
    print(f"\n✓ Analysis complete")
    print(f"✓ Success rate: {len(df_valid)}/{len(df_filtered)} ({100*len(df_valid)/len(df_filtered) if df_filtered.shape[0] > 0 else 0:.1f}%)")
    
    # Calculate statistics
    print("\n" + "="*60)
    print("TRANSDUCTION DETECTION RESULTS")
    print("="*60)
    
    if len(df_valid) > 0:
        # Method 1: Augmentation
        aug_count = df_valid['aug_transductive'].sum()
        print(f"\nMETHOD 1 - AUGMENTATION (Invariance Testing):")
        print(f"Total transductive: {aug_count}/{len(df_valid)} ({aug_count/len(df_valid)*100:.2f}%)")
        
        print("\nBreakdown by model:")
        for model in df_valid['model'].unique():
            model_df = df_valid[df_valid['model'] == model]
            trans_count = model_df['aug_transductive'].sum()
            print(f"  {model}: {trans_count}/{len(model_df)} ({trans_count/len(model_df)*100:.2f}%)")
        
        # Method 2: Code-based
        code_count = df_valid['code_transductive'].sum()
        print(f"\nMETHOD 2 - CODE-BASED (ML Classifier):")
        print(f"Total transductive: {code_count}/{len(df_valid)} ({code_count/len(df_valid)*100:.2f}%)")
        print(f"Average confidence: {df_valid['code_confidence'].mean():.3f}")
        
        print("\nBreakdown by model:")
        for model in df_valid['model'].unique():
            model_df = df_valid[df_valid['model'] == model]
            trans_count = model_df['code_transductive'].sum()
            avg_conf = model_df['code_confidence'].mean()
            print(f"  {model}: {trans_count}/{len(model_df)} ({trans_count/len(model_df)*100:.2f}%), avg conf: {avg_conf:.3f}")
        
        # Agreement analysis
        both_transductive = df_valid['aug_transductive'] & df_valid['code_transductive']
        either_transductive = df_valid['aug_transductive'] | df_valid['code_transductive']
        agreement = (df_valid['aug_transductive'] == df_valid['code_transductive']).mean()
        
        print(f"\n--- METHOD AGREEMENT ---")
        print(f"Methods agree: {agreement*100:.1f}%")
        print(f"Both detect transductive: {both_transductive.sum()}/{len(df_valid)} ({both_transductive.sum()/len(df_valid)*100:.2f}%)")
        print(f"Either detects transductive: {either_transductive.sum()}/{len(df_valid)} ({either_transductive.sum()/len(df_valid)*100:.2f}%)")
        
        # Save results
        output_file = f'experimental/transduction_hf_dataset/{dataset_name.replace("/", "_")}_results.csv'
        df_valid.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to {output_file}")
    
    return df_valid

def main():
    """Analyze both datasets"""
    # Dataset 1
    df1 = analyze_dataset("Trelis/arc-programs-correct-50")
    
    # Dataset 2  
    df2 = analyze_dataset("Trelis/arc-programs-50-full-200-partial")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()