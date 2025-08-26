#!/usr/bin/env python3
"""
Script to check for boolean outputs in the Hugging Face dataset Trelis/arc-agi-1-partial-100.
Analyzes predicted_train_output and predicted_test_output columns for boolean values.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Any, Union, List, Dict
import argparse
from datasets import load_dataset


def is_boolean_value(value: Any) -> bool:
    """Check if a value is a boolean."""
    return isinstance(value, bool) or isinstance(value, np.bool_)


def is_array_of_booleans(value: Any) -> bool:
    """Check if a value is an array/list containing only booleans."""
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return False
        return all(is_boolean_value(item) for item in value)
    return False


def is_nested_array_of_booleans(value: Any) -> bool:
    """Check if a value is a nested array/list containing only booleans."""
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return False
        for item in value:
            if isinstance(item, (list, tuple, np.ndarray)):
                if not is_array_of_booleans(item):
                    return False
            else:
                if not is_boolean_value(item):
                    return False
        return True
    return False


def contains_boolean_output(output: Any) -> Dict[str, bool]:
    """
    Check if an output contains boolean values.
    Returns a dict with flags for different types of boolean content.
    """
    result = {
        'has_boolean': False,
        'has_boolean_array': False,
        'has_nested_boolean_array': False,
        'output_type': str(type(output).__name__),
        'sample_value': str(output)[:100] if output is not None else None
    }
    
    if output is None:
        return result
    
    # Check if the output itself is a boolean
    if is_boolean_value(output):
        result['has_boolean'] = True
        return result
    
    # Check if it's an array of booleans
    if is_array_of_booleans(output):
        result['has_boolean_array'] = True
        return result
    
    # Check if it's a nested array of booleans
    if is_nested_array_of_booleans(output):
        result['has_nested_boolean_array'] = True
        return result
    
    # For complex structures, recursively check
    if isinstance(output, (list, tuple)):
        for item in output:
            item_check = contains_boolean_output(item)
            if any(item_check[key] for key in ['has_boolean', 'has_boolean_array', 'has_nested_boolean_array']):
                result.update(item_check)
                return result
    
    return result


def analyze_hf_dataset(dataset_name: str = "Trelis/arc-agi-1-partial-100") -> List[Dict[str, Any]]:
    """
    Analyze the Hugging Face dataset for boolean outputs.
    Returns a list of findings.
    """
    findings = []
    
    print(f"Loading dataset: {dataset_name}")
    try:
        # Load the dataset
        dataset = load_dataset(dataset_name, split="train")
        df = dataset.to_pandas()
        
        print(f"Dataset loaded with {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        
        # Check which predicted output columns exist
        predicted_columns = [col for col in df.columns if 'predicted' in col.lower() and 'output' in col.lower()]
        print(f"Found predicted output columns: {predicted_columns}")
        
        if not predicted_columns:
            print("No predicted output columns found!")
            return findings
        
        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                print(f"Processing row {idx}/{len(df)}")
                
            row_findings = {
                'row_index': idx,
                'task_id': row.get('task_id', row.get('correct_test_input', 'unknown')),
                'model': row.get('model', 'unknown'),
                'is_transductive': row.get('is_transductive', 'unknown'),
                'issues': {}
            }
            
            has_issues = False
            
            # Check all predicted output columns
            for col in predicted_columns:
                if col in df.columns:
                    output = row[col]
                    output_check = contains_boolean_output(output)
                    if any(output_check[key] for key in ['has_boolean', 'has_boolean_array', 'has_nested_boolean_array']):
                        row_findings['issues'][col] = output_check
                        has_issues = True
            
            if has_issues:
                findings.append(row_findings)
        
        print(f"Found {len(findings)} rows with boolean outputs")
        
    except Exception as e:
        print(f"Error loading/analyzing dataset: {e}")
        import traceback
        traceback.print_exc()
    
    return findings


def main():
    parser = argparse.ArgumentParser(description='Check for boolean outputs in HF dataset')
    parser.add_argument('--dataset', type=str, 
                       default='Trelis/arc-agi-1-partial-100',
                       help='Hugging Face dataset name to analyze')
    parser.add_argument('--output-file', type=str,
                       default='experimental/boolean_output_check/hf_dataset_boolean_findings.json',
                       help='Output file for findings')
    
    args = parser.parse_args()
    
    output_file = Path(args.output_file)
    
    print(f"Analyzing HuggingFace dataset: {args.dataset}")
    
    all_findings = analyze_hf_dataset(args.dataset)
    
    # Summary statistics
    total_issues = len(all_findings)
    
    # Group by issue type
    boolean_count = sum(1 for f in all_findings 
                       if any(issue.get('has_boolean', False) for issue in f['issues'].values()))
    
    boolean_array_count = sum(1 for f in all_findings 
                             if any(issue.get('has_boolean_array', False) for issue in f['issues'].values()))
    
    nested_boolean_array_count = sum(1 for f in all_findings 
                                    if any(issue.get('has_nested_boolean_array', False) for issue in f['issues'].values()))
    
    # Group by column
    column_stats = {}
    for finding in all_findings:
        for col, issue in finding['issues'].items():
            if col not in column_stats:
                column_stats[col] = 0
            column_stats[col] += 1
    
    # Create summary
    summary = {
        'dataset_analyzed': args.dataset,
        'total_rows_with_boolean_outputs': total_issues,
        'breakdown': {
            'direct_boolean_values': boolean_count,
            'boolean_arrays': boolean_array_count,
            'nested_boolean_arrays': nested_boolean_array_count
        },
        'issues_by_column': column_stats
    }
    
    # Save results
    results = {
        'summary': summary,
        'findings': all_findings
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("HF DATASET BOOLEAN OUTPUT ANALYSIS")
    print("="*60)
    print(f"Dataset: {summary['dataset_analyzed']}")
    print(f"Total rows with boolean outputs: {summary['total_rows_with_boolean_outputs']}")
    print(f"\nBreakdown by type:")
    print(f"  - Direct boolean values: {summary['breakdown']['direct_boolean_values']}")
    print(f"  - Boolean arrays: {summary['breakdown']['boolean_arrays']}")
    print(f"  - Nested boolean arrays: {summary['breakdown']['nested_boolean_arrays']}")
    print(f"\nIssues by column:")
    for col, count in summary['issues_by_column'].items():
        print(f"  - {col}: {count}")
    print(f"\nResults saved to: {output_file}")
    
    if total_issues > 0:
        print(f"\nFirst few examples:")
        for i, finding in enumerate(all_findings[:5]):
            print(f"\n{i+1}. Row: {finding['row_index']}")
            print(f"   Task ID: {finding['task_id']}, Model: {finding['model']}")
            print(f"   Is Transductive: {finding['is_transductive']}")
            for col, issue in finding['issues'].items():
                print(f"   {col}: {issue}")


if __name__ == "__main__":
    main()