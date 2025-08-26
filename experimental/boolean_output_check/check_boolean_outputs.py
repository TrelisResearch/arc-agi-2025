#!/usr/bin/env python3
"""
Script to check for boolean outputs in predicted train/test outputs from parquet files.
Searches for programs that have predicted outputs containing booleans or arrays/lists of booleans.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Any, Union, List, Dict
import argparse


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
        'output_type': str(type(output).__name__)
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


def analyze_parquet_file(file_path: Path) -> List[Dict[str, Any]]:
    """
    Analyze a parquet file for boolean outputs.
    Returns a list of findings.
    """
    findings = []
    
    try:
        df = pd.read_parquet(file_path)
        print(f"Analyzing {file_path.name} with {len(df)} rows...")
        
        # Check required columns exist
        required_columns = ['predicted_train_output', 'predicted_test_output']
        available_columns = [col for col in required_columns if col in df.columns]
        
        if not available_columns:
            print(f"  Warning: No predicted output columns found in {file_path.name}")
            return findings
        
        for idx, row in df.iterrows():
            row_findings = {
                'file': file_path.name,
                'row_index': idx,
                'task_id': row.get('task_id', 'unknown'),
                'model': row.get('model', 'unknown'),
                'predicted_train_issues': {},
                'predicted_test_issues': {}
            }
            
            has_issues = False
            
            # Check predicted train output
            if 'predicted_train_output' in df.columns:
                train_output = row['predicted_train_output']
                train_check = contains_boolean_output(train_output)
                if any(train_check[key] for key in ['has_boolean', 'has_boolean_array', 'has_nested_boolean_array']):
                    row_findings['predicted_train_issues'] = train_check
                    has_issues = True
            
            # Check predicted test output
            if 'predicted_test_output' in df.columns:
                test_output = row['predicted_test_output']
                test_check = contains_boolean_output(test_output)
                if any(test_check[key] for key in ['has_boolean', 'has_boolean_array', 'has_nested_boolean_array']):
                    row_findings['predicted_test_issues'] = test_check
                    has_issues = True
            
            if has_issues:
                findings.append(row_findings)
        
        print(f"  Found {len(findings)} rows with boolean outputs")
        
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
    
    return findings


def main():
    parser = argparse.ArgumentParser(description='Check for boolean outputs in parquet files')
    parser.add_argument('--input-dir', type=str, 
                       default='llm_python/datasets/inference',
                       help='Directory containing parquet files to analyze')
    parser.add_argument('--output-file', type=str,
                       default='experimental/boolean_output_check/boolean_output_findings.json',
                       help='Output file for findings')
    parser.add_argument('--pattern', type=str, default='*.parquet',
                       help='File pattern to match')
    
    args = parser.parse_args()
    
    # Convert to Path objects
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    
    if not input_dir.exists():
        print(f"Input directory {input_dir} does not exist")
        return
    
    # Find all parquet files
    parquet_files = list(input_dir.glob(args.pattern))
    
    if not parquet_files:
        print(f"No parquet files found matching pattern '{args.pattern}' in {input_dir}")
        return
    
    print(f"Found {len(parquet_files)} parquet files to analyze")
    
    all_findings = []
    
    # Analyze each file
    for file_path in sorted(parquet_files):
        findings = analyze_parquet_file(file_path)
        all_findings.extend(findings)
    
    # Summary statistics
    total_issues = len(all_findings)
    train_issues = sum(1 for f in all_findings if f['predicted_train_issues'])
    test_issues = sum(1 for f in all_findings if f['predicted_test_issues'])
    
    # Group by issue type
    boolean_count = sum(1 for f in all_findings 
                       if (f['predicted_train_issues'].get('has_boolean') or 
                           f['predicted_test_issues'].get('has_boolean')))
    
    boolean_array_count = sum(1 for f in all_findings 
                             if (f['predicted_train_issues'].get('has_boolean_array') or 
                                 f['predicted_test_issues'].get('has_boolean_array')))
    
    nested_boolean_array_count = sum(1 for f in all_findings 
                                    if (f['predicted_train_issues'].get('has_nested_boolean_array') or 
                                        f['predicted_test_issues'].get('has_nested_boolean_array')))
    
    # Create summary
    summary = {
        'total_files_analyzed': len(parquet_files),
        'total_rows_with_boolean_outputs': total_issues,
        'rows_with_train_boolean_outputs': train_issues,
        'rows_with_test_boolean_outputs': test_issues,
        'breakdown': {
            'direct_boolean_values': boolean_count,
            'boolean_arrays': boolean_array_count,
            'nested_boolean_arrays': nested_boolean_array_count
        },
        'files_analyzed': [f.name for f in parquet_files]
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
    print("BOOLEAN OUTPUT ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total files analyzed: {summary['total_files_analyzed']}")
    print(f"Total rows with boolean outputs: {summary['total_rows_with_boolean_outputs']}")
    print(f"  - Rows with train boolean outputs: {summary['rows_with_train_boolean_outputs']}")
    print(f"  - Rows with test boolean outputs: {summary['rows_with_test_boolean_outputs']}")
    print(f"\nBreakdown by type:")
    print(f"  - Direct boolean values: {summary['breakdown']['direct_boolean_values']}")
    print(f"  - Boolean arrays: {summary['breakdown']['boolean_arrays']}")
    print(f"  - Nested boolean arrays: {summary['breakdown']['nested_boolean_arrays']}")
    print(f"\nResults saved to: {output_file}")
    
    if total_issues > 0:
        print(f"\nFirst few examples:")
        for i, finding in enumerate(all_findings[:3]):
            print(f"\n{i+1}. File: {finding['file']}, Row: {finding['row_index']}")
            print(f"   Task ID: {finding['task_id']}, Model: {finding['model']}")
            if finding['predicted_train_issues']:
                print(f"   Train issues: {finding['predicted_train_issues']}")
            if finding['predicted_test_issues']:
                print(f"   Test issues: {finding['predicted_test_issues']}")


if __name__ == "__main__":
    main()