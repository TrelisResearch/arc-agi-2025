#!/usr/bin/env python3
"""Validate the extracted parquet file."""

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

def validate_parquet(file_path):
    """Validate the parquet file and show key statistics."""
    
    print(f"\n📊 Validating {file_path}")
    print("=" * 60)
    
    # Read the parquet file
    df = pd.read_parquet(file_path)
    
    # Basic info
    print(f"\n📈 Basic Statistics:")
    print(f"  Total rows: {len(df):,}")
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Column info
    print(f"\n📋 Columns:")
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype}")
    
    # Data quality checks
    print(f"\n✅ Data Quality:")
    print(f"  Null values per column:")
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        if count > 0:
            print(f"    - {col}: {count:,} ({count/len(df)*100:.2f}%)")
    if null_counts.sum() == 0:
        print(f"    - No null values found!")
    
    # Unique values
    print(f"\n🔍 Unique Values:")
    print(f"  Unique task_ids: {df['task_id'].nunique():,}")
    print(f"  Unique models: {df['model'].nunique():,}")
    
    # Model distribution
    print(f"\n📊 Model Distribution:")
    model_counts = df['model'].value_counts()
    for model, count in model_counts.head(10).items():
        print(f"  - {model}: {count:,} ({count/len(df)*100:.2f}%)")
    
    # Sample data
    print(f"\n📝 Sample Data (first 3 rows):")
    print("-" * 60)
    
    # Show a few sample rows with key columns
    sample_cols = ['task_id', 'model']
    if 'code' in df.columns:
        # Show code snippet (first 100 chars)
        df_sample = df[sample_cols].head(3).copy()
        df_sample['code_snippet'] = df['code'].head(3).apply(lambda x: x[:100] + '...' if len(x) > 100 else x)
        print(df_sample.to_string())
    else:
        print(df[sample_cols].head(3).to_string())
    
    # Check correctness columns if they exist
    if 'correct_train_input' in df.columns and 'correct_test_input' in df.columns:
        print(f"\n🎯 Correctness Statistics:")
        
        # These columns contain lists, so we need to handle them carefully
        perfect_train = 0
        perfect_test = 0
        
        for idx in range(min(1000, len(df))):  # Sample first 1000 for speed
            row = df.iloc[idx]
            train_correct = row['correct_train_input']
            test_correct = row['correct_test_input']
            
            # Check if all training examples are correct
            if isinstance(train_correct, list) and all(train_correct):
                perfect_train += 1
            
            # Check if all test examples are correct
            if isinstance(test_correct, list) and all(test_correct):
                perfect_test += 1
        
        sample_size = min(1000, len(df))
        print(f"  Sample of {sample_size:,} programs:")
        print(f"    - Programs with 100% train accuracy: {perfect_train:,} ({perfect_train/sample_size*100:.1f}%)")
        print(f"    - Programs with 100% test accuracy: {perfect_test:,} ({perfect_test/sample_size*100:.1f}%)")
    
    # Random sample of task_ids to verify they look correct
    print(f"\n🔤 Sample Task IDs:")
    sample_tasks = df['task_id'].sample(min(10, len(df))).tolist()
    for task_id in sample_tasks[:5]:
        print(f"  - {task_id}")
    
    # Check for any obvious data issues
    print(f"\n⚠️  Potential Issues:")
    issues_found = False
    
    # Check for very short code
    if 'code' in df.columns:
        short_code = df[df['code'].str.len() < 50]
        if len(short_code) > 0:
            print(f"  - {len(short_code):,} programs with code < 50 characters")
            issues_found = True
    
    # Check for duplicate (task_id, code) pairs
    if 'code' in df.columns:
        duplicates = df.duplicated(subset=['task_id', 'code'], keep=False)
        if duplicates.sum() > 0:
            print(f"  - {duplicates.sum():,} duplicate (task_id, code) pairs")
            issues_found = True
    
    if not issues_found:
        print(f"  - No major issues detected!")
    
    print("\n" + "=" * 60)
    print("✅ Validation complete!")
    
    return df

if __name__ == "__main__":
    file_path = "all_programs_20250812.parquet"
    
    if not Path(file_path).exists():
        print(f"❌ File {file_path} not found!")
    else:
        df = validate_parquet(file_path)