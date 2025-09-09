#!/usr/bin/env python3
"""
Script to explore the Trelis/arc-agi-2-mixed-finetuning-20 dataset and analyze program styles.
"""
import random
import pandas as pd
from datasets import load_dataset

def explore_dataset():
    print("Loading dataset...")
    dataset = load_dataset("Trelis/arc-agi-2-mixed-finetuning-20")
    
    # Print dataset info
    print(f"Dataset keys: {list(dataset.keys())}")
    
    # Get the first split (usually 'train')
    split_name = list(dataset.keys())[0]
    data = dataset[split_name]
    
    print(f"Dataset size: {len(data)}")
    print(f"Dataset columns: {data.column_names}")
    
    # Show a few examples of the column structure
    print("\nFirst 3 rows (structure only):")
    for i in range(min(3, len(data))):
        row = data[i]
        print(f"Row {i}:")
        for col in data.column_names:
            if isinstance(row[col], str) and len(row[col]) > 100:
                print(f"  {col}: {row[col][:100]}...")
            else:
                print(f"  {col}: {row[col]}")
        print()

def sample_programs(n=10, iteration=1):
    print(f"\n{'='*50}")
    print(f"ITERATION {iteration}: Sampling {n} random programs")
    print(f"{'='*50}")
    
    dataset = load_dataset("Trelis/arc-agi-2-mixed-finetuning-20")
    split_name = list(dataset.keys())[0]
    data = dataset[split_name]
    
    # Sample random indices
    indices = random.sample(range(len(data)), n)
    
    programs = []
    for i, idx in enumerate(indices):
        row = data[idx]
        
        # Find the column that contains the program code
        program_text = None
        for col in data.column_names:
            if 'program' in col.lower() or 'code' in col.lower() or 'solution' in col.lower():
                if isinstance(row[col], str) and ('import' in row[col] or 'def' in row[col] or 'numpy' in row[col] or 'np.' in row[col]):
                    program_text = row[col]
                    break
        
        # If no obvious program column, look for any column with substantial code-like content
        if program_text is None:
            for col in data.column_names:
                if isinstance(row[col], str) and len(row[col]) > 50:
                    if any(keyword in row[col] for keyword in ['import', 'def ', 'numpy', 'np.', 'array', 'return']):
                        program_text = row[col]
                        break
        
        if program_text:
            programs.append(program_text)
            print(f"\nProgram {i+1} (index {idx}):")
            print("-" * 40)
            print(program_text)
            print("-" * 40)
        else:
            print(f"No program found in row {idx}")
    
    return programs

if __name__ == "__main__":
    # First explore the dataset structure
    explore_dataset()
    
    # Then sample programs for analysis
    programs_iter1 = sample_programs(10, 1)