#!/usr/bin/env python3
"""
Script to perform multiple iterations of program sampling and analysis.
"""
import random
from datasets import load_dataset

def sample_programs(n=10, iteration=1, seed=None):
    if seed:
        random.seed(seed)
    
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
        program_text = row['code']  # We know this is the code column
        
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
    # Run multiple iterations with different seeds for variety
    sample_programs(10, iteration=2, seed=42)