#!/usr/bin/env python3
"""
Debug program data structure.
"""

from datasets import load_dataset

def debug_programs():
    print("Loading HuggingFace dataset...")
    ds = load_dataset('Trelis/arc-agi-1-perfect-2')

    # Debug the dataset structure first
    print(f"Dataset structure: {ds}")
    print(f"Train split keys: {ds['train'].features}")

    # Get first 3 records
    for i in range(3):
        program_data = ds["train"][i]
        print(f"\nProgram {i+1}:")
        print(f"Type: {type(program_data)}")
        if hasattr(program_data, 'keys'):
            print(f"Keys: {list(program_data.keys())}")
            print(f"Code type: {type(program_data['code'])}")
            print(f"Code sample (first 200 chars): {program_data['code'][:200]}")
        else:
            print(f"Data: {program_data}")


if __name__ == "__main__":
    debug_programs()