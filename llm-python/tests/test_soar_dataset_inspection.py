#!/usr/bin/env python3

import json
from datasets import load_dataset

def inspect_soar_dataset():
    """Inspect the first row of the SOAR dataset to understand its structure"""
    
    print("Loading SOAR dataset (streaming mode)...")
    
    # Load dataset in streaming mode to avoid downloading everything
    dataset = load_dataset("julien31/soar_arc_train_5M", streaming=True, split="train")
    
    # Get the first row
    first_row = next(iter(dataset))
    
    print("\n" + "="*80)
    print("FIRST ROW OF SOAR DATASET")
    print("="*80)
    
    # Print the structure
    print(f"Keys in dataset: {list(first_row.keys())}")
    print()
    
    # Print each field with its type and a preview
    for key, value in first_row.items():
        print(f"Field: {key}")
        print(f"Type: {type(value).__name__}")
        
        if isinstance(value, str):
            if len(value) > 200:
                print(f"Value: {value[:200]}... (truncated, total length: {len(value)})")
            else:
                print(f"Value: {value}")
        elif isinstance(value, list):
            print(f"Value: {value} (length: {len(value)})")
        else:
            print(f"Value: {value}")
        print()
    
    # Analyze the structure more deeply
    print("="*80)
    print("STRUCTURE ANALYSIS")
    print("="*80)
    
    # Check if we have the expected fields
    expected_fields = ['code', 'correct_train_input', 'predicted_train_output', 
                      'correct_test_input', 'predicted_test_output', 'task_id', 'model', 'generation']
    
    missing_fields = [field for field in expected_fields if field not in first_row]
    extra_fields = [field for field in first_row.keys() if field not in expected_fields]
    
    print(f"Expected fields present: {len(expected_fields) - len(missing_fields)}/{len(expected_fields)}")
    if missing_fields:
        print(f"Missing fields: {missing_fields}")
    if extra_fields:
        print(f"Extra fields: {extra_fields}")
    
    # Check data types and content
    print("\nField Analysis:")
    for field in expected_fields:
        if field in first_row:
            value = first_row[field]
            if isinstance(value, list):
                print(f"  {field}: list with {len(value)} items")
                if value and isinstance(value[0], list):
                    print(f"    - Nested list structure: {len(value[0])} items in first sublist")
            else:
                print(f"  {field}: {type(value).__name__} = {value}")
        else:
            print(f"  {field}: MISSING")
    
    # Check if we have grid data (train_input, train_output, test_input, test_output)
    grid_fields = ['train_input', 'train_output', 'test_input', 'test_output']
    print(f"\nGrid data fields present: {[f for f in grid_fields if f in first_row]}")
    
    return first_row

if __name__ == "__main__":
    try:
        first_row = inspect_soar_dataset()
        
        # Save the first row to a file for detailed inspection
        with open("tests/soar_first_row.json", "w") as f:
            json.dump(first_row, f, indent=2)
        print("\nFirst row saved to tests/soar_first_row.json for detailed inspection")
        
    except Exception as e:
        print(f"Error inspecting SOAR dataset: {e}")
        import traceback
        traceback.print_exc() 