from datasets import load_dataset
import pandas as pd
import json
import time

max_retries = 3
for attempt in range(max_retries):
    try:
        print(f"Loading dataset (attempt {attempt + 1}/{max_retries})...")
        dataset = load_dataset("Trelis/arc-agi-1-perfect-50", split="train", streaming=False)
        print("Dataset loaded successfully!")
        break
    except Exception as e:
        print(f"Error on attempt {attempt + 1}: {e}")
        if attempt < max_retries - 1:
            print("Retrying in 5 seconds...")
            time.sleep(5)
        else:
            print("Failed to load dataset. Trying alternative approach...")
            # Try downloading just a subset first
            dataset = load_dataset("Trelis/arc-agi-1-perfect-50", split="train[:100]")

print(f"\nDataset info:")
print(f"Number of rows: {len(dataset)}")
print(f"Columns: {dataset.column_names}")

print("\n" + "="*50)
print("First row inspection:")
print("="*50)
first_row = dataset[0]
for col, val in first_row.items():
    if isinstance(val, (list, dict)):
        print(f"\n{col}:")
        if isinstance(val, list) and len(val) > 0:
            print(f"  Type: list of {type(val[0]).__name__ if val else 'empty'}")
            print(f"  Length: {len(val)}")
            if len(str(val)) < 200:
                print(f"  Value: {val}")
            else:
                print(f"  First element: {val[0] if val else 'N/A'}")
        else:
            print(f"  Type: {type(val).__name__}")
            if len(str(val)) < 200:
                print(f"  Value: {val}")
    else:
        print(f"\n{col}:")
        print(f"  Type: {type(val).__name__}")
        if len(str(val)) < 500:
            print(f"  Value: {val}")
        else:
            print(f"  Value (truncated): {str(val)[:500]}...")

df = dataset.to_pandas()
df.to_parquet('dataset.parquet')
print(f"\nDataset saved to dataset.parquet")