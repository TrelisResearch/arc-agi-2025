"""
Check if any rows in the parquet file use refinement.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_python.datasets.io import read_soar_parquet
import pandas as pd

def main():
    # Read the parquet file
    parquet_path = Path("/Users/ronanmcgovern/TR/arc-agi-2025/llm_python/datasets/inference/20250902_122615_openai_gpt-oss-120b_arc-prize-2025_training-hard.parquet")
    
    print(f"Reading parquet file: {parquet_path}")
    df = read_soar_parquet(parquet_path)
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for refinement column
    if 'refined_from_id' in df.columns:
        print(f"\n✓ Found 'refined_from_id' column")
        
        # Check how many rows have refinement
        non_null_refinements = df['refined_from_id'].notna().sum()
        print(f"\nRows with refinement (non-null refined_from_id): {non_null_refinements}/{len(df)}")
        
        if non_null_refinements > 0:
            print("\nFirst few refinement IDs:")
            refined_rows = df[df['refined_from_id'].notna()]
            for idx, row in refined_rows.head(10).iterrows():
                print(f"  Row {idx}: task_id={row['task_id']}, refined_from_id={row['refined_from_id']}")
        else:
            print("No rows use refinement (all refined_from_id values are null/NA)")
    else:
        print("\n✗ No 'refined_from_id' column found")
        print("Available columns:", df.columns.tolist())
    
    # Also check data types
    print(f"\nColumn data types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")

if __name__ == "__main__":
    main()