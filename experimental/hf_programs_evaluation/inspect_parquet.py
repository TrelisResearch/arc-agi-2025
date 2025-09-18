#!/usr/bin/env python3
"""
Inspect the parquet file format first before processing.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_python.datasets.io import read_soar_parquet

def inspect_parquet_format(parquet_path: str):
    """Inspect the structure and format of the parquet file."""
    print(f"Inspecting parquet file: {parquet_path}")

    # Load the dataframe
    df = read_soar_parquet(parquet_path)

    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print(f"\nColumn types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")

    print(f"\nFirst few rows:")
    for idx, row in df.head().iterrows():
        print(f"\nRow {idx}:")
        for col in df.columns:
            value = row[col]
            if isinstance(value, str) and len(str(value)) > 100:
                print(f"  {col}: {str(value)[:100]}...")
            else:
                print(f"  {col}: {value}")

    print(f"\nSample code from first program:")
    if 'code' in df.columns and len(df) > 0:
        first_code = df.iloc[0]['code']
        print(first_code)

    return df

if __name__ == "__main__":
    parquet_file = "/Users/ronanmcgovern/TR/arc-agi-2025/llm_python/datasets/inference/20250918_151237_julien31_Soar-qwen-14b_arc-prize-2025_evaluation.parquet"
    df = inspect_parquet_format(parquet_file)