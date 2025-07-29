#!/usr/bin/env python3
"""
Simple script to load a parquet file and push it to Hugging Face Hub as a dataset.
Assumes user is logged in or has HF_TOKEN environment variable set.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from datasets import Dataset


def load_parquet_and_push_to_hf(input_file: str, dataset_name: str, private: bool = False):
    """
    Load a parquet file and push it to Hugging Face Hub as a dataset.
    
    Args:
        input_file (str): Path to the input parquet file
        dataset_name (str): Name of the dataset on Hugging Face Hub (format: username/dataset-name)
        private (bool): Whether to make the dataset private (default: False)
    """
    # Validate input file exists
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)
    
    if not input_path.suffix.lower() == '.parquet':
        print(f"Error: Input file '{input_file}' is not a parquet file.")
        sys.exit(1)
    
    print(f"Loading parquet file: {input_file}")
    try:
        # Load the parquet file using pandas
        df = pd.read_parquet(input_file)
        print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        
        # Convert pandas DataFrame to Hugging Face Dataset
        dataset = Dataset.from_pandas(df)
        print("Converted to Hugging Face dataset")
        
        # Push to Hugging Face Hub
        print(f"Pushing dataset to Hugging Face Hub: {dataset_name}")
        dataset.push_to_hub(
            dataset_name,
            private=private
        )
        print(f"Successfully pushed dataset to: https://huggingface.co/datasets/{dataset_name}")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Load a parquet file and push it to Hugging Face Hub as a dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "input_file",
        help="Path to the input parquet file"
    )
    
    parser.add_argument(
        "dataset_name",
        help="Name of the dataset on Hugging Face Hub (format: username/dataset-name)"
    )
    
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private (default: public)"
    )
    
    args = parser.parse_args()
    
    # Validate dataset name format
    if "/" not in args.dataset_name:
        print("Error: Dataset name must be in format 'username/dataset-name'")
        sys.exit(1)
    
    load_parquet_and_push_to_hf(
        input_file=args.input_file,
        dataset_name=args.dataset_name,
        private=args.private
    )


if __name__ == "__main__":
    main()
