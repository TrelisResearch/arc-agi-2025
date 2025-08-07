#!/usr/bin/env python3
"""
Efficient script to upload partitioned parquet files to Hugging Face Hub as a dataset.
Uses direct file upload without loading/concatenating in memory.
Assumes user is logged in or has HF_TOKEN environment variable set.
"""

import argparse
import sys
from pathlib import Path
import glob
from typing import List
from huggingface_hub import HfApi, create_repo


def find_parquet_files(input_pattern: str) -> List[str]:
    """
    Find parquet files using a glob pattern or single file path.
    
    Args:
        input_pattern (str): File path or glob pattern to match parquet files
        
    Returns:
        List[str]: List of parquet file paths
    """
    # If it's a direct file path and exists, return it
    if Path(input_pattern).exists() and Path(input_pattern).is_file():
        if not input_pattern.lower().endswith('.parquet'):
            print(f"Error: File '{input_pattern}' is not a parquet file.")
            sys.exit(1)
        return [input_pattern]
    
    # Use glob to find files
    files = glob.glob(input_pattern)
    
    # Filter to only parquet files
    parquet_files = [f for f in files if f.lower().endswith('.parquet')]
    
    if not parquet_files:
        print(f"Error: No parquet files found matching pattern '{input_pattern}'")
        sys.exit(1)
    
    # Sort files for consistent ordering
    parquet_files.sort()
    
    print(f"Found {len(parquet_files)} parquet files:")
    for f in parquet_files:
        print(f"  - {f}")
    
    return parquet_files


def upload_parquet_files(file_paths: List[str], dataset_name: str, private: bool = False):
    """
    Upload parquet files directly to HuggingFace repository without loading into memory.
    This is much more efficient than the previous approach of loading, concatenating and re-uploading.
    
    Args:
        file_paths (List[str]): List of paths to parquet files
        dataset_name (str): Name of the dataset on Hugging Face Hub
        private (bool): Whether to make the dataset private
    """
    print(f"Uploading {len(file_paths)} parquet file(s) directly to HuggingFace Hub")
    print("📝 Note: HuggingFace will automatically treat all uploaded parquet files as a single dataset")
    
    try:
        # Initialize HuggingFace API
        api = HfApi()
        
        # Create repository if it doesn't exist
        print(f"Creating/accessing repository: {dataset_name}")
        try:
            repo_info = create_repo(dataset_name, repo_type="dataset", private=private, exist_ok=True)
            print(f"✓ Repository ready: {repo_info}")
        except Exception as e:
            print(f"Repository creation/access: {e}")
        
        # Upload each parquet file directly without loading into memory
        for i, file_path in enumerate(file_paths):
            file_name = Path(file_path).name
            print(f"\nUploading file {i+1}/{len(file_paths)}: {file_name}")
            
            try:
                # Upload file directly to the repository
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=file_name,
                    repo_id=dataset_name,
                    repo_type="dataset"
                )
                print(f"  ✓ Uploaded {file_name}")
            except Exception as e:
                print(f"  ✗ Failed to upload {file_name}: {e}")
                raise
        
        print("\n🎉 All files uploaded successfully!")
        print(f"📊 Dataset available at: https://huggingface.co/datasets/{dataset_name}")
        print("\n💡 To load the dataset:")
        print("   from datasets import load_dataset")
        print(f'   dataset = load_dataset("{dataset_name}")')
        print("\n📝 Note: load_dataset() will automatically combine all parquet files as a single dataset")
        
    except Exception as e:
        print(f"❌ Error uploading files: {str(e)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Upload partitioned parquet files to Hugging Face Hub as a dataset. Uses efficient direct file upload without loading into memory. HuggingFace automatically combines all parquet files into a single dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "input_pattern",
        help="Path to parquet file or glob pattern (e.g., 'data/*.parquet', 'data/part-*.parquet')"
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
    
    # Find parquet files matching the pattern
    file_paths = find_parquet_files(args.input_pattern)
    
    print("🚀 Using efficient direct upload approach (no memory loading/concatenation)")
    upload_parquet_files(
        file_paths=file_paths,
        dataset_name=args.dataset_name,
        private=args.private
    )


if __name__ == "__main__":
    main()
