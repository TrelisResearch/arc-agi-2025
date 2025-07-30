#!/usr/bin/env python3
"""
Simple script to load a parquet file and push it to Hugging Face Hub as a dataset.
Assumes user is logged in or has HF_TOKEN environment variable set.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
from datasets import Dataset, concatenate_datasets
import pyarrow.parquet as pq
import gc


def load_parquet_and_push_to_hf_streaming(input_file: str, dataset_name: str, private: bool = False, chunk_size: int = 5000):
    """
    Ultra memory-efficient version that streams data directly to HF Hub.
    Creates the dataset incrementally without loading everything into memory.
    
    Args:
        input_file (str): Path to the input parquet file
        dataset_name (str): Name of the dataset on Hugging Face Hub (format: username/dataset-name)
        private (bool): Whether to make the dataset private (default: False)
        chunk_size (int): Number of rows to process at a time (default: 5000)
    """
    # Validate input file exists
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)
    
    if not input_path.suffix.lower() == '.parquet':
        print(f"Error: Input file '{input_file}' is not a parquet file.")
        sys.exit(1)
    
    print(f"Streaming parquet file to HF Hub: {input_file}")
    try:
        # Read parquet file metadata to get total rows
        parquet_file = pq.ParquetFile(input_file)
        total_rows = parquet_file.metadata.num_rows
        print(f"Total rows: {total_rows}")
        
        # Process first chunk to create the dataset
        first_batch = next(parquet_file.iter_batches(batch_size=chunk_size))
        df_first = first_batch.to_pandas()
        dataset = Dataset.from_pandas(df_first)
        
        print(f"Columns: {list(dataset.column_names)}")
        print(f"Pushing initial chunk to HF Hub: {dataset_name}")
        
        # Push initial dataset
        dataset.push_to_hub(dataset_name, private=private)
        
        # Clean up first chunk
        del df_first
        del dataset
        gc.collect()
        
        # Process remaining chunks and append them
        chunk_count = 1
        for i, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size)):
            if i == 0:  # Skip first batch as we already processed it
                continue
                
            chunk_count += 1
            start_row = i * chunk_size
            end_row = min((i+1) * chunk_size, total_rows)
            print(f"Processing and uploading chunk {chunk_count}, rows {start_row} to {end_row}")
            
            # Convert chunk and append to existing dataset
            df_chunk = batch.to_pandas()
            chunk_dataset = Dataset.from_pandas(df_chunk)
            
            # Load existing dataset and concatenate
            from datasets import load_dataset
            existing_dataset = load_dataset(dataset_name, split='train')
            combined_dataset = concatenate_datasets([existing_dataset, chunk_dataset])
            
            # Push updated dataset
            combined_dataset.push_to_hub(dataset_name, private=private)
            
            # Clean up
            del df_chunk
            del chunk_dataset
            del existing_dataset
            del combined_dataset
            gc.collect()
        
        print(f"Successfully streamed {chunk_count} chunks to: https://huggingface.co/datasets/{dataset_name}")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)


def load_parquet_and_push_to_hf_chunked(input_file: str, dataset_name: str, private: bool = False, chunk_size: int = 10000):
    """
    Load a parquet file in chunks and push it to Hugging Face Hub as a dataset.
    Memory-efficient version that processes data in batches.
    
    Args:
        input_file (str): Path to the input parquet file
        dataset_name (str): Name of the dataset on Hugging Face Hub (format: username/dataset-name)
        private (bool): Whether to make the dataset private (default: False)
        chunk_size (int): Number of rows to process at a time (default: 10000)
    """
    # Validate input file exists
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)
    
    if not input_path.suffix.lower() == '.parquet':
        print(f"Error: Input file '{input_file}' is not a parquet file.")
        sys.exit(1)
    
    print(f"Loading parquet file in chunks: {input_file}")
    try:
        # Read parquet file metadata to get total rows
        parquet_file = pq.ParquetFile(input_file)
        total_rows = parquet_file.metadata.num_rows
        print(f"Total rows: {total_rows}")
        
        # Process in chunks
        datasets = []
        for i, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size)):
            print(f"Processing chunk {i+1}, rows {i*chunk_size} to {min((i+1)*chunk_size, total_rows)}")
            
            # Convert Arrow table to pandas DataFrame then to HF Dataset
            df_chunk = batch.to_pandas()
            dataset_chunk = Dataset.from_pandas(df_chunk)
            datasets.append(dataset_chunk)
            
            # Clean up chunk to free memory
            del df_chunk
            del dataset_chunk
            gc.collect()  # Force garbage collection
        
        print(f"Processed {len(datasets)} chunks")
        
        # Concatenate all chunks
        print("Concatenating chunks...")
        final_dataset = concatenate_datasets(datasets)
        
        # Clean up individual chunks
        del datasets
        gc.collect()  # Force garbage collection
        
        print(f"Final dataset has {len(final_dataset)} rows")
        print(f"Columns: {list(final_dataset.column_names)}")
        
        # Push to Hugging Face Hub
        print(f"Pushing dataset to Hugging Face Hub: {dataset_name}")
        final_dataset.push_to_hub(
            dataset_name,
            private=private
        )
        print(f"Successfully pushed dataset to: https://huggingface.co/datasets/{dataset_name}")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        sys.exit(1)


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
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Number of rows to process at a time for memory efficiency (default: 10000)"
    )
    
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use ultra memory-efficient streaming mode (slowest but uses least memory)"
    )
    
    parser.add_argument(
        "--no-chunking",
        action="store_true",
        help="Load entire file into memory at once (not recommended for large files)"
    )
    
    args = parser.parse_args()
    
    # Validate dataset name format
    if "/" not in args.dataset_name:
        print("Error: Dataset name must be in format 'username/dataset-name'")
        sys.exit(1)
    
    if args.no_chunking:
        print("Warning: Loading entire file into memory. This may cause memory issues with large files.")
        load_parquet_and_push_to_hf(
            input_file=args.input_file,
            dataset_name=args.dataset_name,
            private=args.private
        )
    elif args.streaming:
        print(f"Using ultra memory-efficient streaming mode with chunk size: {args.chunk_size}")
        load_parquet_and_push_to_hf_streaming(
            input_file=args.input_file,
            dataset_name=args.dataset_name,
            private=args.private,
            chunk_size=args.chunk_size
        )
    else:
        print(f"Using memory-efficient chunked processing with chunk size: {args.chunk_size}")
        load_parquet_and_push_to_hf_chunked(
            input_file=args.input_file,
            dataset_name=args.dataset_name,
            private=args.private,
            chunk_size=args.chunk_size
        )



if __name__ == "__main__":
    main()
