#!/usr/bin/env python3

import os
from datasets import load_dataset
import pandas as pd

def download_dataset():
    """Download the arc-agi-2-mixed-finetuning-20 dataset from Hugging Face."""

    print("📥 Downloading dataset from Hugging Face...")

    # Load the dataset - using the dataset name from the screenshot
    dataset_name = "Trelis/arc-agi-2-mixed-finetuning-20"

    try:
        # Load dataset
        dataset = load_dataset(dataset_name, split="train")
        print(f"✅ Dataset loaded successfully: {len(dataset)} rows")

        # Convert to pandas DataFrame for easier analysis
        df = dataset.to_pandas()
        print(f"📊 Converted to DataFrame: {df.shape}")

        # Save locally for analysis
        output_path = "experimental/dataset_analysis/dataset.parquet"
        df.to_parquet(output_path, index=False)
        print(f"💾 Saved to: {output_path}")

        # Show basic info about the dataset
        print(f"\n📋 Dataset columns: {list(df.columns)}")
        print(f"📏 Dataset shape: {df.shape}")

        if 'correct_train_input' in df.columns:
            # Look at correctness patterns - handle numpy arrays properly
            def count_correct(x):
                if hasattr(x, 'tolist'):  # numpy array
                    x = x.tolist()
                if isinstance(x, list):
                    return sum(x)
                else:
                    return 1 if x else 0

            correct_stats = df['correct_train_input'].apply(count_correct)
            print(f"\n📈 Correctness statistics:")
            print(f"   Total correct (per row): {correct_stats.describe()}")

        return df

    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return None

if __name__ == "__main__":
    df = download_dataset()
    if df is not None:
        print("\n🎉 Dataset download completed!")
    else:
        print("\n💥 Dataset download failed!")