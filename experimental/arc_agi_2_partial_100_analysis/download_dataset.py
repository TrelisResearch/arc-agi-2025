#!/usr/bin/env python3

import os
from datasets import load_dataset

def download_arc_agi_dataset():
    print("Downloading arc-agi-2-partial-100 dataset from HuggingFace...")

    # Download the dataset
    dataset = load_dataset("Trelis/arc-agi-2-partial-100")

    # Save to local directory
    dataset.save_to_disk("./arc_agi_2_partial_100_data")

    print(f"Dataset downloaded successfully!")
    print(f"Dataset info:")
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} examples")

    return dataset

if __name__ == "__main__":
    dataset = download_arc_agi_dataset()