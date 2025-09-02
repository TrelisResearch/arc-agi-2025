#!/usr/bin/env python3
"""
Analyze the Trelis/arc-agi-2-partial-100-tricky-10 dataset and create a subset file.
"""

import sys
import os
from pathlib import Path

# Add the project root to sys.path so we can import llm_python modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import datasets
    print(f"Using datasets version: {datasets.__version__}")
    
    # Load the dataset
    print("Loading Trelis/arc-agi-2-partial-100-tricky-10...")
    dataset = datasets.load_dataset("Trelis/arc-agi-2-partial-100-tricky-10", split="train")
    
    print(f"Dataset loaded successfully!")
    print(f"Number of rows: {len(dataset)}")
    print(f"Column names: {dataset.column_names}")
    
    # Show first few rows to understand structure
    print(f"\nFirst 3 rows:")
    for i in range(min(3, len(dataset))):
        row = dataset[i]
        print(f"Row {i}:")
        for key, value in row.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"  {key}: {value[:100]}...")
            else:
                print(f"  {key}: {value}")
        print()
    
    # Extract unique task IDs
    if 'task_id' in dataset.column_names:
        task_ids = dataset['task_id']
        unique_task_ids = list(set(task_ids))
        unique_task_ids.sort()
        
        print(f"Total rows: {len(task_ids)}")
        print(f"Unique task IDs: {len(unique_task_ids)}")
        
        # Show some example task IDs
        print(f"\nFirst 10 task IDs: {unique_task_ids[:10]}")
        
        # Create subset file for arc-prize-2025 
        subset_content = "\n".join(unique_task_ids)
        
        # Write to experimental directory
        subset_file = project_root / "experimental" / "trelis_analysis" / "trelis_partial_100_tricky_10_tasks.txt"
        with open(subset_file, 'w') as f:
            f.write(subset_content)
        
        print(f"\n✅ Created subset file: {subset_file}")
        print(f"📊 Contains {len(unique_task_ids)} unique task IDs")
        
    else:
        print("❌ No 'task_id' column found in dataset")
        print("Available columns:", dataset.column_names)
        
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    import traceback
    traceback.print_exc()