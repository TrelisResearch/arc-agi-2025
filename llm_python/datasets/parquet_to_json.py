#!/usr/bin/env python3
"""
Convert SOAR parquet files to JSON format for the web viewer.
Also includes task data from ARC dataset.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from llm_python.datasets.io import read_soar_parquet
from llm_python.utils.task_loader import get_task_loader


def parquet_to_json(parquet_path: str, output_path: str, include_task_data: bool = True) -> None:
    """
    Convert a SOAR parquet file to JSON format for the web viewer.

    Args:
        parquet_path: Path to the input parquet file
        output_path: Path for the output JSON file
        include_task_data: Whether to include ARC task data
    """
    print(f"Loading parquet file: {parquet_path}")
    df = read_soar_parquet(parquet_path)

    print(f"Loaded {len(df)} records")

    # Convert DataFrame to list of dictionaries
    data = df.to_dict('records')

    # Prepare the output structure
    output_data = {
        "metadata": {
            "total_records": len(data),
            "columns": list(df.columns),
            "created_from": str(parquet_path)
        },
        "data": data,
        "task_data": {}
    }

    # Load task data if requested
    if include_task_data:
        print("Loading ARC task data...")
        task_loader = get_task_loader()
        unique_task_ids = df['task_id'].unique()

        for task_id in unique_task_ids:
            try:
                task_data = task_loader.get_task(task_id)
                output_data["task_data"][task_id] = task_data
                print(f"  Loaded task: {task_id}")
            except Exception as e:
                print(f"  Warning: Failed to load task {task_id}: {e}")
                # Create minimal task data structure
                output_data["task_data"][task_id] = {
                    "train": [],
                    "test": []
                }

    # Write JSON file
    print(f"Writing JSON file: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Successfully converted {len(data)} records to JSON")
    if include_task_data:
        print(f"Included task data for {len(output_data['task_data'])} tasks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SOAR parquet files to JSON for web viewer")
    parser.add_argument("input", help="Path to input parquet file")
    parser.add_argument("output", help="Path to output JSON file")
    parser.add_argument(
        "--no-task-data",
        action="store_true",
        help="Don't include ARC task data (makes file smaller but grids won't show inputs/expected outputs)"
    )

    args = parser.parse_args()

    parquet_to_json(
        args.input,
        args.output,
        include_task_data=not args.no_task_data
    )