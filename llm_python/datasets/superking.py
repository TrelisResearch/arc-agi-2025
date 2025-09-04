from typing import List
from google.cloud import storage
from pathlib import Path
import pandas as pd
import os

from llm_python.datasets.io import read_soar_parquet, write_soar_parquet


def sync_superking() -> List[str]:
    # GCS bucket and path
    gcs_bucket = "trelis-arc"
    gcs_prefix = "datasets/superking/"
    local_download_dir = "/tmp/superking_data"

    print("Downloading superking dataset from Google Cloud Storage...")
    print(f"Bucket: gs://{gcs_bucket}/{gcs_prefix}")

    # Create local directory
    Path(local_download_dir).mkdir(parents=True, exist_ok=True)

    # Initialize GCS client
    storage_client = storage.Client()
    bucket = storage_client.bucket(gcs_bucket)

    # List all parquet files in the superking dataset
    blobs = list(bucket.list_blobs(prefix=gcs_prefix))
    parquet_blobs = [blob for blob in blobs if blob.name.endswith(".parquet")]

    print(f"Found {len(parquet_blobs)} parquet files in {gcs_prefix}")

    # Download all parquet files, skipping if already present and unchanged
    local_files = []
    skipped = 0
    downloaded = 0
    for blob in parquet_blobs:
        filename = blob.name.split("/")[-1]
        local_path = f"{local_download_dir}/{filename}"

        # Check if file exists and matches remote size
        if os.path.exists(local_path):
            local_size = os.path.getsize(local_path)
            if local_size == blob.size:
                local_files.append(local_path)
                skipped += 1
                continue
            else:
                print(
                    f"  Redownloading {blob.name} (local size {local_size} != remote size {blob.size})"
                )
        else:
            print(f"  Downloading {blob.name} -> {local_path}")
        blob.download_to_filename(local_path)
        local_files.append(local_path)
        downloaded += 1

        print(
            f"Downloaded {downloaded} files, skipped {skipped} files (already present and unchanged)"
        )
    return local_files


def load_superking() -> pd.DataFrame:
    # Load the superking dataset from local parquet files
    local_files = sync_superking()
    print("Loading superking dataset...")
    dataframes = [read_soar_parquet(file_path) for file_path in local_files]
    return pd.concat(dataframes, ignore_index=True)


def download_superking(output_path: str = "/tmp/superking.parquet") -> str:
    try:
        df = load_superking()
        print(f"Combined dataset: {len(df):,} programs")
        print(f"Columns: {list(df.columns)}")
        print(f"Number of rows: {len(df)}")

        # Save combined dataset to local parquet for analysis
        write_soar_parquet(df, output_path)
        print(f"Saved combined dataset to: {output_path}")

    except Exception as e:
        print(f"Error downloading data: {e}")
        raise
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and combine superking dataset from Google Cloud Storage"
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        default="/tmp/superking.parquet",
        help="Output path for the combined parquet file (default: /tmp/superking.parquet)",
    )

    args = parser.parse_args()

    output_file = download_superking(args.output_path)
    print(f"Successfully created dataset at: {output_file}")
