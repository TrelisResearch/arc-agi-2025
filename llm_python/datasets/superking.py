from google.cloud import storage
from pathlib import Path
import pandas as pd
import os


def download_superking(output_path: str = "/tmp/superking.parquet") -> str:
    # GCS bucket and path
    gcs_bucket = "trelis-arc"
    gcs_prefix = "datasets/superking/"
    local_download_dir = "/tmp/superking_data"

    print("Downloading superking dataset from Google Cloud Storage...")
    print(f"Bucket: gs://{gcs_bucket}/{gcs_prefix}")

    # Create local directory
    Path(local_download_dir).mkdir(parents=True, exist_ok=True)

    try:
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(gcs_bucket)

        # List all parquet files in the superking dataset
        blobs = list(bucket.list_blobs(prefix=gcs_prefix))
        parquet_blobs = [blob for blob in blobs if blob.name.endswith(".parquet")]

        print(f"Found {len(parquet_blobs)} parquet files in {gcs_prefix}")

        # Download all parquet files
        local_files = []
        for blob in parquet_blobs:
            filename = blob.name.split("/")[-1]  # Get just the filename
            local_path = f"{local_download_dir}/{filename}"

            print(f"  Downloading {blob.name} -> {local_path}")
            blob.download_to_filename(local_path)
            local_files.append(local_path)

        print(f"Downloaded {len(local_files)} files")

        # Combine all parquet files into a single DataFrame
        print("Combining parquet files...")
        dataframes = []
        for file_path in local_files:
            df_part = pd.read_parquet(file_path)
            print(f"  {file_path}: {len(df_part):,} rows")
            dataframes.append(df_part)

        # Concatenate all dataframes
        raw_data = pd.concat(dataframes, ignore_index=True)

        print(f"Combined dataset: {len(raw_data):,} programs")
        print(f"Columns: {list(raw_data.columns)}")
        print(f"Number of rows: {len(raw_data)}")

        # Save combined dataset to local parquet for analysis
        raw_data.to_parquet(output_path, index=False)
        print(f"Saved combined dataset to: {output_path}")

        for file_path in local_files:
            os.remove(file_path)
        os.rmdir(local_download_dir)

    except Exception as e:
        print(f"Error downloading data: {e}")
        raise
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and combine superking dataset from Google Cloud Storage")
    parser.add_argument(
        "--output-path", 
        "-o",
        type=str, 
        default="/tmp/superking.parquet",
        help="Output path for the combined parquet file (default: /tmp/superking.parquet)"
    )
    
    args = parser.parse_args()
    
    output_file = download_superking(args.output_path)
    print(f"Successfully created dataset at: {output_file}")