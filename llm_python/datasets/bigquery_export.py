"""
BigQuery export and download utilities.
Handles exporting BigQuery tables to GCS and downloading to local files.
"""

import pandas as pd
from pathlib import Path
from google.cloud import bigquery, storage
from typing import Optional


def export_bigquery_table_to_local(
    client: bigquery.Client,
    table_name: str,
    local_path: Optional[str] = None,
    gcs_bucket: str = "trelis-arc",
    gcs_prefix: str = "tmp",
    use_sharding: bool = False
) -> str:
    """Export a BigQuery table to local parquet file via GCS.
    
    This is much faster than querying BigQuery directly for large datasets.
    
    Args:
        client: BigQuery client
        table_name: Full table name (e.g., 'project.dataset.table')
        local_path: Local file path. If None, uses /tmp/{table_basename}.parquet
        gcs_bucket: GCS bucket name for temporary storage
        gcs_prefix: GCS prefix/folder for temporary files
        use_sharding: If True, uses BigQuery's automatic sharding with * wildcard
        
    Returns:
        Path to the downloaded local file (or directory if sharded)
        
    Raises:
        Exception: If export or download fails
    """
    # Extract just the table name for file naming
    file_name = table_name.split('.')[-1]
    
    if local_path is None:
        local_path = f"/tmp/{file_name}.parquet"
    
    # Ensure local directory exists
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    
    if use_sharding:
        # Use BigQuery's automatic sharding with * wildcard
        gcs_uri = f"gs://{gcs_bucket}/{gcs_prefix}/{file_name}_*.parquet"
        gcs_blob_prefix = f"{gcs_prefix}/{file_name}_"
        
        print(f"Exporting BigQuery table '{table_name}' to GCS with sharding...")
        
        # Export to Cloud Storage with sharding
        export_job_config = bigquery.ExtractJobConfig()
        export_job_config.destination_format = bigquery.DestinationFormat.PARQUET
        
        extract_job = client.extract_table(
            table_name,
            gcs_uri,
            job_config=export_job_config
        )
        
        print("Waiting for BigQuery export to complete...")
        extract_job.result()  # Wait for export to complete
        print("✓ Export to GCS completed successfully")
        
        # Download all sharded files
        print("Downloading sharded files from GCS...")
        storage_client = storage.Client()
        bucket = storage_client.bucket(gcs_bucket)
        
        # List all sharded files
        blobs = list(bucket.list_blobs(prefix=gcs_blob_prefix))
        parquet_blobs = [blob for blob in blobs if blob.name.endswith('.parquet')]
        
        if not parquet_blobs:
            raise Exception(f"No parquet files found with prefix: {gcs_blob_prefix}")
        
        print(f"Found {len(parquet_blobs)} sharded files")
        
        # Create a directory for sharded files
        shard_dir = f"/tmp/{file_name}_shards"
        Path(shard_dir).mkdir(parents=True, exist_ok=True)
        
        # Download each shard
        shard_paths = []
        for i, blob in enumerate(parquet_blobs):
            shard_path = f"{shard_dir}/shard_{i:03d}.parquet"
            blob.download_to_filename(shard_path)
            shard_paths.append(shard_path)
            print(f"✓ Downloaded shard {i+1}/{len(parquet_blobs)}")
        
        # Combine all shards into a single file
        print("Combining sharded files...")
        import pandas as pd
        
        all_dfs = []
        for shard_path in shard_paths:
            df = pd.read_parquet(shard_path)
            all_dfs.append(df)
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_parquet(local_path, index=False)
        
        # Clean up shard files
        for shard_path in shard_paths:
            Path(shard_path).unlink(missing_ok=True)
        Path(shard_dir).rmdir()
        
        # Clean up GCS files
        for blob in parquet_blobs:
            blob.delete()
        
        print(f"✓ Combined {len(combined_df)} rows into {local_path}")
        
    else:
        # Single file export (original behavior)
        gcs_uri = f"gs://{gcs_bucket}/{gcs_prefix}/{file_name}.parquet"
        gcs_blob_path = f"{gcs_prefix}/{file_name}.parquet"
        
        print(f"Exporting BigQuery table '{table_name}' to GCS...")
        
        # Export to Cloud Storage (much faster for large datasets)
        export_job_config = bigquery.ExtractJobConfig()
        export_job_config.destination_format = bigquery.DestinationFormat.PARQUET
        
        extract_job = client.extract_table(
            table_name,
            gcs_uri,
            job_config=export_job_config
        )
        
        print("Waiting for BigQuery export to complete...")
        extract_job.result()  # Wait for export to complete
        print("✓ Export to GCS completed successfully")
        
        # Download from GCS to local file
        print(f"Downloading from GCS to {local_path}...")
        storage_client = storage.Client()
        bucket = storage_client.bucket(gcs_bucket)
        blob = bucket.blob(gcs_blob_path)
        blob.download_to_filename(local_path)
        print("✓ Download completed")
    
    return local_path


def load_bigquery_table_as_dataframe(
    client: bigquery.Client,
    table_name: str,
    gcs_bucket: str = "trelis-arc",
    gcs_prefix: str = "tmp",
    use_sharding: bool = False
) -> pd.DataFrame:
    """Load a BigQuery table as a pandas DataFrame via GCS export.
    
    For large tables that exceed BigQuery's single-file export limit,
    use sharding to automatically split into multiple files.
    
    Args:
        client: BigQuery client
        table_name: Full table name (e.g., 'project.dataset.table')
        gcs_bucket: GCS bucket name for temporary storage
        gcs_prefix: GCS prefix/folder for temporary files
        use_sharding: If True, uses BigQuery's automatic sharding with * wildcard
        
    Returns:
        DataFrame containing the table data
    """
    # Try single file export first, fall back to sharding if it fails
    if not use_sharding:
        try:
            local_path = export_bigquery_table_to_local(
                client=client,
                table_name=table_name,
                gcs_bucket=gcs_bucket,
                gcs_prefix=gcs_prefix,
                use_sharding=False
            )
            
            print("Reading parquet file...")
            df = pd.read_parquet(local_path)
            print(f"Loaded {len(df)} rows from BigQuery table")
            return df
            
        except Exception as e:
            if "too large to be exported to a single file" in str(e):
                print("Table too large for single file export, retrying with sharding...")
            else:
                raise e
    
    # Use BigQuery's built-in sharding
    local_path = export_bigquery_table_to_local(
        client=client,
        table_name=table_name,
        gcs_bucket=gcs_bucket,
        gcs_prefix=gcs_prefix,
        use_sharding=True
    )
    
    print("Reading combined parquet file...")
    df = pd.read_parquet(local_path)
    print(f"Loaded {len(df)} rows from sharded BigQuery table")
    
    return df


def cleanup_gcs_temp_file(
    gcs_bucket: str,
    gcs_blob_path: str
) -> None:
    """Clean up temporary files from GCS.
    
    Args:
        gcs_bucket: GCS bucket name
        gcs_blob_path: Full blob path to delete
    """
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(gcs_bucket)
        blob = bucket.blob(gcs_blob_path)
        blob.delete()
        print(f"✓ Cleaned up temporary file: gs://{gcs_bucket}/{gcs_blob_path}")
    except Exception as e:
        print(f"Warning: Could not clean up temporary file: {e}")


def cleanup_local_temp_file(local_path: str) -> None:
    """Clean up local temporary files.
    
    Args:
        local_path: Path to local file to delete
    """
    try:
        Path(local_path).unlink(missing_ok=True)
        print(f"✓ Cleaned up local temp file: {local_path}")
    except Exception as e:
        print(f"Warning: Could not clean up local file: {e}")
