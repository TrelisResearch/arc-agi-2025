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
    gcs_prefix: str = "tmp"
) -> str:
    """Export a BigQuery table to local parquet file via GCS.
    
    This is much faster than querying BigQuery directly for large datasets.
    
    Args:
        client: BigQuery client
        table_name: Full table name (e.g., 'project.dataset.table')
        local_path: Local file path. If None, uses /tmp/{table_basename}.parquet
        gcs_bucket: GCS bucket name for temporary storage
        gcs_prefix: GCS prefix/folder for temporary files
        
    Returns:
        Path to the downloaded local file
        
    Raises:
        Exception: If export or download fails
    """
    # Extract just the table name for file naming
    file_name = table_name.split('.')[-1]
    
    if local_path is None:
        local_path = f"/tmp/{file_name}.parquet"
    
    # Ensure local directory exists
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    
    # GCS paths
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
    gcs_prefix: str = "tmp"
) -> pd.DataFrame:
    """Load a BigQuery table as a pandas DataFrame via GCS export.
    
    Args:
        client: BigQuery client
        table_name: Full table name (e.g., 'project.dataset.table')
        gcs_bucket: GCS bucket name for temporary storage
        gcs_prefix: GCS prefix/folder for temporary files
        
    Returns:
        DataFrame containing the table data
    """
    local_path = export_bigquery_table_to_local(
        client=client,
        table_name=table_name,
        gcs_bucket=gcs_bucket,
        gcs_prefix=gcs_prefix
    )
    
    print("Reading parquet file...")
    df = pd.read_parquet(local_path)
    print(f"Loaded {len(df)} rows from BigQuery table")
    
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
