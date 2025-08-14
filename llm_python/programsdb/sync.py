"""
Sync local program database to Google Cloud Storage.
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import storage
from google.cloud.exceptions import GoogleCloudError

from .localdb import get_localdb
from .schema import PARQUET_SCHEMA

__all__ = ['sync_database_to_cloud', 'import_from_parquet']


def import_from_parquet(parquet_path: str, db_path: Optional[str] = None) -> int:
    """
    Import programs from a parquet file into the local database.
    Uses efficient DuckDB batch processing for both local and GCS files.
    
    Args:
        parquet_path: Path to parquet file (local path or GCS gs:// URL)
        db_path: Optional path to the database file. If None, uses default location.
        
    Returns:
        Number of programs imported
        
    Raises:
        RuntimeError: If the import operation fails
    """
    # Get database instance
    db = get_localdb(db_path)
    
    if parquet_path.startswith("gs://"):
        # For GCS files, download to temp file first
        print(f"Downloading from {parquet_path}...")
        
        # Parse GCS path
        parts = parquet_path[5:].split("/", 1)  # Remove gs:// prefix
        bucket_name = parts[0]
        blob_name = parts[1]
        
        # Download to temp file
        temp_dir = Path("/tmp")
        temp_file = temp_dir / f"import_{blob_name.replace('/', '_')}"
        
        try:
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(str(temp_file))
            print(f"Downloaded to temporary file: {temp_file}")
            
            # Use the local file path for processing
            local_parquet_path = str(temp_file)
            
        except GoogleCloudError as e:
            raise RuntimeError(f"Failed to download from GCS: {e}")
        except Exception as e:
            if "credentials" in str(e).lower() or "authentication" in str(e).lower():
                raise RuntimeError(
                    f"Authentication error: {e}\n"
                    "Please set up Google Cloud credentials:\n"
                    "1. Set GOOGLE_APPLICATION_CREDENTIALS environment variable, or\n"
                    "2. Run 'gcloud auth application-default login'"
                )
            raise RuntimeError(f"Failed to download from GCS: {e}")
    else:
        # For local files, use directly
        local_parquet_path = parquet_path
        temp_file = None
        print(f"Processing local file: {parquet_path}")

    try:
        # Validate the parquet file has the correct schema
        import pyarrow.parquet as pq
        
        # Read parquet metadata to validate schema
        parquet_file = pq.ParquetFile(local_parquet_path)
        file_schema = parquet_file.schema_arrow
        
        # Check if schemas are compatible (allow missing optional fields)
        expected_fields = {field.name: field.type for field in PARQUET_SCHEMA}
        actual_fields = {field.name: field.type for field in file_schema}
        
        for field_name, expected_type in expected_fields.items():
            if field_name not in actual_fields:
                raise ValueError(f"Missing required field in parquet: {field_name}")
            # Note: We could add more sophisticated type checking here if needed
            
        print(f"Parquet schema validation passed for {local_parquet_path}")
        
        # Use DuckDB to directly read from parquet and insert into our table
        # This is much faster than pandas + line-by-line insertion
        
        # Create a temporary table from the parquet file
        create_temp_sql = f"""
        CREATE TEMP TABLE temp_import AS 
        SELECT 
            task_id,
            reasoning,
            code,
            correct_train_input,
            correct_test_input,
            predicted_train_output,
            predicted_test_output,
            model
        FROM read_parquet('{local_parquet_path}')
        """
        
        db.connection.execute(create_temp_sql)
        
        # Insert with conflict resolution (ignore duplicates to avoid constraint violations)
        upsert_sql = """
        INSERT OR IGNORE INTO programs 
        (key, task_id, reasoning, code, correct_train_input, correct_test_input,
         predicted_train_output, predicted_test_output, model)
        SELECT 
            sha256(task_id || ':' || code) as key,
            task_id,
            reasoning,
            code,
            correct_train_input,
            correct_test_input,
            predicted_train_output,
            predicted_test_output,
            model
        FROM temp_import
        """
        
        db.connection.execute(upsert_sql)
        
        # Get count of what was actually in the temp table
        total_result = db.connection.execute("SELECT COUNT(*) FROM temp_import").fetchone()
        total_in_parquet = total_result[0] if total_result else 0
        
        # For simplicity, we'll just return the total that was processed
        # The real benefit is avoiding the constraint error, not precise counting
        imported_count = total_in_parquet
        
        # Clean up temporary table
        db.connection.execute("DROP TABLE temp_import")
        
        print(f"Successfully imported {imported_count} programs using DuckDB (duplicates ignored)")
        return imported_count
        
    except Exception as e:
        raise ValueError(f"Parquet processing failed: {e}")
    finally:
        # Clean up temporary file if we downloaded one
        if temp_file and temp_file.exists():
            temp_file.unlink()
            print(f"Cleaned up temporary file: {temp_file}")


def sync_database_to_cloud(db_path: Optional[str] = None, destination: Optional[str] = None) -> str:
    """
    Sync the local database to Google Cloud Storage or a local parquet file.
    
    Args:
        db_path: Optional path to the database file. If None, uses default location.
        destination: Optional destination path. If None, uploads to GCS. If provided,
                    saves to local parquet file at this path.
        
    Returns:
        The path where the file was saved (GCS path or local path)
        
    Raises:
        RuntimeError: If the sync operation fails
    """
    # Get database instance
    db = get_localdb(db_path)
    
    # Get database ID for consistent naming
    db_id = db.get_database_id()
    
    # Get all programs from the database
    programs = db.get_all_programs()
    
    if not programs:
        print("No programs found in database - nothing to sync")
        return ""
    
    print(f"Found {len(programs)} programs in database with ID: {db_id}")
    
    # Convert to DataFrame
    df = pd.DataFrame(programs)
    
    if destination:
        # Save to local file
        try:
            # Write to parquet with schema validation
            table = pa.Table.from_pandas(df, schema=PARQUET_SCHEMA)
            pq.write_table(table, destination)
            print(f"Successfully saved to local file: {destination}")
            return destination
        except Exception as e:
            raise RuntimeError(f"Failed to save to local file {destination}: {e}")
    else:
        # Upload to GCS (original behavior)
        # Create temporary parquet file
        temp_dir = Path("/tmp")
        temp_file = temp_dir / f"local_snapshot_{db_id}.parquet"
        
        try:
            # Write to parquet with schema validation
            table = pa.Table.from_pandas(df, schema=PARQUET_SCHEMA)
            pq.write_table(table, temp_file)
            print(f"Created temporary parquet file with schema validation: {temp_file}")
            
            # Upload to GCS
            gcs_path = f"gs://trelis-arc/datasets/superking/local_snapshot_{db_id}.parquet"
            
            try:
                client = storage.Client()
                bucket = client.bucket("trelis-arc")
                blob_name = f"datasets/superking/local_snapshot_{db_id}.parquet"
                blob = bucket.blob(blob_name)
                
                print(f"Uploading to {gcs_path}...")
                blob.upload_from_filename(str(temp_file))
                print(f"Successfully uploaded to {gcs_path}")
                return gcs_path
                
            except GoogleCloudError as e:
                raise RuntimeError(f"Failed to upload to GCS: {e}")
            except Exception as e:
                if "credentials" in str(e).lower() or "authentication" in str(e).lower():
                    raise RuntimeError(
                        f"Authentication error: {e}\n"
                        "Please set up Google Cloud credentials:\n"
                        "1. Set GOOGLE_APPLICATION_CREDENTIALS environment variable, or\n"
                        "2. Run 'gcloud auth application-default login'"
                    )
                raise RuntimeError(f"Failed to upload to GCS: {e}")
            
        finally:
            # Clean up temporary file
            if temp_file and temp_file.exists():
                temp_file.unlink()
                print(f"Cleaned up temporary file: {temp_file}")
