"""
Super simple, no-conversion-needed parquet handling.

Just use the built-in pandas and pyarrow features properly!
"""

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Union
from llm_python.datasets.schema import PARQUET_SCHEMA
import pyarrow as pa

def read_soar_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """
    Read parquet file with proper type handling and schema enforcement.
    
    Uses dtype_backend='pyarrow' to preserve proper Python types in nested lists.
    Schema validation is automatic via the schema parameter.
    
    Args:
        path: Path to parquet file
        
    Returns:
        DataFrame with proper Python types (no numpy scalars)
    """
    
    # Read with schema enforcement and dtype_backend='pyarrow'
    # Schema validation happens automatically here
    df = pd.read_parquet(path, dtype_backend='pyarrow', schema=PARQUET_SCHEMA)
    
    return df

def write_soar_parquet(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """
    Write parquet file with proper schema preservation and enforcement.
    
    Schema validation happens automatically when converting to PyArrow table.
    
    Args:
        df: DataFrame to write
        path: Output path
    """
    
    # Convert to PyArrow table with strict schema enforcement
    # Schema validation happens automatically here
    table = pa.Table.from_pandas(df, schema=PARQUET_SCHEMA)
    
    # Write with PyArrow to ensure schema compliance
    pq.write_table(table, path)


def validate_soar_dataframe_schema(df: pd.DataFrame) -> None:
    """
    Validate DataFrame matches expected ProgramSample schema using PyArrow.
    
    This is useful for validating DataFrames before processing when not
    reading/writing parquet files. Validates all rows for complete validation.
    
    Args:
        df: DataFrame to validate
        
    Raises:
        ValueError: If schema validation fails
    """
    try:
        # Convert the full DataFrame with schema validation
        # This validates:
        # - Column names and types match schema
        # - Nullability constraints across all rows
        # - Data type compatibility for all values
        pa.Table.from_pandas(df, schema=PARQUET_SCHEMA)
            
    except (pa.ArrowInvalid, pa.ArrowTypeError, KeyError, ValueError) as e:
        # PyArrow provides detailed error messages about schema mismatches
        raise ValueError(f"Schema validation failed: {e}")

