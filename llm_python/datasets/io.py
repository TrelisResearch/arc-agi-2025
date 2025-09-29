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

def read_soar_parquet(path: Union[str, Path], schema=PARQUET_SCHEMA) -> pd.DataFrame:
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
    df = pd.read_parquet(path, dtype_backend='pyarrow', schema=schema)
    
    return df

def _add_missing_nullable_columns(df: pd.DataFrame, schema=PARQUET_SCHEMA) -> pd.DataFrame:
    """
    Return a shallow copy of df with missing nullable columns from PARQUET_SCHEMA added as nulls.
    """
    df_copy = df.copy()
    for field in schema:
        if field.name not in df_copy.columns and field.nullable:
            df_copy[field.name] = pd.NA
    return df_copy

def write_soar_parquet(df: pd.DataFrame, path: Union[str, Path], schema=PARQUET_SCHEMA) -> None:
    """
    Write parquet file with proper schema preservation and enforcement.
    
    Schema validation happens automatically when converting to PyArrow table.
    
    Args:
        df: DataFrame to write
        path: Output path
    """
    
    # Use a shallow copy with missing nullable columns added
    df_for_write = _add_missing_nullable_columns(df)
    table = pa.Table.from_pandas(df_for_write, schema=schema)
    pq.write_table(table, path)


def validate_soar_dataframe_schema(df: pd.DataFrame, schema=PARQUET_SCHEMA) -> None:
    """
    Validate DataFrame matches expected ProgramSample schema using PyArrow.
    
    This is useful for validating DataFrames before processing when not
    reading/writing parquet files. Validates all rows for complete validation.
    
    Args:
        df: DataFrame to validate
        
    Raises:
        ValueError: If schema validation fails
    """
    # Use a shallow copy with missing nullable columns added
    df_for_validate = _add_missing_nullable_columns(df)
    try:
        pa.Table.from_pandas(df_for_validate, schema=schema)
    except (pa.ArrowInvalid, pa.ArrowTypeError, KeyError, ValueError) as e:
        raise ValueError(f"Schema validation failed: {e}")

