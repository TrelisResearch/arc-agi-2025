"""
BigQuery to SOAR format converter.
Handles conversion of BigQuery nested structures to proper SOAR parquet format.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

from .schema import PARQUET_SCHEMA


def convert_bq_nested_structure(bq_data) -> List:
    """Convert BigQuery nested structure to proper list format.
    
    Handles the complex nested structure from BigQuery exports where
    arrays are stored as {"list": [{"element": value}, ...]} format.
    
    Args:
        bq_data: BigQuery nested structure
        
    Returns:
        Properly formatted list
    """
    if bq_data is None:
        return []
    
    # If it's already a simple list, return it
    if isinstance(bq_data, list):
        # Check if it's a list of BigQuery element structures
        if len(bq_data) > 0 and isinstance(bq_data[0], dict) and 'element' in bq_data[0]:
            result = []
            for item in bq_data:
                if isinstance(item, dict) and 'element' in item:
                    element = item['element']
                    # Recursively convert nested structures
                    result.append(convert_bq_nested_structure(element))
                else:
                    result.append(item)
            return result
        else:
            return bq_data
    
    # Handle BigQuery's nested structure
    if isinstance(bq_data, dict):
        if 'list' in bq_data:
            list_data = bq_data['list']
            
            # Convert numpy array to list if needed
            if hasattr(list_data, 'tolist'):
                list_data = list_data.tolist()
            
            # Recursively process the list
            return convert_bq_nested_structure(list_data)
        else:
            # Not a standard BigQuery list structure, return as is
            raise ValueError(f"Unexpected BigQuery format: {bq_data}")
    
    # For primitive values, return as is
    return bq_data


def extract_boolean_values(bool_array) -> List[bool]:
    """Extract boolean values from the {'element': bool} format.
    
    Args:
        bool_array: Array potentially containing {'element': bool} structures
        
    Returns:
        List of boolean values
    """
    if not isinstance(bool_array, list):
        return []
    
    result = []
    for item in bool_array:
        if isinstance(item, dict) and 'element' in item:
            result.append(bool(item['element']))
        else:
            result.append(bool(item))
    return result


def validate_soar_data(data_dict: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate a single converted data dict against the SOAR schema.
    
    Args:
        data_dict: Dictionary containing the converted data
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check required fields exist
        required_fields = ['task_id', 'code', 'model', 'predicted_train_output', 
                          'predicted_test_output', 'correct_train_input', 'correct_test_input']
        for field in required_fields:
            if field not in data_dict:
                return False, f"Missing field: {field}"
        
        # Check string types
        for field in ['task_id', 'code', 'model']:
            if not isinstance(data_dict[field], str):
                return False, f"{field} should be str, got {type(data_dict[field])}"
        
        # Check 3D arrays (List[List[List[int]]])
        for field in ['predicted_train_output', 'predicted_test_output']:
            arr = data_dict[field]
            if not isinstance(arr, list):
                return False, f"{field} should be list, got {type(arr)}"
            for i, grid in enumerate(arr):
                if not isinstance(grid, list):
                    return False, f"{field}[{i}] should be list (2D grid), got {type(grid)}"
                for j, row in enumerate(grid):
                    if not isinstance(row, list):
                        return False, f"{field}[{i}][{j}] should be list (row), got {type(row)}"
                    for k, cell in enumerate(row):
                        if not isinstance(cell, int):
                            return False, f"{field}[{i}][{j}][{k}] should be int, got {type(cell)}"
        
        # Check boolean arrays
        for field in ['correct_train_input', 'correct_test_input']:
            arr = data_dict[field]
            if not isinstance(arr, list):
                return False, f"{field} should be list, got {type(arr)}"
            for i, val in enumerate(arr):
                if not isinstance(val, bool):
                    return False, f"{field}[{i}] should be bool, got {type(val)}"
        
        return True, "Valid"
    except Exception as e:
        return False, f"Validation error: {e}"


def convert_bigquery_to_soar(raw_data: pd.DataFrame, show_progress: bool = True) -> pd.DataFrame:
    """Convert BigQuery raw data to SOAR format.
    
    Args:
        raw_data: DataFrame from BigQuery export
        show_progress: Whether to show progress bar
        
    Returns:
        DataFrame in SOAR format with proper schema
        
    Raises:
        ValueError: If no valid data could be converted
    """
    converted_data = []
    validation_errors = []
    
    iterator = tqdm(range(len(raw_data)), desc="Converting BQ to SOAR") if show_progress else range(len(raw_data))
    
    for idx in iterator:
        row = raw_data.iloc[idx]
        
        try:
            converted_row = {
                'task_id': row['task_id'],
                'code': row['code'], 
                'model': row['model'],
                'predicted_train_output': convert_bq_nested_structure(row['predicted_train_output']),
                'predicted_test_output': convert_bq_nested_structure(row['predicted_test_output']),
                'correct_train_input': extract_boolean_values(convert_bq_nested_structure(row['correct_train_input'])),
                'correct_test_input': extract_boolean_values(convert_bq_nested_structure(row['correct_test_input']))
            }
            
            # Validate the converted row
            is_valid, error_msg = validate_soar_data(converted_row)
            if is_valid:
                converted_data.append(converted_row)
            else:
                validation_errors.append(f"Row {idx}: {error_msg}")
        
        except Exception as e:
            validation_errors.append(f"Row {idx}: Conversion error: {e}")
    
    if not converted_data:
        raise ValueError(f"No valid data could be converted. Errors: {validation_errors[:5]}")
    
    # Create DataFrame from successfully converted data
    final_dataset = pd.DataFrame(converted_data)
    
    # Add missing columns with default values for schema compliance
    final_dataset['reasoning'] = ''  # Empty reasoning for now
    final_dataset['train_input'] = [[] for _ in range(len(final_dataset))]  # Empty for now
    final_dataset['test_input'] = [[] for _ in range(len(final_dataset))]   # Empty for now
    final_dataset['generation'] = 0  # Default generation

    # Reorder columns to match schema
    schema_columns = ['task_id', 'reasoning', 'code', 'correct_train_input', 'correct_test_input',
                     'predicted_train_output', 'predicted_test_output', 'train_input', 'test_input',
                     'model', 'generation']
    final_dataset = final_dataset[schema_columns]
    
    print(f"Successfully converted {len(final_dataset)} programs from {len(raw_data)} input rows")
    if validation_errors:
        print(f"Had {len(validation_errors)} validation/conversion errors")
    
    return final_dataset


def save_soar_parquet(df: pd.DataFrame, output_path: str) -> None:
    """Save DataFrame to parquet with proper SOAR schema.
    
    Args:
        df: DataFrame to save
        output_path: Path to save the parquet file
    """
    try:
        # Convert to PyArrow table with explicit schema
        table = pa.Table.from_pandas(df, schema=PARQUET_SCHEMA)
        pq.write_table(table, output_path)
        print(f"✓ Saved {len(df)} programs to {output_path} with proper PyArrow schema")
    except Exception as e:
        print(f"PyArrow save failed ({e}), using pandas fallback")
        df.to_parquet(output_path, index=False)
        print(f"✓ Saved {len(df)} programs to {output_path} with pandas")
