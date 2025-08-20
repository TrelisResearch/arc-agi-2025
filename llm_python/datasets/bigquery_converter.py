"""
BigQuery to SOAR format converter.
Handles conversion of BigQuery nested structures to proper SOAR parquet format.
"""

from typing import List

import pandas as pd
from tqdm import tqdm


def convert_bigquery_to_soar(
    raw_data: pd.DataFrame, show_progress: bool = True
) -> pd.DataFrame:
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

    iterator = (
        tqdm(range(len(raw_data)), desc="Converting BQ to SOAR")
        if show_progress
        else range(len(raw_data))
    )

    for idx in iterator:
        row = raw_data.iloc[idx]

        try:
            converted_data.append(
                {
                    "task_id": row["task_id"],
                    "code": row["code"],
                    "model": row["model"],
                    "predicted_train_output": _convert_bq_nested_structure(
                        row["predicted_train_output"]
                    ),
                    "predicted_test_output": _convert_bq_nested_structure(
                        row["predicted_test_output"]
                    ),
                    "correct_train_input": _extract_boolean_values(
                        _convert_bq_nested_structure(row["correct_train_input"])
                    ),
                    "correct_test_input": _extract_boolean_values(
                        _convert_bq_nested_structure(row["correct_test_input"])
                    ),
                }
            )

        except Exception as e:
            validation_errors.append(f"Row {idx}: Conversion error: {e}")

    if not converted_data:
        raise ValueError(
            f"No valid data could be converted. Errors: {validation_errors[:5]}"
        )

    # Create DataFrame from successfully converted data
    final_dataset = pd.DataFrame(converted_data)

    # Add missing columns with default values for schema compliance
    final_dataset["reasoning"] = ""  # Empty reasoning for now

    # Reorder columns to match schema
    schema_columns = [
        "task_id",
        "reasoning",
        "code",
        "correct_train_input",
        "correct_test_input",
        "predicted_train_output",
        "predicted_test_output",
        "model",
    ]
    final_dataset = final_dataset[schema_columns]

    print(
        f"Successfully converted {len(final_dataset)} programs from {len(raw_data)} input rows"
    )
    if validation_errors:
        print(f"Had {len(validation_errors)} validation/conversion errors")

    return final_dataset


def _convert_bq_nested_structure(bq_data) -> List:
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
        if (
            len(bq_data) > 0
            and isinstance(bq_data[0], dict)
            and "element" in bq_data[0]
        ):
            result = []
            for item in bq_data:
                if isinstance(item, dict) and "element" in item:
                    element = item["element"]
                    # Recursively convert nested structures
                    result.append(_convert_bq_nested_structure(element))
                else:
                    result.append(item)
            return result
        else:
            return bq_data

    # Handle BigQuery's nested structure
    if isinstance(bq_data, dict):
        if "list" in bq_data:
            list_data = bq_data["list"]

            # Convert numpy array to list if needed
            if hasattr(list_data, "tolist"):
                list_data = list_data.tolist()

            # Recursively process the list
            return _convert_bq_nested_structure(list_data)
        else:
            # Not a standard BigQuery list structure, return as is
            raise ValueError(f"Unexpected BigQuery format: {bq_data}")

    # For primitive values, return as is
    return bq_data


def _extract_boolean_values(bool_array) -> List[bool]:
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
        if isinstance(item, dict) and "element" in item:
            result.append(bool(item["element"]))
        else:
            result.append(bool(item))
    return result
