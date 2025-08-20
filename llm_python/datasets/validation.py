from typing import List, Tuple, Union

import pandas as pd

from llm_python.programsdb.schema import ProgramSample


def validate_soar_dataframe(data_frame: pd.DataFrame) -> Tuple[bool, str]:
    """Validate a DataFrame against the SOAR schema.

    Args:
        data_frame: DataFrame containing the converted data

    Returns:
        Tuple of (is_valid, error_message)
    """
    errors: List[str] = []
    for idx, row in data_frame.iterrows():
        is_valid, error_msg = validate_soar_sample(row.to_dict())
        if not is_valid:
            errors.append(f"Row {idx}: {error_msg}")

    if errors:
        return False, f"{len(errors)} validation error sample:\n" + "\n".join(
            errors[:10]
        )
    return True, f"All {len(data_frame)} rows are valid"


def validate_soar_sample(data_dict: Union[ProgramSample, dict]) -> Tuple[bool, str]:  # noqa: F821
    """Validate a single converted data dict against the SOAR schema.

    Args:
        data_dict: Dictionary containing the converted data

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Check required fields exist
        required_fields = [
            "task_id",
            "code",
            "model",
            "predicted_train_output",
            "predicted_test_output",
            "correct_train_input",
            "correct_test_input",
        ]
        for field in required_fields:
            if field not in data_dict:
                return False, f"Missing field: {field}"

        # Check string types
        for field in ["task_id", "code", "model"]:
            if not isinstance(data_dict[field], str):
                return False, f"{field} should be str, got {type(data_dict[field])}"

        # Check 3D arrays (List[List[List[int]]])
        for field in ["predicted_train_output", "predicted_test_output"]:
            arr = data_dict[field]
            if not isinstance(arr, list):
                return False, f"{field} should be list, got {type(arr)}"
            for i, grid in enumerate(arr):
                if not isinstance(grid, list):
                    return (
                        False,
                        f"{field}[{i}] should be list (2D grid), got {type(grid)}",
                    )
                for j, row in enumerate(grid):
                    if not isinstance(row, list):
                        return (
                            False,
                            f"{field}[{i}][{j}] should be list (row), got {type(row)}",
                        )
                    for k, cell in enumerate(row):
                        if not isinstance(cell, int):
                            return (
                                False,
                                f"{field}[{i}][{j}][{k}] should be int, got {type(cell)}",
                            )

        # Check boolean arrays
        for field in ["correct_train_input", "correct_test_input"]:
            arr = data_dict[field]
            if not isinstance(arr, list):
                return False, f"{field} should be list, got {type(arr)}"
            for i, val in enumerate(arr):
                if not isinstance(val, bool):
                    return False, f"{field}[{i}] should be bool, got {type(val)}"

        return True, "Valid"
    except Exception as e:
        return False, f"Validation error: {e}"
