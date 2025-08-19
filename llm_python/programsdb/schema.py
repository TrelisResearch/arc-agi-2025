"""
Schema types for the programs database.
"""

from typing import TypedDict, List
import pyarrow as pa


class ProgramSample(TypedDict):
    """Schema for SOAR program examples stored in the database"""
    task_id: str  # Task ID from ARC
    reasoning: str  # Reasoning trace if provided (optional)
    code: str  # Program code that should define a `generate` function
    correct_train_input: List[bool]  # Training inputs where program produced correct output
    correct_test_input: List[bool]  # Test inputs where program produced correct output
    predicted_train_output: List[List[List[int]]]  # Program's predicted outputs for training inputs
    predicted_test_output: List[List[List[int]]]  # Program's predicted outputs for test inputs
    model: str  # What model generated this example

# PyArrow schema for parquet serialization/deserialization
PARQUET_SCHEMA = pa.schema(
    [
        ("task_id", pa.string()),
        ("reasoning", pa.string()),
        ("code", pa.string()),
        ("correct_train_input", pa.list_(pa.bool_())),
        ("correct_test_input", pa.list_(pa.bool_())),
        ("predicted_train_output", pa.list_(pa.list_(pa.list_(pa.int64())))),
        ("predicted_test_output", pa.list_(pa.list_(pa.list_(pa.int64())))),
        ("model", pa.string()),
    ]
)

__all__ = ['ProgramSample', 'PARQUET_SCHEMA']
