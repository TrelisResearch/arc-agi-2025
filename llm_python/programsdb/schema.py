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
# All fields are required (nullable=False) except reasoning
PARQUET_SCHEMA = pa.schema(
    [
        ("task_id", pa.string(), False),  # Required
        ("reasoning", pa.string(), True),  # Optional - can be null
        ("code", pa.string(), False),  # Required
        ("correct_train_input", pa.list_(pa.bool_()), False),  # Required
        ("correct_test_input", pa.list_(pa.bool_()), False),  # Required
        ("predicted_train_output", pa.list_(pa.list_(pa.list_(pa.int64()))), False),  # Required
        ("predicted_test_output", pa.list_(pa.list_(pa.list_(pa.int64()))), False),  # Required
        ("model", pa.string(), False),  # Required
    ]
)

__all__ = ['ProgramSample', 'PARQUET_SCHEMA']
