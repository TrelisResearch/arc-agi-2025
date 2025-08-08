from typing import TypedDict, List

import pyarrow as pa

class SoarProgramExample(TypedDict):
    """Schema for training examples enriched for actual training with separate train/test data"""
    task_id: str # Task ID from ARC
    reasoning: str # Reasoning trace if provided
    code: str # Program code that should define a `generate` function
    correct_train_input: List[bool] # Training inputs where program produced correct output
    correct_test_input: List[bool] # Test inputs where program produced correct output
    predicted_train_output: List[List[List[int]]] # Program's predicted outputs for training inputs
    predicted_test_output: List[List[List[int]]] # Program's predicted outputs for test inputs
    model: str # What model generated this example


# Define explicit PyArrow schema for our parquet file
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

