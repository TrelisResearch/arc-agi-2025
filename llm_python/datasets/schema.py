from typing import List, Optional, TypedDict
import pyarrow as pa


PARQUET_SCHEMA = pa.schema(
    (
        pa.field("row_id", pa.string(), nullable=False),  # Required
        pa.field("task_id", pa.string(), nullable=False),  # Required
        pa.field("reasoning", pa.large_string(), nullable=True),  # Optional - can be null
        pa.field("program", pa.large_string(), nullable=False),  # Required
        pa.field(
            "correct_train_input", pa.list_(pa.bool_()), nullable=False
        ),  # Required
        pa.field(
            "correct_test_input", pa.list_(pa.bool_()), nullable=False
        ),  # Required
        pa.field(
            "predicted_train_output",
            pa.list_(pa.list_(pa.list_(pa.int64()))),
            nullable=False,
        ),  # Required
        pa.field(
            "predicted_test_output",
            pa.list_(pa.list_(pa.list_(pa.int64()))),
            nullable=False,
        ),  # Required
        pa.field("model", pa.string(), nullable=False),  # Required
        pa.field("is_transductive", pa.bool_(), nullable=False),  # Required
        pa.field("refined_from_id", pa.string(), nullable=True),  # Optional
        pa.field("compound_inspiration_id", pa.string(), nullable=True),  # Optional
    )
)

REFINEMENT_PARQUET_SCHEMA = pa.schema(
    (
        [field for field in PARQUET_SCHEMA]
        + [
            pa.field("program_original", pa.string(), nullable=True),
            pa.field(
                "predicted_train_output_original",
                pa.list_(pa.list_(pa.list_(pa.int64()))),
                nullable=True,
            ),  # Required
            pa.field(
                "predicted_test_output_original",
                pa.list_(pa.list_(pa.list_(pa.int64()))),
                nullable=True,
            ),  # Required
        ]
    )
)


class ProgramSample(TypedDict):
    """Schema for SOAR program examples stored in the database"""

    row_id: str  # Unique row ID for this example
    task_id: str  # Task ID from ARC
    reasoning: Optional[str]  # Reasoning trace if provided (optional)
    program: str  # Natural language program description
    correct_train_input: List[
        bool
    ]  # Training inputs where program produced correct output
    correct_test_input: List[bool]  # Test inputs where program produced correct output
    predicted_train_output: List[
        List[List[int]]
    ]  # Program's predicted outputs for training inputs
    predicted_test_output: List[
        List[List[int]]
    ]  # Program's predicted outputs for test inputs
    model: str  # What model generated this example
    is_transductive: bool  # Whether program hardcodes outputs (transductive)
    refined_from_id: Optional[
        str
    ]  # Row ID of the example this was refined from (if applicable)
