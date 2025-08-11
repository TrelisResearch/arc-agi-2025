import pytest
import pandas as pd
import tempfile
import os

from llm_python.datasets.schema import validate_soar_dataset, ValidationError


def create_test_dataset(valid=True, **overrides):
    """Create a test dataset for validation testing"""
    base_data = {
        "task_id": ["task_001", "task_002"],
        "reasoning": ["Test reasoning 1", "Test reasoning 2"],
        "code": ["def generate(x): return x", "def generate(x): return [[0]]"],
        "model": ["test_model", "test_model"],
        "predicted_train_output": [
            [[[1, 0], [0, 1]]],  # Single 2x2 grid
            [[[0]]],  # Single 1x1 grid
        ],
        "predicted_test_output": [
            [[[0, 1], [1, 0]]],  # Single 2x2 grid
            [[[1]]],  # Single 1x1 grid
        ],
        "correct_train_input": [
            [True, False],  # Two training examples
            [True],  # One training example
        ],
        "correct_test_input": [
            [True],  # One test example
            [False],  # One test example
        ],
    }

    if not valid:
        # Apply overrides to make data invalid
        for key, value in overrides.items():
            if key in base_data:
                base_data[key] = value

    return base_data


def save_test_data_to_parquet(data, file_path):
    """Save test data to parquet with proper schema"""
    from llm_python.datasets.schema import PARQUET_SCHEMA
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(data, schema=PARQUET_SCHEMA)
    pq.write_table(table, file_path)


@pytest.fixture
def temp_parquet_file():
    """Create a temporary parquet file"""
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        yield f.name
    if os.path.exists(f.name):
        os.unlink(f.name)


def test_valid_dataset_passes(temp_parquet_file):
    """Test that a valid dataset passes validation"""
    data = create_test_dataset(valid=True)
    save_test_data_to_parquet(data, temp_parquet_file)

    df = pd.read_parquet(temp_parquet_file)
    validate_soar_dataset(df, silent=True)  # Should not raise


def test_empty_dataset_fails(temp_parquet_file):
    """Test that empty dataset causes validation failure"""
    empty_data = {
        "task_id": [],
        "reasoning": [],
        "code": [],
        "model": [],
        "predicted_train_output": [],
        "predicted_test_output": [],
        "correct_train_input": [],
        "correct_test_input": [],
    }
    save_test_data_to_parquet(empty_data, temp_parquet_file)

    df = pd.read_parquet(temp_parquet_file)
    with pytest.raises(ValidationError):
        validate_soar_dataset(df, silent=True)


def test_oversized_grids_fail(temp_parquet_file):
    """Test that oversized grids cause validation failure"""
    large_grid = [[[0] * 50] * 50]  # 50x50 grid
    data = create_test_dataset(
        valid=False, predicted_train_output=[large_grid, [[[0]]]]
    )
    save_test_data_to_parquet(data, temp_parquet_file)

    df = pd.read_parquet(temp_parquet_file)
    with pytest.raises(ValidationError):
        validate_soar_dataset(df, max_grid_size=40, silent=True)
