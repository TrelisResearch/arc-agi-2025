import pytest
import tempfile
from pathlib import Path
import time

from llm_python.datasets.collector import SoarDatasetCollector


@pytest.fixture
def temp_output_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def minimal_sample():
    return {
        "task_id": "task_001",
        "reasoning": None,
        "code": "def generate(): return [[1]]",
        "correct_train_input": [True],
        "correct_test_input": [True],
        "predicted_train_output": [[[1]]],
        "predicted_test_output": [[[1]]],
        "model": "test_model",
        "is_transductive": False,
    }


def test_collector_flush_and_output(temp_output_dir):
    collector = SoarDatasetCollector(
        name="pytest", flush_every=2, output_dir=temp_output_dir
    )
    # Collect two samples to trigger flush
    assert collector.collect(minimal_sample())
    assert collector.collect(minimal_sample())
    # Wait for flush thread to finish
    time.sleep(0.5)
    output_file = collector.output_path()
    assert output_file.exists()
    # Check that the output file is a parquet file
    assert output_file.suffix == ".parquet"
