import logging
from pathlib import Path
from typing import Optional
import threading

import pandas as pd

from llm_python.datasets.io import write_soar_parquet
from llm_python.datasets.schema import ProgramSample
from llm_python.datasets.validation import validate_soar_dataframe, validate_soar_row
from llm_python.utils.code import normalize_code
import secrets
_DEFAULT_DIR = Path(__file__).parent / "inference"


logger = logging.getLogger(__name__)

def generate_unique_hex_id(hex_length=32):
    # 32 hex chars = 128 bits, collision probability is negligible for 10M rows
    return secrets.token_hex(hex_length // 2)

class SoarDatasetCollector:
    def __init__(self, name: Optional[str], flush_every=1000, output_dir: Optional[Path] = None):
        self.data: list[ProgramSample] = []
        self.name = name
        self.timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        self.flush_every = flush_every
        self.flush_at = flush_every
        self.output_dir = output_dir or _DEFAULT_DIR

        self._flush_at_lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._flush_thread: Optional[threading.Thread] = None

    def collect(self, item: ProgramSample) -> bool:
        normalized = self.normalize_row(item)
        validation_result = validate_soar_row(dict(normalized))
        if not validation_result.is_valid:
            logger.info(f"Invalid row, skipping collection due to errors: {validation_result.errors}")
            return False

        self.data.append(normalized)
        self.maybe_flush_async()
        return True

    def normalize_row(self, item: ProgramSample) -> ProgramSample:
        copy = item.copy()

        # Add a UUID.
        copy["row_id"] = generate_unique_hex_id()

        # Normalize the code
        if copy["code"]:
            copy["code"] = normalize_code(copy["code"])
        return copy

    def get_data(self) -> list[ProgramSample]:
        return self.data

    def output_path(self) -> Path:
        return (
            self.output_dir
            / f"{self.timestamp}{'_' if self.name else ''}{self.name}.parquet"
        )

    def maybe_flush_async(self):
        if len(self.data) < self.flush_at:
            return
        with self._flush_at_lock:
            if len(self.data) < self.flush_at:
                return
            self.flush_at += self.flush_every
            def flush_worker():
                self.flush()
            self._flush_thread = threading.Thread(target=flush_worker, daemon=True)
            self._flush_thread.start()

    def flush(self):
        with self._flush_lock:
            # Make a copy of the data to avoid mutation during flush
            data_copy = self.data.copy()
            expected_keys = {"task_id", "reasoning", "code", "correct_train_input", 
                            "correct_test_input", "predicted_train_output", 
                            "predicted_test_output", "model", "is_transductive"}
            
            # Debug: Check for inconsistent samples before normalization
            for i, sample in enumerate(data_copy):
                sample_keys = set(sample.keys())
                if sample_keys != expected_keys:
                    missing = expected_keys - sample_keys
                    extra = sample_keys - expected_keys
                    logger.warning(
                        f"Sample {i} has inconsistent keys - "
                        f"missing: {missing}, extra: {extra}, "
                        f"task_id: {sample.get('task_id', 'UNKNOWN')}"
                    )
            
            normalized_data = [
                {
                    "row_id": sample.get("row_id"),
                    "task_id": sample.get("task_id"),
                    "reasoning": sample.get("reasoning"),  # Optional field - can be None
                    "code": sample.get("code"),
                    "correct_train_input": sample.get("correct_train_input"),
                    "correct_test_input": sample.get("correct_test_input"),
                    "predicted_train_output": sample.get("predicted_train_output"),
                    "predicted_test_output": sample.get("predicted_test_output"),
                    "model": sample.get("model"),
                    "is_transductive": sample.get("is_transductive"),
                }
                for sample in data_copy
            ]
            
            df = pd.DataFrame(normalized_data)
            
            # Check validation result
            validation_result = validate_soar_dataframe(df)
            if not validation_result.is_valid():
                logger.warning(f"DataFrame validation failed: {validation_result.summary()}")
                # Continue anyway but log the issue
            
            try:
                self.output_path().parent.mkdir(parents=True, exist_ok=True)
                write_soar_parquet(df, self.output_path())
                logger.warning(f"✅ Checkpointed {len(df)} programs to {self.output_path()}")
            except Exception as e:
                logger.error(f"❌ FAILED to write parquet: {e}")
                # Don't clear data on failure so it can be retried
                return
