import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from llm_python.datasets.io import write_soar_parquet
from llm_python.datasets.schema import ProgramSample
from llm_python.datasets.validation import validate_soar_dataframe, validate_soar_row
from llm_python.utils.code import normalize_code

_DEFAULT_DIR = Path(__file__).parent / "inference"


logger = logging.getLogger(__name__)

class SoarDatasetCollector:
    def __init__(self, name: Optional[str], flush_every=100):
        self.data: list[ProgramSample] = []
        self.name = name
        self.timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        self.flush_every = flush_every
        self.flush_at = flush_every

    def collect(self, item: ProgramSample) -> bool:
        validation_result = validate_soar_row(dict(item))
        if not validation_result.is_valid:
            logger.info(f"Invalid row, skipping collection due to errors: {validation_result.errors}")
            return False

        normalized = self.normalize_row(item)
        self.data.append(normalized)
        self.maybe_flush()
        return True

    def normalize_row(self, item: ProgramSample) -> ProgramSample:
        copy = item.copy()

        # Normalize the code
        if copy["code"]:
            copy["code"] = normalize_code(copy["code"])
        return copy

    def get_data(self) -> list[ProgramSample]:
        return self.data

    def output_path(self) -> Path:
        return (
            _DEFAULT_DIR
            / f"{self.timestamp}{'_' if self.name else ''}{self.name}.parquet"
        )

    def maybe_flush(self):
        if len(self.data) >= self.flush_at:
            self.flush()
            self.flush_at += self.flush_every

    def flush(self):
        df = pd.DataFrame(self.data)
        validate_soar_dataframe(df)
        self.output_path().parent.mkdir(parents=True, exist_ok=True)
        write_soar_parquet(df, self.output_path())
