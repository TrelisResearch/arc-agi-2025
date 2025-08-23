from pathlib import Path
from typing import Optional, Union

import pandas as pd
from llm_python.datasets.io import write_soar_parquet
from llm_python.datasets.validation import validate_soar_dataframe, validate_soar_row
from llm_python.programsdb.schema import ProgramSample
import uuid

_DEFAULT_DIR = Path(__file__).parent / "samples"


class SoarDatasetCollector:
    def __init__(self, name: Optional[str], flush_every=100):
        self.data: list[ProgramSample] = []
        self.name = name
        self.timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M_%S")
        self.flush_every = flush_every
        self.flush_at = flush_every

    def collect(self, item: Union[ProgramSample, dict]) -> bool:
        if not validate_soar_row(item):
            return False
        self.data.append(item)
        if len(self.data) >= self.flush_at:
            self.flush()
            self.flush_at += self.flush_every
        return True

    def get_data(self) -> list[ProgramSample]:
        return self.data

    def output_path(self) -> Path:
        return (
            _DEFAULT_DIR
            / f"{self.timestamp}{'_' if self.name else ''}{self.name}.parquet"
        )

    def flush(self):
        df = pd.DataFrame(self.data)
        validate_soar_dataframe(df)
        write_soar_parquet(df, self.output_path())
