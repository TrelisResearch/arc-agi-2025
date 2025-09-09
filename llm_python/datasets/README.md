## llm_python/datasets

This folder contains utilities for working with ARC/Soar datasets.

### Collector Class

Use the `Collector` class to gather and process dataset samples. See the source for usage patterns and customization options.

This is used by default by the task runner. All sampled programs are dumped by default into `llm_python/datasets/inference/` as parquet files with timestamps and run names.

### Reading & Writing SOAR Parquet Files

Use the IO utilities to read and write SOAR-format parquet files:

```python
from llm_python.datasets.io import read_soar_parquet, write_soar_parquet
df = read_soar_parquet('path/to/file.parquet')
write_soar_parquet(df, 'path/to/output.parquet')
```

### Viewing SOAR Parquet Files

To quickly view a SOAR parquet file, use the viewer script:

```bash
uv run python -m llm_python.datasets.viewer path/to/file.parquet
```

### Syncing local inference parquet files to superking (GCS)

```bash
gcloud auth login --no-launch-browser
gcloud config set project trelis-arc
gsutil -m cp llm_python/datasets/inference/* gs://trelis-arc/datasets/superking/
```
