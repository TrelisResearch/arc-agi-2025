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
# --- Install Google Cloud CLI (Ubuntu/Debian container) ---
apt-get update && apt-get install -y curl

# Install into /opt so the path is predictable
cd /opt
curl -sSL https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-537.0.0-linux-x86_64.tar.gz \
  -o google-cloud-cli.tar.gz
tar -xzf google-cloud-cli.tar.gz && rm google-cloud-cli.tar.gz
/opt/google-cloud-sdk/install.sh --quiet

# Make it available now and on future shells
export PATH="/opt/google-cloud-sdk/bin:$PATH"
echo 'source /opt/google-cloud-sdk/path.bash.inc' >> ~/.bashrc
echo 'source /opt/google-cloud-sdk/completion.bash.inc' >> ~/.bashrc

# Optional: verify install
gcloud --version
gsutil --version

gcloud auth login --no-launch-browser
gcloud config set project trelis-arc
gsutil -m cp llm_python/datasets/inference/* gs://trelis-arc/datasets/superking/
gsutil -m cp llm_python/datasets/rewritten/* gs://trelis-arc/datasets/superking/
```