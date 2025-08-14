# Programs Database

Local DuckDB for storing ARC program instances with auto-deduplication and cloud sync to the superking dataset.

## How It Works

Automatically captures successful programs from arc runs from `run_arc_tasks.py` in a local duckdb, prevents duplicates using normalized code comparison, and provides a cli to sync to GCS. Schema details in [`schema.py`](schema.py).

## CLI Commands

```bash
# Show database stats
uv run python -m llm_python.programsdb.cli stats

# Clear database
uv run python -m llm_python.programsdb.cli clear

# Sync to GCS
uv run python -m llm_python.programsdb.cli sync

# Import from parquet
uv run python -m llm_python.programsdb.cli import /path/to/programs.parquet
uv run python -m llm_python.programsdb.cli import gs://bucket/path/programs.parquet
```

## Authentication

For cloud sync:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
# or
gcloud auth application-default login
```

## Database Location

Default: `llm_python/programsdb/local.db`

Custom path with: `--db-path /custom/path/db.duckdb`
