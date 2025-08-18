# Programs Database

Local DuckDB for storing ARC program instances with auto-deduplication and cloud sync to the superking dataset.

## How It Works

Automatically captures successful programs from arc runs from `run_arc_tasks.py` in a local duckdb, prevents duplicates using normalized code comparison, and provides a cli to sync to GCS. Schema details in [`schema.py`](schema.py).

⚠️ **WARNING**: Currently, programs that are test-transductive (cheating on test data) are still saved to the database WITHOUT any flag indicating they are transductive. This needs to be fixed by adding transduction flags to the database schema. Train-transductive programs are now filtered out and not saved.

## Database Management Scripts

### validate_duckdb.py
Validates DuckDB databases and checks their schemas and data integrity.

```bash
# Validate all .db files in current directory
uv run python validate_duckdb.py

# Validate specific database files
uv run python validate_duckdb.py db1.db db2.db

# Show detailed schema information
uv run python validate_duckdb.py -v

# Validate databases in a specific directory
uv run python validate_duckdb.py -d /path/to/databases
```

### merge_duckdb.py
Merges multiple DuckDB databases into a single consolidated database.

```bash
# Merge all .db files in current directory into local.db
uv run python merge_duckdb.py local.db

# Merge specific databases
uv run python merge_duckdb.py consolidated.db source1.db source2.db source3.db

# Merge with verbose output
uv run python merge_duckdb.py local.db -v

# Merge databases from a specific directory
uv run python merge_duckdb.py local.db -d /path/to/databases

# Merge a different table (default is 'programs')
uv run python merge_duckdb.py local.db -t my_table

# Disable deduplication (include all rows)
uv run python merge_duckdb.py local.db --no-dedup

# Use a different column for deduplication (default is 'key')
uv run python merge_duckdb.py local.db -k task_id
```

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
