# Project tips for Claude Code

- 🐍 **Python**: Always use `uv` — use `uv pip` (not `pip`) and `uv run` (not `python`).
- 🗂️ **Working dir**: Assume commands run from the **repo root**.
- 🦆 **Databases**: We use **DuckDB** (`.duckdb` files). Prefer DuckDB SQL features and Python’s `duckdb` package.

## Examples
- Install: `uv pip install -r requirements.txt`
- Run:     `uv run python scripts/foo.py --arg ...`
- Query:   `uv run python -c "import duckdb as d; con=d.connect('data/app.duckdb'); print(con.sql('select 1').df())"`