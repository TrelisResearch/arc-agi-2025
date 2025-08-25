# Project tips for Claude Code

- 🐍 **Python**: Always use `uv` — use `uv pip` (not `pip`) and `uv run` (not `python`).
- 🗂️ **Working dir**: Assume commands run from the **repo root**.

## Examples
- Install: `uv pip install -r requirements.txt`
- Run:     `uv run python scripts/foo.py --arg ...`