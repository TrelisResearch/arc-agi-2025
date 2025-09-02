# Partial Improvement Analysis

This folder contains scripts to analyze the correctness distribution of programs across different parquet files, focusing on partial correctness improvements between models.

## Scripts

### `analyze_correctness_distribution.py`
Main analysis script that compares correctness distributions between two parquet files for a specific task ID.

**Features:**
- Filters out transductive programs (focuses on `is_transductive == False`)
- Shows complete breakdown from 0-correct to max-correct for each file
- Displays results as both counts and percentages
- Compares base model vs fine-tuned model performance

**Usage:**
```bash
# Edit the task_id variable in the script, then run:
uv run python experimental/partial-improvement/analyze_correctness_distribution.py
```

**Output format:**
- Total non-transductive programs
- Programs with 0 correct, 1 correct, 2 correct, etc.
- Percentages of total and percentages of programs with ≥1 correct

### `check_task_135a2760.py`
Legacy script for basic correctness checking (kept for reference).

## Analyzed Parquet Files

1. **Base model**: `20250830_082909_Trelis_Qwen3-4B_ds-arc-agi-2-partial-100-c2806_arc-prize-2025_evaluation.parquet`
2. **Fine-tuned model**: `20250830_114154__workspace_arc-agi-2025_llm_python_fine-tuning_Qwen3-4B_ds-arc-agi-2-partial-100-c2806_ds-inference-final_arc-prize-2025_evaluation.parquet`

## Results Summary

### Task 135a2760 (Max 3 correct per program)
- **Base model**: 44.8% success rate, only 1-correct solutions
- **Fine-tuned**: 99.2% success rate, only 1-correct solutions

### Task 981571dc (Max 5 correct per program)
- **Base model**: 57.3% success rate, diverse distribution (1-4 correct)
- **Fine-tuned**: 89.8% success rate, improved 4-correct performance

### Task 4c7dc4dd (Max 4 correct per program)
- **Base model**: 2.4% success rate, very challenging
- **Fine-tuned**: 16.7% success rate, enables 1-correct and 2-correct

## Key Insights

1. Fine-tuning consistently improves success rates across all analyzed tasks
2. Different tasks show varying levels of partial correctness
3. Some tasks (like 135a2760) cap out at 1-correct, while others (like 981571dc) allow for higher correctness counts
4. Task difficulty varies significantly (4c7dc4dd is much harder than the others)

## Dependencies

- Uses `llm_python.datasets.io.read_soar_parquet()` for reading parquet files
- Requires the project's virtual environment with `uv run`