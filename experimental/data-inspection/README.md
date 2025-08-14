# ARC Programs Dataset Analysis

This folder contains analysis of the `Trelis/arc-programs-50-full-200-partial` dataset from HuggingFace.

## Files

- `analyze_dataset.py` - Main analysis script that downloads data and generates plots
- `task_analysis_plots.png` - Generated visualization plots
- `processed_data.csv` - Full dataset with calculated metrics (69,483 rows)
- `task_metrics.csv` - Task-level aggregated metrics (399 rows)

## Dataset Summary

- **Total programs**: 69,483 across 399 ARC tasks
- **Average programs per task**: 174.1
- **Task difficulty range**: 20-100% correct (mean: 56.14%)
- **Program length range**: 42-48,422 characters (mean: 1,099)

## Key Metrics

### Task Difficulty
Average percentage correct across all programs for each task.

### Reference Minimum Program Length
For each task, the shortest program among those with the highest correctness percentage. This avoids using short but low-quality programs as the baseline.

### Mean/Min Length Ratio
Ratio of average program length to reference minimum length for each task. Shows how "bloated" typical solutions are compared to the best compact solution.

## Key Findings

1. **Task Difficulty vs Average Program Length**: Weak negative correlation (-0.255) - harder tasks tend to have slightly longer programs.

2. **Task Difficulty vs Mean/Min Ratio**: Positive correlation (0.166) - interesting pattern where mid-difficulty tasks (40-60%) show the highest ratios (8-12x), suggesting these tasks have the most solution diversity in terms of code length.

## Usage

```bash
uv run python analyze_dataset.py
```

The script will download the dataset, calculate metrics, generate plots, and save processed data files.