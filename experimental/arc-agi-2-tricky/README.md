# ARC-AGI-2 Tricky Tasks Analysis

This directory contains analysis tools for identifying and visualizing "tricky" ARC-AGI tasks - those with limited numbers of programs achieving perfect correctness.

## Overview

The `analyze_tricky_tasks.py` script downloads the `Trelis/arc-agi-2-partial-100` dataset, filters it to identify challenging tasks, applies deduplication, and creates visualizations of program correctness distribution.

## Key Features

### 1. Task Filtering
- **Criteria**: Tasks with ≤10 programs achieving 100% correctness (percentage-based)
- **Percentage Calculation**: (train_correct + test_correct) / (train_total + test_total) * 100
- **Result**: 249 "tricky" tasks identified from the original dataset

### 2. Deduplication
- **Method**: Normalizes code by removing newlines, tabs, and extra whitespace
- **Scope**: Applied per-task after filtering
- **Utility**: Uses `llm_python.utils.deduplication` module
- **Results**: 0 duplicates found (dataset already well-deduplicated)

### 3. Program Selection
- **Strategy**: Top 10 highest-performing programs per task
- **Sorting**: By percentage correctness (desc), then code length (asc) for tie-breaking
- **Output**: 1,890 total programs across 249 tasks (avg 7.6 per task)

### 4. Visualization
- **Chart Type**: Stacked column chart with percentage buckets
- **Buckets**: <25%, 25-50%, 50-75%, >75% but <100%, 100%
- **Sorting**: By total programs, then by 100% programs (hierarchical)
- **Colors**: Dark green (100%) → Blue-gray (>75%) → Yellow → Orange → Red

## Usage

### Basic Analysis
```bash
uv run python analyze_tricky_tasks.py
```

### With Dataset Upload
```bash
uv run python analyze_tricky_tasks.py --upload
```

## Output Files

### Generated Files
- `output/correctness_distribution.png` - Visualization of program correctness by task
- `output/filtered_tricky_tasks.csv` - Filtered and deduplicated dataset
- `output/hf_cache/` - Cached Hugging Face dataset files

### Dataset Upload
- **Target Repository**: `Trelis/arc-agi-2-partial-100-tricky-10`
- **Format**: Hugging Face Dataset
- **Content**: Cleaned dataset without temporary analysis columns

## Key Statistics

### Final Dataset Composition
- **Tasks**: 249 tricky tasks (≤10 programs with 100% correctness)
- **Programs**: 1,890 total programs (top 10 per task)
- **Quality Distribution**:
  - 47.4% achieve 100% correctness (896 programs)
  - 25.8% achieve 50-75% correctness (488 programs)
  - 17.7% achieve 25-50% correctness (335 programs)
  - 4.1% achieve >75% but <100% (78 programs)
  - 4.9% achieve <25% correctness (93 programs)

### Deduplication Results
- **Original Programs**: 8,909
- **After Deduplication**: 8,909
- **Duplicates Removed**: 0 (0.0%)
- **Programs with Duplicates**: 0

## Technical Details

### Dependencies
- `pandas` - Data manipulation
- `matplotlib` - Visualization
- `numpy` - Numerical operations
- `datasets` - Hugging Face dataset handling
- `llm_python.utils.deduplication` - Custom deduplication utilities

### Filtering Logic
1. Download full `Trelis/arc-agi-2-partial-100` dataset (44,880 programs)
2. Calculate percentage correctness per program
3. Identify tasks with ≤10 programs achieving 100% correctness
4. Apply deduplication within each filtered task
5. Select top 10 programs per task by performance
6. Generate visualization and save results

### Color Scheme
- **Dark Green (#2e7d32)**: 100% perfect correctness
- **Blue-Gray (#90a4ae)**: >75% but <100% correctness
- **Yellow (#ffd54f)**: 50-75% correctness
- **Orange (#ffab40)**: 25-50% correctness
- **Red (#ff6b6b)**: <25% correctness

## Use Cases

This filtered dataset is ideal for:
- **Challenge Analysis**: Understanding what makes certain ARC tasks difficult
- **Model Evaluation**: Testing on tasks where even good models struggle
- **Algorithm Development**: Focusing improvement efforts on genuinely hard problems
- **Research**: Studying patterns in program synthesis failures and partial successes