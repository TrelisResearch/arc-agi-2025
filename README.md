# ARC-AGI 2025

This repository contains resources for working with the ARC-AGI (Abstraction and Reasoning Corpus for Artificial General Intelligence) datasets.

>[!IMPORTANT]
> This branch contains a frozen version of the o3-tools branch with a prompt to generate input and output grids AS WELL as the transformation function. It uses openai responses api so is only compatible with openai models.

## Branches

`o3-tools-images` a frozen version of the openai o3-tools branch that still maintains functionality for adding images to the prompts and feedback.

## Data

The `data/` folder contains:
- Complete ARC-AGI-1 dataset (800 tasks)
- Complete ARC-AGI-2 dataset (1,120 tasks)
- Predefined subsets for quick experimentation

For detailed information about the data structure, file formats, and how to work with the datasets, see the [data folder README](data/README.md).

## o3-tools

The `o3-tools/` folder contains a comprehensive testing framework for evaluating OpenAI models (o3, o4, GPT-4, etc.) on ARC-AGI tasks:

### Key Features

- **Parallel Execution**: Run up to 30 tasks simultaneously for dramatic speedup
- **Multiple Models**: Support for o3/o4, GPT-4 family, and other OpenAI models
- **Tool Integration**: Optional code interpreter tools for iterative problem solving
- **Three-Code Analysis**: Models generate input generators, output generators, and transformation functions for comprehensive pattern understanding
- **Comprehensive Scoring**: Binary correctness, pixel accuracy, and pattern learning metrics
- **Cost Tracking**: Detailed token usage and cost calculation with 6-decimal precision
- **Flexible Subsets**: Pre-defined task subsets by difficulty and size

### Three-Code Block Analysis

The framework now prompts models to generate three separate code blocks for each task:

1. **Input Grid Generator**: Reproduces all training input grids + the test input grid
2. **Output Grid Generator**: Reproduces all training output grids  
3. **Transformation Function**: Maps individual input grids to output grids

This provides detailed insight into model capabilities:
- **Input Pattern Understanding**: Can the model understand and reproduce input patterns?
- **Output Pattern Understanding**: Can the model understand and reproduce output patterns?
- **Transformation Logic**: Can the model correctly implement the input→output mapping?

Example output showing all three metrics:
```
📊 Training: 0/2 solved, 0.0% accuracy | Input Grids: 3/3 correctly described | Output Grids: 2/2 correctly described
```

This reveals that while the model perfectly understands input/output patterns (100% accuracy), it struggles with the transformation logic (0% accuracy), providing valuable diagnostic information for model improvement.

### Quick Start

```bash
# Install dependencies
cd o3-tools && uv sync

# Run a single task (fast test)
uv run python run_arc_tasks.py --dataset arc-agi-1 --subset shortest_1

# Run 10 tasks in parallel with 5 workers
uv run python run_arc_tasks.py --dataset arc-agi-2 --subset shortest_training_10 --max_workers 5

# Run with tools enabled for more capable solving
uv run python run_arc_tasks.py --subset shortest_10 --tools --max_workers 3
```

### Performance Benefits

Parallelization can provide significant speedups:
- **Sequential**: 30 tasks × 10 seconds = 5 minutes
- **10 workers**: 30 tasks ÷ 10 workers = 30 seconds  
- **30 workers**: 30 tasks ÷ 30 workers = 10 seconds

For complete documentation, usage examples, and detailed configuration options, see the [o3-tools README](o3-tools/README.md).