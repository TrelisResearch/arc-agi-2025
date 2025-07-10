# ARC-AGI Data Directory

This directory contains the complete datasets for ARC-AGI-1 and ARC-AGI-2, along with predefined subsets for efficient experimentation.

## Directory Structure

```
data/
├── arc-agi-1/              # ARC-AGI-1 dataset (original ARC)
│   ├── training/           # 400 training tasks
│   └── evaluation/         # 400 evaluation tasks
├── arc-agi-2/              # ARC-AGI-2 dataset (2024 release)
│   ├── training/           # 1,000 training tasks
│   └── evaluation/         # 120 evaluation tasks
└── subsets/                # Predefined subsets for both datasets
    ├── arc-agi-1/
    │   ├── shortest_evaluation_1.txt           # Task ID of the shortest evaluation task
    │   ├── shortest_evaluation_10.txt          # Task IDs of 10 shortest evaluation tasks
    │   ├── shortest_evaluation_30.txt          # Task IDs of 30 shortest evaluation tasks
    │   ├── shortest_evaluation_100.txt         # Task IDs of 100 shortest evaluation tasks
    │   ├── shortest_training_1.txt             # Task ID of the shortest training task
    │   ├── shortest_training_10.txt            # Task IDs of 10 shortest training tasks
    │   ├── shortest_training_30.txt            # Task IDs of 30 shortest training tasks
    │   ├── shortest_training_100.txt           # Task IDs of 100 shortest training tasks
    │   ├── middle_evaluation_1.txt             # Task ID of the median evaluation task
    │   ├── middle_evaluation_10.txt            # Task IDs of 10 median evaluation tasks
    │   ├── middle_evaluation_30.txt            # Task IDs of 30 median evaluation tasks
    │   ├── middle_training_1.txt               # Task ID of the median training task
    │   ├── middle_training_10.txt              # Task IDs of 10 median training tasks
    │   ├── middle_training_30.txt              # Task IDs of 30 median training tasks
    │   ├── longest_evaluation_1.txt            # Task ID of the longest evaluation task
    │   ├── longest_evaluation_10.txt           # Task IDs of 10 longest evaluation tasks
    │   ├── longest_evaluation_30.txt           # Task IDs of 30 longest evaluation tasks
    │   ├── longest_training_1.txt              # Task ID of the longest training task
    │   ├── longest_training_10.txt             # Task IDs of 10 longest training tasks
    │   ├── longest_training_30.txt             # Task IDs of 30 longest training tasks
    │   ├── grid_size_distributed_30_evaluation.txt # 30 evaluation tasks evenly distributed by grid size
    │   ├── grid_size_distributed_30_training.txt   # 30 training tasks evenly distributed by grid size
    │   ├── *_details.json files                # Corresponding detailed info files for each subset
    │   └── tasks_with_multiple_tests.json      # Tasks with >1 test example
    └── arc-agi-2/
        ├── shortest_evaluation_1.txt           # Task ID of the shortest evaluation task
        ├── shortest_evaluation_10.txt          # Task IDs of 10 shortest evaluation tasks
        ├── shortest_evaluation_30.txt          # Task IDs of 30 shortest evaluation tasks
        ├── shortest_training_1.txt             # Task ID of the shortest training task
        ├── shortest_training_10.txt            # Task IDs of 10 shortest training tasks
        ├── shortest_training_30.txt            # Task IDs of 30 shortest training tasks
        ├── middle_evaluation_1.txt             # Task ID of the median evaluation task
        ├── middle_evaluation_10.txt            # Task IDs of 10 median evaluation tasks
        ├── middle_evaluation_30.txt            # Task IDs of 30 median evaluation tasks
        ├── middle_training_1.txt               # Task ID of the median training task
        ├── middle_training_10.txt              # Task IDs of 10 median training tasks
        ├── middle_training_30.txt              # Task IDs of 30 median training tasks
        ├── longest_evaluation_1.txt            # Task ID of the longest evaluation task
        ├── longest_evaluation_10.txt           # Task IDs of 10 longest evaluation tasks
        ├── longest_evaluation_30.txt           # Task IDs of 30 longest evaluation tasks
        ├── longest_training_1.txt              # Task ID of the longest training task
        ├── longest_training_10.txt             # Task IDs of 10 longest training tasks
        ├── longest_training_30.txt             # Task IDs of 30 longest training tasks
        ├── grid_size_distributed_30_evaluation.txt # 30 evaluation tasks evenly distributed by grid size
        ├── grid_size_distributed_30_training.txt   # 30 training tasks evenly distributed by grid size
        ├── *_details.json files                # Corresponding detailed info files for each subset
        └── tasks_with_multiple_tests.json      # Tasks with >1 test example
```

## File Naming Convention

All task files follow the naming pattern: `{task_id}.json`
- Task IDs are 8-character hexadecimal strings (e.g., `007bbfb7.json`)
- Each JSON file contains exactly one task

## Task Format

Each task file is a JSON object with the following structure:

```json
{
  "train": [
    {
      "input": [[grid_values]],
      "output": [[grid_values]]
    },
    // ... more training examples
  ],
  "test": [
    {
      "input": [[grid_values]],
      "output": [[grid_values]]
    },
    // ... more test examples (usually just 1)
  ]
}
```

### Key Properties:
- **Grid Values**: Integers from 0-9 representing different colors
- **Grid Size**: Minimum 1x1, Maximum 30x30
- **Training Examples**: Typically 2-5 examples per task (max observed: 10)
- **Test Examples**: Usually 1 example per task (see below for exceptions)

## Dataset Statistics

### ARC-AGI-1 (800 tasks total)
- **Training set**: 400 tasks
- **Evaluation set**: 400 tasks
- **Training examples per task**: 2-10 (most common: 3 examples)
- **Test examples per task**: 1-3 (most have 1)
- **Tasks with multiple test examples**: 33 tasks
  - 2 test examples: 31 tasks
  - 3 test examples: 2 tasks (`ff28f65a`, `27a28665`)

### ARC-AGI-2 (1,120 tasks total)
- **Training set**: 1,000 tasks
- **Evaluation set**: 120 tasks
- **Training examples per task**: 2-10 (most common: 3 examples)
- **Test examples per task**: 1-4 (most have 1)
- **Tasks with multiple test examples**: 114 tasks
  - 2 test examples: 106 tasks
  - 3 test examples: 7 tasks
  - 4 test examples: 1 task (`8dab14c2`)

## Subset Naming Convention (2025+)

Subsets are now split by training and evaluation for each dataset:
- `shortest_training_1`, `shortest_training_10`, `shortest_training_30`, `shortest_training_100` (arc-agi-1 only)
- `shortest_evaluation_1`, `shortest_evaluation_10`, `shortest_evaluation_30`, `shortest_evaluation_100` (arc-agi-1 only)
- ... and similarly for `middle` and `longest` (up to 30 tasks)
- `grid_size_distributed_30_training`: 30 training tasks evenly distributed by grid size
- `grid_size_distributed_30_evaluation`: 30 evaluation tasks evenly distributed by grid size

Each subset contains only task IDs from the relevant split. Legacy mixed subsets are in the `archive/` folder.

**Grid Size Distributed Subsets:**
These subsets select 30 tasks evenly spaced across the range of grid sizes (total cells in all input/output grids). They provide a balanced representation of task complexity within each split.

**Example usage:**

Run the 10 shortest evaluation tasks from ARC-AGI-2:
```bash
uv run python o3-tools/run_arc_tasks.py --dataset arc-agi-2 --subset shortest_evaluation_10
```

Run the 30 longest training tasks from ARC-AGI-1:
```bash
uv run python o3-tools/run_arc_tasks.py --dataset arc-agi-1 --subset longest_training_30
```

Run the 100 shortest evaluation tasks from ARC-AGI-1:
```bash
uv run python o3-tools/run_arc_tasks.py --dataset arc-agi-1 --subset shortest_evaluation_100
```

Run 30 grid size distributed evaluation tasks from ARC-AGI-2:
```bash
uv run python o3-tools/run_arc_tasks.py --dataset arc-agi-2 --subset grid_size_distributed_30_evaluation
```

## Task Size Definition
**Task size** is calculated as the total number of grid cells across all inputs and outputs in a task:
- Size = sum of (width × height) for all input grids + sum of (width × height) for all output grids
- This includes all training examples and all test examples
- For example, a task with 3 training examples of 3×3 grids (input+output each) and 1 test example of 3×3 grids (input+output) would have size: 3×(3×3×2) + 1×(3×3×2) = 54 + 18 = 72 cells

### Available Subsets

The following subset files are available:

**For arc-agi-1 (in `data/subsets/arc-agi-1/`):**
- **shortest_training_1.txt**: Single shortest training task
- **shortest_training_10.txt**: 10 shortest training tasks  
- **shortest_training_30.txt**: 30 shortest training tasks
- **shortest_training_100.txt**: 100 shortest training tasks
- **shortest_evaluation_1.txt**: Single shortest evaluation task
- **shortest_evaluation_10.txt**: 10 shortest evaluation tasks
- **shortest_evaluation_30.txt**: 30 shortest evaluation tasks
- **shortest_evaluation_100.txt**: 100 shortest evaluation tasks
- Similar files for **middle** and **longest** (up to 30 tasks each)
- **grid_size_distributed_30_training.txt**: 30 training tasks evenly distributed by grid size
- **grid_size_distributed_30_evaluation.txt**: 30 evaluation tasks evenly distributed by grid size

**For arc-agi-2 (in `data/subsets/arc-agi-2/`):**
- **shortest_training_1.txt**: Single shortest training task
- **shortest_training_10.txt**: 10 shortest training tasks  
- **shortest_training_30.txt**: 30 shortest training tasks
- **shortest_evaluation_1.txt**: Single shortest evaluation task
- **shortest_evaluation_10.txt**: 10 shortest evaluation tasks
- **shortest_evaluation_30.txt**: 30 shortest evaluation tasks
- Similar files for **middle** and **longest** (up to 30 tasks each)
- **grid_size_distributed_30_training.txt**: 30 training tasks evenly distributed by grid size
- **grid_size_distributed_30_evaluation.txt**: 30 evaluation tasks evenly distributed by grid size

**For both datasets:**
- **tasks_with_multiple_tests.json**: Tasks with more than one test example (JSON, not a subset for evaluation)

Each `.txt` file contains one task ID per line. The corresponding `_details.json` files include additional metadata like the computed size and which split (training/evaluation) the task belongs to.

## Working with Tasks

### Loading a Single Task
```python
import json

with open('data/arc-agi-1/training/007bbfb7.json', 'r') as f:
    task = json.load(f)

# Access training examples
for i, example in enumerate(task['train']):
    input_grid = example['input']
    output_grid = example['output']
    print(f"Training example {i+1}: {len(input_grid)}x{len(input_grid[0])} -> {len(output_grid)}x{len(output_grid[0])}")

# Access test examples
for i, example in enumerate(task['test']):
    input_grid = example['input']
    output_grid = example['output']
    print(f"Test example {i+1}: {len(input_grid)}x{len(input_grid[0])} -> {len(output_grid)}x{len(output_grid[0])}")
```

### Loading a Subset
```python
# Read task IDs from subset file
with open('data/subsets/arc-agi-1/shortest_10.txt', 'r') as f:
    task_ids = [line.strip() for line in f]

# Load tasks
tasks = []
for task_id in task_ids:
    # Check both training and evaluation directories
    for split in ['training', 'evaluation']:
        filepath = f'data/arc-agi-1/{split}/{task_id}.json'
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                tasks.append(json.load(f))
            break
```

## Important Notes

1. **Grid coordinates**: Grids are 2D arrays where the first index is the row (y-coordinate) and the second is the column (x-coordinate).

2. **Color mapping**: The integers 0-9 typically represent:
   - 0: Black (background)
   - 1-9: Various colors

3. **Task complexity**: "Shortest" tasks (by total grid size) are not necessarily the easiest to solve. They're useful for quick testing and debugging.

## Data Sources

- ARC-AGI-1: https://github.com/fchollet/ARC-AGI
- ARC-AGI-2: https://github.com/arcprize/ARC-AGI-2

## grid_size_distributed_30 Subset

This subset contains 30 tasks from each of the arc-agi-1 and arc-agi-2 evaluation sets, selected to be evenly distributed by grid size. For each task, the grid size is defined as the sum of the number of cells in the first input and first output grid (from the first training example). This provides a balanced benchmark across a range of task complexities.

- The selected task IDs are listed in:
  - `data/subsets/arc-agi-1/grid_size_distributed_30.txt`
  - `data/subsets/arc-agi-2/grid_size_distributed_30.txt`
- The actual task files are in:
  - `data/subsets/grid_size_distributed_30/arc-agi-1/`
  - `data/subsets/grid_size_distributed_30/arc-agi-2/`
- A manifest with filenames and grid sizes is in:
  - `data/subsets/grid_size_distributed_30/manifest.json`

### How to use this subset

To run with this subset, use the following command:

```
uv run python o3-tools/run_arc_tasks.py --dataset arc-agi-1 --subset grid_size_distributed_30 --model o3 --tools
```

or for arc-agi-2:

```
uv run python o3-tools/run_arc_tasks.py --dataset arc-agi-2 --subset grid_size_distributed_30 --model o3 --tools
```

This will run the 30 selected tasks for the specified dataset, evenly distributed by grid size.