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
    │   ├── shortest_1.txt              # Task ID of the shortest task
    │   ├── shortest_10.txt             # Task IDs of 10 shortest tasks
    │   ├── shortest_100.txt            # Task IDs of 100 shortest tasks
    │   ├── shortest_1_details.json     # Detailed info for shortest task
    │   ├── shortest_10_details.json    # Detailed info for 10 shortest tasks
    │   ├── shortest_100_details.json   # Detailed info for 100 shortest tasks
    │   └── tasks_with_multiple_tests.json  # Tasks with >1 test example
    └── arc-agi-2/
        ├── shortest_1.txt              # Task ID of the shortest task
        ├── shortest_10.txt             # Task IDs of 10 shortest tasks
        ├── shortest_100.txt            # Task IDs of 100 shortest tasks
        ├── shortest_1_details.json     # Detailed info for shortest task
        ├── shortest_10_details.json    # Detailed info for 10 shortest tasks
        ├── shortest_100_details.json   # Detailed info for 100 shortest tasks
        └── tasks_with_multiple_tests.json  # Tasks with >1 test example
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

## Subset Files

The `subsets/` directory contains predefined task subsets based on **task size**.

### Task Size Definition
**Task size** is calculated as the total number of grid cells across all inputs and outputs in a task:
- Size = sum of (width × height) for all input grids + sum of (width × height) for all output grids
- This includes all training examples and all test examples
- For example, a task with 3 training examples of 3×3 grids (input+output each) and 1 test example of 3×3 grids (input+output) would have size: 3×(3×3×2) + 1×(3×3×2) = 54 + 18 = 72 cells

### Available Subsets
- **shortest_1.txt**: Single shortest task
  - ARC-AGI-1: `6150a2bd` (54 total cells)
  - ARC-AGI-2: `833966f4` (30 total cells)
- **shortest_10.txt**: 10 shortest tasks
- **shortest_100.txt**: 100 shortest tasks

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