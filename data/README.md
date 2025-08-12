# ARC-AGI Data Directory

This directory contains the complete datasets for ARC-AGI-1, ARC-AGI-1r (reverse), and ARC-AGI-2, along with predefined subsets for efficient experimentation.

## Directory Structure

```
data/
├── arc-agi-1/              # ARC-AGI-1 dataset (original ARC)
│   ├── training/           # 400 training tasks
│   └── evaluation/         # 400 evaluation tasks
├── arc-agi-1r/             # ARC-AGI-1r dataset (reverse of original ARC)
│   ├── training/           # 400 training tasks (input/output swapped)
│   └── evaluation/         # 400 evaluation tasks (input/output swapped)
├── arc-agi-2/              # ARC-AGI-2 dataset (2024 release)
│   ├── training/           # 1,000 training tasks
│   └── evaluation/         # 120 evaluation tasks
└── subsets/                # Predefined subsets for all datasets
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
    ├── arc-agi-1r/
    │   ├── shortest_evaluation_1r.txt          # Reverse task IDs for shortest evaluation tasks
    │   ├── shortest_evaluation_10r.txt         # (All subset files with 'r' suffix)
    │   ├── shortest_evaluation_30r.txt
    │   ├── shortest_training_1r.txt            # (Task IDs have 'r' appended)
    │   ├── shortest_training_10r.txt
    │   ├── shortest_training_30r.txt
    │   ├── middle_evaluation_*r.txt            # All middle subsets with 'r' suffix
    │   ├── middle_training_*r.txt
    │   ├── longest_evaluation_*r.txt           # All longest subsets with 'r' suffix
    │   ├── longest_training_*r.txt
    │   ├── grid_size_distributed_30_evaluationr.txt
    │   ├── grid_size_distributed_30_trainingr.txt
    │   └── ... (all other subsets with 'r' suffix)
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

### ARC-AGI-1r (800 tasks total - reverse of ARC-AGI-1)
- **Training set**: 400 tasks (input/output swapped from ARC-AGI-1 training)
- **Evaluation set**: 400 tasks (input/output swapped from ARC-AGI-1 evaluation)
- **Task IDs**: Original IDs with 'r' appended (e.g., `007bbfb7` → `007bbfb7r`)
- **Grid swapping**: For each example, original input becomes output and original output becomes input
- **Use case**: Testing model performance on inverse reasoning tasks

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
- `shortest_training_1`, `shortest_training_10`, `shortest_training_30`, `shortest_training_100` (both datasets)
- `shortest_evaluation_1`, `shortest_evaluation_10`, `shortest_evaluation_30`, `shortest_evaluation_100` (both datasets)
- ... and similarly for `middle` and `longest` (up to 30 tasks)
- `grid_size_distributed_30_training`: 30 training tasks evenly distributed by grid size
- `grid_size_distributed_30_evaluation`: 30 evaluation tasks evenly distributed by grid size

Each subset contains only task IDs from the relevant split. Legacy mixed subsets are in the `archive/` folder.

**Grid Size Distributed Subsets:**
These subsets select 30 tasks evenly spaced across the range of grid sizes (total cells in all input/output grids). They provide a balanced representation of task complexity within each split.

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
- **random_split_1_training.txt** through **random_split_8_training.txt**: Eight random splits of 50 training tasks each (total 400 tasks)

**For arc-agi-1r (in `data/subsets/arc-agi-1r/`):**
- **All subset files from arc-agi-1 with 'r' suffix**: e.g., `shortest_training_10r.txt`
- **Task IDs with 'r' appended**: e.g., `6150a2bd` → `6150a2bdr`
- **Same organizational structure**: All 44 subset files replicated with transformed task IDs
- **Complete coverage**: Every subset from arc-agi-1 has a corresponding reverse version

**For arc-agi-2 (in `data/subsets/arc-agi-2/`):**
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
- **unique_training_tasks.txt**: 233 ARC-AGI-2 training tasks that are NOT present in ARC-AGI-1 (unique to ARC-AGI-2)
- **remaining_tasks_no_512_attempts.txt**: 110 ARC-AGI-2 unique training tasks EXCLUDING the 123 tasks that already have 512 attempts of data collected (created because a run stopped midway when working on the unique_training_tasks subset)

**For both datasets:**
- **tasks_with_multiple_tests.json**: Tasks with more than one test example (JSON, not a subset for evaluation)

Each `.txt` file contains one task ID per line. The corresponding `_details.json` files include additional metadata like the computed size and which split (training/evaluation) the task belongs to.

## Random Training Splits (ARC-AGI-1)

Eight random splits of the ARC-AGI-1 training set have been created for various experimental purposes:

**In `data/subsets/arc-agi-1/`:**

- **random_split_1_training.txt** through **random_split_8_training.txt**: Eight non-overlapping random splits of 50 training tasks each
  - Total coverage: All 400 ARC-AGI-1 training tasks
  - Selection method: Random shuffling with seed 42 for reproducibility
  - Split size: 50 tasks per split
  - Use cases: Cross-validation, ablation studies, training subset experiments, model comparison

### Random Split Properties:
- **Reproducible**: Uses fixed random seed (42) for consistent results
- **Non-overlapping**: Each task appears in exactly one split
- **Balanced**: Each split contains exactly 50 tasks
- **Complete coverage**: All 400 training tasks are included across the 8 splits

## Model Performance Subsets (ARC-AGI-1)

These subsets contain task IDs that were successfully solved by specific models, useful for performance analysis and creating calibration sets:

**In `data/subsets/arc-agi-1/`:**

- **all_evaluation.txt**: Complete list of all 400 evaluation task IDs from ARC-AGI-1
- **gpt_4_1.txt**: 22 tasks solved correctly by gpt-4.1 (original baseline)
- **o4-mini.txt**: 94 tasks solved correctly by o4-mini (independent attempts, single shot)
- **gpt-4.1-nano.txt**: 3 tasks solved correctly by gpt-4.1-nano (when tested on gpt-4.1-o4-mini subset)
- **gpt-4.1-o4-mini.txt**: 98 tasks solved by either gpt-4.1 OR o4-mini (union/combined set)
  - Contains 18 tasks solved by both models (overlap)
  - Contains 4 tasks solved only by gpt-4.1 
  - Contains 76 tasks solved only by o4-mini
- **gpt-4.1-mini-calib.txt**: 95 task calibration subset for testing gpt-4.1-mini
  - Created by removing gpt-4.1-nano solved tasks from gpt-4.1-o4-mini subset
  - Excludes the 3 "easy" tasks that gpt-4.1-nano could solve
  - Designed for more challenging evaluation of gpt-4.1-mini capabilities
- **gpt-4.1-mini-calib-train.txt**: 46 task training calibration subset for gpt-4.1-mini experiments
  - Created from shortest_training_100 subset (100 smallest training tasks)
  - Contains tasks solved by EITHER o4-mini OR gpt-4.1 but NOT by gpt-4.1-mini OR gpt-4.1-nano
  - Designed for training experiments with ideal difficulty range for gpt-4.1-mini

### Model Performance Summary:
- **gpt-4.1**: 22/400 tasks solved (5.5%)
- **o4-mini**: 94/400 tasks solved (23.5%) 
- **gpt-4.1-nano**: 3/98 tasks solved (3.1% on combined subset)
- **Combined coverage**: 98/400 unique tasks solved by either gpt-4.1 or o4-mini (24.5%)

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
- ARC-AGI-1r: Programmatically generated reverse of ARC-AGI-1 (input/output swapped)
- ARC-AGI-2: https://github.com/arcprize/ARC-AGI-2

## ARC-AGI-1r (Reverse Dataset)

The ARC-AGI-1r dataset is a programmatically generated reverse version of ARC-AGI-1, where the input and output grids are swapped for every training and test example. This creates an interesting challenge for testing model performance on inverse reasoning tasks.

### Key Features:
- **Complete reversal**: Every training and test example has input and output grids swapped
- **Preserved structure**: Same number of examples, same grid dimensions, same color patterns
- **New task IDs**: Original task ID + 'r' (e.g., `007bbfb7` → `007bbfb7r`)
- **Complete subset coverage**: All 44 subset files from arc-agi-1 replicated with 'r' suffix

### Example Transformation:
**Original task `007bbfb7.json`:**
```json
{
  "train": [{"input": [[0,7,7]], "output": [[0,0,0,0,7,7,0,7,7]]}],
  "test": [{"input": [[7,0,7]], "output": [[7,0,7,0,0,0,7,0,7]]}]
}
```

**Becomes `007bbfb7r.json`:**
```json
{
  "train": [{"input": [[0,0,0,0,7,7,0,7,7]], "output": [[0,7,7]]}],
  "test": [{"input": [[7,0,7,0,0,0,7,0,7]], "output": [[7,0,7]]}]
}
```

## ARC-AGI-2 Unique Training Tasks

A comprehensive analysis of dataset overlap between ARC-AGI-1 and ARC-AGI-2 revealed significant overlap in the training sets. The unique tasks subset provides access to truly novel tasks for experimentation.

### Overlap Analysis Summary:

**ARC-AGI-2 Training Set (1,000 tasks):**
- **Overlap with ARC-AGI-1 Training**: 391 tasks (39.1%)
- **Overlap with ARC-AGI-1 Evaluation**: 376 tasks (37.6%)
- **Total Overlap with ARC-AGI-1**: 767 tasks (76.7%)
- **Unique to ARC-AGI-2**: 233 tasks (23.3%)

**ARC-AGI-2 Evaluation Set (120 tasks):**
- **Overlap with ARC-AGI-1**: 6 tasks (5.0%)
  - Task IDs: `0934a4d8`, `136b0064`, `16b78196`, `981571dc`, `aa4ec2a5`, `da515329`
- **Unique to ARC-AGI-2**: 114 tasks (95.0%)

### Files:
- **unique_training_tasks.txt**: List of 233 task IDs unique to ARC-AGI-2
- **unique_training_tasks_details.json**: Complete overlap analysis with task lists
- **analyze_dataset_overlap.py**: Script that generated the analysis

This analysis was performed by the `analyze_dataset_overlap.py` script and ensures researchers can work with truly novel tasks when experimenting with ARC-AGI-2.

## Solution Counts and Performance Data

The `data/arc-agi-1-training/` directory contains solution count files that track the number of solution programs found for each ARC-AGI-1 training task:

- **soar_arc_training_solution_counts_enhanced_YYYYMMDD_HHMMSS.json**: Multiple versions with timestamps
- **training_new_tricky_YYYYMMDD_HHMMSS.txt**: Task lists for particularly challenging tasks

These files contain solution program counts in JSON format, with some tasks having null counts indicating no solutions were found. The most recent enhanced file (20250806_184213) provides the current solution status for ARC-AGI-1 training tasks.

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