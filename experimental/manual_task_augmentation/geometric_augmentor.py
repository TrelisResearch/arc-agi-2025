#!/usr/bin/env python3
"""
Geometric and recolor augmentation for manual tasks.
Applies horizontal flip, vertical flip, rotations (90°, 180°, 270°), and recoloring
uniformly to all input and output grids in train and test examples.
"""

import random
from typing import List, Dict, Any, Tuple
from copy import deepcopy


def horizontal_flip(grid: List[List[int]]) -> List[List[int]]:
    """Flip grid horizontally (left-right mirror)."""
    return [row[::-1] for row in grid]


def vertical_flip(grid: List[List[int]]) -> List[List[int]]:
    """Flip grid vertically (top-bottom mirror)."""
    return grid[::-1]


def rotate_90(grid: List[List[int]]) -> List[List[int]]:
    """Rotate grid 90 degrees clockwise."""
    rows, cols = len(grid), len(grid[0])
    return [[grid[rows - 1 - j][i] for j in range(rows)] for i in range(cols)]


def rotate_180(grid: List[List[int]]) -> List[List[int]]:
    """Rotate grid 180 degrees."""
    return [row[::-1] for row in grid[::-1]]


def rotate_270(grid: List[List[int]]) -> List[List[int]]:
    """Rotate grid 270 degrees clockwise (90 degrees counterclockwise)."""
    rows, cols = len(grid), len(grid[0])
    return [[grid[j][cols - 1 - i] for j in range(rows)] for i in range(cols)]


def recolor_grid(grid: List[List[int]], increment: int) -> List[List[int]]:
    """
    Recolor grid by incrementing all non-zero colors by the specified amount.
    Colors wrap from 9 back to 1 (0 stays as 0).

    Args:
        grid: Input grid
        increment: Amount to increment colors (1-8)

    Returns:
        Recolored grid
    """
    recolored = []
    for row in grid:
        new_row = []
        for cell in row:
            if cell == 0:
                new_row.append(0)  # Keep black cells as black
            else:
                # Increment and wrap: 1-9 -> 1-9
                new_color = ((cell - 1 + increment) % 9) + 1
                new_row.append(new_color)
        recolored.append(new_row)
    return recolored


def apply_transformations(grid: List[List[int]], flip_op: str, rotate_op: str, recolor_op: int) -> List[List[int]]:
    """
    Apply the specified transformations to a grid in order: flip -> rotate -> recolor.

    Args:
        grid: Input grid
        flip_op: 'none', 'horizontal', or 'vertical'
        rotate_op: 'none', '90', '180', or '270'
        recolor_op: 0 for none, or 1-8 for increment amount

    Returns:
        Transformed grid
    """
    result = deepcopy(grid)

    # Apply flip
    if flip_op == 'horizontal':
        result = horizontal_flip(result)
    elif flip_op == 'vertical':
        result = vertical_flip(result)

    # Apply rotation
    if rotate_op == '90':
        result = rotate_90(result)
    elif rotate_op == '180':
        result = rotate_180(result)
    elif rotate_op == '270':
        result = rotate_270(result)

    # Apply recolor
    if recolor_op > 0:
        result = recolor_grid(result, recolor_op)

    return result


def generate_random_transformation(seed: int = None) -> Tuple[str, str, int]:
    """
    Generate a random combination of transformations.

    Args:
        seed: Random seed for reproducibility

    Returns:
        Tuple of (flip_op, rotate_op, recolor_op)
    """
    if seed is not None:
        random.seed(seed)

    # Flip operations: none, horizontal, vertical (equal probability)
    flip_ops = ['none', 'horizontal', 'vertical']
    flip_op = random.choice(flip_ops)

    # Rotation operations: none, 90, 180, 270 (equal probability)
    rotate_ops = ['none', '90', '180', '270']
    rotate_op = random.choice(rotate_ops)

    # Recolor operations: none (0), or increment by 1-8 (equal probability with no-op)
    recolor_ops = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    recolor_op = random.choice(recolor_ops)

    return flip_op, rotate_op, recolor_op


def augment_task(task_data: Dict[str, Any], augmentation_id: int, seed: int = None) -> Tuple[str, Dict[str, Any]]:
    """
    Create an augmented version of a task by applying the same transformations to all grids.

    Args:
        task_data: Original task data with 'train' and 'test' examples
        augmentation_id: Unique ID for this augmentation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (new_task_id, augmented_task_data)
    """
    original_task_id = task_data.get('task_id', 'unknown')
    new_task_id = f"{original_task_id}_aug_{augmentation_id:03d}"

    # Generate random transformation for this augmentation
    flip_op, rotate_op, recolor_op = generate_random_transformation(seed)

    # Deep copy original task
    augmented_task = deepcopy(task_data)

    # Apply same transformations to all training grids (input and output)
    for example in augmented_task['train']:
        example['input'] = apply_transformations(example['input'], flip_op, rotate_op, recolor_op)
        example['output'] = apply_transformations(example['output'], flip_op, rotate_op, recolor_op)

    # Apply same transformations to all test grids (input and output)
    for example in augmented_task['test']:
        example['input'] = apply_transformations(example['input'], flip_op, rotate_op, recolor_op)
        example['output'] = apply_transformations(example['output'], flip_op, rotate_op, recolor_op)

    return new_task_id, augmented_task


def generate_augmentations(task_data: Dict[str, Any], num_augmentations: int, base_seed: int = None) -> Dict[str, Dict[str, Any]]:
    """
    Generate multiple augmented versions of a task.

    Args:
        task_data: Original task data
        num_augmentations: Number of augmented versions to create
        base_seed: Base seed for reproducibility

    Returns:
        Dictionary mapping new_task_id -> augmented_task_data
    """
    augmentations = {}

    for i in range(num_augmentations):
        aug_seed = base_seed + i if base_seed is not None else None
        new_task_id, aug_task = augment_task(task_data, i + 1, seed=aug_seed)
        augmentations[new_task_id] = aug_task

    return augmentations


def describe_transformation(flip_op: str, rotate_op: str, recolor_op: int) -> str:
    """Create a human-readable description of the transformation."""
    parts = []

    if flip_op != 'none':
        parts.append(f"{flip_op} flip")

    if rotate_op != 'none':
        parts.append(f"rotate {rotate_op}°")

    if recolor_op > 0:
        parts.append(f"recolor +{recolor_op}")

    if not parts:
        return "no transformation"

    return ", ".join(parts)


if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pathlib import Path

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from llm_python.utils.task_loader import get_task_loader

    def test_transformations():
        """Test the transformation functions with a sample grid."""
        print("=== Testing Geometric Transformations ===\n")

        # Create a simple test grid
        test_grid = [
            [1, 2, 0],
            [3, 4, 5],
            [0, 6, 7]
        ]

        print("Original grid:")
        for row in test_grid:
            print(row)
        print()

        # Test each transformation
        print("Horizontal flip:")
        for row in horizontal_flip(test_grid):
            print(row)
        print()

        print("Vertical flip:")
        for row in vertical_flip(test_grid):
            print(row)
        print()

        print("Rotate 90°:")
        for row in rotate_90(test_grid):
            print(row)
        print()

        print("Rotate 180°:")
        for row in rotate_180(test_grid):
            print(row)
        print()

        print("Rotate 270°:")
        for row in rotate_270(test_grid):
            print(row)
        print()

        print("Recolor +3:")
        for row in recolor_grid(test_grid, 3):
            print(row)
        print()

        print("Combined (h-flip + rotate 90° + recolor +2):")
        combined = apply_transformations(test_grid, 'horizontal', '90', 2)
        for row in combined:
            print(row)
        print()

        # Test random generation
        print("Random transformations:")
        for i in range(5):
            flip_op, rotate_op, recolor_op = generate_random_transformation(seed=i)
            desc = describe_transformation(flip_op, rotate_op, recolor_op)
            print(f"  {i+1}: {desc}")

    def augment_sample_task():
        """Augment a sample task to demonstrate functionality."""
        print("\n=== Sample Task Augmentation ===\n")

        # Load a task
        task_loader = get_task_loader()
        task_ids = ["007bbfb7", "00d62c1b", "025d127b"]  # Some known tasks

        for task_id in task_ids:
            try:
                task_data = task_loader.get_task(task_id)
                task_data['task_id'] = task_id
                print(f"Loaded task: {task_id}")

                # Generate a few augmentations
                augmentations = generate_augmentations(task_data, num_augmentations=3, base_seed=42)

                print(f"Generated {len(augmentations)} augmentations:")
                for aug_id, aug_data in augmentations.items():
                    # Show what transformation was applied by recreating it
                    flip_op, rotate_op, recolor_op = generate_random_transformation(seed=42)
                    desc = describe_transformation(flip_op, rotate_op, recolor_op)
                    print(f"  {aug_id}: {desc}")
                    break  # Just show the first one for brevity
                break
            except Exception as e:
                print(f"Could not load task {task_id}: {e}")
                continue

    def augment_dataset(dataset: str, subset: str, num_augmentations: int, seed: int, output_dir: str):
        """Augment tasks from a dataset/subset and save the results."""
        from pathlib import Path

        print(f"=== Augmenting Dataset: {dataset}/{subset} ===\n")

        # Load tasks based on dataset
        task_loader = get_task_loader()
        tasks_to_augment = {}

        if dataset.lower() == 'manual':
            # Load from manual tasks
            challenges_path = project_root / "data/manual/arc-agi_training_challenges.json"
            solutions_path = project_root / "data/manual/arc-agi_training_solutions.json"

            if not challenges_path.exists():
                print(f"Error: {challenges_path} not found")
                return

            with open(challenges_path, 'r') as f:
                challenges = json.load(f)

            solutions = {}
            if solutions_path.exists():
                with open(solutions_path, 'r') as f:
                    solutions = json.load(f)

            # Filter by subset if specified
            if subset and subset != 'all':
                if subset == 'shortest_1':
                    # Take first task (alphabetically)
                    task_ids = sorted(challenges.keys())[:1]
                else:
                    # Try to parse as number
                    try:
                        num_tasks = int(subset)
                        task_ids = sorted(challenges.keys())[:num_tasks]
                    except ValueError:
                        print(f"Unknown subset: {subset}")
                        return
            else:
                task_ids = list(challenges.keys())

            for task_id in task_ids:
                task_data = deepcopy(challenges[task_id])
                task_data['task_id'] = task_id

                # Add test outputs if available
                if task_id in solutions:
                    for i, solution in enumerate(solutions[task_id]):
                        if i < len(task_data['test']):
                            task_data['test'][i]['output'] = solution

                tasks_to_augment[task_id] = task_data

        else:
            # Load from ARC dataset
            try:
                if subset == 'all':
                    task_ids = task_loader.get_all_task_ids(dataset)
                elif subset == 'shortest_1':
                    task_ids = task_loader.get_all_task_ids(dataset)[:1]
                else:
                    try:
                        num_tasks = int(subset)
                        task_ids = task_loader.get_all_task_ids(dataset)[:num_tasks]
                    except ValueError:
                        print(f"Unknown subset: {subset}")
                        return

                for task_id in task_ids:
                    task_data = task_loader.get_task(task_id)
                    task_data['task_id'] = task_id
                    tasks_to_augment[task_id] = task_data

            except Exception as e:
                print(f"Error loading dataset {dataset}: {e}")
                return

        print(f"Loaded {len(tasks_to_augment)} tasks from {dataset}/{subset}")

        # Generate augmentations
        all_augmentations = {}
        for task_id, task_data in tasks_to_augment.items():
            print(f"Augmenting {task_id}...")
            task_seed = seed + hash(task_id) % 10000
            augmentations = generate_augmentations(task_data, num_augmentations, task_seed)
            all_augmentations.update(augmentations)
            print(f"  Generated {len(augmentations)} augmentations")

        print(f"\nTotal: {len(all_augmentations)} augmented tasks")

        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Prepare challenges and solutions
        challenges = {}
        solutions = {}

        for task_id, task_data in all_augmentations.items():
            challenge_data = {
                'train': [{'input': ex['input'], 'output': ex['output']} for ex in task_data['train']],
                'test': [{'input': ex['input']} for ex in task_data['test']]
            }
            challenges[task_id] = challenge_data

            # Save test solutions if available
            test_solutions = []
            for ex in task_data['test']:
                if 'output' in ex and ex['output'] is not None:
                    test_solutions.append(ex['output'])
            if test_solutions:
                solutions[task_id] = test_solutions

        # Write files
        challenges_file = output_path / f'arc-agi_geometric_augmented_{dataset}_{subset}_challenges.json'
        solutions_file = output_path / f'arc-agi_geometric_augmented_{dataset}_{subset}_solutions.json'

        with open(challenges_file, 'w') as f:
            json.dump(challenges, f, indent=2)

        with open(solutions_file, 'w') as f:
            json.dump(solutions, f, indent=2)

        print(f"\n✅ Saved augmented tasks:")
        print(f"  Challenges: {challenges_file}")
        print(f"  Solutions: {solutions_file}")
        print(f"  Total tasks: {len(challenges)}")

    parser = argparse.ArgumentParser(description="Geometric and recolor augmentation for ARC tasks")
    parser.add_argument(
        '--dataset',
        type=str,
        help="Dataset name (e.g., 'manual', 'arc-agi-1')"
    )
    parser.add_argument(
        '--subset',
        type=str,
        help="Subset name (e.g., 'shortest_1', 'all')"
    )
    parser.add_argument(
        '--num-augmentations',
        type=int,
        default=10,
        help="Number of augmentations per task (default: 10)"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default="data/manual/",
        help="Output directory for augmented tasks (default: data/manual/)"
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help="Run transformation tests on sample grids"
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help="Demonstrate augmentation on a sample task"
    )

    args = parser.parse_args()

    if args.test:
        test_transformations()
    elif args.demo:
        augment_sample_task()
    elif args.dataset and args.subset:
        augment_dataset(args.dataset, args.subset, args.num_augmentations, args.seed, args.output_dir)
    else:
        print("Geometric and Recolor Augmentation")
        print("\nUsage:")
        print("  --dataset manual --subset shortest_1       Create augmentations from manual tasks")
        print("  --dataset arc-agi-1 --subset 10           Create augmentations from first 10 ARC tasks")
        print("  --test                                     Test transformation functions")
        print("  --demo                                     Demo augmentation on sample task")
        print("  -h                                         Show detailed help")
        print("\nOptions:")
        print("  --num-augmentations N    Number of augmentations per task (default: 10)")
        print("  --seed N                 Random seed (default: 42)")
        print("  --output-dir PATH        Output directory (default: data/manual/)")
        print("\nLibrary usage:")
        print("  from geometric_augmentor import generate_augmentations")