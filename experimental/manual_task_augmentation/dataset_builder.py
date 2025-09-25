#!/usr/bin/env python3
"""
Build augmented datasets and expand parquet files with validated programs.
"""

import json
import sys
import uuid
from pathlib import Path
from typing import Dict, Any, List, Tuple
from copy import deepcopy

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from llm_python.datasets.io import read_soar_parquet, write_soar_parquet
from llm_python.datasets.schema import ProgramSample
from noise_augmentor import generate_augmentations as generate_noise_augmentations
from geometric_augmentor import generate_augmentations as generate_geometric_augmentations
from program_validator import ProgramValidator, load_fully_correct_programs
import pandas as pd


class AugmentedDatasetBuilder:
    """Builds augmented datasets and expands parquet files."""

    def __init__(self, parquet_path: str, backup: bool = True):
        """
        Initialize builder with parquet file.

        Args:
            parquet_path: Path to original parquet file
            backup: Whether to create backup of original parquet
        """
        self.parquet_path = parquet_path
        self.validator = ProgramValidator()

        # Create backup if requested
        if backup:
            backup_path = parquet_path.replace('.parquet', '_backup.parquet')
            self._create_backup(parquet_path, backup_path)
            print(f"Created backup: {backup_path}")

        # Load original data
        self.original_df = read_soar_parquet(parquet_path)
        self.fully_correct_programs = load_fully_correct_programs(parquet_path)

        print(f"Loaded {len(self.original_df)} programs from parquet")
        print(f"Found fully correct programs for {len(self.fully_correct_programs)} tasks: {list(self.fully_correct_programs.keys())}")

    def _create_backup(self, source_path: str, backup_path: str):
        """Create backup of original parquet file."""
        df = read_soar_parquet(source_path)
        write_soar_parquet(df, backup_path)

    def load_manual_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Load manual tasks from JSON files."""
        challenges_path = project_root / "data/manual/arc-agi_training_challenges.json"
        solutions_path = project_root / "data/manual/arc-agi_training_solutions.json"

        with open(challenges_path, 'r') as f:
            challenges = json.load(f)

        with open(solutions_path, 'r') as f:
            solutions = json.load(f)

        # Merge challenges and solutions
        tasks = {}
        for task_id, challenge_data in challenges.items():
            task_data = deepcopy(challenge_data)
            task_data['task_id'] = task_id

            # Add expected outputs for test examples if available
            if task_id in solutions:
                test_solutions = solutions[task_id]
                for i, test_example in enumerate(task_data['test']):
                    if i < len(test_solutions):
                        test_example['output'] = test_solutions[i]

            tasks[task_id] = task_data

        return tasks

    def augment_tasks(self, num_augmentations: int = 10, augmentation_type: str = "noise", noise_percentage: float = 0.1, base_seed: int = 42) -> Tuple[Dict[str, Dict], List[ProgramSample]]:
        """
        Create augmented tasks and validate programs against them.

        Args:
            num_augmentations: Number of augmentations to try per task
            augmentation_type: Type of augmentation ("noise" or "geometric")
            noise_percentage: Fraction of black cells to replace (default 0.1 = 10%) - only for noise type
            base_seed: Base seed for reproducibility

        Returns:
            Tuple of (augmented_tasks, new_program_samples)
        """
        manual_tasks = self.load_manual_tasks()
        augmented_tasks = {}
        new_program_samples = []

        # Only process tasks that have fully correct programs
        tasks_to_process = {k: v for k, v in manual_tasks.items() if k in self.fully_correct_programs}

        print(f"Processing {len(tasks_to_process)} tasks with fully correct programs...")

        for task_id, task_data in tasks_to_process.items():
            print(f"\nProcessing task {task_id}...")

            # Generate augmentations for this task
            task_seed = base_seed + hash(task_id) % 10000

            if augmentation_type == "noise":
                task_augmentations = generate_noise_augmentations(task_data, num_augmentations, noise_percentage, task_seed)
            elif augmentation_type == "geometric":
                task_augmentations = generate_geometric_augmentations(task_data, num_augmentations, task_seed)
            else:
                raise ValueError(f"Unknown augmentation type: {augmentation_type}. Use 'noise' or 'geometric'")

            # Test each fully correct program against each augmentation
            programs = self.fully_correct_programs[task_id]
            print(f"  Testing {len(programs)} programs against {len(task_augmentations)} augmentations...")

            successful_augmentations = {}

            for aug_task_id, aug_task_data in task_augmentations.items():
                programs_that_pass = []

                for program_record in programs:
                    # Test program on augmented task
                    all_correct, train_correct, test_correct = self.validator.validate_program_on_task(
                        program_record['code'], aug_task_data
                    )

                    if all_correct:
                        programs_that_pass.append(program_record)

                # If at least one program passes, keep this augmentation
                if programs_that_pass:
                    successful_augmentations[aug_task_id] = aug_task_data
                    print(f"    ✅ {aug_task_id}: {len(programs_that_pass)} programs pass")

                    # Create new program samples for each passing program
                    for program_record in programs_that_pass:
                        # Get new predicted outputs on augmented task
                        train_preds, test_preds = self.validator.get_predicted_outputs(
                            program_record['code'], aug_task_data
                        )

                        # Create new program sample
                        new_sample = ProgramSample(
                            row_id=str(uuid.uuid4().hex),
                            task_id=aug_task_id,
                            reasoning=program_record['reasoning'],
                            code=program_record['code'],
                            correct_train_input=train_correct,
                            correct_test_input=test_correct,
                            predicted_train_output=train_preds,
                            predicted_test_output=test_preds,
                            model=program_record['model'],
                            is_transductive=program_record['is_transductive'],
                            refined_from_id=program_record.get('refined_from_id')
                        )
                        new_program_samples.append(new_sample)
                else:
                    print(f"    ❌ {aug_task_id}: no programs pass")

            print(f"  Task {task_id}: {len(successful_augmentations)}/{num_augmentations} augmentations successful")
            augmented_tasks.update(successful_augmentations)

        print(f"\nTotal: {len(augmented_tasks)} successful augmentations, {len(new_program_samples)} new program samples")
        return augmented_tasks, new_program_samples

    def save_augmented_tasks(self, augmented_tasks: Dict[str, Dict], augmentation_type: str = "noise", output_dir: str = None):
        """Save augmented tasks to JSON files in the manual folder, combining with existing augmented tasks."""
        # Save to manual folder with consistent naming
        manual_path = project_root / "data/manual"
        manual_path.mkdir(parents=True, exist_ok=True)

        challenges_file = manual_path / 'arc-agi_augmented_challenges.json'
        solutions_file = manual_path / 'arc-agi_augmented_solutions.json'

        # Load existing augmented tasks if they exist
        existing_challenges = {}
        existing_solutions = {}

        if challenges_file.exists():
            with open(challenges_file, 'r') as f:
                existing_challenges = json.load(f)

        if solutions_file.exists():
            with open(solutions_file, 'r') as f:
                existing_solutions = json.load(f)

        # Prepare new challenges and solutions
        new_challenges = {}
        new_solutions = {}

        for task_id, task_data in augmented_tasks.items():
            challenge_data = {
                'train': [{'input': ex['input'], 'output': ex['output']} for ex in task_data['train']],
                'test': [{'input': ex['input']} for ex in task_data['test']]
            }
            new_challenges[task_id] = challenge_data

            # Save test solutions separately
            test_solutions = []
            for ex in task_data['test']:
                if 'output' in ex and ex['output'] is not None:
                    test_solutions.append(ex['output'])
            if test_solutions:
                new_solutions[task_id] = test_solutions

        # Combine existing and new tasks
        combined_challenges = {**existing_challenges, **new_challenges}
        combined_solutions = {**existing_solutions, **new_solutions}

        # Write combined files
        with open(challenges_file, 'w') as f:
            json.dump(combined_challenges, f, indent=2)

        with open(solutions_file, 'w') as f:
            json.dump(combined_solutions, f, indent=2)

        print(f"Added {len(new_challenges)} {augmentation_type} augmented tasks to {manual_path}")
        print(f"  - arc-agi_augmented_challenges.json: {len(combined_challenges)} total tasks (+{len(new_challenges)})")
        print(f"  - arc-agi_augmented_solutions.json: {len(combined_solutions)} total tasks (+{len(new_solutions)})")

    def expand_parquet(self, new_program_samples: List[ProgramSample], augmentation_type: str = "noise", output_path: str = None):
        """Add new program samples to parquet file."""
        if not new_program_samples:
            print("No new program samples to add")
            return

        # Convert new samples to DataFrame
        new_data = []
        for sample in new_program_samples:
            new_data.append(dict(sample))

        new_df = pd.DataFrame(new_data)

        # Combine with original data
        combined_df = pd.concat([self.original_df, new_df], ignore_index=True)

        # Write to output file - use consistent naming for all augmentation types
        if output_path is None:
            output_path = self.parquet_path.replace('.parquet', '_augmented.parquet')

        write_soar_parquet(combined_df, output_path)
        print(f"Expanded parquet saved to {output_path}")
        print(f"Original: {len(self.original_df)} programs, New: {len(combined_df)} programs (+{len(new_df)})")

        return output_path