#!/usr/bin/env python3

import argparse
from typing import List
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from .schema import TrainingExample, EnrichedTrainingExample
from ..utils.task_loader import Grid, TaskLoader
from ..utils.transduction import is_transduction_cheating

# Initialize utilities
task_loader = TaskLoader()


def has_valid_outputs(sample_outputs: List[Grid]) -> bool:
    """Check if all sample outputs are proper lists of lists with values (not None/null)."""
    for output in sample_outputs:
        if output is None:
            return False
        if not isinstance(output, list):
            return False
        if not output:  # Empty list
            return False
        for row in output:
            if row is None:
                return False
            if not isinstance(row, list):
                return False
            if not row:  # Empty row
                return False
            # Check that all values are not None
            for val in row:
                if val is None:
                    return False
    return True


def _find_matching_outputs(
    inputs: List[Grid], sample_inputs: List[Grid], sample_outputs: List[Grid]
) -> List[Grid]:
    predicted_outputs = []

    for _, input_grid in enumerate(inputs):
        # Find matching sample input
        matching_sample_idx = None
        for j, sample_input in enumerate(sample_inputs):
            if input_grid == sample_input:
                matching_sample_idx = j
                break

        if matching_sample_idx is None:
            raise ValueError("Could not find matching sample input for task.")

        predicted_output = sample_outputs[matching_sample_idx]
        predicted_outputs.append(predicted_output)

    return predicted_outputs


def enrich_single_program(program_data: TrainingExample) -> EnrichedTrainingExample:
    """Convert a simple TrainingExample to an EnrichedTrainingExample with separate train/test data."""
    task_id = program_data["task_id"]
    program = program_data["code"]
    reasoning = program_data["reasoning"]
    model = program_data["model"]
    sample_inputs = program_data["sample_inputs"]
    sample_outputs = program_data["sample_outputs"]

    # Load task data to get proper train/test splits
    task_data = task_loader.load_task(task_id)

    # Extract training inputs and outputs
    train_inputs = [example["input"] for example in task_data["train"]]
    train_outputs = [example["output"] for example in task_data["train"]]

    # Extract test inputs and outputs
    test_inputs = [example["input"] for example in task_data["test"]]
    test_outputs = [example["output"] for example in task_data["test"]]

    # Match training inputs with sample inputs to get predicted outputs
    predicted_train_outputs = _find_matching_outputs(
        train_inputs, sample_inputs, sample_outputs
    )

    # Find which train inputs produced correct outputs
    correct_train_inputs = []
    for i, (train_input, predicted_output) in enumerate(
        zip(train_inputs, predicted_train_outputs)
    ):
        if predicted_output == train_outputs[i]:
            correct_train_inputs.append(train_input)

    # Match test inputs with sample inputs to get predicted outputs
    predicted_test_outputs = _find_matching_outputs(
        test_inputs, sample_inputs, sample_outputs
    )

    # Find which test inputs produced correct outputs
    correct_test_inputs = []
    for i, (test_input, predicted_output) in enumerate(
        zip(test_inputs, predicted_test_outputs)
    ):
        if predicted_output == test_outputs[i]:
            correct_test_inputs.append(test_input)

    return EnrichedTrainingExample(
        reasoning=reasoning,
        code=program,
        correct_train_input=correct_train_inputs,
        train_input=train_inputs,
        train_output=train_outputs,
        predicted_train_output=predicted_train_outputs,
        correct_test_input=correct_test_inputs,
        test_input=test_inputs,
        test_output=test_outputs,
        predicted_test_output=predicted_test_outputs,
        task_id=task_id,
        model=model,
        generation=0,  # Default generation if not available
    )


def main():
    parser = argparse.ArgumentParser(
        description="Merge simple training examples into enriched format for actual training"
    )
    parser.add_argument(
        "input_parquet",
        type=str,
        help="Input parquet file with simple TrainingExample format",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output parquet file (default: input_enriched.parquet)",
    )
    parser.add_argument(
        "--no-filter-valid-outputs",
        action="store_true",
        help="Disable filter for valid sample outputs (default: enabled)",
    )
    parser.add_argument(
        "--no-filter-transduction",
        action="store_true", 
        help="Disable transduction cheating filter (default: enabled)",
    )

    args = parser.parse_args()

    # Generate output filename if not provided
    if args.output is None:
        input_path = Path(args.input_parquet)
        args.output = str(input_path.parent / (input_path.stem + "_enriched.parquet"))

    # Load the input data
    print(f"Loading data from {args.input_parquet}...")
    df = pd.read_parquet(args.input_parquet)
    print(f"Loaded {len(df)} programs")

    # Apply filters if enabled
    if not args.no_filter_valid_outputs:
        initial_count = len(df)
        df = df[df["sample_outputs"].apply(has_valid_outputs)]
        print(f"After valid outputs filter: {len(df)} programs (removed {initial_count - len(df)})")

    if not args.no_filter_transduction:
        initial_count = len(df)
        filtered_indices = []
        for idx, row in df.iterrows():
            # Load task data for transduction check
            task_data = task_loader.load_task(row["task_id"])
            is_cheating, _ = is_transduction_cheating(row["code"], task_data)
            if not is_cheating:
                filtered_indices.append(idx)
        df = df.loc[filtered_indices]
        print(f"After transduction filter: {len(df)} programs (removed {initial_count - len(df)})")

    if len(df) == 0:
        print("No programs remain after filtering!")
        return

    # Convert DataFrame rows to TrainingExample objects
    training_examples = []
    for _, row in df.iterrows():
        training_examples.append(
            TrainingExample(
                code=row["code"],
                reasoning=row["reasoning"],
                model=row["model"],
                task_id=row["task_id"],
                train_correct_fraction=row["train_correct_fraction"],
                test_correct_fraction=row["test_correct_fraction"],
                sample_inputs=row["sample_inputs"],
                sample_outputs=row["sample_outputs"],
            )
        )

    # Calculate number of workers
    max_workers = max(1, multiprocessing.cpu_count() - 2)
    print(
        f"Using {max_workers} worker processes to enrich {len(training_examples)} programs..."
    )

    # Process programs in parallel
    enriched_programs = []
    processed_count = 0
    failed_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_program = {
            executor.submit(enrich_single_program, program): program
            for program in training_examples
        }

        # Collect results as they complete
        for future in as_completed(future_to_program):
            program = future_to_program[future]
            processed_count += 1

            # Progress reporting every 100 programs
            if processed_count % 100 == 0:
                print(
                    f"  Processed {processed_count}/{len(training_examples)} programs..."
                )

            try:
                enriched_program = future.result()
                enriched_programs.append(enriched_program)
            except Exception as e:
                failed_count += 1
                print(
                    f"  Warning: Failed to enrich program for task {program['task_id']}: {e}"
                )

    print(
        f"Successfully enriched {len(enriched_programs)} programs ({failed_count} failed)"
    )

    # Convert to DataFrame and save
    if enriched_programs:
        enriched_df = pd.DataFrame(enriched_programs)

        # Ensure output directory exists
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        enriched_df.to_parquet(output_path, index=False)
        print(f"Saved enriched dataset to {output_path}")

        # Print basic stats
        print(f"\nSaved {len(enriched_programs)} enriched programs")
        print(f"Unique tasks: {enriched_df['task_id'].nunique()}")
        print(f"Unique models: {enriched_df['model'].nunique()}")
    else:
        print("No programs to save!")


if __name__ == "__main__":
    main()
