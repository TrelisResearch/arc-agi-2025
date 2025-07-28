#!/usr/bin/env python3

import argparse
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Import utilities
try:
    # Try relative imports first (when run as module)
    from .schema import TrainingExample, EnrichedTrainingExample
    from ..utils.scoring import ProgramExecutor
    from ..utils.task_loader import TaskLoader
except ImportError:
    # Fall back to absolute imports (when run directly)
    from schema import TrainingExample, EnrichedTrainingExample
    from utils.scoring import ProgramExecutor
    from utils.task_loader import TaskLoader

# Initialize utilities
task_loader = TaskLoader()
program_executor = ProgramExecutor(timeout=0.5)


def enrich_single_program(program_data: TrainingExample) -> EnrichedTrainingExample:
    """Convert a simple TrainingExample to an EnrichedTrainingExample with separate train/test data."""
    task_id = program_data["task_id"]
    program = program_data["code"]
    reasoning = program_data["reasoning"]
    model = program_data["model"]
    
    # Load task data to get proper train/test splits
    task_data = task_loader.load_task(task_id, "arc-agi-1")
    
    # Extract training inputs and outputs
    train_inputs = [example["input"] for example in task_data["train"]]
    train_outputs = [example["output"] for example in task_data["train"]]
    
    # Extract test inputs and outputs
    test_inputs = [example["input"] for example in task_data["test"]]
    test_outputs = [example["output"] for example in task_data["test"]]
    
    # Run the program on each training input to get predictions
    predicted_train_outputs = []
    correct_train_inputs = []
    
    for i, train_input in enumerate(train_inputs):
        predicted_output, error, timed_out = program_executor.execute_program_with_timeout(program, train_input)
        
        if predicted_output is not None and not error and not timed_out:
            predicted_train_outputs.append(predicted_output)
            if predicted_output == train_outputs[i]:
                correct_train_inputs.append(train_input)
        else:
            predicted_train_outputs.append([])  # Empty grid for failed executions
    
    # Run the program on each test input to get predictions
    predicted_test_outputs = []
    correct_test_inputs = []
    
    for i, test_input in enumerate(test_inputs):
        predicted_output, error, timed_out = program_executor.execute_program_with_timeout(program, test_input)
        
        if predicted_output is not None and not error and not timed_out:
            predicted_test_outputs.append(predicted_output)
            if predicted_output == test_outputs[i]:
                correct_test_inputs.append(test_input)
        else:
            predicted_test_outputs.append([])  # Empty grid for failed executions
    
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
        generation=0  # Default generation if not available
    )


def main():
    parser = argparse.ArgumentParser(description="Merge simple training examples into enriched format for actual training")
    parser.add_argument(
        "input_parquet", 
        type=str,
        help="Input parquet file with simple TrainingExample format"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None,
        help="Output parquet file (default: input_enriched.parquet)"
    )
    parser.add_argument(
        "--filter-min-train-accuracy",
        type=float,
        default=None,
        help="Filter to only programs with at least this training accuracy"
    )
    parser.add_argument(
        "--filter-min-test-accuracy",
        type=float,
        default=None,
        help="Filter to only programs with at least this test accuracy"
    )
    parser.add_argument(
        "--filter-perfect-train",
        action="store_true",
        help="Filter to only programs with perfect training accuracy"
    )
    parser.add_argument(
        "--filter-perfect-test",
        action="store_true",
        help="Filter to only programs with perfect test accuracy"
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

    # Apply filters if specified
    if args.filter_min_train_accuracy is not None:
        df = df[df['train_correct_fraction'] >= args.filter_min_train_accuracy]
        print(f"After min training accuracy filter ({args.filter_min_train_accuracy}): {len(df)} programs")
    
    if args.filter_min_test_accuracy is not None:
        df = df[df['test_correct_fraction'] >= args.filter_min_test_accuracy]
        print(f"After min test accuracy filter ({args.filter_min_test_accuracy}): {len(df)} programs")
    
    if args.filter_perfect_train:
        df = df[df['train_correct_fraction'] == 1.0]
        print(f"After perfect training accuracy filter: {len(df)} programs")
    
    if args.filter_perfect_test:
        df = df[df['test_correct_fraction'] == 1.0]
        print(f"After perfect test accuracy filter: {len(df)} programs")

    if len(df) == 0:
        print("No programs remain after filtering!")
        return

    # Convert DataFrame rows to TrainingExample objects
    training_examples = []
    for _, row in df.iterrows():
        training_examples.append(TrainingExample(
            code=row['code'],
            reasoning=row['reasoning'],
            model=row['model'],
            task_id=row['task_id'],
            train_correct_fraction=row['train_correct_fraction'],
            test_correct_fraction=row['test_correct_fraction'],
            sample_inputs=row['sample_inputs'],
            sample_outputs=row['sample_outputs']
        ))

    # Calculate number of workers
    max_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"Using {max_workers} worker processes to enrich {len(training_examples)} programs...")

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
                print(f"  Processed {processed_count}/{len(training_examples)} programs...")

            try:
                enriched_program = future.result()
                enriched_programs.append(enriched_program)
            except Exception as e:
                failed_count += 1
                print(f"  Warning: Failed to enrich program for task {program['task_id']}: {e}")

    print(f"Successfully enriched {len(enriched_programs)} programs ({failed_count} failed)")

    # Convert to DataFrame and save
    if enriched_programs:
        enriched_df = pd.DataFrame(enriched_programs)
        
        # Ensure output directory exists
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        enriched_df.to_parquet(output_path, index=False)
        print(f"Saved enriched dataset to {output_path}")
        
        # Print some stats
        print("\nEnriched Dataset Stats:")
        print(f"  Total programs: {len(enriched_programs)}")
        print(f"  Unique tasks: {enriched_df['task_id'].nunique()}")
        print(f"  Unique models: {enriched_df['model'].nunique()}")
        
        # Calculate actual training and test accuracy from the enriched data
        train_accuracies = []
        test_accuracies = []
        
        for _, row in enriched_df.iterrows():
            # Training accuracy: correct_train_input / train_input
            train_total = len(row['train_input'])
            train_correct = len(row['correct_train_input'])
            train_acc = train_correct / train_total if train_total > 0 else 0
            train_accuracies.append(train_acc)
            
            # Test accuracy: correct_test_input / test_input
            test_total = len(row['test_input'])
            test_correct = len(row['correct_test_input'])
            test_acc = test_correct / test_total if test_total > 0 else 0
            test_accuracies.append(test_acc)
        
        if train_accuracies:
            print(f"  Average training accuracy: {sum(train_accuracies) / len(train_accuracies):.3f}")
            perfect_train = sum(1 for acc in train_accuracies if acc == 1.0)
            print(f"  Perfect training accuracy: {perfect_train}/{len(train_accuracies)} ({perfect_train/len(train_accuracies)*100:.1f}%)")
        
        if test_accuracies:
            print(f"  Average test accuracy: {sum(test_accuracies) / len(test_accuracies):.3f}")
            perfect_test = sum(1 for acc in test_accuracies if acc == 1.0)
            print(f"  Perfect test accuracy: {perfect_test}/{len(test_accuracies)} ({perfect_test/len(test_accuracies)*100:.1f}%)")
    else:
        print("No programs to save!")


if __name__ == "__main__":
    main()
