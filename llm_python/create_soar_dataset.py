#!/usr/bin/env python3

import os
import json
import argparse
import datetime
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi

# Import utilities
try:
    # Try relative imports first (when run as module)
    from .utils.task_loader import TaskLoader
except ImportError:
    # Fall back to absolute imports (when run directly)
    from utils.task_loader import TaskLoader

# Initialize utilities
task_loader = TaskLoader()

def load_grid_data_for_task(task_id: str, dataset: str) -> Optional[Dict]:
    """Load grid data for a specific task from the data directory"""
    try:
        # Try to load from both training and evaluation directories
        for split in ['training', 'evaluation']:
            filepath = f"../data/{dataset}/{split}/{task_id}.json"
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
        
        # If not found in either split, return None
        return None
    except Exception as e:
        print(f"Error loading grid data for task {task_id}: {e}")
        return None

def create_combined_example(soar_row: Dict, grid_data: Dict) -> Dict:
    """Create a combined example with SOAR data and grid data"""
    
    # Extract grid data
    train_inputs = [example['input'] for example in grid_data['train']]
    train_outputs = [example['output'] for example in grid_data['train']]
    test_inputs = [example['input'] for example in grid_data['test']]
    test_outputs = [example['output'] for example in grid_data['test']]
    
    # Extract SOAR data
    code = soar_row.get('code', '')
    model = soar_row.get('model', '')
    generation = soar_row.get('generation', 0)
    task_id = soar_row.get('task_id', '')
    
    # Use SOAR predictions and correctness flags
    predicted_train_output = soar_row.get('predicted_train_output', [])
    predicted_test_output = soar_row.get('predicted_test_output', [])
    correct_train_input = soar_row.get('correct_train_input', [])
    correct_test_input = soar_row.get('correct_test_input', [])
    
    # Create the combined example
    combined_example = {
        "reasoning": "",  # SOAR dataset doesn't have reasoning
        "code": code,
        "correct_train_input": correct_train_input,
        "train_input": train_inputs,
        "train_output": train_outputs,
        "predicted_train_output": predicted_train_output,
        "correct_test_input": correct_test_input,
        "test_input": test_inputs,
        "test_output": test_outputs,
        "predicted_test_output": predicted_test_output,
        "task_id": task_id,
        "model": model,
        "generation": generation
    }
    
    return combined_example

def calculate_correctness_percentage(correct_flags: List[bool]) -> float:
    """Calculate the percentage of True values in a list of boolean flags"""
    if not correct_flags:
        return 0.0
    return sum(correct_flags) / len(correct_flags) * 100

def filter_soar_rows_by_greedy(soar_rows: List[Dict], max_rows_per_task: int, max_total_rows: int = None) -> List[Dict]:
    """Filter SOAR rows using greedy method: prioritize perfect solutions, then by training accuracy"""
    
    # Group rows by task_id
    task_rows = defaultdict(list)
    for row in soar_rows:
        task_id = row['task_id']
        task_rows[task_id].append(row)
    
    filtered_rows = []
    
    for task_id, rows in task_rows.items():
        # Sort rows by priority:
        # 1. All test correct AND all train correct
        # 2. All test correct, then by train accuracy
        # 3. Not all test correct, by train accuracy
        # 4. If still need more, by code length (shortest first)
        
        def sort_key(row):
            test_all_correct = all(row.get('correct_test_input', []))
            train_all_correct = all(row.get('correct_train_input', []))
            train_accuracy = calculate_correctness_percentage(row.get('correct_train_input', []))
            code_length = len(row.get('code', ''))
            
            # Priority 1: Both test and train all correct
            if test_all_correct and train_all_correct:
                return (0, 0, 0, code_length)
            # Priority 2: Test all correct, then by train accuracy
            elif test_all_correct:
                return (1, -train_accuracy, 0, code_length)
            # Priority 3: Not all test correct, by train accuracy
            else:
                return (2, -train_accuracy, 0, code_length)
        
        # Sort rows by priority
        sorted_rows = sorted(rows, key=sort_key)
        
        # Take up to max_rows_per_task, but respect total limit
        max_for_this_task = max_rows_per_task
        if max_total_rows is not None:
            remaining_slots = max_total_rows - len(filtered_rows)
            max_for_this_task = min(max_for_this_task, remaining_slots)
        
        if max_for_this_task > 0:
            selected_rows = sorted_rows[:max_for_this_task]
            filtered_rows.extend(selected_rows)
    
    return filtered_rows

def filter_soar_rows_by_balanced(soar_rows: List[Dict], max_rows_per_task: int, max_total_rows: int = None) -> List[Dict]:
    """Filter SOAR rows using balanced method: half greedy, half random from failures"""
    
    # Group rows by task_id
    task_rows = defaultdict(list)
    for row in soar_rows:
        task_id = row['task_id']
        task_rows[task_id].append(row)
    
    filtered_rows = []
    
    for task_id, rows in task_rows.items():
        # Separate rows into successful and failed
        successful_rows = []
        failed_rows = []
        
        for row in rows:
            test_all_correct = all(row.get('correct_test_input', []))
            train_all_correct = all(row.get('correct_train_input', []))
            
            if test_all_correct and train_all_correct:
                successful_rows.append(row)
            elif not test_all_correct and not train_all_correct:
                failed_rows.append(row)
            else:
                # Mixed results - put in successful for now
                successful_rows.append(row)
        
        # Sort successful rows by priority (same as greedy)
        def sort_key(row):
            test_all_correct = all(row.get('correct_test_input', []))
            train_all_correct = all(row.get('correct_train_input', []))
            train_accuracy = calculate_correctness_percentage(row.get('correct_train_input', []))
            code_length = len(row.get('code', ''))
            
            if test_all_correct and train_all_correct:
                return (0, 0, 0, code_length)
            elif test_all_correct:
                return (1, -train_accuracy, 0, code_length)
            else:
                return (2, -train_accuracy, 0, code_length)
        
        successful_rows.sort(key=sort_key)
        
        # Calculate how many to take from each category, but respect total limit
        max_for_this_task = max_rows_per_task
        if max_total_rows is not None:
            remaining_slots = max_total_rows - len(filtered_rows)
            max_for_this_task = min(max_for_this_task, remaining_slots)
        
        if max_for_this_task <= 0:
            break
        
        greedy_count = min(max_for_this_task // 2, len(successful_rows))
        random_count = min(max_for_this_task - greedy_count, len(failed_rows))
        
        # Take greedy rows
        selected_rows = successful_rows[:greedy_count]
        
        # Take random failed rows
        if random_count > 0:
            random.shuffle(failed_rows)
            selected_rows.extend(failed_rows[:random_count])
        
        # If we still need more, take from remaining successful rows
        remaining_needed = max_for_this_task - len(selected_rows)
        if remaining_needed > 0:
            remaining_successful = successful_rows[greedy_count:]
            selected_rows.extend(remaining_successful[:remaining_needed])
        
        filtered_rows.extend(selected_rows)
    
    return filtered_rows

def main():
    parser = argparse.ArgumentParser(description="Create SOAR dataset with grid data and filtering options")
    parser.add_argument("--max-rows", type=int, default=1024,
                       help="Maximum number of rows in final dataset (default: 1024)")
    parser.add_argument("--chunk-size", type=int, default=32000,
                       help="Number of rows to stream per chunk from SOAR dataset (default: 2000)")
    parser.add_argument("--max-rows-per-task", type=int, default=8,
                       help="Maximum rows per task in final dataset (default: 10)")
    parser.add_argument("--filter-method", choices=["greedy", "balanced"], default="greedy",
                       help="Filtering method: greedy (prioritize perfect solutions) or balanced (half greedy, half random failures)")
    parser.add_argument("--dataset", type=str, default="arc-agi-1",
                       help="Dataset to use for grid data (default: arc-agi-1)")
    parser.add_argument("--hf-org", type=str, default="Trelis",
                       help="Hugging Face organization (default: Trelis)")
    parser.add_argument("--hf-private", action="store_true",
                       help="Make the Hugging Face dataset private")
    parser.add_argument("--save-local", action="store_true",
                       help="Save dataset locally as JSONL file")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSONL file name (optional)")
    
    args = parser.parse_args()
    
    print(f"Loading SOAR dataset (chunked streaming, target: {args.max_rows} final rows)...")
    
    # Load SOAR dataset in streaming mode
    soar_dataset = load_dataset("julien31/soar_arc_train_5M", streaming=True, split="train")
    soar_iterator = iter(soar_dataset)  # Create iterator to maintain position
    
    # Collect rows until we have enough data
    selected_rows = []  # Final selected rows
    task_rows = defaultdict(list)  # Group rows by task_id
    completed_tasks = set()  # Tasks we've filled quota for
    missing_grid_data = []
    chunk_count = 0
    total_streamed = 0
    
    while len(selected_rows) < args.max_rows:  # Stop when we have enough rows
        chunk_count += 1
        chunk_rows = []
        
        print(f"  Streaming chunk {chunk_count} ({args.chunk_size} rows)...")
        
        # Track all tasks encountered in this chunk for debugging
        chunk_all_tasks = set()
        chunk_with_grid = set()
        chunk_completed_skipped = set()
        
        # Stream one chunk
        for i in range(args.chunk_size):
            try:
                row = next(soar_iterator)
            except StopIteration:
                print(f"    Reached end of dataset after {total_streamed + i} total rows")
                break
            
            task_id = row.get('task_id', '')
            if not task_id:
                continue
            
            chunk_all_tasks.add(task_id)
            
            # Skip if we've already filled quota for this task
            if task_id in completed_tasks:
                chunk_completed_skipped.add(task_id)
                continue
            
            # Check if we have grid data for this task
            grid_data = load_grid_data_for_task(task_id, args.dataset)
            if grid_data is None:
                missing_grid_data.append(task_id)
                continue
            
            chunk_with_grid.add(task_id)
            chunk_rows.append(row)
            task_rows[task_id].append(row)
            
            # Check if we've filled quota for this task
            if len(task_rows[task_id]) >= args.max_rows_per_task:
                # Apply filtering to this task's rows
                if args.filter_method == "greedy":
                    task_selected = filter_soar_rows_by_greedy(task_rows[task_id], args.max_rows_per_task)
                else:  # balanced
                    task_selected = filter_soar_rows_by_balanced(task_rows[task_id], args.max_rows_per_task)
                
                selected_rows.extend(task_selected)
                completed_tasks.add(task_id)
                
                # Clear the task data to save memory
                del task_rows[task_id]
                
                print(f"    Completed task {task_id}: selected {len(task_selected)} rows (total: {len(selected_rows)})")
                
                # Check if we've reached our target
                if len(selected_rows) >= args.max_rows:
                    break
        
        total_streamed += len(chunk_rows)
        
        # Debug: show what we found in this chunk
        chunk_no_grid = chunk_all_tasks - chunk_with_grid - chunk_completed_skipped
        
        print(f"    Chunk {chunk_count}: {len(chunk_rows)} new rows with grid data from {len(chunk_with_grid)} new tasks")
        print(f"    Debug: {len(chunk_all_tasks)} total tasks, {len(chunk_completed_skipped)} already completed, {len(chunk_no_grid)} without grid data")
        print(f"    Progress: {len(selected_rows)}/{args.max_rows} rows from {len(completed_tasks)} completed tasks")
        
        # If we've streamed too much without getting enough data, stop
        if total_streamed > 2750000:  # Safety limit: ~2.75M rows
            print(f"  Warning: Streamed {total_streamed} rows but only got {len(selected_rows)} after filtering")
            print(f"  Stopping to prevent excessive streaming")
            break
    
    # Handle any remaining incomplete tasks
    for task_id, rows in task_rows.items():
        if task_id not in completed_tasks and len(selected_rows) < args.max_rows:
            if args.filter_method == "greedy":
                task_selected = filter_soar_rows_by_greedy(rows, args.max_rows_per_task)
            else:  # balanced
                task_selected = filter_soar_rows_by_balanced(rows, args.max_rows_per_task)
            
            # Take only what we need to reach target
            remaining_slots = args.max_rows - len(selected_rows)
            task_selected = task_selected[:remaining_slots]
            selected_rows.extend(task_selected)
            
            if task_selected:
                print(f"    Final task {task_id}: selected {len(task_selected)} rows (total: {len(selected_rows)})")
    
    # Trim to exact target if we went over
    if len(selected_rows) > args.max_rows:
        selected_rows = selected_rows[:args.max_rows]
    
    filtered_rows = selected_rows
    task_id_set = set(row['task_id'] for row in filtered_rows)
    
    print(f"Final: {len(filtered_rows)} SOAR rows with matching grid data")
    print(f"Unique tasks: {len(task_id_set)}")
    print(f"Total streamed: {total_streamed} rows in {chunk_count} chunks")
    
    if missing_grid_data:
        print(f"Warning: {len(missing_grid_data)} tasks missing grid data (first 10: {missing_grid_data[:10]})")
    
    # Create combined examples
    print("Creating combined examples...")
    combined_examples = []
    task_counts = defaultdict(int)
    
    for soar_row in filtered_rows:
        task_id = soar_row.get('task_id', '')
        grid_data = load_grid_data_for_task(task_id, args.dataset)
        
        if grid_data is None:
            print(f"Error: Missing grid data for task {task_id} during processing")
            continue
        
        # Validate that grid data matches SOAR predictions
        soar_train_count = len(soar_row.get('correct_train_input', []))
        soar_test_count = len(soar_row.get('correct_test_input', []))
        grid_train_count = len(grid_data.get('train', []))
        grid_test_count = len(grid_data.get('test', []))
        
        if soar_train_count != grid_train_count or soar_test_count != grid_test_count:
            print(f"Warning: Mismatch for task {task_id}: SOAR has {soar_train_count} train, {soar_test_count} test; Grid has {grid_train_count} train, {grid_test_count} test")
            continue
        
        combined_example = create_combined_example(soar_row, grid_data)
        combined_examples.append(combined_example)
        task_counts[task_id] += 1
    
    print(f"Created {len(combined_examples)} combined examples")
    
    # Calculate quality statistics
    unique_tasks = len(task_counts)
    if unique_tasks > 0:
        # Calculate row-level statistics
        all_train_test_correct_rows = 0
        at_least_one_train_correct_rows = 0
        partial_success_rows = 0  # Some train correct but not all test correct
        
        # Calculate task-level statistics
        tasks_with_all_correct = set()
        tasks_with_at_least_one_train_correct = set()
        
        for example in combined_examples:
            task_id = example['task_id']
            correct_train = example['correct_train_input']
            correct_test = example['correct_test_input']
            
            # Row-level stats
            if all(correct_train) and all(correct_test):
                all_train_test_correct_rows += 1
                tasks_with_all_correct.add(task_id)
            
            if any(correct_train):
                at_least_one_train_correct_rows += 1
                tasks_with_at_least_one_train_correct.add(task_id)
                
                # Check for partial success (some train correct but not all test correct)
                if not all(correct_test):
                    partial_success_rows += 1
        
        # Calculate percentages
        total_rows = len(combined_examples)
        pct_all_correct_rows = (all_train_test_correct_rows / total_rows * 100) if total_rows > 0 else 0
        pct_at_least_one_train_rows = (at_least_one_train_correct_rows / total_rows * 100) if total_rows > 0 else 0
        pct_tasks_all_correct = (len(tasks_with_all_correct) / unique_tasks * 100) if unique_tasks > 0 else 0
        pct_tasks_at_least_one_train = (len(tasks_with_at_least_one_train_correct) / unique_tasks * 100) if unique_tasks > 0 else 0
        
        print(f"\nDataset Statistics:")
        print(f"  Unique tasks: {unique_tasks}")
        print(f"  Total examples: {len(combined_examples)}")
        print(f"  Rows per task: {list(task_counts.values())}")
        print(f"  Rows with all train AND test correct: {all_train_test_correct_rows}/{total_rows} ({pct_all_correct_rows:.1f}%)")
        print(f"  Rows with at least one train correct: {at_least_one_train_correct_rows}/{total_rows} ({pct_at_least_one_train_rows:.1f}%)")
        print(f"  Rows with partial success (train correct, test wrong): {partial_success_rows}/{total_rows} ({partial_success_rows/total_rows*100:.1f}%)")
        print(f"  Tasks with at least one all-correct row: {len(tasks_with_all_correct)}/{unique_tasks} ({pct_tasks_all_correct:.1f}%)")
        print(f"  Tasks with at least one train-correct row: {len(tasks_with_at_least_one_train_correct)}/{unique_tasks} ({pct_tasks_at_least_one_train:.1f}%)")
    else:
        print("No examples created!")
        return
    
    # Generate dataset name
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset_name = f"soar-{timestamp}-{len(combined_examples)}"
    
    # Save locally if requested
    if args.save_local or args.output:
        output_file = args.output or f"training_data/{dataset_name}.jsonl"
        os.makedirs("training_data", exist_ok=True)
        
        with open(output_file, 'w') as f:
            for example in combined_examples:
                f.write(json.dumps(example) + '\n')
        
        print(f"Saved {len(combined_examples)} examples to {output_file}")
    
    # Ask user if they want to push to Hugging Face
    print(f"\nReady to push dataset to {args.hf_org}/{dataset_name}")
    response = input("Push to Hugging Face? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        print(f"Pushing to Hugging Face...")
        
        # Create and push dataset
        dataset = Dataset.from_list(combined_examples)
        dataset.push_to_hub(
            f"{args.hf_org}/{dataset_name}",
            private=args.hf_private
        )
        
        visibility = "private" if args.hf_private else "public"
        print(f"Successfully pushed dataset to https://huggingface.co/datasets/{args.hf_org}/{dataset_name} ({visibility})")
    else:
        print("Dataset not pushed to Hugging Face")

if __name__ == "__main__":
    main() 