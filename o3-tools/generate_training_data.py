#!/usr/bin/env python3

import os
import json
import glob
import sys
import tempfile
import subprocess
import argparse
import datetime
import random
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def load_task_data(task_id: str, dataset: str = "arc-agi-1") -> Optional[Dict]:
    """Load task data from the data folder"""
    # Try both training and evaluation folders
    for folder in ["training", "evaluation"]:
        task_path = Path(f"../data/{dataset}/{folder}/{task_id}.json")
        if task_path.exists():
            with open(task_path, 'r') as f:
                return json.load(f)
    
    # Also try arc-agi-2
    if dataset == "arc-agi-1":
        return load_task_data(task_id, "arc-agi-2")
    return None

def format_grid(grid: List[List[int]]) -> str:
    """Format a grid as a string"""
    return '\n'.join(' '.join(str(cell) for cell in row) for row in grid)

def format_task_for_prompt(task_data: Dict, include_test: bool = False) -> str:
    """Format task data into a string suitable for prompting"""
    lines = []
    
    # Format training examples
    lines.append("Training Examples:")
    for i, example in enumerate(task_data.get('train', [])):
        lines.append(f"\nExample {i+1}:")
        lines.append("Input:")
        lines.append(format_grid(example['input']))
        lines.append("Output:")
        lines.append(format_grid(example['output']))
    
    # Only include test if explicitly requested
    if include_test and task_data.get('test'):
        lines.append("\nTest Input:")
        lines.append(format_grid(task_data['test'][0]['input']))
    
    return '\n'.join(lines)

def execute_program(program: str, input_grid: List[List[int]], timeout: float = 0.5) -> Tuple[Optional[List[List[int]]], Optional[str], bool]:
    """Execute a program on an input grid and return the result"""
    try:
        # Create a temporary file with the program
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(program)
            f.write(f"\n\ntest_input = {input_grid}\n")
            f.write("try:\n")
            f.write("    output = transform(test_input)\n")
            f.write("    print('SUCCESS:', output)\n")
            f.write("except Exception as e:\n")
            f.write("    print('ERROR:', str(e))\n")
            temp_path = f.name
        
        # Execute the program
        try:
            result = subprocess.run(
                ['python', temp_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # Parse the output
            if result.returncode == 0:
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines:
                    if line.startswith('SUCCESS:'):
                        output_str = line[8:].strip()
                        output = eval(output_str)  # Be careful with eval in production
                        return output, None, False
                    elif line.startswith('ERROR:'):
                        error_msg = line[6:].strip()
                        return None, error_msg, False
                
                return None, "No output produced", False
            else:
                return None, result.stderr.strip() or "Unknown error", False
                
        except subprocess.TimeoutExpired:
            return None, "Execution timed out", True
            
    except Exception as e:
        return None, str(e), False
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass

def evaluate_program_on_task(program: str, task_data: Dict) -> Dict:
    """Evaluate a program on all training examples of a task"""
    training_results = []
    
    for i, example in enumerate(task_data['train']):
        input_grid = example['input']
        expected_output = example['output']
        
        predicted_output, error, timed_out = execute_program(program, input_grid)
        
        if predicted_output is not None and not error and not timed_out:
            # Check if output matches expected
            is_correct = predicted_output == expected_output
        else:
            is_correct = False
        
        training_results.append({
            'example_index': i,
            'correct': is_correct,
            'error': error,
            'timed_out': timed_out,
            'predicted_output': predicted_output,
            'expected_output': expected_output
        })
    
    # Calculate summary stats
    solved_count = sum(1 for r in training_results if r['correct'])
    total_examples = len(training_results)
    
    return {
        'solved_count': solved_count,
        'total_examples': total_examples,
        'training_results': training_results
    }

def create_prompt_for_task(task_data: Dict) -> str:
    """Create the full prompt for a task as used in run_arc_tasks.py"""
    # Get the consistent output grid dimensions from the first training example
    if task_data['train'] and task_data['train'][0]['output']:
        output_grid = task_data['train'][0]['output']
        output_height = len(output_grid)
        output_width = len(output_grid[0]) if output_grid else 0
        grid_size_info = f"\n**IMPORTANT: Your transformation must always produce a {output_height}×{output_width} output grid.**\n"
    else:
        grid_size_info = ""
    
    # Create the text content with text grid representation
    task_str = format_task_for_prompt(task_data, include_test=True)
    
    text_content = f"""You are solving an ARC (Abstraction and Reasoning Corpus) task. 
I will show you training examples with input and output grids, plus a test input grid. Your task is to:

1. **Analyze the training examples** to discover patterns that map input grids to output grids
2. **Write a Python program** that implements your best understanding of the transformation  
3. **DO NOT predict or generate the test output** - your job is only to write the transformation program
4. **Attempt a solution** - even if the pattern isn't completely clear, provide your best hypothesis
5. **Do not repeat the same transformation** - if you have already tried a transformation, do not repeat it.
{grid_size_info}
The test input is shown for context so you understand what type of grid your program will eventually process. Focus on learning patterns from training examples and writing code that captures your understanding.

{task_str}

Analyze the patterns in the training examples and write a Python function that performs this transformation.

**Approach Guidelines:**
- Look for patterns in shapes, colors, positions, sizes, rotations, reflections, etc.
- Even if you can't solve all training examples perfectly, implement what patterns you do observe
- A partial solution that captures some aspects is better than returning the input unchanged
- If the pattern is unclear, make your best educated guess based on what you can see

Requirements:
- The function takes a 2D list (grid) where grid[row][col] gives the value at that position
- Values are integers from 0-9
- Return a new grid (2D list) with the transformation applied
- You can use numpy if needed - just add 'import numpy as np' at the start of your function
- Aim to handle the training examples as well as possible, even if not perfectly
- Your function should attempt some meaningful transformation based on the patterns you observe

You MUST end your response with the following exact format:

Final answer:
```python
def transform(grid):
    # Your transformation logic here (implement your best understanding)
    return transformed_grid
```
"""
    
    return text_content

def extract_programs_from_log(log_path: str) -> List[Dict]:
    """Extract valid programs from a log file"""
    with open(log_path, 'r') as f:
        log_data = json.load(f)
    
    task_id = log_data.get('task_id')
    api_type = log_data.get('api_type', '')
    
    programs = []
    
    if 'multiturn' in api_type:
        # Multi-turn: only use first turn
        multiturn_data = log_data.get('multiturn_data', {})
        turn_details = multiturn_data.get('turn_details', [])
        
        if turn_details:
            first_turn = turn_details[0]
            program = first_turn.get('program', '')
            if program and first_turn.get('program_extracted', False):
                training_feedback = first_turn.get('training_feedback', {})
                solved_count = training_feedback.get('solved_count', 0)
                total_examples = training_feedback.get('total_training_examples', 0)
                
                # Only include if at least one training example is wrong
                if total_examples > 0 and solved_count < total_examples:
                    programs.append({
                        'task_id': task_id,
                        'program': program,
                        'turn_number': 1,
                        'logged_solved_count': solved_count,
                        'logged_total_examples': total_examples,
                        'log_path': log_path
                    })
    
    elif 'independent_attempts' in api_type:
        # Independent attempts: use all attempts
        independent_data = log_data.get('independent_attempts_data', {})
        attempt_details = independent_data.get('attempt_details', [])
        
        for attempt in attempt_details:
            program = attempt.get('program', '')
            if program and attempt.get('program_extracted', False):
                programs.append({
                    'task_id': task_id,
                    'program': program,
                    'attempt_number': attempt.get('attempt_number', 1),
                    'log_path': log_path
                })
    
    return programs

def validate_single_program(prog_data: Dict) -> Optional[Dict]:
    """Validate a single program and return qualified program data or None"""
    task_id = prog_data['task_id']
    program = prog_data['program']
    
    # Load task data
    task_data = load_task_data(task_id)
    if not task_data:
        return None
    
    # Always re-evaluate program to ensure consistency with training example creation
    evaluation = evaluate_program_on_task(program, task_data)
    solved_count = evaluation['solved_count']
    total_examples = evaluation['total_examples']
    
    # Check if program executes successfully on at least one example
    successful_executions = sum(1 for result in evaluation['training_results'] 
                               if result.get('predicted_output') is not None 
                               and not result.get('error') 
                               and not result.get('timed_out'))
    
    # Store the re-evaluated counts for later use
    prog_data['validated_solved_count'] = solved_count
    prog_data['validated_total_examples'] = total_examples
    prog_data['successful_executions'] = successful_executions
    
    # Only include if:
    # 1. Program runs without error (at least one successful execution)
    # 2. At least one training example is wrong
    if successful_executions > 0 and solved_count < total_examples:
        return {
            'program_data': prog_data,
            'task_data': task_data
        }
    
    return None

def create_training_example(program_data: Dict, task_data: Dict) -> Dict:
    """Create a training example in JSONL format"""
    # Create modified task data with program-generated outputs
    modified_task_data = task_data.copy()
    modified_task_data['train'] = []
    
    program = program_data['program']
    
    # Run the program on each training input to get what it actually produces
    for example in task_data['train']:
        train_input = example['input']
        predicted_output, error, timed_out = execute_program(program, train_input)
        
        # If the program runs successfully, validate and use its output
        if predicted_output is not None and not error and not timed_out:
            # Strict validation: reject entire program if ANY output is not a proper 2D grid
            if not isinstance(predicted_output, list):
                raise ValueError(f"Program returned invalid output format: expected list, got {type(predicted_output).__name__}")
            if len(predicted_output) == 0:
                raise ValueError(f"Program returned empty grid")
            if not isinstance(predicted_output[0], list):
                raise ValueError(f"Program returned 1D list instead of 2D grid")
                
            modified_example = {
                'input': train_input,
                'output': predicted_output  # Use program's actual output, not ground truth
            }
            modified_task_data['train'].append(modified_example)
    
    # Only proceed if we have at least one successful execution
    if not modified_task_data['train']:
        raise ValueError(f"Program failed to run on all {len(task_data['train'])} training examples")
    
    # Create system message
    system_message = {
        "role": "system", 
        "content": "You are an expert at solving abstract reasoning puzzles. Write clean, efficient Python code."
    }
    
    # Create user message with the modified prompt (using program's actual outputs)
    user_message = {
        "role": "user",
        "content": create_prompt_for_task(modified_task_data)
    }
    
    # Create assistant message with just the program
    assistant_message = {
        "role": "assistant",
        "content": f"Final answer:\n```python\n{program_data['program']}\n```"
    }
    
    # Create the training example
    training_example = {
        "messages": [system_message, user_message, assistant_message]
    }
    
    return training_example

def main():
    parser = argparse.ArgumentParser(description="Generate training data from log files")
    parser.add_argument("--limit", type=int, default=None, 
                       help="Limit to the last N log files (sorted by timestamp)")
    parser.add_argument("--output", type=str, 
                       default=f"training_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
                       help="Output JSONL file name")
    parser.add_argument("--validation", action="store_true",
                       help="Create a validation split (10% or 32 examples, whichever is smaller)")
    
    args = parser.parse_args()
    
    # Create training_data directory if it doesn't exist
    output_dir = Path("training_data")
    output_dir.mkdir(exist_ok=True)
    
    # Calculate number of workers (total cores - 2, minimum 1)
    max_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"Using {max_workers} worker processes (total cores: {multiprocessing.cpu_count()})")
    
    # Get all log files (exclude summary files)
    log_files = glob.glob("logs/*.json")
    log_files = [f for f in log_files if 'summary' not in f]
    
    # Sort by timestamp (filename contains timestamp)
    log_files.sort()
    
    if args.limit:
        log_files = log_files[-args.limit:]
    
    print(f"Processing {len(log_files)} log files...")
    
    all_programs = []
    processed_count = 0
    
    # Process log files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_file = {executor.submit(extract_programs_from_log, log_file): log_file 
                         for log_file in log_files}
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            log_file = future_to_file[future]
            processed_count += 1
            
            # Progress reporting every 100 files
            if processed_count % 100 == 0:
                print(f"  Processed {processed_count}/{len(log_files)} files...")
            
            try:
                programs = future.result()
                all_programs.extend(programs)
            except Exception as e:
                if processed_count % 500 == 0:  # Only print errors occasionally to avoid spam
                    print(f"  Warning: Error processing {log_file}: {e}")
    
    print(f"\nFound {len(all_programs)} total programs")
    
    # Filter programs to only those that qualify (in parallel)
    qualified_programs = []
    validated_count = 0
    
    print(f"Validating programs in parallel...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all validation jobs
        future_to_prog = {executor.submit(validate_single_program, prog_data): prog_data 
                         for prog_data in all_programs}
        
        # Collect results as they complete
        for future in as_completed(future_to_prog):
            validated_count += 1
            
            # Progress reporting every 50 programs
            if validated_count % 50 == 0:
                print(f"  Validated {validated_count}/{len(all_programs)} programs...")
            
            try:
                result = future.result()
                if result is not None:
                    qualified_programs.append(result)
            except Exception as e:
                if validated_count % 500 == 0:  # Only print errors occasionally
                    print(f"  Warning: Error validating program: {e}")
    
    print(f"\nQualified programs: {len(qualified_programs)}")
    
    # Generate training examples (in parallel)
    training_examples = []
    programs_with_at_least_one_correct = 0
    validation_mismatches = 0
    invalid_output_programs = 0
    processed_examples = 0
    
    print(f"Generating training examples from {len(qualified_programs)} qualified programs in parallel...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all training example generation jobs
        future_to_info = {executor.submit(create_training_example, prog_info['program_data'], prog_info['task_data']): prog_info 
                         for prog_info in qualified_programs}
        
        # Collect results as they complete
        for future in as_completed(future_to_info):
            prog_info = future_to_info[future]
            processed_examples += 1
            
            # Progress reporting every 100 programs
            if processed_examples % 100 == 0:
                print(f"  Generated {len(training_examples)} examples from {processed_examples}/{len(qualified_programs)} programs ({processed_examples/len(qualified_programs)*100:.1f}%)...")
            
            try:
                training_example = future.result()
                training_examples.append(training_example)
                
                # Track programs with at least one originally correct answer
                prog_data = prog_info['program_data']
                solved_count = prog_data.get('validated_solved_count', 0)
                
                if solved_count > 0:
                    programs_with_at_least_one_correct += 1
                
                # Check for validation mismatches (if we have logged data to compare)
                if 'logged_solved_count' in prog_data:
                    logged_count = prog_data['logged_solved_count']
                    if logged_count != solved_count:
                        validation_mismatches += 1
                        if validation_mismatches <= 3:  # Only print first few to avoid spam
                            print(f"  ⚠️  Validation mismatch - Task {prog_data.get('task_id', 'unknown')}: logged={logged_count}, validated={solved_count}")
                    
            except Exception as e:
                if any(phrase in str(e) for phrase in ["invalid output format", "1D list instead", "empty grid"]):
                    invalid_output_programs += 1
                if processed_examples % 500 == 0:  # Only print errors occasionally to avoid spam
                    print(f"  Warning: Error creating training example: {e}")
    
    print(f"Generated {len(training_examples)} training examples")
    
    # Calculate percentage with at least one originally correct answer
    if len(training_examples) > 0:
        pct_with_correct = (programs_with_at_least_one_correct / len(training_examples)) * 100
        print(f"Programs with at least one originally correct answer: {programs_with_at_least_one_correct}/{len(training_examples)} ({pct_with_correct:.1f}%)")
    else:
        print(f"Programs with at least one originally correct answer: 0/0 (0.0%)")
    
    # Report validation mismatches
    if validation_mismatches > 0:
        print(f"⚠️  Validation mismatches found: {validation_mismatches} programs had different results than logged")
        print(f"   This suggests code extraction or execution inconsistencies")
    else:
        print(f"✅ No validation mismatches found - all programs behaved consistently")
        
    # Report invalid output programs
    if invalid_output_programs > 0:
        print(f"⚠️  Invalid output format: {invalid_output_programs} programs returned non-2D-grid outputs")
        print(f"   These programs were rejected entirely for format violations")
    else:
        print(f"✅ All programs returned valid 2D grid formats")
    
    # Handle validation split if requested
    if args.validation and len(training_examples) > 1:
        # Set random seed for reproducible splits
        random.seed(42)
        
        # Group examples by task_id
        task_to_examples = {}
        for i, example in enumerate(training_examples):
            # Extract task_id from the qualified_programs
            task_id = qualified_programs[i]['program_data']['task_id']
            if task_id not in task_to_examples:
                task_to_examples[task_id] = []
            task_to_examples[task_id].append(example)
        
        # Get all unique task_ids
        all_task_ids = list(task_to_examples.keys())
        random.shuffle(all_task_ids)
        
        # Calculate validation size (10% or 32 examples, whichever is smaller)
        target_validation_size = min(int(len(training_examples) * 0.1), 32)
        
        # Select tasks for validation by accumulating examples until we reach target size
        validation_task_ids = []
        validation_examples = []
        
        for task_id in all_task_ids:
            task_examples = task_to_examples[task_id]
            if len(validation_examples) + len(task_examples) <= target_validation_size:
                validation_task_ids.append(task_id)
                validation_examples.extend(task_examples)
            elif len(validation_examples) == 0:
                # If we haven't selected any tasks yet and this task would exceed the limit,
                # include it anyway to ensure we have at least some validation data
                validation_task_ids.append(task_id)
                validation_examples.extend(task_examples)
                break
        
        # All remaining tasks go to training
        train_examples = []
        train_task_ids = []
        for task_id in all_task_ids:
            if task_id not in validation_task_ids:
                train_task_ids.append(task_id)
                train_examples.extend(task_to_examples[task_id])
        
        # Create filenames
        base_name = args.output.replace('.jsonl', '')
        train_filename = output_dir / f"{base_name}_train.jsonl"
        val_filename = output_dir / f"{base_name}_val.jsonl"
        
        # Save training set
        with open(train_filename, 'w') as f:
            for example in train_examples:
                f.write(json.dumps(example) + '\n')
        
        # Save validation set
        with open(val_filename, 'w') as f:
            for example in validation_examples:
                f.write(json.dumps(example) + '\n')
        
        print(f"Saved training data to: {train_filename} ({len(train_examples)} examples from {len(train_task_ids)} tasks)")
        print(f"Saved validation data to: {val_filename} ({len(validation_examples)} examples from {len(validation_task_ids)} tasks)")
        print(f"Validation tasks: {sorted(validation_task_ids)}")
        
    else:
        # Save all as training data
        output_file = output_dir / args.output
        with open(output_file, 'w') as f:
            for example in training_examples:
                f.write(json.dumps(example) + '\n')
        
        print(f"Saved training data to: {output_file}")
    
    # Print some stats
    task_counts = {}
    for prog_info in qualified_programs:
        task_id = prog_info['program_data']['task_id']
        task_counts[task_id] = task_counts.get(task_id, 0) + 1
    
    print(f"\nStatistics:")
    print(f"  Unique tasks: {len(task_counts)}")
    if len(task_counts) > 0:
        print(f"  Average examples per task: {len(training_examples) / len(task_counts):.1f}")
        print(f"  Tasks with most examples: {sorted(task_counts.items(), key=lambda x: x[1], reverse=True)[:5]}")
    else:
        print(f"  No qualified programs found")

if __name__ == "__main__":
    main() 