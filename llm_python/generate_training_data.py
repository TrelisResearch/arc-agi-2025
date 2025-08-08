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
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datasets import Dataset
from huggingface_hub import HfApi

# Import utilities
try:
    # Try relative imports first (when run as module)
    from .utils.task_loader import TaskLoader
    from .progdb.arc_tester import ArcTester
    from .utils.transduction import is_transduction_cheating
except ImportError:
    # Fall back to absolute imports (when run directly)
    from utils.task_loader import TaskLoader
    from progdb.arc_tester import ArcTester
    from utils.transduction import is_transduction_cheating

# Initialize utilities
task_loader = TaskLoader()

def get_arc_tester():
    """Get an ArcTester instance (lazy initialization)"""
    return ArcTester(timeout=0.5)

def format_grid(grid: List[List[int]]) -> str:
    """Format a grid as a string, preserving empty rows with special marker"""
    lines = []
    for row in grid:
        if len(row) == 0:
            lines.append('[EMPTY_ROW]')
        else:
            lines.append(' '.join(str(cell) for cell in row))
    return '\n'.join(lines)

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

def strip_comments_aggressive(source_code: str) -> str:
    """
    Aggressive comment stripping - removes comments and cleans up whitespace.
    This version removes comment lines entirely and normalizes whitespace.
    """
    if not source_code.strip():
        return source_code
    
    try:
        # First pass: identify which lines are comment-only or empty
        lines = source_code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            # Skip empty lines and comment-only lines
            if not stripped or stripped.startswith('#'):
                continue
            
            # Remove inline comments but preserve the code part
            if '#' in line:
                # Find the # that's not inside a string literal
                in_string = False
                quote_char = None
                for i, char in enumerate(line):
                    if char in ['"', "'"] and (i == 0 or line[i-1] != '\\'):
                        if not in_string:
                            in_string = True
                            quote_char = char
                        elif char == quote_char:
                            in_string = False
                            quote_char = None
                    elif char == '#' and not in_string:
                        line = line[:i].rstrip()
                        break
            
            cleaned_lines.append(line)
        
        # Join lines and normalize whitespace
        result = '\n'.join(cleaned_lines)
        
        # Remove excessive blank lines (more than 2 consecutive)
        result = re.sub(r'\n\s*\n\s*\n+', '\n\n', result)
        
        return result.strip()
        
    except Exception as e:
        print(f"Warning: Error in aggressive comment stripping: {e}")
        return source_code

def evaluate_program_on_task(program: str, task_data: Dict) -> Dict:
    """Evaluate a program on all training examples of a task"""
    training_results = []
    
    for i, example in enumerate(task_data['train']):
        input_grid = example['input']
        expected_output = example['output']
        
        predicted_output, error, timed_out = get_arc_tester().execute_program_with_timeout(program, input_grid)
        
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
                
                # Extract reasoning content from standardized 'reasoning' field
                reasoning_content = None
                raw_response = first_turn.get('raw_response', {})
                if isinstance(raw_response, dict):
                    # Check for standardized reasoning field in message
                    if 'choices' in raw_response:
                        choices = raw_response.get('choices', [])
                        if choices and isinstance(choices[0], dict):
                            message = choices[0].get('message', {})
                            reasoning_content = message.get('reasoning')
                
                # Include if we have training examples and the program was extracted
                if total_examples > 0:
                    programs.append({
                        'task_id': task_id,
                        'program': program,
                        'turn_number': 1,
                        'logged_solved_count': solved_count,
                        'logged_total_examples': total_examples,
                        'log_path': log_path,
                        'model': log_data.get('model', ''),
                        'dataset': log_data.get('dataset', ''),
                        'subset': log_data.get('subset', ''),
                        'api_type': api_type,
                        'reasoning_content': reasoning_content
                    })
    
    elif 'independent_attempts' in api_type:
        # Independent attempts: use all attempts
        independent_data = log_data.get('independent_attempts_data', {})
        attempt_details = independent_data.get('attempt_details', [])
        
        for attempt in attempt_details:
            program = attempt.get('program', '')
            if program and attempt.get('program_extracted', False):
                # Extract reasoning content from standardized 'reasoning' field
                reasoning_content = None
                
                # First try the attempt's raw_response
                raw_response = attempt.get('raw_response', {})
                if isinstance(raw_response, dict):
                    # Check for standardized reasoning field in message
                    if 'choices' in raw_response:
                        choices = raw_response.get('choices', [])
                        if choices and isinstance(choices[0], dict):
                            message = choices[0].get('message', {})
                            reasoning_content = message.get('reasoning')
                
                programs.append({
                    'task_id': task_id,
                    'program': program,
                    'attempt_number': attempt.get('attempt_number', 1),
                    'log_path': log_path,
                    'model': log_data.get('model', ''),
                    'dataset': log_data.get('dataset', ''),
                    'subset': log_data.get('subset', ''),
                    'api_type': api_type,
                    'reasoning_content': reasoning_content
                })
    
    elif 'all_attempts' in api_type:
        # All attempts: use all attempts (new format from run_arc_tasks_soar.py)
        attempt_details = log_data.get('attempt_details', [])
        
        for attempt in attempt_details:
            program = attempt.get('program', '')
            if program and attempt.get('program_extracted', False):
                # Extract reasoning content from standardized 'reasoning' field
                reasoning_content = None
                
                # First try the attempt's raw_response
                raw_response = attempt.get('raw_response', {})
                if isinstance(raw_response, dict):
                    # Check for standardized reasoning field in message
                    if 'choices' in raw_response:
                        choices = raw_response.get('choices', [])
                        if choices and isinstance(choices[0], dict):
                            message = choices[0].get('message', {})
                            reasoning_content = message.get('reasoning')
                
                programs.append({
                    'task_id': task_id,
                    'program': program,
                    'attempt_number': attempt.get('attempt_number', 1),
                    'log_path': log_path,
                    'model': log_data.get('model', ''),
                    'dataset': log_data.get('dataset', ''),
                    'subset': log_data.get('subset', ''),
                    'api_type': api_type,
                    'reasoning_content': reasoning_content
                })
    
    return programs

def validate_single_program(prog_data: Dict, args) -> Optional[Dict]:
    """Validate a single program and return qualified program data or None"""
    task_id = prog_data['task_id']
    program = prog_data['program']
    dataset = prog_data.get('dataset', 'arc-agi-1')
    
    # Load task data
    try:
        task_data = task_loader.load_task(task_id, dataset)
    except FileNotFoundError:
        return None
    
    # Check for transduction/cheating (unless disabled)
    if not args.no_transduction_filter:
        is_cheating, cheat_reason = is_transduction_cheating(program, task_data, debug=args.debug)
        if is_cheating:
            prog_data['transduction_reason'] = cheat_reason
            return None  # Reject cheating programs
    
    # Always re-evaluate program to ensure consistency with training example creation
    evaluation = evaluate_program_on_task(program, task_data)
    solved_count = evaluation['solved_count']
    total_examples = evaluation['total_examples']
    
    # Check if program executes successfully on ALL training examples
    successful_executions = sum(1 for result in evaluation['training_results'] 
                               if result.get('predicted_output') is not None 
                               and not result.get('error') 
                               and not result.get('timed_out'))
    
    all_successful = successful_executions == total_examples
    
    # Store the re-evaluated counts for later use
    prog_data['validated_solved_count'] = solved_count
    prog_data['validated_total_examples'] = total_examples
    prog_data['successful_executions'] = successful_executions
    prog_data['all_examples_execute'] = all_successful
    
    # Only include if program runs without error on ALL training examples
    if all_successful:
        return {
            'program_data': prog_data,
            'task_data': task_data
        }
    
    return None

def create_training_example(program_data: Dict, task_data: Dict, args) -> Dict:
    """Create a training example in competition dataset format"""
    program = program_data['program']
    
    # Extract training inputs and outputs
    train_inputs = [example['input'] for example in task_data['train']]
    train_outputs = [example['output'] for example in task_data['train']]
    
    # Extract test inputs and outputs
    test_inputs = [example['input'] for example in task_data['test']]
    test_outputs = [example['output'] for example in task_data['test']]
    
    # Run the program on each training input to get predictions
    predicted_train_outputs = []
    correct_train_inputs = []
    
    for i, train_input in enumerate(train_inputs):
        predicted_output, error, timed_out = get_arc_tester().execute_program_with_timeout(program, train_input)
        
        if predicted_output is not None and not error and not timed_out:
            # Validate output format
            if not isinstance(predicted_output, list) or len(predicted_output) == 0 or not isinstance(predicted_output[0], list):
                raise ValueError(f"Program returned invalid output format for training input {i}")
            
            # Validate cell values
            for row_idx, row in enumerate(predicted_output):
                if not isinstance(row, list):
                    raise ValueError(f"Row {row_idx} is not a list in training output {i}")
                for col_idx, cell in enumerate(row):
                    if isinstance(cell, bool) or not isinstance(cell, int) or not (0 <= cell <= 9):
                        raise ValueError(f"Invalid cell value at [{row_idx}][{col_idx}] in training output {i}: {cell}")
            
            predicted_train_outputs.append(predicted_output)
            # Check if prediction matches ground truth
            correct_train_inputs.append(predicted_output == train_outputs[i])
        else:
            raise ValueError(f"Program failed to run on training input {i}")
    
    # Run the program on each test input to get predictions
    predicted_test_outputs = []
    correct_test_inputs = []
    
    for i, test_input in enumerate(test_inputs):
        predicted_output, error, timed_out = get_arc_tester().execute_program_with_timeout(program, test_input)
        
        if predicted_output is not None and not error and not timed_out:
            # Validate output format
            if not isinstance(predicted_output, list) or len(predicted_output) == 0 or not isinstance(predicted_output[0], list):
                raise ValueError(f"Program returned invalid output format for test input {i}")
            
            # Validate cell values
            for row_idx, row in enumerate(predicted_output):
                if not isinstance(row, list):
                    raise ValueError(f"Row {row_idx} is not a list in test output {i}")
                for col_idx, cell in enumerate(row):
                    if isinstance(cell, bool) or not isinstance(cell, int) or not (0 <= cell <= 9):
                        raise ValueError(f"Invalid cell value at [{row_idx}][{col_idx}] in test output {i}: {cell}")
            
            predicted_test_outputs.append(predicted_output)
            # Check if prediction matches ground truth
            correct_test_inputs.append(predicted_output == test_outputs[i])
        else:
            raise ValueError(f"Program failed to run on test input {i}")
    
    # Get reasoning content if available and requested
    reasoning_content = ""
    if args.reasoning and program_data.get('reasoning_content'):
        reasoning_content = program_data.get('reasoning_content', '')
    
    # Create the flat format training example
    training_example = {
        "reasoning": reasoning_content,
        "code": program,
        "correct_train_input": correct_train_inputs,
        "train_input": train_inputs,
        "train_output": train_outputs,
        "predicted_train_output": predicted_train_outputs,
        "correct_test_input": correct_test_inputs,
        "test_input": test_inputs,
        "test_output": test_outputs,
        "predicted_test_output": predicted_test_outputs,
        "task_id": program_data['task_id'],
        "model": program_data.get('model', 'unknown'),
        "generation": program_data.get('generation', 0)
    }
    
    return training_example

def create_and_push_hf_dataset(training_examples: List[Dict], validation_examples: List[Dict], args) -> None:
    """Create and push Hugging Face dataset to the hub"""
    
    # Generate dataset name if not provided
    if args.hf_dataset_name:
        dataset_name = args.hf_dataset_name
    else:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_part = args.dataset or "unknown"
        subset_part = args.subset or "unknown"
        dataset_name = f"synth_{dataset_part}_{subset_part}_{timestamp}"
    
    print(f"Creating Hugging Face dataset: {args.hf_org}/{dataset_name}")
    
    # Create datasets
    if validation_examples:
        # Create train and validation splits
        train_dataset = Dataset.from_list(training_examples)
        val_dataset = Dataset.from_list(validation_examples)
        
        print(f"Created training dataset with {len(training_examples)} examples")
        print(f"Created validation dataset with {len(validation_examples)} examples")
        
        # Push both splits
        train_dataset.push_to_hub(
            f"{args.hf_org}/{dataset_name}",
            split="train",
            private=args.hf_private
        )
        val_dataset.push_to_hub(
            f"{args.hf_org}/{dataset_name}",
            split="validation",
            private=args.hf_private
        )
        
        print(f"Successfully pushed training and validation splits to {args.hf_org}/{dataset_name}")
    else:
        # Create single training split
        train_dataset = Dataset.from_list(training_examples)
        
        print(f"Created training dataset with {len(training_examples)} examples")
        
        # Push training split
        train_dataset.push_to_hub(
            f"{args.hf_org}/{dataset_name}",
            split="train",
            private=args.hf_private
        )
        
        print(f"Successfully pushed training split to {args.hf_org}/{dataset_name}")
    
    # Print dataset URL
    visibility = "private" if args.hf_private else "public"
    print(f"Dataset URL: https://huggingface.co/datasets/{args.hf_org}/{dataset_name} ({visibility})")

def deduplicate_programs_by_task(qualified_programs: List[Dict], args) -> Tuple[List[Dict], Dict]:
    """Deduplicate programs within each task based on test correctness and output similarity"""
    
    # Group programs by task_id
    task_programs = {}
    for prog_info in qualified_programs:
        task_id = prog_info['program_data']['task_id']
        if task_id not in task_programs:
            task_programs[task_id] = []
        task_programs[task_id].append(prog_info)
    
    deduplication_stats = {
        'total_tasks': len(task_programs),
        'test_correct_deduped': 0,
        'output_deduped': 0,
        'total_programs_before': len(qualified_programs),
        'total_programs_after': 0
    }
    
    deduplicated_programs = []
    
    for task_id, programs in task_programs.items():
        # Show detailed debug info for tasks with multiple programs
        show_debug = len(programs) > 1
        if show_debug:
            print(f"  🔍 Deduplicating task {task_id}: {len(programs)} programs")
        
        task_deduped = []
        
        # Step 1: Deduplicate programs that correctly solve the test
        test_correct_programs = []
        test_incorrect_programs = []
        
        for prog_info in programs:
            prog_data = prog_info['program_data']
            task_data = prog_info['task_data']
            program = prog_data['program']
            
            # Check if program correctly solves the test
            if task_data.get('test') and len(task_data['test']) > 0:
                test_input = task_data['test'][0]['input']
                expected_test_output = task_data['test'][0]['output']
                predicted_test_output, error, timed_out = get_arc_tester().execute_program_with_timeout(program, test_input)
                
                if (predicted_test_output is not None and not error and not timed_out and 
                    predicted_test_output == expected_test_output):
                    test_correct_programs.append(prog_info)
                else:
                    test_incorrect_programs.append(prog_info)
            else:
                test_incorrect_programs.append(prog_info)
        
        # For test-correct programs, deduplicate by cleaned code string match
        if len(test_correct_programs) > 0:
            code_signatures = {}
            for prog_info in test_correct_programs:
                program_code = prog_info['program_data']['program']
                
                # Keep first program with this exact code signature
                if program_code not in code_signatures:
                    code_signatures[program_code] = prog_info
                    task_deduped.append(prog_info)
                else:
                    deduplication_stats['test_correct_deduped'] += 1
            
            if show_debug and len(test_correct_programs) > len(code_signatures):
                unique_codes = len(code_signatures)
                print(f"    ✅ Test-correct code dedup: {len(test_correct_programs)} → {unique_codes} (removed {len(test_correct_programs) - unique_codes} duplicates)")
        
        # Step 2: For test-incorrect programs, deduplicate by output similarity
        if test_incorrect_programs:
            output_signatures = {}
            
            for prog_info in test_incorrect_programs:
                prog_data = prog_info['program_data']
                task_data = prog_info['task_data']
                program = prog_data['program']
                
                # Generate output signature by running program on all training inputs
                outputs = []
                for example in task_data['train']:
                    predicted_output, error, timed_out = get_arc_tester().execute_program_with_timeout(program, example['input'])
                    if predicted_output is not None and not error and not timed_out:
                        try:
                            # Validate that predicted_output is a valid 2D list structure with hashable elements
                            if (isinstance(predicted_output, list) and 
                                all(isinstance(row, list) for row in predicted_output) and
                                all(all(isinstance(cell, (int, float, str, bool, type(None))) for cell in row) 
                                    for row in predicted_output if isinstance(row, list))):
                                # Convert to tuple for hashing
                                output_tuple = tuple(tuple(row) for row in predicted_output)
                            else:
                                if args.debug:
                                    print(f"    🚫 Invalid output format for task {task_id}: {type(predicted_output)} - {predicted_output}")
                                output_tuple = ('ERROR', f'Invalid output format: {type(predicted_output)}')
                        except (TypeError, ValueError, AttributeError) as e:
                            if args.debug:
                                print(f"    🚫 Failed to convert output for task {task_id}: {str(e)} - {predicted_output}")
                            # Fallback: create a safe string representation for hashing
                            try:
                                output_tuple = ('ERROR', f'Conversion failed: {str(predicted_output)[:100]}')
                            except:
                                output_tuple = ('ERROR', 'Failed to convert output (unprintable)')
                    else:
                        output_tuple = ('ERROR', str(error) if error else 'TIMEOUT')
                    outputs.append(output_tuple)
                
                signature = tuple(outputs)
                
                # Keep first program with this signature
                if signature not in output_signatures:
                    output_signatures[signature] = prog_info
                    task_deduped.append(prog_info)
                else:
                    deduplication_stats['output_deduped'] += 1
            
            if show_debug and len(test_incorrect_programs) > len([sig for sig in output_signatures]):
                unique_outputs = len(output_signatures)
                print(f"    🔍 Output-based dedup: {len(test_incorrect_programs)} → {unique_outputs} (removed {len(test_incorrect_programs) - unique_outputs} duplicates)")
        
        deduplicated_programs.extend(task_deduped)
    
    deduplication_stats['total_programs_after'] = len(deduplicated_programs)
    
    # Always show deduplication summary
    print(f"\n📊 Deduplication Summary:")
    print(f"  Tasks processed: {deduplication_stats['total_tasks']}")
    print(f"  Programs before: {deduplication_stats['total_programs_before']}")
    print(f"  Programs after: {deduplication_stats['total_programs_after']}")
    print(f"  Test-correct deduped: {deduplication_stats['test_correct_deduped']}")
    print(f"  Output-similarity deduped: {deduplication_stats['output_deduped']}")
    total_deduped = deduplication_stats['test_correct_deduped'] + deduplication_stats['output_deduped']
    if deduplication_stats['total_programs_before'] > 0:
        dedup_pct = (total_deduped / deduplication_stats['total_programs_before']) * 100
        print(f"  Total deduplication: {total_deduped} programs ({dedup_pct:.1f}%)")
    
    return deduplicated_programs, deduplication_stats

def main():
    parser = argparse.ArgumentParser(description="Generate training data from log files")
    parser.add_argument("--limit", type=int, default=None, 
                       help="Limit to the last N log files (sorted by timestamp)")
    parser.add_argument("--output", type=str, 
                       default=f"training_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
                       help="Output JSONL file name")
    parser.add_argument("--validation", action="store_true",
                       help="Create a validation split (10 percent or 32 examples, whichever is smaller)")
    parser.add_argument("--pattern", type=str, default=None,
                       help="Filter log files by pattern (e.g., '20250721_112639' for specific timestamp)")
    parser.add_argument("--model", type=str, action='append', default=None,
                       help="Filter by model name(s). Can be used multiple times or as comma-separated list (e.g., 'google/gemini-2.5-flash,gpt-4.1-mini')")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Filter by dataset (e.g., 'arc-agi-1')")
    parser.add_argument("--subset", type=str, default=None,
                       help="Filter by subset (e.g., 'all_training')")
    parser.add_argument("--date-from", type=str, default=None,
                       help="Filter files from this date onwards (format: YYYYMMDD)")
    parser.add_argument("--date-to", type=str, default=None,
                       help="Filter files up to this date (format: YYYYMMDD)")
    parser.add_argument("--reasoning", action="store_true", default=True,
                       help="Include reasoning content for programs that correctly solve the test output (default: True)")
    parser.add_argument("--no-reasoning", action="store_false", dest="reasoning",
                       help="Disable reasoning content inclusion")
    parser.add_argument("--clean-code", action="store_true",
                       help="Strip comments and clean up code before processing")
    parser.add_argument("--no-dedup", action="store_true",
                       help="Disable deduplication of programs within each task")
    parser.add_argument("--no-transduction-filter", action="store_true",
                       help="Disable filtering of transduction/cheating programs")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with detailed logging for transduction detection")
    parser.add_argument("--hf-dataset-name", type=str, default=None,
                       help="Name for the Hugging Face dataset. If not provided, will use synth_{dataset}_{subset}_DATETIME format")
    parser.add_argument("--hf-org", type=str, default="Trelis",
                       help="Hugging Face organization to push the dataset to (default: Trelis)")
    parser.add_argument("--hf-private", action="store_true",
                       help="Make the Hugging Face dataset private")
    
    args = parser.parse_args()
    
    # Process models list to handle comma-separated values
    if args.model:
        # Flatten the list and split comma-separated values
        models_list = []
        for model_arg in args.model:
            models_list.extend([m.strip() for m in model_arg.split(',')])
        args.model = [m for m in models_list if m]  # Remove empty strings
    
    # Create training_data directory if it doesn't exist
    output_dir = Path("training_data")
    output_dir.mkdir(exist_ok=True)
    
    # Calculate number of workers (total cores - 2, minimum 1)
    max_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"Using {max_workers} worker processes (total cores: {multiprocessing.cpu_count()})")
    
    # Get all log files (exclude summary files)
    # Support both flat logs/*.json and datetime subdirectories logs/*/
    log_files = glob.glob("llm_python/logs/*.json") + glob.glob("llm_python/logs/*/*.json")
    log_files = [f for f in log_files if 'summary' not in f]
    
    # Filter by pattern if provided
    if args.pattern:
        log_files = [f for f in log_files if args.pattern in f]
        print(f"Filtered to {len(log_files)} log files matching pattern '{args.pattern}'")
    
    # Filter by date range if provided
    if args.date_from or args.date_to:
        def extract_date_from_filename(filename):
            # Extract date from filename or subdirectory
            # New structure: logs/20250721_112639/20250721_112639_123456_task_id.json
            # Old structure: logs/20250721_112639_task_id.json
            
            # First try to extract from subdirectory name (more reliable for new structure)
            path_parts = Path(filename).parts
            if len(path_parts) >= 2:  # logs/timestamp/file.json
                subdir = path_parts[-2]  # Get the subdirectory name
                if len(subdir) >= 8 and subdir[:8].isdigit():
                    return subdir[:8]
            
            # Fall back to extracting from filename (for old structure)
            basename = os.path.basename(filename)
            if len(basename) >= 8 and basename[:8].isdigit():
                return basename[:8]
            return None
        
        filtered_files = []
        for log_file in log_files:
            file_date = extract_date_from_filename(log_file)
            if file_date:
                include_file = True
                if args.date_from and file_date < args.date_from:
                    include_file = False
                if args.date_to and file_date > args.date_to:
                    include_file = False
                if include_file:
                    filtered_files.append(log_file)
        
        print(f"Filtered to {len(filtered_files)} log files by date range")
        log_files = filtered_files
    
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
    
    # Filter programs by model, dataset, subset if specified
    if args.model or args.dataset or args.subset:
        print(f"Applying program filters:")
        if args.model:
            print(f"  - Models: {', '.join(args.model)}")
        if args.dataset:
            print(f"  - Dataset: {args.dataset}")
        if args.subset:
            print(f"  - Subset: {args.subset}")
            
        filtered_programs = []
        for program in all_programs:
            include_program = True
            
            if args.model:
                program_model = program.get('model', '').lower()
                if not any(model.lower() == program_model for model in args.model):
                    include_program = False
            if args.dataset and program.get('dataset', '').lower() != args.dataset.lower():
                include_program = False
            if args.subset and program.get('subset', '').lower() != args.subset.lower():
                include_program = False
                
            if include_program:
                filtered_programs.append(program)
        
        print(f"Filtered to {len(filtered_programs)} programs by model/dataset/subset criteria")
        all_programs = filtered_programs
    
    # Clean code if requested
    if args.clean_code:
        print(f"Cleaning code (stripping comments and whitespace)...")
        cleaned_programs = []
        cleaning_stats = {'original_chars': 0, 'cleaned_chars': 0, 'failed_cleans': 0}
        
        for program_data in all_programs:
            original_program = program_data['program']
            cleaned_program = strip_comments_aggressive(original_program)
            
            # Test that cleaned code still compiles
            try:
                compile(cleaned_program, '<cleaned>', 'exec')
                # Update the program with cleaned version
                program_data['program'] = cleaned_program
                program_data['original_program'] = original_program  # Keep original for reference
                cleaned_programs.append(program_data)
                
                # Track stats
                cleaning_stats['original_chars'] += len(original_program)
                cleaning_stats['cleaned_chars'] += len(cleaned_program)
                
            except (SyntaxError, Exception) as e:
                # If cleaning breaks the code, keep the original
                print(f"  Warning: Code cleaning failed for task {program_data.get('task_id', 'unknown')}: {e}")
                cleaned_programs.append(program_data)  # Keep original
                cleaning_stats['failed_cleans'] += 1
        
        all_programs = cleaned_programs
        
        # Report cleaning results
        if cleaning_stats['original_chars'] > 0:
            reduction_pct = (1 - cleaning_stats['cleaned_chars'] / cleaning_stats['original_chars']) * 100
            print(f"  Code cleaning results:")
            print(f"    - {len(all_programs)} programs processed")
            print(f"    - {cleaning_stats['failed_cleans']} cleaning failures (kept original)")
            print(f"    - {cleaning_stats['original_chars']:,} → {cleaning_stats['cleaned_chars']:,} characters")
            print(f"    - {reduction_pct:.1f}% size reduction achieved")
    
    # Filter programs to only those that qualify (in parallel)
    qualified_programs = []
    validated_count = 0
    transduction_rejected = 0
    transduction_stats = {}  # task_id -> list of rejection reasons
    
    validation_message = "Validating programs"
    if not args.no_transduction_filter:
        validation_message += " and filtering transduction/cheating"
    validation_message += " in parallel..."
    print(validation_message)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all validation jobs
        future_to_prog = {executor.submit(validate_single_program, prog_data, args): prog_data 
                         for prog_data in all_programs}
        
        # Collect results as they complete
        for future in as_completed(future_to_prog):
            prog_data = future_to_prog[future]
            validated_count += 1
            
            # Progress reporting every 50 programs
            if validated_count % 50 == 0:
                print(f"  Validated {validated_count}/{len(all_programs)} programs...")
            
            try:
                result = future.result()
                if result is not None:
                    qualified_programs.append(result)
                elif 'transduction_reason' in prog_data:
                    # Track transduction rejections
                    transduction_rejected += 1
                    task_id = prog_data['task_id']
                    reason = prog_data['transduction_reason']
                    
                    if task_id not in transduction_stats:
                        transduction_stats[task_id] = []
                    transduction_stats[task_id].append(reason)
                    
                    if args.debug:
                        print(f"  🚫 Rejected transduction in task {task_id}: {reason}")
                        
            except Exception as e:
                if validated_count % 500 == 0:  # Only print errors occasionally
                    print(f"  Warning: Error validating program: {e}")
    
    print(f"\nQualified programs: {len(qualified_programs)}")
    
    # Report transduction filtering results
    if not args.no_transduction_filter:
        print(f"🛡️  Transduction/Cheating Filter Results:")
        print(f"  Programs rejected for cheating: {transduction_rejected}")
        if transduction_rejected > 0:
            print(f"  Tasks with cheating programs: {len(transduction_stats)}")
            if args.debug:
                print(f"  📋 Detailed breakdown by task:")
                for task_id, reasons in transduction_stats.items():
                    print(f"    Task {task_id}: {len(reasons)} programs rejected")
                    for reason in reasons:
                        print(f"      - {reason}")
            else:
                # Show summary without debug details
                reason_counts = {}
                for reasons in transduction_stats.values():
                    for reason in reasons:
                        # Categorize reasons
                        if "exceeds 200 characters" in reason:
                            category = "Long lines (>200 chars)"
                        elif "hardcoded in program" in reason:
                            category = "Hardcoded outputs"
                        else:
                            category = "Other"
                        reason_counts[category] = reason_counts.get(category, 0) + 1
                
                print(f"  📊 Rejection categories:")
                for category, count in reason_counts.items():
                    print(f"    {category}: {count}")
    else:
        print(f"⚠️  Transduction filtering disabled by --no-transduction-filter flag")
    
    # Deduplicate programs within each task (unless disabled)
    if not args.no_dedup:
        print(f"Deduplicating programs within each task...")
        qualified_programs, dedup_stats = deduplicate_programs_by_task(qualified_programs, args)
        print(f"After deduplication: {len(qualified_programs)} programs")
    else:
        print(f"Deduplication disabled by --no-dedup flag")
    
    # Generate training examples (in parallel)
    training_examples = []
    programs_with_at_least_one_correct = 0
    programs_with_all_correct = 0
    programs_with_reasoning = 0
    programs_with_test_correct = 0
    programs_with_all_training_and_test_correct = 0
    validation_mismatches = 0
    invalid_output_programs = 0
    processed_examples = 0
    
    print(f"Generating training examples from {len(qualified_programs)} qualified programs in parallel...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all training example generation jobs
        future_to_info = {executor.submit(create_training_example, prog_info['program_data'], prog_info['task_data'], args): prog_info 
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
                
                # Track test correctness before removing metadata
                test_correct = training_example.get('_originally_test_correct', False)
                if test_correct:
                    programs_with_test_correct += 1
                
                # Remove metadata before adding to final dataset
                if '_originally_test_correct' in training_example:
                    del training_example['_originally_test_correct']
                
                training_examples.append(training_example)
                
                # Check if reasoning was included in this example
                assistant_content = training_example['messages'][2]['content']
                if '<think>' in assistant_content:
                    programs_with_reasoning += 1
                
                # Track programs with at least one originally correct answer
                prog_data = prog_info['program_data']
                solved_count = prog_data.get('validated_solved_count', 0)
                total_examples = prog_data.get('validated_total_examples', 0)
                
                if solved_count > 0:
                    programs_with_at_least_one_correct += 1
                
                all_training_correct = (solved_count == total_examples and total_examples > 0)
                if all_training_correct:
                    programs_with_all_correct += 1
                
                # Track programs with both all training correct AND test correct
                if all_training_correct and test_correct:
                    programs_with_all_training_and_test_correct += 1
                
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
        pct_all_correct = (programs_with_all_correct / len(training_examples)) * 100
        pct_test_correct = (programs_with_test_correct / len(training_examples)) * 100
        pct_all_training_and_test = (programs_with_all_training_and_test_correct / len(training_examples)) * 100
        pct_with_reasoning = (programs_with_reasoning / len(training_examples)) * 100
        print(f"Programs with at least one originally correct answer: {programs_with_at_least_one_correct}/{len(training_examples)} ({pct_with_correct:.1f}%)")
        print(f"Programs with all training examples correct: {programs_with_all_correct}/{len(training_examples)} ({pct_all_correct:.1f}%)")
        print(f"Programs that originally solved the test case: {programs_with_test_correct}/{len(training_examples)} ({pct_test_correct:.1f}%)")
        print(f"Programs with all training AND test correct: {programs_with_all_training_and_test_correct}/{len(training_examples)} ({pct_all_training_and_test:.1f}%)")
        if args.reasoning:
            print(f"Programs with reasoning content included: {programs_with_reasoning}/{len(training_examples)} ({pct_with_reasoning:.1f}%)")
    else:
        print(f"Programs with at least one originally correct answer: 0/0 (0.0%)")
        print(f"Programs with all training examples correct: 0/0 (0.0%)")
        print(f"Programs that originally solved the test case: 0/0 (0.0%)")
        print(f"Programs with all training AND test correct: 0/0 (0.0%)")
        if args.reasoning:
            print(f"Programs with reasoning content included: 0/0 (0.0%)")
    
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
        
        # Group examples by task_id and correctness status
        task_to_examples = {}
        task_correctness = {}  # task_id -> has_at_least_one_correct
        
        for i, example in enumerate(training_examples):
            # Extract task_id and correctness from qualified_programs
            task_id = qualified_programs[i]['program_data']['task_id']
            solved_count = qualified_programs[i]['program_data'].get('validated_solved_count', 0)
            has_correct = solved_count > 0
            
            if task_id not in task_to_examples:
                task_to_examples[task_id] = []
                task_correctness[task_id] = has_correct
            task_to_examples[task_id].append(example)
        
        # Separate tasks by correctness status
        correct_tasks = [tid for tid, has_correct in task_correctness.items() if has_correct]
        incorrect_tasks = [tid for tid, has_correct in task_correctness.items() if not has_correct]
        
        random.shuffle(correct_tasks)
        random.shuffle(incorrect_tasks)
        
        print(f"  Task breakdown: {len(correct_tasks)} with correct examples, {len(incorrect_tasks)} with no correct examples")
        
        # Balance the dataset first by dropping excess tasks from the larger group
        min_group_size = min(len(correct_tasks), len(incorrect_tasks))
        
        dropped_correct = 0
        dropped_incorrect = 0
        
        if len(correct_tasks) > min_group_size:
            dropped_correct = len(correct_tasks) - min_group_size
            correct_tasks = correct_tasks[:min_group_size]
            print(f"  Balanced dataset: dropped {dropped_correct} excess correct-example tasks")
            
        if len(incorrect_tasks) > min_group_size:
            dropped_incorrect = len(incorrect_tasks) - min_group_size
            incorrect_tasks = incorrect_tasks[:min_group_size]
            print(f"  Balanced dataset: dropped {dropped_incorrect} excess no-correct-example tasks")
        
        print(f"  Balanced breakdown: {len(correct_tasks)} with correct examples, {len(incorrect_tasks)} with no correct examples")
        
        # Filter training examples to only include balanced tasks
        balanced_task_ids = set(correct_tasks + incorrect_tasks)
        balanced_training_examples = []
        balanced_qualified_programs = []
        
        for i, example in enumerate(training_examples):
            task_id = qualified_programs[i]['program_data']['task_id']
            if task_id in balanced_task_ids:
                balanced_training_examples.append(example)
                balanced_qualified_programs.append(qualified_programs[i])
        
        print(f"  Filtered to {len(balanced_training_examples)} examples from balanced tasks")
        
        # Update our working variables to use balanced data
        training_examples = balanced_training_examples
        qualified_programs = balanced_qualified_programs
        
        # Rebuild task_to_examples with balanced data
        task_to_examples = {}
        for i, example in enumerate(training_examples):
            task_id = qualified_programs[i]['program_data']['task_id']
            if task_id not in task_to_examples:
                task_to_examples[task_id] = []
            task_to_examples[task_id].append(example)
        
        # Calculate target validation size
        target_validation_size = min(int(len(training_examples) * 0.1), 32)
        
        # Now do a simple 50/50 split from the balanced groups
        target_val_correct_tasks = min(len(correct_tasks), target_validation_size // 2)
        target_val_incorrect_tasks = min(len(incorrect_tasks), target_validation_size - target_val_correct_tasks)
        
        print(f"  Target validation tasks: {target_val_correct_tasks} correct, {target_val_incorrect_tasks} incorrect")
        
        # Helper function to select tasks for validation from a group
        def select_validation_tasks_from_group(task_group, target_size):
            validation_tasks = []
            validation_examples = []
            
            for task_id in task_group:
                task_examples = task_to_examples[task_id]
                if len(validation_examples) + len(task_examples) <= target_size:
                    validation_tasks.append(task_id)
                    validation_examples.extend(task_examples)
                elif len(validation_examples) == 0:
                    # Include at least one task even if it exceeds target
                    validation_tasks.append(task_id)
                    validation_examples.extend(task_examples)
                    break
            
            return validation_tasks, validation_examples
        
        # Select validation tasks from each group
        val_correct_tasks, val_correct_examples = select_validation_tasks_from_group(correct_tasks, target_val_correct_tasks)
        val_incorrect_tasks, val_incorrect_examples = select_validation_tasks_from_group(incorrect_tasks, target_val_incorrect_tasks)
        
        # Combine validation sets
        validation_task_ids = val_correct_tasks + val_incorrect_tasks
        validation_examples = val_correct_examples + val_incorrect_examples
        
        # All remaining tasks go to training
        train_examples = []
        train_task_ids = []
        
        for task_id in task_to_examples.keys():
            if task_id not in validation_task_ids:
                train_task_ids.append(task_id)
                train_examples.extend(task_to_examples[task_id])
        
        # Calculate balance statistics
        val_correct_count = len(val_correct_examples)
        val_incorrect_count = len(val_incorrect_examples)
        train_correct_tasks_count = len([tid for tid in train_task_ids if task_correctness[tid]])
        train_incorrect_tasks_count = len([tid for tid in train_task_ids if not task_correctness[tid]])
        
        print(f"  Validation balance: {val_correct_count}/{len(validation_examples)} ({val_correct_count/len(validation_examples)*100:.1f}%) from tasks with correct examples")
        print(f"  Training balance: {train_correct_tasks_count}/{len(train_task_ids)} ({train_correct_tasks_count/len(train_task_ids)*100:.1f}%) tasks with correct examples")
        print(f"Validation tasks: {sorted(validation_task_ids)}")
        
        # Create and push Hugging Face dataset with train/validation splits
        create_and_push_hf_dataset(train_examples, validation_examples, args)
        
    else:
        # Create and push Hugging Face dataset with only training data
        create_and_push_hf_dataset(training_examples, [], args)
    
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