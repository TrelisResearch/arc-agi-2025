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
                ['python3', temp_path],
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
                
                # Extract reasoning content from various possible fields
                reasoning_content = None
                raw_response = first_turn.get('raw_response', {})
                if isinstance(raw_response, dict):
                    # Try top-level fields first (for Gemini and some other models)
                    reasoning_content = (raw_response.get('reasoning') or 
                                       raw_response.get('reasoning_content') or
                                       raw_response.get('thinking'))
                    
                    # If not found, try OpenAI-style nested structure
                    if not reasoning_content and 'choices' in raw_response:
                        choices = raw_response.get('choices', [])
                        if choices and isinstance(choices[0], dict):
                            message = choices[0].get('message', {})
                            reasoning_content = (message.get('reasoning') or 
                                               message.get('reasoning_content') or
                                               message.get('thinking'))
                
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
                # Extract reasoning content from various possible fields
                reasoning_content = None
                
                # First try the attempt's raw_response
                raw_response = attempt.get('raw_response', {})
                if isinstance(raw_response, dict):
                    # Try top-level fields first (for Gemini and some other models)
                    reasoning_content = (raw_response.get('reasoning') or 
                                       raw_response.get('reasoning_content') or
                                       raw_response.get('thinking'))
                    
                    # If not found, try OpenAI-style nested structure
                    if not reasoning_content and 'choices' in raw_response:
                        choices = raw_response.get('choices', [])
                        if choices and isinstance(choices[0], dict):
                            message = choices[0].get('message', {})
                            reasoning_content = (message.get('reasoning') or 
                                               message.get('reasoning_content') or
                                               message.get('thinking'))
                
                # If not found in attempt details, try main log data (for some API responses)
                if not reasoning_content:
                    main_raw_response = log_data.get('raw_response', {})
                    if isinstance(main_raw_response, dict):
                        reasoning_content = (main_raw_response.get('reasoning') or 
                                           main_raw_response.get('reasoning_content') or
                                           main_raw_response.get('thinking'))
                        
                        # If not found, try OpenAI-style nested structure in main response
                        if not reasoning_content and 'choices' in main_raw_response:
                            choices = main_raw_response.get('choices', [])
                            if choices and isinstance(choices[0], dict):
                                message = choices[0].get('message', {})
                                reasoning_content = (message.get('reasoning') or 
                                                   message.get('reasoning_content') or
                                                   message.get('thinking'))
                
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

def is_transduction_cheating(program: str, task_data: Dict, debug: bool = False) -> Tuple[bool, str]:
    """
    Detect if a program is cheating by hardcoding outputs (transduction).
    Returns (is_cheating, reason).
    """
    
    # Check 1: Very long lines (likely hardcoded values)
    lines = program.split('\n')
    for line_num, line in enumerate(lines, 1):
        if len(line) > 200:
            reason = f"Line {line_num} exceeds 200 characters (likely hardcoded)"
            if debug:
                print(f"    🚫 Transduction detected: {reason}")
                print(f"       Line: {line[:100]}...")
            return True, reason
    
    # Check 2: Hardcoded output values in code
    # Determine if task has 1x1 outputs (special case)
    flag_one = any((1, 1) == (len(example["output"]), len(example["output"][0]) if example["output"] else 0) 
                   for example in task_data.get("train", []))
    
    # Collect all outputs (training + test)
    all_outputs = []
    
    # Add training outputs
    for example in task_data.get("train", []):
        if example.get("output"):
            all_outputs.append(example["output"])
    
    # Add test outputs if available
    for test_example in task_data.get("test", []):
        if test_example.get("output"):
            all_outputs.append(test_example["output"])
    
    if not all_outputs:
        return False, ""
    
    # Create string representations of outputs
    if flag_one:
        # For 1x1 outputs, only remove spaces
        def clean_string(s):
            return str(s).replace(' ', '')
    else:
        # For larger outputs, remove spaces and brackets
        def clean_string(s):
            return str(s).replace(' ', '').replace('[', '').replace(']', '')
    
    output_strings = [clean_string(output) for output in all_outputs]
    cleaned_code = clean_string(program)
    
    # Check if any output appears in the code
    for i, output_str in enumerate(output_strings):
        if len(output_str) > 2 and output_str in cleaned_code:  # Only check non-trivial outputs
            reason = f"Output {i+1} hardcoded in program: {output_str[:50]}..."
            if debug:
                print(f"    🚫 Transduction detected: {reason}")
                # Show context around the hardcoded value
                code_idx = cleaned_code.find(output_str)
                context_start = max(0, code_idx - 30)
                context_end = min(len(cleaned_code), code_idx + len(output_str) + 30)
                context = cleaned_code[context_start:context_end]
                print(f"       Code context: ...{context}...")
            return True, reason
    
    return False, ""

def validate_single_program(prog_data: Dict, args) -> Optional[Dict]:
    """Validate a single program and return qualified program data or None"""
    task_id = prog_data['task_id']
    program = prog_data['program']
    
    # Load task data
    task_data = load_task_data(task_id)
    if not task_data:
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
    """Create a training example in JSONL format"""
    # Create modified task data with program-generated outputs
    modified_task_data = task_data.copy()
    modified_task_data['train'] = []
    
    program = program_data['program']
    
    # Check if program originally solved the test correctly
    originally_test_correct = False
    if task_data.get('test') and len(task_data['test']) > 0:
        test_input = task_data['test'][0]['input']
        expected_test_output = task_data['test'][0]['output']
        predicted_test_output, error, timed_out = execute_program(program, test_input)
        
        if (predicted_test_output is not None and not error and not timed_out and 
            predicted_test_output == expected_test_output):
            originally_test_correct = True
    
    # Check if program correctly solves the test output (for reasoning inclusion)
    test_correct = originally_test_correct  # Use the same check for reasoning
    
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
    
    # Critical validation: If program originally solved the test, verify it still does after relabeling
    if originally_test_correct:
        # Re-run the test to make sure relabeling didn't break the solution
        test_input = task_data['test'][0]['input']
        expected_test_output = task_data['test'][0]['output']
        predicted_test_output_after_relabel, error, timed_out = execute_program(program, test_input)
        
        still_test_correct = (predicted_test_output_after_relabel is not None and 
                             not error and not timed_out and 
                             predicted_test_output_after_relabel == expected_test_output)
        
        if not still_test_correct:
            raise ValueError(f"Program originally solved test but fails after relabeling - this suggests execution inconsistency")
    
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
    
    # Create assistant message with program and optionally reasoning
    assistant_content = ""
    
    # Include reasoning if flag is set and test is correct
    if args.reasoning and test_correct and program_data.get('reasoning_content'):
        reasoning_content = program_data['reasoning_content']
        assistant_content = f"<think>\n{reasoning_content}\n</think>\n\n"
    
    assistant_content += f"Final answer:\n```python\n{program_data['program']}\n```"
    
    assistant_message = {
        "role": "assistant",
        "content": assistant_content
    }
    
    # Create the training example
    training_example = {
        "messages": [system_message, user_message, assistant_message]
    }
    
    return training_example

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
                predicted_test_output, error, timed_out = execute_program(program, test_input)
                
                if (predicted_test_output is not None and not error and not timed_out and 
                    predicted_test_output == expected_test_output):
                    test_correct_programs.append(prog_info)
                else:
                    test_incorrect_programs.append(prog_info)
            else:
                test_incorrect_programs.append(prog_info)
        
        # For test-correct programs, keep only the first one (they all solve the test correctly)
        if len(test_correct_programs) > 1:
            deduplication_stats['test_correct_deduped'] += len(test_correct_programs) - 1
            if show_debug:
                print(f"    ✅ Test-correct dedup: {len(test_correct_programs)} → 1 (kept first)")
            task_deduped.append(test_correct_programs[0])  # Keep only the first test-correct program
        elif len(test_correct_programs) == 1:
            task_deduped.append(test_correct_programs[0])
        
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
                    predicted_output, error, timed_out = execute_program(program, example['input'])
                    if predicted_output is not None and not error and not timed_out:
                        # Convert to tuple for hashing
                        output_tuple = tuple(tuple(row) for row in predicted_output)
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
                       help="Create a validation split (10% or 32 examples, whichever is smaller)")
    parser.add_argument("--pattern", type=str, default=None,
                       help="Filter log files by pattern (e.g., '20250721_112639' for specific timestamp)")
    parser.add_argument("--model", type=str, default=None,
                       help="Filter by model name (e.g., 'google/gemini-2.5-flash')")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Filter by dataset (e.g., 'arc-agi-1')")
    parser.add_argument("--subset", type=str, default=None,
                       help="Filter by subset (e.g., 'all_training')")
    parser.add_argument("--date-from", type=str, default=None,
                       help="Filter files from this date onwards (format: YYYYMMDD)")
    parser.add_argument("--date-to", type=str, default=None,
                       help="Filter files up to this date (format: YYYYMMDD)")
    parser.add_argument("--reasoning", action="store_true",
                       help="Include reasoning content for programs that correctly solve the test output")
    parser.add_argument("--clean-code", action="store_true",
                       help="Strip comments and clean up code before processing")
    parser.add_argument("--no-dedup", action="store_true",
                       help="Disable deduplication of programs within each task")
    parser.add_argument("--no-transduction-filter", action="store_true",
                       help="Disable filtering of transduction/cheating programs")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with detailed logging for transduction detection")
    
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
    
    # Filter by pattern if provided
    if args.pattern:
        log_files = [f for f in log_files if args.pattern in f]
        print(f"Filtered to {len(log_files)} log files matching pattern '{args.pattern}'")
    
    # Filter by date range if provided
    if args.date_from or args.date_to:
        def extract_date_from_filename(filename):
            # Extract date from filename like logs/20250721_112639_task_id.json
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
            print(f"  - Model: {args.model}")
        if args.dataset:
            print(f"  - Dataset: {args.dataset}")
        if args.subset:
            print(f"  - Subset: {args.subset}")
            
        filtered_programs = []
        for program in all_programs:
            include_program = True
            
            if args.model and program.get('model', '').lower() != args.model.lower():
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
                
                if solved_count == total_examples and total_examples > 0:
                    programs_with_all_correct += 1
                
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
        pct_with_reasoning = (programs_with_reasoning / len(training_examples)) * 100
        print(f"Programs with at least one originally correct answer: {programs_with_at_least_one_correct}/{len(training_examples)} ({pct_with_correct:.1f}%)")
        print(f"Programs with all training examples correct: {programs_with_all_correct}/{len(training_examples)} ({pct_all_correct:.1f}%)")
        if args.reasoning:
            print(f"Programs with reasoning content included: {programs_with_reasoning}/{len(training_examples)} ({pct_with_reasoning:.1f}%)")
    else:
        print(f"Programs with at least one originally correct answer: 0/0 (0.0%)")
        print(f"Programs with all training examples correct: 0/0 (0.0%)")
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