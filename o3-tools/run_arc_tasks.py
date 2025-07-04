#!/usr/bin/env python3

import os
import json
import argparse
import datetime
import requests
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from task_loader import TaskLoader
from scoring import GridScorer, ProgramExecutor

load_dotenv()

class ARCTaskRunner:
    """Run ARC tasks using the OpenAI Responses API (single-shot with tool execution)"""
    
    def __init__(self, model: str = "gpt-4.1-nano", use_tools: bool = False, max_tool_calls: int = 64, reasoning_effort: str = "medium", max_workers: int = 1, rate_limit_delay: float = 0.0):
        self.model = model
        self.use_tools = use_tools
        self.max_tool_calls = max_tool_calls
        self.reasoning_effort = reasoning_effort
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.task_loader = TaskLoader()
        self.scorer = GridScorer()
        self.executor = ProgramExecutor(timeout=0.1)
        
        # Create logs directory
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Thread-safe cost and token tracking
        self._cost_lock = threading.Lock()
        self.total_cost = 0.0
        self.total_tokens = 0
        
        # Thread-safe progress tracking
        self._progress_lock = threading.Lock()
        self.completed_tasks = 0
        
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
    
    def _update_costs(self, cost: float, tokens: int):
        """Thread-safe method to update total costs and tokens"""
        with self._cost_lock:
            self.total_cost += cost
            self.total_tokens += tokens
    
    def _update_progress(self, total_tasks: int, task_id: str = None, success: bool = True):
        """Thread-safe method to update and print progress"""
        with self._progress_lock:
            self.completed_tasks += 1
            if self.max_workers > 1:  # Only show progress for parallel execution
                progress_pct = (self.completed_tasks / total_tasks) * 100
                status = "✅ COMPLETED" if success else "❌ FAILED"
                task_info = f" ({task_id})" if task_id else ""
                print(f"Progress: {self.completed_tasks}/{total_tasks} tasks processed ({progress_pct:.1f}%) - {status}{task_info}")
    
    def get_model_pricing(self, model: str) -> tuple[float, float]:
        """Get input and output pricing rates for a model in $/1M tokens"""
        model_lower = model.lower()
        
        # Reasoning models
        if model_lower.startswith('o3-pro'):
            return (20.00, 80.00)
        elif model_lower.startswith('o3-deep-research'):
            return (10.00, 40.00)
        elif model_lower.startswith('o3-mini'):
            return (1.10, 4.40)
        elif model_lower.startswith('o3'):
            return (2.00, 8.00)
        elif model_lower.startswith('o4-mini-deep-research'):
            return (2.00, 8.00)
        elif model_lower.startswith('o4-mini'):
            return (1.10, 4.40)
        elif model_lower.startswith('o1-pro'):
            return (150.00, 600.00)
        elif model_lower.startswith('o1-mini'):
            return (1.10, 4.40)
        elif model_lower.startswith('o1'):
            return (15.00, 60.00)
        
        # GPT-4 models
        elif model_lower.startswith('gpt-4.1-nano'):
            return (0.10, 0.40)
        elif model_lower.startswith('gpt-4.1-mini'):
            return (0.40, 1.60)
        elif model_lower.startswith('gpt-4.1'):
            return (2.00, 8.00)
        elif model_lower.startswith('gpt-4.5-preview'):
            return (75.00, 150.00)
        elif model_lower.startswith('gpt-4o-mini-realtime-preview'):
            return (0.60, 2.40)
        elif model_lower.startswith('gpt-4o-mini'):
            return (0.15, 0.60)
        elif model_lower.startswith('gpt-4o-realtime-preview'):
            return (5.00, 20.00)
        elif model_lower.startswith('gpt-4o-search-preview'):
            return (2.50, 10.00)
        elif model_lower.startswith('gpt-4o'):
            return (2.50, 10.00)
        
        # Other models
        elif model_lower.startswith('codex-mini'):
            return (1.50, 6.00)
        elif model_lower.startswith('computer-use-preview'):
            return (3.00, 12.00)
        elif model_lower.startswith('gpt-image-1'):
            return (5.00, 0.00)  # No output pricing for image model
        
        # Default fallback (gpt-4o-mini pricing)
        else:
            return (0.15, 0.60)
    
    def create_prompt(self, task_data: Dict) -> str:
        """Create a prompt for the model to solve an ARC task"""
        # Include test input in the prompt for context
        task_str = self.task_loader.format_task_for_prompt(task_data, include_test=True)
        
        # Different instructions based on tools availability
        if self.use_tools:
            tools_instruction = """

IMPORTANT: You have access to a live Python code interpreter. You MUST:
1. First complete **Step 1: Pattern Analysis** by describing the pattern you observe
2. Use the code interpreter to analyze and understand the training examples
3. Develop your transform function iteratively, testing it on EVERY training example
4. Keep refining until your function correctly transforms ALL training examples
5. Once you find a working solution, you MUST end your response with the "Final answer:" section

Use the code interpreter to verify your solution works, but ALWAYS conclude with:

Final answer:
```python
def transform(grid):
    # Your verified transformation logic here
    return transformed_grid
```"""
        else:
            tools_instruction = ""
        
        prompt = f"""You are solving an ARC (Abstraction and Reasoning Corpus) task. 
I will show you training examples with input and output grids, plus a test input grid. Your task is to:

1. **Analyze the training examples** to discover the transformation pattern that maps each input grid to its corresponding output grid
2. **Write a Python program** that implements this transformation pattern  
3. **DO NOT predict or generate the test output** - your job is only to write the transformation program
4. **Ensure your program generalizes** - it will be applied to the test input grid (which may differ in size or complexity from training examples) after you provide it

The test input is shown for context so you understand what type of grid your program will eventually process. Focus on learning the pattern from training examples and writing robust code that can handle the test case.

{task_str}

Please analyze the pattern in the training examples step by step. First, provide a brief summary of your reasoning approach and what pattern you discovered. Then write a Python function that performs this transformation.{tools_instruction}

**Step 1: Pattern Analysis**
Briefly describe:
- What pattern you see across the training examples
- How the input grids are transformed to create the output grids
- Any key insights about the transformation rule

**Step 2: Implementation**
Your function should work correctly on all the training examples shown above and be robust enough to handle the test input.

Final answer:
```python
def transform(grid):
    # Your transformation logic here
    return transformed_grid
```

Requirements:
- The function takes a 2D list (grid) where grid[row][col] gives the value at that position
- Values are integers from 0-9
- Return a new grid (2D list) with the transformation applied
- You can use numpy if needed - just add 'import numpy as np' at the start of your function
- Your function MUST produce the correct output for all training examples
- Your function must be robust enough to work on the test input grid
"""
        
        return prompt
    
    def call_responses_api(self, messages: List[Dict]) -> Dict:
        """Call the OpenAI Responses API with retry logic for prompt violations"""
        data = {
            "model": self.model,
            "input": messages
        }
        
        # Only add reasoning effort for reasoning models
        model_lower = self.model.lower()
        if (model_lower.startswith(('o3', 'o4', 'o1'))):
            data["reasoning"] = {"effort": self.reasoning_effort}
        
        if self.use_tools:
            data["tools"] = [{"type": "code_interpreter", "container": {"type": "auto"}}]
            data["include"] = ["code_interpreter_call.outputs"]
            data["max_tool_calls"] = self.max_tool_calls

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    'https://api.openai.com/v1/responses',
                    headers=self.headers,
                    json=data
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    error_data = response.json() if response.content else {}
                    
                    # Check for rate limit error (429)
                    if response.status_code == 429:
                        error_msg = error_data.get('error', {}).get('message', '')
                        print(f"[WARN] Rate limit hit (attempt {attempt}/{max_retries}): {error_msg}")
                        
                        if attempt < max_retries:
                            # Extract delay from error message if available
                            import re
                            delay_match = re.search(r'Please try again in (\d+)ms', error_msg)
                            if delay_match:
                                delay_ms = int(delay_match.group(1))
                                delay_seconds = delay_ms / 1000.0 + 0.5  # Add 500ms buffer
                                print(f"[WARN] Waiting {delay_seconds:.1f}s before retry...")
                                time.sleep(delay_seconds)
                            else:
                                # Exponential backoff: 2, 4, 8 seconds
                                delay = 2 ** attempt
                                print(f"[WARN] Using exponential backoff: {delay}s")
                                time.sleep(delay)
                            continue
                    
                    # Check for prompt violation error
                    elif response.status_code == 400 and error_data.get('error', {}).get('code') == 'invalid_prompt':
                        print(f"[WARN] Prompt violation detected (attempt {attempt}/{max_retries}). Retrying...")
                        if attempt < max_retries:
                            time.sleep(2)
                            continue
                    
                    raise Exception(f"API error {response.status_code}: {error_data}")
            except Exception as e:
                if attempt < max_retries:
                    print(f"[WARN] API call failed (attempt {attempt}/{max_retries}): {e}. Retrying...")
                    # Exponential backoff for general errors
                    delay = 2 ** attempt
                    time.sleep(delay)
                    continue
                else:
                    raise Exception(f"API call failed after {max_retries} attempts: {e}")
    
    
    def extract_code_from_response(self, response_data: Dict) -> str:
        """Extract Python code from the Responses API result using simple regex"""
        import re
        
        # Get the full text from response
        full_text = ""
        
        # Extract from Responses API structure
        for output_item in response_data.get('output', []):
            if output_item.get('type') == 'message':
                content_items = output_item.get('content', [])
                for content_item in content_items:
                    if content_item.get('type') == 'output_text':
                        full_text += content_item.get('text', '') + "\n"
        
        # Fallback: check legacy format if no text found
        if not full_text.strip():
            for choice in response_data.get('choices', []):
                message = choice.get('message', {})
                full_text += message.get('content', '') + "\n"
        
        # First priority: look for code after "Final answer:"
        final_answer_match = re.search(r'Final answer:\s*```python\s*\n(.*?)\n```', full_text, re.DOTALL | re.IGNORECASE)
        if final_answer_match:
            return final_answer_match.group(1).strip()
        
        # Second priority: last ```python block 
        python_blocks = re.findall(r'```python\s*\n(.*?)\n```', full_text, re.DOTALL)
        if python_blocks:
            return python_blocks[-1].strip()
        
        # Third priority: any ``` block with def transform
        code_blocks = re.findall(r'```\s*\n(.*?)\n```', full_text, re.DOTALL)
        for block in reversed(code_blocks):  # Check from last to first
            if 'def transform' in block:
                return block.strip()
        
        # Last resort: extract def transform function without code blocks
        transform_match = re.search(r'(def transform.*?)(?=\n\S|\n*$)', full_text, re.DOTALL)
        if transform_match:
            return transform_match.group(1).strip()
        
        return ""
    
    def count_tool_calls(self, response_data: Dict) -> int:
        """Count the number of tool calls made in the response"""
        count = 0
        
        # For built-in tools in Responses API, count items in output array by type
        for output_item in response_data.get('output', []):
            output_type = output_item.get('type', '')
            
            # Built-in tools show up as their specific type in the output array
            if output_type == 'code_interpreter_call':
                count += 1
            elif output_type == 'web_search_preview_call':
                count += 1
            elif output_type == 'image_generation_call':
                count += 1
            elif output_type == 'file_search_call':
                count += 1
            # Add other built-in tool types as needed
            
            # For custom function tools, they appear in message.tool_calls
            elif output_type == 'message':
                content_items = output_item.get('content', [])
                for content_item in content_items:
                    if content_item.get('type') == 'tool_call':
                        count += 1
        
        return count
    
    def run_task(self, task_id: str, task_data: Dict, total_tasks: int = 1) -> Dict:
        """Run a single ARC task using Responses API (single-shot for built-in tools)"""
        if self.max_workers == 1:  # Only print for sequential execution
            print(f"\nProcessing task: {task_id}")
        
        # Apply rate limiting if configured
        if self.rate_limit_delay > 0:
            time.sleep(self.rate_limit_delay)
        
        # Create initial prompt
        prompt = self.create_prompt(task_data)
        
        # Prepare initial messages
        messages = [
            {"role": "system", "content": "You are an expert at solving abstract reasoning puzzles. Write clean, efficient Python code."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Single API call - built-in tools execute internally at OpenAI
            print(f"  Making single Responses API call...")
            response_data = self.call_responses_api(messages)
            
            # Extract usage data from response
            usage = response_data.get('usage', {})
            total_tokens = usage.get('total_tokens', 0)
            
            # Calculate cost based on model - with fallback calculation
            input_rate, output_rate = self.get_model_pricing(self.model)
            
            # Get token counts (Responses API uses input_tokens/output_tokens)
            input_tokens = usage.get('input_tokens', usage.get('prompt_tokens', 0))
            output_tokens = usage.get('output_tokens', usage.get('completion_tokens', 0))
            
            # Calculate request cost with proper input/output breakdown
            if input_tokens > 0 or output_tokens > 0:
                prompt_cost = (input_tokens / 1_000_000) * input_rate
                completion_cost = (output_tokens / 1_000_000) * output_rate
                request_cost = prompt_cost + completion_cost
                
                # Show reasoning token breakdown for reasoning models
                reasoning_tokens = usage.get('output_tokens_details', {}).get('reasoning_tokens', 0)
                if reasoning_tokens > 0:
                    visible_tokens = output_tokens - reasoning_tokens
                    print(f"  💰 Cost: ${request_cost:.6f} (input: {input_tokens} @ ${input_rate}, output: {output_tokens} @ ${output_rate})")
                    print(f"     ↳ Output breakdown: {reasoning_tokens} reasoning + {visible_tokens} visible tokens")
                else:
                    print(f"  💰 Cost: ${request_cost:.6f} (input: {input_tokens} @ ${input_rate}, output: {output_tokens} @ ${output_rate})")
            else:
                # Fallback: estimate 50/50 split if individual token counts missing
                if total_tokens > 0:
                    estimated_input = total_tokens // 2
                    estimated_output = total_tokens - estimated_input
                    prompt_cost = (estimated_input / 1_000_000) * input_rate
                    completion_cost = (estimated_output / 1_000_000) * output_rate
                    request_cost = prompt_cost + completion_cost
                    print(f"  💰 Cost: ${request_cost:.6f} (estimated from {total_tokens} total tokens)")
                else:
                    request_cost = 0.0
                    print(f"  ⚠️  No usage data available - cost calculation failed")
            
            self._update_costs(request_cost, total_tokens)
            
            # Extract code and count tool calls
            program = self.extract_code_from_response(response_data)
            tool_calls_count = self.count_tool_calls(response_data)
            
            if not program:
                print(f"  No code found in response")
                # Count pixels even when no code is generated
                actual_output = task_data['test'][0]['output']
                total_pixels = len(actual_output) * len(actual_output[0]) if actual_output else 0
                
                # Update progress tracking - successful API call but no code
                self._update_progress(total_tasks, task_id, success=True)
                
                return {
                    'task_id': task_id,
                    'model': self.model,
                    'use_tools': self.use_tools,
                    'api_type': 'responses_api',
                    'program': '',
                    'execution_error': 'No code generated',
                    'timed_out': False,
                    'tokens_used': usage.get('total_tokens', 0),
                    'tool_calls_count': tool_calls_count,
                    'request_cost': request_cost,
                    'raw_response': response_data,
                    'score': {
                        'correct': False, 
                        'pixel_accuracy': 0.0, 
                        'total_pixels': total_pixels,
                        'correct_pixels': 0,
                        'error': 'No code generated'
                    },
                    'mdl': None,
                    'actual_output': actual_output,
                    'api_success': True  # API call succeeded, just no code
                }
            
            # Execute program on test input
            test_input = task_data['test'][0]['input']
            predicted_output, error, timed_out = self.executor.execute_program(program, test_input)
            
            results = {
                'task_id': task_id,
                'model': self.model,
                'use_tools': self.use_tools,
                'api_type': 'responses_api',
                'program': program,
                'execution_error': error,
                'timed_out': timed_out,
                'tokens_used': usage.get('total_tokens', 0),
                'tool_calls_count': tool_calls_count,
                'request_cost': request_cost,
                'raw_response': response_data,
                'api_success': True  # API call succeeded
            }
            
            if predicted_output is not None and not timed_out and not error:
                # Score the result
                actual_output = task_data['test'][0]['output']
                score = self.scorer.score_grid(predicted_output, actual_output)
                results['score'] = score
                
                # Calculate residual reduction (pattern learning)
                training_examples = task_data.get('train', [])
                reduction = self.scorer.calculate_residual_reduction(program, training_examples, self.executor)
                results['residual_reduction'] = reduction
                results['mdl'] = reduction
                
                # Print training execution summary
                executed = reduction.get('training_executions', 0)
                correct = reduction.get('training_correct', 0)
                total_training = reduction.get('training_examples_count', 0)
                failed = len(reduction.get('training_errors', []))
                
                if failed > 0:
                    print(f"  ⚠️  Program crashed on {failed}/{total_training} training examples")
                else:
                    print(f"  ✅ Program executed on all {total_training} training examples")
                
                if executed > 0:
                    print(f"  🎯 Training accuracy: {correct}/{executed} ({correct/executed:.1%}) correct outputs")
                
                results['predicted_output'] = predicted_output
                results['actual_output'] = actual_output
                
                if score.get('correct', False):
                    print(f"  ✅ Perfect solution found!")
                else:
                    print(f"  📊 Pixel accuracy: {score.get('pixel_accuracy', 0):.1%}")
            else:
                # Count pixels even when execution fails
                actual_output = task_data['test'][0]['output']
                total_pixels = len(actual_output) * len(actual_output[0]) if actual_output else 0
                results['score'] = {
                    'correct': False,
                    'pixel_accuracy': 0.0,
                    'total_pixels': total_pixels,
                    'correct_pixels': 0,
                    'error': 'Program execution failed'
                }
                results['mdl'] = None
                results['actual_output'] = actual_output
                
                # Still calculate null baseline for comparison
                training_examples = task_data.get('train', [])
                null_residuals = self.scorer.calculate_null_program_training_residuals(training_examples)
                null_residual_bytes = self.scorer.gzip_compress_grid(null_residuals)
                results['null_residual_bytes'] = null_residual_bytes
                
                print(f"  ❌ Execution failed: {error}")
                print(f"  📊 Null baseline: {null_residual_bytes} residual bytes to beat")
            
            # Update progress tracking - successful task
            self._update_progress(total_tasks, task_id, success=True)
            
            return results
            
        except Exception as e:
            # Count pixels even for exceptions
            try:
                actual_output = task_data['test'][0]['output']
                total_pixels = len(actual_output) * len(actual_output[0]) if actual_output else 0
            except:
                total_pixels = 0
            
            # Print explicit error message for console visibility
            print(f"  ❌ TASK FAILED: {task_id}")
            print(f"     Error: {str(e)}")
            
            # Update progress tracking - failed task
            self._update_progress(total_tasks, task_id, success=False)
                
            return {
                'task_id': task_id,
                'error': str(e),
                'score': {
                    'correct': False, 
                    'pixel_accuracy': 0.0,
                    'total_pixels': total_pixels,
                    'correct_pixels': 0
                },
                'api_success': False  # API call failed
            }
    
    def run_subset(self, subset_name: str, dataset: str = "arc-agi-1", limit: Optional[int] = None) -> List[Dict]:
        """Run all tasks in a subset with optional parallelization"""
        tasks = self.task_loader.load_tasks_from_subset(subset_name, dataset)
        
        if limit:
            tasks = tasks[:limit]
        
        total_tasks = len(tasks)
        
        # Print configuration info
        print(f"\nRunning {total_tasks} tasks from {dataset}/{subset_name}")
        print(f"Model: {self.model}")
        print(f"API: Responses API (single-shot)")
        if self.use_tools:
            print("Tools: ENABLED (code interpreter - OpenAI runs code internally, model can iterate)")
        else:
            print("Tools: DISABLED (model outputs final code, we execute it locally)")
        if self.max_workers > 1:
            print(f"Parallelization: ENABLED ({self.max_workers} workers)")
            if self.rate_limit_delay > 0:
                print(f"Rate limiting: {self.rate_limit_delay}s delay between requests")
        else:
            print("Parallelization: DISABLED (sequential execution)")
        print("-" * 50)
        
        # Reset progress counter for this subset
        with self._progress_lock:
            self.completed_tasks = 0
        
        results = []
        
        if self.max_workers == 1:
            # Sequential execution (original behavior)
            for task_id, task_data in tasks:
                result = self.run_task(task_id, task_data, total_tasks)
                results.append(result)
                
                # Save individual result
                self.save_result(result)
        else:
            # Parallel execution
            print(f"Starting parallel execution with {self.max_workers} workers...")
            
            def process_task(task_info):
                task_id, task_data = task_info
                result = self.run_task(task_id, task_data, total_tasks)
                # Save individual result
                self.save_result(result)
                return result
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_task = {executor.submit(process_task, task_info): task_info[0] 
                                for task_info in tasks}
                
                # Collect results as they complete
                for future in as_completed(future_to_task):
                    task_id = future_to_task[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"Task {task_id} failed with error: {e}")
                        # Create a minimal error result
                        error_result = {
                            'task_id': task_id,
                            'error': str(e),
                            'score': {'correct': False, 'pixel_accuracy': 0.0, 'total_pixels': 0, 'correct_pixels': 0}
                        }
                        results.append(error_result)
                        # Still update progress for failed tasks
                        self._update_progress(total_tasks, task_id, success=False)
            
            # Sort results by task_id to maintain consistent order
            results.sort(key=lambda x: x.get('task_id', ''))
            print(f"\nParallel execution completed. All {total_tasks} tasks processed.")
        
        # Save summary
        self.save_summary(results, subset_name, dataset)
        
        return results
    
    def save_result(self, result: Dict):
        """Save individual task result with thread-safe unique filename"""
        # Add microseconds and thread ID to ensure unique filenames in parallel execution
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        thread_id = threading.get_ident()
        filename = f"{timestamp}_{thread_id}_{result['task_id']}.json"
        filepath = self.logs_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
    
    def save_summary(self, results: List[Dict], subset_name: str, dataset: str):
        """Save summary of all results"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate statistics
        total_tasks = len(results)
        correct_tasks = sum(1 for r in results if r.get('score', {}).get('correct', False))
        
        # Separate successful API calls from complete failures
        api_successes = [r for r in results if r.get('api_success', True)]  # Default True for backward compatibility
        api_failures = [r for r in results if not r.get('api_success', True)]
        failed_tasks = len(api_failures)
        successful_api_calls = len(api_successes)
        
        total_pixels = sum(r.get('score', {}).get('total_pixels', 0) for r in results)
        correct_pixels = sum(r.get('score', {}).get('correct_pixels', 0) for r in results)
        
        # Calculate pattern learning statistics (only for successful tasks)
        pattern_learning_scores = [r.get('residual_reduction', {}).get('pattern_learning_score') for r in api_successes if r.get('residual_reduction')]
        avg_pattern_learning = sum(pattern_learning_scores) / len(pattern_learning_scores) if pattern_learning_scores else 0
        
        program_residual_bytes = [r.get('residual_reduction', {}).get('program_residual_bytes', 0) for r in api_successes if r.get('residual_reduction')]
        avg_program_residual_bytes = sum(program_residual_bytes) / len(program_residual_bytes) if program_residual_bytes else 0
        
        null_residual_bytes = [r.get('residual_reduction', {}).get('null_residual_bytes', 0) for r in api_successes if r.get('residual_reduction')]
        avg_null_residual_bytes = sum(null_residual_bytes) / len(null_residual_bytes) if null_residual_bytes else 0
        
        # Calculate training execution and correctness statistics (only for successful tasks)
        training_executions = sum(r.get('residual_reduction', {}).get('training_executions', 0) for r in api_successes if r.get('residual_reduction'))
        training_correct = sum(r.get('residual_reduction', {}).get('training_correct', 0) for r in api_successes if r.get('residual_reduction'))
        total_training_examples = sum(r.get('residual_reduction', {}).get('training_examples_count', 0) for r in api_successes if r.get('residual_reduction'))
        
        training_execution_rate = training_executions / total_training_examples if total_training_examples > 0 else 0
        training_correctness_rate = training_correct / training_executions if training_executions > 0 else 0
        
        # Count how many programs achieved good pattern learning (>50%)
        good_pattern_learners = sum(1 for score in pattern_learning_scores if score > 50)
        excellent_pattern_learners = sum(1 for score in pattern_learning_scores if score > 80)
        
        # Calculate tool usage statistics
        total_tool_calls = sum(r.get('tool_calls_count', 0) for r in results)
        avg_tool_calls = total_tool_calls / total_tasks if total_tasks > 0 else 0
        
        summary = {
            'timestamp': timestamp,
            'dataset': dataset,
            'subset': subset_name,
            'model': self.model,
            'use_tools': self.use_tools,
            'api_type': 'responses_api',
            'total_tasks': total_tasks,
            'successful_api_calls': successful_api_calls,
            'failed_api_calls': failed_tasks,
            'correct_tasks': correct_tasks,
            'task_accuracy': correct_tasks / total_tasks if total_tasks > 0 else 0.0,
            'success_rate': successful_api_calls / total_tasks if total_tasks > 0 else 0.0,
            'total_pixels': total_pixels,
            'correct_pixels': correct_pixels,
            'pixel_accuracy': correct_pixels / total_pixels if total_pixels > 0 else 0.0,
            'avg_pattern_learning_score': avg_pattern_learning,
            'avg_program_residual_bytes': avg_program_residual_bytes,
            'avg_null_residual_bytes': avg_null_residual_bytes,
            'training_execution_rate': training_execution_rate,
            'training_correctness_rate': training_correctness_rate,
            'good_pattern_learners': good_pattern_learners,
            'excellent_pattern_learners': excellent_pattern_learners,
            'total_tool_calls': total_tool_calls,
            'avg_tool_calls': avg_tool_calls,
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost,
            'results': results
        }
        
        filename = f"{timestamp}_summary_{dataset}_{subset_name}.json"
        filepath = self.logs_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*50)
        print("SUMMARY")
        print("="*50)
        print(f"Dataset: {dataset}")
        print(f"Subset: {subset_name}")
        print(f"Model: {self.model}")
        print(f"API: Responses (single-shot)")
        print(f"Tools enabled: {self.use_tools}")
        print(f"Total tasks attempted: {total_tasks}")
        print(f"Successful API calls: {successful_api_calls}/{total_tasks} ({summary['success_rate']:.1%})")
        if failed_tasks > 0:
            print(f"Failed API calls: {failed_tasks}/{total_tasks} ({failed_tasks/total_tasks:.1%}) ❌")
        print(f"Tasks solved correctly: {correct_tasks}/{total_tasks} ({summary['task_accuracy']:.1%})")
        print(f"Pixel accuracy: {correct_pixels}/{total_pixels} ({summary['pixel_accuracy']:.1%})")
        print(f"Average pattern learning: {avg_pattern_learning:.1f}%")
        print(f"Training execution rate: {training_execution_rate:.1%} ({training_executions}/{total_training_examples})")
        print(f"Training correctness rate: {training_correctness_rate:.1%} ({training_correct}/{training_executions})" if training_executions > 0 else "Training correctness rate: N/A")
        print(f"Programs with >50% pattern learning: {good_pattern_learners}/{len(pattern_learning_scores)}")
        print(f"Programs with >80% pattern learning: {excellent_pattern_learners}/{len(pattern_learning_scores)}")
        print(f"Average program residual: {avg_program_residual_bytes:.1f} bytes")
        print(f"Average null baseline: {avg_null_residual_bytes:.1f} bytes")
        print(f"Total tool calls made: {total_tool_calls}")
        print(f"Average tool calls per task: {avg_tool_calls:.1f}")
        print(f"Total tokens used: {self.total_tokens:,}")
        print(f"Total cost: ${self.total_cost:.6f}")
        
        # List failed tasks for debugging
        if failed_tasks > 0:
            print(f"\n❌ FAILED TASKS ({failed_tasks}):")
            for result in api_failures:
                task_id = result.get('task_id', 'unknown')
                error = result.get('error', 'Unknown error')
                print(f"  - {task_id}: {error}")
        
        print(f"\nResults saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Run ARC tasks with OpenAI o3/o4 models")
    parser.add_argument("--dataset", default="arc-agi-1", choices=["arc-agi-1", "arc-agi-2"],
                       help="Dataset to use")
    parser.add_argument("--subset", default="shortest_1",
                       help="Subset name (e.g., shortest_1, shortest_10, shortest_100)")
    parser.add_argument("--model", default="gpt-4.1-mini",
                       help="OpenAI model to use")
    parser.add_argument("--tools", action="store_true",
                       help="Enable code interpreter tools")
    parser.add_argument("--limit", type=int,
                       help="Limit number of tasks to run")
    parser.add_argument("--max_tool_calls", type=int, default=64,
                       help="Maximum number of tool calls allowed for the model (default: 64, only applies if --tools is set)")
    parser.add_argument("--reasoning_effort", type=str, default="medium", choices=["low", "medium", "high"],
                       help="Reasoning effort for the model (default: medium)")
    parser.add_argument("--max_workers", type=int, default=1,
                       help="Maximum number of parallel workers (default: 1)")
    parser.add_argument("--rate_limit_delay", type=float, default=0.0,
                       help="Delay between API calls in seconds (default: 0.0)")
    
    args = parser.parse_args()
    
    # Validate max_workers
    if args.max_workers < 1:
        parser.error("--max_workers must be at least 1")
    if args.max_workers > 30:
        parser.error("--max_workers cannot exceed 30 (OpenAI rate limits)")
    
    # Validate rate_limit_delay
    if args.rate_limit_delay < 0:
        parser.error("--rate_limit_delay cannot be negative")
    
    # Create runner and run tasks
    runner = ARCTaskRunner(model=args.model, use_tools=args.tools, max_tool_calls=args.max_tool_calls, reasoning_effort=args.reasoning_effort, max_workers=args.max_workers, rate_limit_delay=args.rate_limit_delay)
    runner.run_subset(args.subset, args.dataset, args.limit)


if __name__ == "__main__":
    main()