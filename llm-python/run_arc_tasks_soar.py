#!/usr/bin/env python3

import os
import json
import argparse
import datetime
import time
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

from utils.task_loader import TaskLoader
from utils.scoring import GridScorer, ProgramExecutor

load_dotenv()

def execute_with_timeout(func, *args, timeout=1000, **kwargs):
    """Execute a function with timeout using ThreadPoolExecutor"""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout)
            return result
        except Exception as e:
            # Cancel the future if it's still running
            future.cancel()
            raise e

class ARCTaskRunnerSimple:
    """Simple ARC task runner using direct prompts without feedback"""
    
    def __init__(self, model: str = "gpt-4.1-nano", max_workers: int = 1, rate_limit_delay: float = 0.0, max_attempts: int = 8, run_number: int = 0, base_url: str = None, debug: bool = False, max_tokens: int = None, temperature: float = None):
        self.model = model
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        self.max_attempts = max_attempts
        self.run_number = run_number
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.base_url = base_url
        self.debug = debug
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize OpenAI client with optional base URL
        if base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
            print(f"📝 Using custom endpoint: {base_url}")
        else:
            self.client = OpenAI(api_key=self.api_key)
            print(f"📝 Using OpenAI endpoint")
        
        print(f"📝 Simple mode - direct prompts, no feedback")
        
        self.task_loader = TaskLoader()
        self.scorer = GridScorer()
        self.executor = ProgramExecutor(timeout=0.5)
        
        # Create logs directory
        self.logs_dir = Path("llm-python/logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Thread-safe cost and token tracking
        self._cost_lock = threading.Lock()
        self.total_cost = 0.0
        self.total_tokens = 0
        
        # Thread-safe progress tracking
        self._progress_lock = threading.Lock()
        self.completed_tasks = 0
    
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
        
        # Google models
        elif model_lower.startswith('google/gemini-2.5-flash'):
            return (0.30, 2.50)
        elif model_lower.startswith('google/gemini'):
            return (0.30, 2.50)  # Default for other Gemini models
        
        # Default fallback (gpt-4o-mini pricing)
        else:
            return (0.15, 0.60)
    
    def create_prompt(self, task_data: Dict) -> str:
        """Create the simple, direct prompt for solving ARC tasks"""
        
        # Format the task data into the required format
        task_content = ""
        
        # Add training examples
        for i, example in enumerate(task_data['train'], 1):
            input_grid = example['input']
            output_grid = example['output']
            
            # Get grid shapes
            input_shape = f"{len(input_grid[0])} by {len(input_grid)}"
            output_shape = f"{len(output_grid[0])} by {len(output_grid)}"
            
            # Format grids as numpy-style arrays
            input_str = str(input_grid).replace('[', '[[').replace(']', ']]')
            output_str = str(output_grid).replace('[', '[[').replace(']', ']]')
            
            task_content += f"## Input {i} (grid shape: {input_shape}):\n{input_str}\n"
            task_content += f"## Output {i} (grid shape: {output_shape}):\n{output_str}\n"
        
        # Add test input
        if task_data['test']:
            test_input = task_data['test'][0]['input']
            test_shape = f"{len(test_input[0])} by {len(test_input)}"
            test_str = str(test_input).replace('[', '[[').replace(']', ']]')
            task_content += f"## Test Input 1 (grid shape: {test_shape}):\n{test_str}\n"
        
        # Create the full prompt
        system_content = """You are an AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks by
reasoning and generating Python code."""

        user_content = f"""You are an AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks by
generating Python code.
Your goal is to analyze input-output grid pairs. The outputs were produced by applying a
transformation rule to the inputs. Implement the transformation rules as a Python function.
You should only write the implemented the transformation in code.
You must write code in triple backticks (```python and then ```). You must write a function
called 'transform' which takes a single argument, the input grid as 'list[list[int]]', and
returns the transformed grid (also as 'list[list[int]]').
You should make sure that you implement a version of the transformation which works in general
(at least for all given input-output pairs and test input pairs).
The number in the input grid can be mapped to the following colors: 0:Black; 1:Blue; 2:Red; 3:
Green; 4:Yellow; 5:Grey; 6:Pink; 7:Orange; 8:Purple; 9:Brown
Now, solve the following ARC-AGI task:
# Task to solve:
{task_content}```"""

        return system_content, user_content
    
    def call_chat_completions_api(self, messages: List[Dict]) -> Dict:
        """Call the OpenAI Chat Completions API"""
        try:
            # Prepare the request
            kwargs = {
                "model": self.model,
                "messages": messages
            }
            
            # Add max_tokens if specified by user
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            
            # Add temperature if specified by user
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            
            # Make the API call
            response = self.client.chat.completions.create(**kwargs)
            
            return response
            
        except Exception as e:
            raise Exception(f"API call failed: {e}")
    
    def extract_code_from_response(self, response) -> str:
        """Extract Python code from the Chat Completions API result"""
        import re
        
        # Get the full text from response
        full_text = ""
        
        # Extract from Chat Completions API structure
        if hasattr(response, 'choices') and len(response.choices) > 0:
            message = response.choices[0].message
            if hasattr(message, 'content') and message.content:
                full_text = message.content
        
        # Look for python code blocks
        python_blocks = re.findall(r'```python\s*\n(.*?)\n```', full_text, re.DOTALL)
        if python_blocks:
            return python_blocks[-1].strip()
        
        # Look for any code blocks with def transform
        code_blocks = re.findall(r'```\s*\n(.*?)\n```', full_text, re.DOTALL)
        for block in reversed(code_blocks):  # Check from last to first
            if 'def transform' in block:
                return block.strip()
        
        # Last resort: extract def transform function without code blocks
        transform_match = re.search(r'(def transform.*?)(?=\n\S|\n*$)', full_text, re.DOTALL)
        if transform_match:
            return transform_match.group(1).strip()
        
        return ""
    
    def create_success_result(self, task_id: str, program: str, response, test_score: Dict, total_cost: float, total_tokens: int, attempts_used: int, task_data: Dict, all_responses: List = None, attempt_details: List = None, dataset: str = None, subset: str = None) -> Dict:
        """Create a successful task result"""

        # Convert response to JSON-serializable format
        response_dict = None
        if response:
            try:
                response_dict = {
                    'id': response.id,
                    'model': response.model,
                    'usage': {
                        'prompt_tokens': response.usage.prompt_tokens if response.usage else 0,
                        'completion_tokens': response.usage.completion_tokens if response.usage else 0,
                        'total_tokens': response.usage.total_tokens if response.usage else 0,
                    },
                    'content': response.choices[0].message.content if response.choices else "",
                }
            except Exception as e:
                response_dict = {'error': f'Failed to serialize response: {str(e)}'}

        # Convert all responses to JSON-serializable format
        all_responses_dict = []
        if all_responses:
            for resp in all_responses:
                try:
                    all_responses_dict.append({
                        'id': resp.id,
                        'model': resp.model,
                        'usage': {
                            'prompt_tokens': resp.usage.prompt_tokens if resp.usage else 0,
                            'completion_tokens': resp.usage.completion_tokens if resp.usage else 0,
                            'total_tokens': resp.usage.total_tokens if resp.usage else 0,
                        },
                        'content': resp.choices[0].message.content if resp.choices else "",
                    })
                except Exception as e:
                    all_responses_dict.append({'error': f'Failed to serialize response: {str(e)}'})
        
        result = {
            'task_id': task_id,
            'model': self.model,
            'api_type': 'chat_completions_simple',
            'dataset': dataset,
            'subset': subset,
            'program': program,
            'task_failure_reason': '',
            'timed_out': False,
            'tokens_used': total_tokens,
            'request_cost': total_cost,
            'attempts_used': attempts_used,
            'raw_response': response_dict,
            'score': test_score,
            'predicted_output': test_score.get('predicted_output'),
            'actual_output': test_score.get('actual_output'),
            'api_success': True,
            'simple_attempts_data': {
                'attempt_details': attempt_details or [],
                'total_attempts': attempts_used,
                'all_responses': all_responses_dict,
                'mode': 'simple_attempts'
            }
        }
            
        return result
    
    def create_failure_result(self, task_id: str, program: str, all_responses: List, total_cost: float, total_tokens: int, attempts_used: int, task_data: Dict, error_msg: str, attempt_details: List = None, dataset: str = None, subset: str = None) -> Dict:
        """Create a failed task result"""
        # Get test output for pixel counting
        actual_output = task_data['test'][0]['output']
        total_pixels = len(actual_output) * len(actual_output[0]) if actual_output else 0
        
        # Convert last response to JSON-serializable format
        response_dict = None
        if all_responses:
            try:
                response = all_responses[-1]
                response_dict = {
                    'id': response.id,
                    'model': response.model,
                    'usage': {
                        'prompt_tokens': response.usage.prompt_tokens if response.usage else 0,
                        'completion_tokens': response.usage.completion_tokens if response.usage else 0,
                        'total_tokens': response.usage.total_tokens if response.usage else 0,
                    },
                    'content': response.choices[0].message.content if response.choices else "",
                }
            except Exception as e:
                response_dict = {'error': f'Failed to serialize response: {str(e)}'}

        # Convert all responses to JSON-serializable format
        all_responses_dict = []
        if all_responses:
            for resp in all_responses:
                try:
                    all_responses_dict.append({
                        'id': resp.id,
                        'model': resp.model,
                        'usage': {
                            'prompt_tokens': resp.usage.prompt_tokens if resp.usage else 0,
                            'completion_tokens': resp.usage.completion_tokens if resp.usage else 0,
                            'total_tokens': resp.usage.total_tokens if resp.usage else 0,
                        },
                        'content': resp.choices[0].message.content if resp.choices else "",
                    })
                except Exception as e:
                    all_responses_dict.append({'error': f'Failed to serialize response: {str(e)}'})
        
        result = {
            'task_id': task_id,
            'model': self.model,
            'api_type': 'chat_completions_simple',
            'dataset': dataset,
            'subset': subset,
            'program': program,
            'task_failure_reason': error_msg,
            'timed_out': False,
            'tokens_used': total_tokens,
            'request_cost': total_cost,
            'attempts_used': attempts_used,
            'raw_response': response_dict,
            'score': {
                'correct': False,
                'pixel_accuracy': 0.0,
                'total_pixels': total_pixels,
                'correct_pixels': 0,
                'error': error_msg
            },
            'actual_output': actual_output,
            'api_success': True,
            'simple_attempts_data': {
                'attempt_details': attempt_details or [],
                'total_attempts': attempts_used,
                'all_responses': all_responses_dict,
                'mode': 'simple_attempts'
            }
        }
            
        return result
    
    def create_timeout_failure_result(self, task_id: str, total_cost: float, total_tokens: int, attempts_completed: int, task_data: Dict, all_responses: List = None, attempt_details: List = None, dataset: str = None, subset: str = None) -> Dict:
        """Create a timeout failure result"""
        # Get test output for pixel counting
        actual_output = task_data['test'][0]['output']
        total_pixels = len(actual_output) * len(actual_output[0]) if actual_output else 0
        
        # Convert all responses to JSON-serializable format
        all_responses_dict = []
        if all_responses:
            for resp in all_responses:
                try:
                    all_responses_dict.append({
                        'id': resp.id,
                        'model': resp.model,
                        'usage': {
                            'prompt_tokens': resp.usage.prompt_tokens if resp.usage else 0,
                            'completion_tokens': resp.usage.completion_tokens if resp.usage else 0,
                            'total_tokens': resp.usage.total_tokens if resp.usage else 0,
                        },
                        'content': resp.choices[0].message.content if resp.choices else "",
                    })
                except Exception as e:
                    all_responses_dict.append({'error': f'Failed to serialize response: {str(e)}'})
        
        return {
            'task_id': task_id,
            'model': self.model,
            'api_type': 'chat_completions_simple',
            'dataset': dataset,
            'subset': subset,
            'program': '',
            'task_failure_reason': 'API timeout after retries',
            'timed_out': True,
            'tokens_used': total_tokens,
            'request_cost': total_cost,
            'attempts_used': attempts_completed,
            'raw_response': None,
            'score': {
                'correct': False,
                'pixel_accuracy': 0.0,
                'total_pixels': total_pixels,
                'correct_pixels': 0,
                'error': 'API timeout after retries'
            },
            'actual_output': actual_output,
            'api_success': False,
            'timeout_failure': True,
            'simple_attempts_data': {
                'attempt_details': attempt_details or [],
                'total_attempts': attempts_completed,
                'all_responses': all_responses_dict,
                'mode': 'simple_attempts'
            }
        }

    def run_task(self, task_id: str, task_data: Dict, total_tasks: int = 1, dataset: str = None, subset: str = None) -> Dict:
        """Run a single ARC task using simple independent attempts"""
        if self.debug:
            print(f"🔍 DEBUG TASK: Starting run_task for {task_id}")
        
        try:
            # Validate task_data structure
            if not isinstance(task_data, dict):
                raise ValueError(f"task_data is not a dict, got {type(task_data)}")
            
            if 'train' not in task_data:
                raise ValueError("task_data missing 'train' key")
            
            if 'test' not in task_data:
                raise ValueError("task_data missing 'test' key")
            
            if not isinstance(task_data['train'], list):
                raise ValueError(f"task_data['train'] is not a list, got {type(task_data['train'])}")
            
            if not isinstance(task_data['test'], list):
                raise ValueError(f"task_data['test'] is not a list, got {type(task_data['test'])}")
            
            if len(task_data['test']) == 0:
                raise ValueError("task_data['test'] is empty")
            
            if self.debug:
                print(f"🔍 DEBUG TASK: Task data validation passed for {task_id}")
                print(f"🔍 DEBUG TASK: Train examples: {len(task_data['train'])}, Test examples: {len(task_data['test'])}")
            
            if self.max_workers == 1:  # Only print for sequential execution
                print(f"\nProcessing task: {task_id}")
            
            # Apply rate limiting if configured
            if self.rate_limit_delay > 0:
                time.sleep(self.rate_limit_delay)
            
            return self.run_task_simple_attempts(task_id, task_data, total_tasks, dataset, subset)
            
        except Exception as e:
            if self.debug:
                print(f"🔍 DEBUG TASK: Exception in run_task for {task_id}: {type(e).__name__}: {str(e)}")
                import traceback
                print(f"🔍 DEBUG TASK: Full traceback:")
                traceback.print_exc()
            
            # Create a failure result for the exception
            return {
                'task_id': task_id,
                'model': self.model,
                'api_type': 'chat_completions_simple',
                'program': '',
                'task_failure_reason': f'Task setup failed: {str(e)}',
                'timed_out': False,
                'tokens_used': 0,
                'request_cost': 0.0,
                'attempts_used': 0,
                'raw_response': None,
                'score': {
                    'correct': False,
                    'pixel_accuracy': 0.0,
                    'total_pixels': 0,
                    'correct_pixels': 0,
                    'error': f'Task setup failed: {str(e)}'
                },
                'api_success': False,
            }
    
    def run_task_simple_attempts(self, task_id: str, task_data: Dict, total_tasks: int = 1, dataset: str = None, subset: str = None) -> Dict:
        """Run simple independent attempts without feedback - fresh starts with the same prompt"""
        if self.debug:
            print(f"🔍 DEBUG SIMPLE: Starting simple attempts for task {task_id}")
            print(f"🔍 DEBUG SIMPLE: max_attempts = {self.max_attempts}")
        
        total_cost = 0.0
        total_tokens = 0
        all_responses = []
        attempt_details = []
        
        # Create the prompt once
        system_content, user_content = self.create_prompt(task_data)
        
        try:
            if self.debug:
                print(f"🔍 DEBUG SIMPLE: About to start attempt loop for {task_id}")
            for attempt in range(self.max_attempts):
                if self.debug:
                    print(f"🔍 DEBUG SIMPLE: Starting attempt {attempt + 1}/{self.max_attempts} for {task_id}")
                attempt_start_time = datetime.datetime.now()
                if self.max_workers == 1:  # Only print detailed logs for sequential execution
                    print(f"  🔄 Attempt {attempt + 1}/{self.max_attempts}...")
                
                # Create fresh conversation for each attempt (same prompt each time)
                conversation_history = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ]
                
                # Make API call with timeout and retry logic
                if self.debug:
                    print(f"🔍 DEBUG SIMPLE: About to make API call for {task_id} attempt {attempt + 1}")
                response = None
                api_call_successful = False
                
                for retry_attempt in range(3):  # 3 attempts total (initial + 2 retries)
                    try:
                        if self.debug:
                            if retry_attempt == 0:
                                print(f"🔍 DEBUG SIMPLE: API call (first attempt) for {task_id} attempt {attempt + 1}")
                            else:
                                print(f"🔍 DEBUG SIMPLE: API call (retry {retry_attempt}/2) for {task_id} attempt {attempt + 1}")
                        
                        if self.max_workers == 1 and retry_attempt > 0:
                            print(f"     🔄 Attempt {attempt + 1} retry {retry_attempt}/2...")
                        
                        response = execute_with_timeout(self.call_chat_completions_api, conversation_history, timeout=1000)
                        api_call_successful = True
                        break  # Success!
                        
                    except Exception as e:
                        error_type = type(e).__name__
                        error_msg = str(e)
                        if retry_attempt < 2:  # Can still retry
                            if self.max_workers == 1:
                                print(f"     ⏰ Attempt {attempt + 1} retry {retry_attempt + 1} failed ({error_type}: {error_msg}), retrying in 2s...")
                            if self.debug:
                                print(f"🔍 DEBUG SIMPLE: API retry {retry_attempt + 1}/2 for {task_id} attempt {attempt + 1} - Error: {error_type}: {error_msg}")
                            time.sleep(2)  # Brief backoff
                            continue
                        else:  # All retries exhausted
                            if self.max_workers == 1:
                                print(f"     ❌ Attempt {attempt + 1} failed after 3 API retries: {error_type}: {error_msg}")
                                print(f"     🔍 STOPPING TASK - API failures prevent further attempts")
                            if self.debug:
                                print(f"🔍 DEBUG SIMPLE: All retries exhausted for {task_id} attempt {attempt + 1} - Final error: {error_type}: {error_msg}")
                            # Return timeout failure result
                            self._update_costs(total_cost, total_tokens)
                            return self.create_timeout_failure_result(task_id, total_cost, total_tokens, attempt, task_data, all_responses, attempt_details, dataset, subset)
                
                if not api_call_successful or response is None:
                    # This shouldn't happen, but just in case
                    self._update_costs(total_cost, total_tokens)
                    return self.create_timeout_failure_result(task_id, total_cost, total_tokens, attempt, task_data, all_responses, attempt_details, dataset, subset)
                
                all_responses.append(response)
                
                # Track costs
                usage = response.usage
                input_rate, output_rate = self.get_model_pricing(self.model)
                input_tokens = usage.prompt_tokens if usage else 0
                output_tokens = usage.completion_tokens if usage else 0
                attempt_cost = (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate
                total_cost += attempt_cost
                total_tokens += usage.total_tokens if usage else 0
                
                if self.max_workers == 1:
                    print(f"     💰 Attempt cost: ${attempt_cost:.6f} (input: {input_tokens}, output: {output_tokens})")
                
                # Extract code
                program = self.extract_code_from_response(response)
                
                # Initialize attempt detail
                attempt_detail = {
                    'attempt_number': attempt + 1,
                    'timestamp': attempt_start_time.isoformat(),
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'attempt_cost': attempt_cost,
                    'program_extracted': bool(program),
                    'program': program,
                    'test_result': None,
                    'status': 'in_progress'
                }
                
                if not program:
                    if self.max_workers == 1:
                        print(f"     ❌ No code found in response")
                    
                    attempt_detail['status'] = 'no_code_found'
                    attempt_details.append(attempt_detail)
                    
                    # Continue to next attempt
                    continue
                
                # Test on test input
                test_input = task_data['test'][0]['input']
                test_expected = task_data['test'][0]['output']
                predicted_output, error, timed_out = self.executor.execute_program(program, test_input)
                
                if predicted_output is not None and not error and not timed_out:
                    # Check if test is correct
                    test_score = self.scorer.score_grid(predicted_output, test_expected)
                    attempt_detail['test_result'] = test_score
                    
                    if test_score['correct']:
                        # SUCCESS! 
                        attempt_detail['status'] = 'success'
                        attempt_details.append(attempt_detail)
                        
                        if self.max_workers == 1:
                            print(f"     ✅ Perfect solution found on attempt {attempt + 1}!")
                        
                        # Update costs only (progress handled by parallel executor)
                        self._update_costs(total_cost, total_tokens)
                        
                        # Add actual outputs to test_score for logging
                        test_score['predicted_output'] = predicted_output
                        test_score['actual_output'] = test_expected
                        
                        return self.create_success_result(task_id, program, response, test_score, total_cost, total_tokens, attempt + 1, task_data, all_responses, attempt_details, dataset, subset)
                else:
                    # Execution failed - record the failure and continue to next attempt
                    attempt_detail['test_result'] = {
                        'task_failure_reason': error,
                        'timed_out': timed_out,
                        'predicted_output': predicted_output
                    }
                    attempt_detail['status'] = 'failed_test'
                    attempt_details.append(attempt_detail)
                    
                    if self.max_workers == 1:
                        print(f"     📊 Attempt {attempt + 1} failed test - continuing to next attempt")
                    
                    # Continue to next attempt
                    continue
                
                # If we get here, the test passed but wasn't correct - also continue to next attempt
                attempt_detail['status'] = 'failed_test'
                attempt_details.append(attempt_detail)
                
                if self.max_workers == 1:
                    print(f"     📊 Attempt {attempt + 1} failed test - continuing to next attempt")
            
            # Failed after all attempts
            if self.debug:
                print(f"🔍 DEBUG SIMPLE: Completed all {self.max_attempts} attempts for {task_id}")
                print(f"🔍 DEBUG SIMPLE: Total responses collected: {len(all_responses)}")
                print(f"🔍 DEBUG SIMPLE: Total cost: ${total_cost:.6f}")
            self._update_costs(total_cost, total_tokens)
            
            return self.create_failure_result(task_id, program if 'program' in locals() else "", all_responses, total_cost, total_tokens, self.max_attempts, task_data, "All attempts failed", attempt_details, dataset, subset)
        
        except Exception as e:
            if self.debug:
                print(f"🔍 DEBUG SIMPLE: Exception in simple attempts for {task_id}: {type(e).__name__}: {str(e)}")
                import traceback
                print(f"🔍 DEBUG SIMPLE: Full traceback:")
                traceback.print_exc()
            print(f"     ❌ Simple attempts execution failed: {e}")
            self._update_costs(total_cost, total_tokens)
            
            return self.create_failure_result(task_id, "", all_responses, total_cost, total_tokens, len(attempt_details), task_data, str(e), attempt_details, dataset, subset)

    def run_subset(self, subset_name: str, dataset: str = "arc-agi-1", limit: Optional[int] = None) -> List[Dict]:
        """Run all tasks in a subset with optional parallelization"""
        if self.debug:
            print(f"🔍 DEBUG SUBSET: Loading tasks from {dataset}/{subset_name}")
        
        try:
            tasks = self.task_loader.load_tasks_from_subset(subset_name, dataset)
            if self.debug:
                print(f"🔍 DEBUG SUBSET: Loaded {len(tasks)} tasks successfully")
            
            if limit:
                tasks = tasks[:limit]
                if self.debug:
                    print(f"🔍 DEBUG SUBSET: Limited to {len(tasks)} tasks")
            
            total_tasks = len(tasks)
            if self.debug:
                print(f"🔍 DEBUG SUBSET: Total tasks to process: {total_tasks}")
                
                # Debug first few task IDs
                if tasks:
                    first_few = [task_id for task_id, _ in tasks[:3]]
                    print(f"🔍 DEBUG SUBSET: First few task IDs: {first_few}")
            
        except Exception as e:
            if self.debug:
                print(f"🔍 DEBUG SUBSET: Error loading tasks: {type(e).__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
            return []
        
        # Print configuration info
        print(f"\nRunning {total_tasks} tasks from {dataset}/{subset_name}")
        print(f"Model: {self.model}")
        print(f"API: Simple Chat Completions (max {self.max_attempts} attempts)")
        print("Mode: Simple attempts - fresh prompt each time, no feedback")
        print("Input mode: Text-only")
            
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
                result = self.run_task(task_id, task_data, total_tasks, dataset, subset_name)
                results.append(result)
                
                # Save individual result
                self.save_result(result)
        else:
            # Parallel execution
            print(f"Starting parallel execution with {self.max_workers} workers...")
            
            def process_task(task_info):
                task_id, task_data = task_info
                if self.debug:
                    print(f"🔍 DEBUG PARALLEL: [Thread {threading.get_ident()}] Starting process_task for {task_id}")
                
                try:
                    if self.max_workers > 1:
                        print(f"[{task_id}] Starting...")
                    
                    if self.debug:
                        print(f"🔍 DEBUG PARALLEL: [Thread {threading.get_ident()}] About to call run_task for {task_id}")
                    result = self.run_task(task_id, task_data, total_tasks, dataset, subset_name)
                    if self.debug:
                        print(f"🔍 DEBUG PARALLEL: [Thread {threading.get_ident()}] run_task completed for {task_id}")
                    
                    # Save individual result
                    self.save_result(result)
                    if self.debug:
                        print(f"🔍 DEBUG PARALLEL: [Thread {threading.get_ident()}] Result saved for {task_id}")
                    
                    if self.max_workers > 1:
                        status = "✅ SOLVED" if result.get('score', {}).get('correct', False) else "❌ FAILED"
                        cost = result.get('request_cost', 0.0)
                        attempts = result.get('attempts_used', 1)
                        print(f"[{task_id}] {status} (${cost:.6f}, {attempts} attempts)")
                    
                    return result
                    
                except Exception as e:
                    if self.debug:
                        print(f"🔍 DEBUG PARALLEL: [Thread {threading.get_ident()}] Exception in process_task for {task_id}: {type(e).__name__}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                    
                    # Return a minimal error result
                    return {
                        'task_id': task_id,
                        'model': self.model,
                        'api_type': 'chat_completions_simple',
                        'task_failure_reason': f'process_task failed: {str(e)}',
                        'tokens_used': 0,
                        'request_cost': 0.0,
                        'attempts_used': 0,
                        'score': {'correct': False, 'pixel_accuracy': 0.0, 'total_pixels': 0, 'correct_pixels': 0},
                        'api_success': False,
                    }
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_task = {executor.submit(process_task, task_info): task_info[0] 
                                for task_info in tasks}
                
                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_task):
                    task_id = future_to_task[future]
                    completed += 1
                    try:
                        result = future.result()
                        results.append(result)
                        status = "✅ COMPLETED" if result.get('score', {}).get('correct', False) else "❌ FAILED"
                        attempts = result.get('attempts_used', 1)
                        status_with_attempts = f"{status} ({attempts} attempts)"
                    except Exception as e:
                        print(f"[{task_id}] ❌ SAVE FAILED: {e}")
                        # Create a minimal error result
                        error_result = {
                            'task_id': task_id,
                            'error': str(e),
                            'score': {'correct': False, 'pixel_accuracy': 0.0, 'total_pixels': 0, 'correct_pixels': 0}
                        }
                        results.append(error_result)
                        status_with_attempts = "❌ SAVE FAILED"
                    
                    # Show overall progress
                    progress_pct = (completed / total_tasks) * 100
                    print(f"Progress: {completed}/{total_tasks} tasks processed ({progress_pct:.1f}%) - {status_with_attempts}")
            
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
        
        # Include run number in filename when doing repeated runs
        if self.run_number > 0:
            filename = f"{timestamp}_{thread_id}_{result['task_id']}_simple_run{self.run_number}.json"
        else:
            filename = f"{timestamp}_{thread_id}_{result['task_id']}_simple.json"
        
        filepath = self.logs_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
    
    def save_summary(self, results: List[Dict], subset_name: str, dataset: str):
        """Save summary of all results"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate statistics
        total_tasks = len(results)
        correct_tasks = sum(1 for r in results if r.get('score', {}).get('correct', False))
        
        # Separate successful API calls from complete failures and timeout failures
        api_successes = [r for r in results if r.get('api_success', True)]  # Default True for backward compatibility
        api_failures = [r for r in results if not r.get('api_success', True) and not r.get('timeout_failure', False)]
        timeout_failures = [r for r in results if r.get('timeout_failure', False)]
        failed_tasks = len(api_failures)
        timeout_tasks = len(timeout_failures)
        successful_api_calls = len(api_successes)
        
        total_pixels = sum(r.get('score', {}).get('total_pixels', 0) for r in results)
        correct_pixels = sum(r.get('score', {}).get('correct_pixels', 0) for r in results)
        
        # Calculate attempt usage statistics
        total_attempts_used = sum(r.get('attempts_used', 1) for r in results)  # Default to 1 for single-shot
        avg_attempts_used = total_attempts_used / total_tasks if total_tasks > 0 else 0
        
        summary = {
            'timestamp': timestamp,
            'dataset': dataset,
            'subset': subset_name,
            'model': self.model,
            'api_type': 'chat_completions_simple',
            'run_number': self.run_number,
            'total_tasks': total_tasks,
            'successful_api_calls': successful_api_calls,
            'failed_api_calls': failed_tasks,
            'timeout_failed_tasks': timeout_tasks,
            'correct_tasks': correct_tasks,
            'task_accuracy': correct_tasks / total_tasks if total_tasks > 0 else 0.0,
            'success_rate': successful_api_calls / total_tasks if total_tasks > 0 else 0.0,
            'total_pixels': total_pixels,
            'correct_pixels': correct_pixels,
            'pixel_accuracy': correct_pixels / total_pixels if total_pixels > 0 else 0.0,
            'total_attempts_used': total_attempts_used,
            'avg_attempts_used': avg_attempts_used,
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost,
            'results': results
        }
        
        # Include run number in filename when doing repeated runs
        if self.run_number > 0:
            filename = f"{timestamp}_summary_{dataset}_{subset_name}_simple_run{self.run_number}.json"
        else:
            filename = f"{timestamp}_summary_{dataset}_{subset_name}_simple.json"
        
        filepath = self.logs_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        run_info = f" (Run {self.run_number})" if self.run_number > 0 else ""
        print("\n" + "="*50)
        print(f"SUMMARY{run_info}")
        print("="*50)
        print(f"Dataset: {dataset}")
        print(f"Subset: {subset_name}")
        print(f"Model: {self.model}")
        print(f"API: Simple Chat Completions (max {self.max_attempts} attempts)")
        print(f"Total tasks attempted: {total_tasks}")
        print(f"Successful API calls: {successful_api_calls}/{total_tasks} ({summary['success_rate']:.1%})")
        if failed_tasks > 0:
            print(f"Failed API calls: {failed_tasks}/{total_tasks} ({failed_tasks/total_tasks:.1%}) ❌")
        if timeout_tasks > 0:
            print(f"Timeout failures: {timeout_tasks}/{total_tasks} ({timeout_tasks/total_tasks:.1%}) ⏰")
        print(f"Tasks solved correctly: {correct_tasks}/{total_tasks} ({summary['task_accuracy']:.1%})")
        print(f"Pixel accuracy: {correct_pixels}/{total_pixels} ({summary['pixel_accuracy']:.1%})")
        print(f"Total attempts used: {total_attempts_used}")
        print(f"Average attempts per task: {avg_attempts_used:.1f}")
        print(f"Total tokens used: {self.total_tokens:,}")
        print(f"Total cost: ${self.total_cost:.6f}")
        
        # List failed tasks for debugging
        if failed_tasks > 0:
            print(f"\n❌ FAILED TASKS ({failed_tasks}):")
            for result in api_failures:
                task_id = result.get('task_id', 'unknown')
                error = result.get('error', 'Unknown error')
                print(f"  - {task_id}: {error}")
        
        if timeout_tasks > 0:
            print(f"\n⏰ TIMEOUT FAILURES ({timeout_tasks}):")
            for result in timeout_failures:
                task_id = result.get('task_id', 'unknown')
                attempts_completed = result.get('attempts_used', 0)
                print(f"  - {task_id}: API timeout after {attempts_completed} attempts and 3 retries")
        
        print(f"\nResults saved to: {filepath}")

    def run_repeated_subset(self, subset_name: str, dataset: str = "arc-agi-1", limit: Optional[int] = None, repeat_runs: int = 3) -> List[List[Dict]]:
        """Run the same subset multiple times and calculate aggregate statistics"""
        print(f"\nRunning {repeat_runs} repeated tests of {dataset}/{subset_name}")
        print(f"Model: {self.model}")
        print(f"API: Simple Chat Completions (max {self.max_attempts} attempts)")
        print("Mode: Simple attempts - fresh prompt each time, no feedback")
        print("Input mode: Text-only")
            
        if self.max_workers > 1:
            print(f"Parallelization: ENABLED ({self.max_workers} workers)")
        else:
            print(f"Parallelization: DISABLED (sequential execution)")
        print("="*70)
        
        all_run_results = []
        
        # Run the subset multiple times
        for run_num in range(1, repeat_runs + 1):
            print(f"\n🚀 STARTING RUN {run_num}/{repeat_runs}")
            print("-" * 50)
            
            # Create a new runner for this run to ensure clean state
            runner = ARCTaskRunnerSimple(
                model=self.model,
                max_workers=self.max_workers,
                rate_limit_delay=self.rate_limit_delay,
                max_attempts=self.max_attempts,
                run_number=run_num,
                base_url=self.base_url,
                debug=self.debug,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            # Run the subset
            results = runner.run_subset(subset_name, dataset, limit)
            all_run_results.append(results)
            
            print(f"\n✅ COMPLETED RUN {run_num}/{repeat_runs}")
        
        # Calculate and display aggregate statistics
        self._calculate_and_display_aggregate_stats(all_run_results, subset_name, dataset, repeat_runs)
        
        return all_run_results
    
    def _calculate_and_display_aggregate_stats(self, all_run_results: List[List[Dict]], subset_name: str, dataset: str, repeat_runs: int):
        """Calculate and display mean and standard deviation across multiple runs"""
        
        # Calculate statistics for each run
        run_stats = []
        
        for run_num, results in enumerate(all_run_results, 1):
            # Filter out API failures (timeout failures)
            successful_api_results = [r for r in results if r.get('api_success', True)]
            
            if not successful_api_results:
                # Handle case where all API calls failed
                run_stats.append({
                    'run_number': run_num,
                    'attempted_tasks': 0,
                    'attempt1_solved': 0,
                    'all_attempts_solved': 0,
                    'attempt1_success_rate': 0.0,
                    'all_attempts_success_rate': 0.0
                })
                continue
            
            attempted_tasks = len(successful_api_results)
            
            # Count tasks solved on attempt 1 only
            attempt1_solved = sum(1 for r in successful_api_results 
                                if r.get('score', {}).get('correct', False) and r.get('attempts_used', 1) == 1)
            
            # Count tasks solved by end of all attempts
            all_attempts_solved = sum(1 for r in successful_api_results 
                                    if r.get('score', {}).get('correct', False))
            
            # Calculate success rates
            attempt1_success_rate = attempt1_solved / attempted_tasks if attempted_tasks > 0 else 0.0
            all_attempts_success_rate = all_attempts_solved / attempted_tasks if attempted_tasks > 0 else 0.0
            
            run_stats.append({
                'run_number': run_num,
                'attempted_tasks': attempted_tasks,
                'attempt1_solved': attempt1_solved,
                'all_attempts_solved': all_attempts_solved,
                'attempt1_success_rate': attempt1_success_rate,
                'all_attempts_success_rate': all_attempts_success_rate
            })
        
        # Calculate aggregate statistics
        if run_stats:
            attempt1_rates = [s['attempt1_success_rate'] for s in run_stats]
            all_attempts_rates = [s['all_attempts_success_rate'] for s in run_stats]
            
            attempt1_mean = np.mean(attempt1_rates)
            attempt1_std = np.std(attempt1_rates, ddof=1) if len(attempt1_rates) > 1 else 0.0
            
            all_attempts_mean = np.mean(all_attempts_rates)
            all_attempts_std = np.std(all_attempts_rates, ddof=1) if len(all_attempts_rates) > 1 else 0.0
            
            # Save aggregate summary
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            aggregate_summary = {
                'timestamp': timestamp,
                'dataset': dataset,
                'subset': subset_name,
                'model': self.model,
                'api_type': 'chat_completions_simple',
                'repeat_runs': repeat_runs,
                'run_statistics': run_stats,
                'attempt1_success_rate_mean': attempt1_mean,
                'attempt1_success_rate_std': attempt1_std,
                'all_attempts_success_rate_mean': all_attempts_mean,
                'all_attempts_success_rate_std': all_attempts_std
            }
            
            filename = f"{timestamp}_aggregate_summary_{dataset}_{subset_name}_simple_{repeat_runs}runs.json"
            filepath = self.logs_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(aggregate_summary, f, indent=2)
            
            # Display results
            print("\n" + "="*70)
            print("AGGREGATE STATISTICS ACROSS MULTIPLE RUNS")
            print("="*70)
            print(f"Dataset: {dataset}")
            print(f"Subset: {subset_name}")
            print(f"Model: {self.model}")
            print(f"Number of runs: {repeat_runs}")
            print(f"API failures excluded from analysis: YES")
            print("")
            
            # Individual run results
            print("INDIVIDUAL RUN RESULTS:")
            print("-" * 70)
            print(f"{'Run':<4} {'Attempted':<10} {'Attempt 1 Only':<14} {'All Attempts':<14} {'Attempt 1 Rate':<14} {'All Attempts Rate':<14}")
            print("-" * 70)
            
            for stats in run_stats:
                run_num = stats['run_number']
                attempted = stats['attempted_tasks']
                attempt1_solved = stats['attempt1_solved']
                all_attempts_solved = stats['all_attempts_solved']
                attempt1_rate = stats['attempt1_success_rate']
                all_attempts_rate = stats['all_attempts_success_rate']
                
                print(f"{run_num:<4} {attempted:<10} {attempt1_solved:<14} {all_attempts_solved:<14} {attempt1_rate:<14.1%} {all_attempts_rate:<14.1%}")
            
            print("")
            print("AGGREGATE STATISTICS:")
            print("-" * 70)
            print(f"Attempt 1 Only Success Rate:")
            print(f"  Mean: {attempt1_mean:.1%}")
            print(f"  Std Dev: {attempt1_std:.1%}")
            print(f"  95% CI: [{attempt1_mean - 1.96*attempt1_std:.1%}, {attempt1_mean + 1.96*attempt1_std:.1%}]")
            print("")
            print(f"All Attempts Success Rate:")
            print(f"  Mean: {all_attempts_mean:.1%}")
            print(f"  Std Dev: {all_attempts_std:.1%}")
            print(f"  95% CI: [{all_attempts_mean - 1.96*all_attempts_std:.1%}, {all_attempts_mean + 1.96*all_attempts_std:.1%}]")
            print("")
            print(f"Aggregate results saved to: {filepath}")
        else:
            print("\n❌ No valid run statistics to aggregate")

def main():
    parser = argparse.ArgumentParser(description="Run ARC tasks with simple direct prompts")
    parser.add_argument("--dataset", default="arc-agi-1", choices=["arc-agi-1", "arc-agi-2"],
                       help="Dataset to use")
    parser.add_argument("--subset", default="shortest_1",
                       help="Subset name (e.g., shortest_1, shortest_10, shortest_100)")
    parser.add_argument("--model", default="gpt-4.1-mini",
                       help="OpenAI model to use")
    parser.add_argument("--limit", type=int,
                       help="Limit number of tasks to run")
    parser.add_argument("--base-url", type=str,
                       help="Base URL for OpenAI-compatible API endpoint (default: OpenAI)")
    parser.add_argument("--max_workers", type=int, default=1,
                       help="Maximum number of parallel workers (default: 1)")
    parser.add_argument("--rate_limit_delay", type=float, default=0.0,
                       help="Delay between API calls in seconds (default: 0.0)")
    parser.add_argument("--max_attempts", type=int, default=8,
                       help="Maximum number of attempts per task (default: 8)")
    parser.add_argument("--repeat-runs", type=int, default=1,
                       help="Number of times to repeat the entire test (default: 1)")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug output (default: disabled)")
    parser.add_argument("--max-tokens", type=int,
                       help="Maximum tokens for model responses")
    parser.add_argument("--temperature", type=float,
                       help="Temperature for model responses (0.0 to 2.0)")
    
    args = parser.parse_args()
    
    # Validate max_workers
    if args.max_workers < 1:
        parser.error("--max_workers must be at least 1")
    if args.max_workers > 128:
        parser.error("--max_workers cannot exceed 128 (practical limit)")
    
    # Validate rate_limit_delay
    if args.rate_limit_delay < 0:
        parser.error("--rate_limit_delay cannot be negative")
    
    # Validate repeat_runs
    if args.repeat_runs < 1:
        parser.error("--repeat-runs must be at least 1")
    if args.repeat_runs > 10:
        parser.error("--repeat-runs cannot exceed 10 (practical limit)")
    
    # Validate temperature
    if args.temperature is not None:
        if args.temperature < 0.0 or args.temperature > 2.0:
            parser.error("--temperature must be between 0.0 and 2.0")
    
    # Create runner and run tasks
    runner = ARCTaskRunnerSimple(
        model=args.model, 
        max_workers=args.max_workers, 
        rate_limit_delay=args.rate_limit_delay, 
        max_attempts=args.max_attempts, 
        run_number=0,
        base_url=getattr(args, 'base_url', None),
        debug=args.debug,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    if args.repeat_runs > 1:
        # Run repeated tests with aggregate statistics
        runner.run_repeated_subset(args.subset, args.dataset, args.limit, args.repeat_runs)
    else:
        # Single run (original behavior)
        runner.run_subset(args.subset, args.dataset, args.limit)


if __name__ == "__main__":
    main() 