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
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

try:
    # Try relative imports first (when run as module)
    from .utils.task_loader import TaskLoader
    from .utils.scoring import GridScorer, ProgramExecutor
    from .utils.prompt_utils import create_arc_prompt, extract_python_code
    from .utils.metrics_utils import calculate_task_metrics, format_metrics_display, metrics_to_percentages
    from .utils.timeout_utils import execute_with_timeout
    from .prompt_loader import PromptLoader
except ImportError:
    # Fall back to absolute imports (when run directly)
    from utils.task_loader import TaskLoader
    from utils.scoring import GridScorer, ProgramExecutor
    from utils.prompt_utils import create_arc_prompt, extract_python_code
    from utils.metrics_utils import calculate_task_metrics, format_metrics_display, metrics_to_percentages
    from utils.timeout_utils import execute_with_timeout
    from prompt_loader import PromptLoader

load_dotenv()

def serialize_response(response):
    """Convert OpenAI response to JSON-serializable format"""
    if not response:
        return None
    
    try:
        return {
            'id': getattr(response, 'id', None),
            'model': getattr(response, 'model', None),
            'usage': {
                'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0) if hasattr(response, 'usage') and response.usage else 0,
                'completion_tokens': getattr(response.usage, 'completion_tokens', 0) if hasattr(response, 'usage') and response.usage else 0,
                'total_tokens': getattr(response.usage, 'total_tokens', 0) if hasattr(response, 'usage') and response.usage else 0,
            },
            'choices': [
                {
                    'index': getattr(choice, 'index', None),
                    'message': {
                        'role': getattr(choice.message, 'role', None) if hasattr(choice, 'message') else None,
                        'content': getattr(choice.message, 'content', None) if hasattr(choice, 'message') else None,
                    },
                    'finish_reason': getattr(choice, 'finish_reason', None),
                } for choice in (getattr(response, 'choices', []))
            ],
        }
    except Exception as e:
        return {'error': f'Failed to serialize response: {str(e)}'}

class ARCTaskRunnerSimple:
    """ARC task runner with all-attempts, rolling execution, and voting-based evaluation"""
    
    def __init__(self, model: str = "gpt-4.1-nano", max_workers: int = 1, rate_limit_delay: float = 0.0, 
                 max_attempts: int = 8, run_number: int = 0, base_url: str = None, debug: bool = False, 
                 max_tokens: int = None, temperature: float = None, reasoning_effort: str = "low", 
                 qwen_no_think: bool = False, prompt_version: str = "soar"):
        self.model = model
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        self.max_attempts = max_attempts
        self.run_number = run_number
        self.reasoning_effort = reasoning_effort
        self.qwen_no_think = qwen_no_think
        self.prompt_version = prompt_version
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.base_url = base_url
        self.debug = debug
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize OpenAI client
        if base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
            print(f"📝 Using custom endpoint: {base_url}")
        else:
            self.client = OpenAI(api_key=self.api_key)
            print(f"📝 Using OpenAI endpoint")
        
        self.task_loader = TaskLoader()
        self.scorer = GridScorer()
        self.executor = ProgramExecutor(timeout=0.5)
        self.prompt_loader = PromptLoader()
        
        # Create logs directory
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        
        # Thread-safe cost and token tracking
        self._cost_lock = threading.Lock()
        self.total_cost = 0.0
        self.total_tokens = 0
    
    def _update_costs(self, cost: float, tokens: int):
        """Thread-safe method to update total costs and tokens"""
        with self._cost_lock:
            self.total_cost += cost
            self.total_tokens += tokens
    
    def get_model_pricing(self, model: str) -> tuple[float, float]:
        """Get input and output pricing rates for a model in $/1M tokens"""
        model_lower = model.lower()
        
        # Reasoning models
        if model_lower.startswith('o3-pro'):
            return (20.00, 80.00)
        elif model_lower.startswith('o3-mini'):
            return (1.10, 4.40)
        elif model_lower.startswith('o3'):
            return (2.00, 8.00)
        elif model_lower.startswith('o4-mini'):
            return (1.10, 4.40)
        
        # GPT-4 models
        elif model_lower.startswith('gpt-4.1-nano'):
            return (0.10, 0.40)
        elif model_lower.startswith('gpt-4.1-mini'):
            return (0.40, 1.60)
        elif model_lower.startswith('gpt-4.1'):
            return (2.00, 8.00)
        elif model_lower.startswith('gpt-4o-mini'):
            return (0.15, 0.60)
        elif model_lower.startswith('gpt-4o'):
            return (2.50, 10.00)
        
        # Google models
        elif model_lower.startswith('google/gemini-2.5-flash'):
            return (0.30, 2.50)
        
        # Default fallback
        else:
            return (0.15, 0.60)
    
    def create_prompt(self, task_data: Dict) -> tuple[str, str]:
        """Create a prompt for the model to solve an ARC task"""
        return create_arc_prompt(task_data, self.prompt_loader, self.prompt_version)
    
    def get_sampling_parameters(self) -> Dict:
        """Get the sampling parameters that will be used for API calls"""
        kwargs = {}
        
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        
        # Set temperature (use instance value or default to 1.0)
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        else:
            kwargs["temperature"] = 1.0  # Default temperature
        
        # Add reasoning parameters for OpenRouter
        if self.base_url and "openrouter" in self.base_url.lower():
            reasoning_tokens = {"low": 2000, "medium": 8000, "high": 32000}
            if self.reasoning_effort in reasoning_tokens:
                if "gemini" in self.model.lower():
                    kwargs["extra_body"] = {"reasoning": {"max_tokens": reasoning_tokens[self.reasoning_effort]}}
                else:
                    if self.max_tokens is None:
                        kwargs["max_tokens"] = reasoning_tokens[self.reasoning_effort]
        
        # Add sampling parameters based on endpoint type
        # Only apply defaults if not already set by model-specific logic
        if "top_p" not in kwargs and "min_p" not in kwargs:
            # For TCP endpoints, use min_p instead of top_p/top_k
            if self.base_url and ":" in self.base_url:
                if "extra_body" not in kwargs:
                    kwargs["extra_body"] = {}
                kwargs["extra_body"]["min_p"] = 0.05
            else:
                # For most endpoints, use top_p and top_k defaults
                kwargs["top_p"] = 0.9
                # Put top_k in extra_body to avoid API errors
                if "extra_body" not in kwargs:
                    kwargs["extra_body"] = {}
                if "top_k" not in kwargs["extra_body"]:
                    kwargs["extra_body"]["top_k"] = 50
        
        # Add Qwen-specific parameters (only for no-think flag)
        if "qwen" in self.model.lower() and self.base_url and self.qwen_no_think:
            if "extra_body" not in kwargs:
                kwargs["extra_body"] = {}
            kwargs["extra_body"]["chat_template_kwargs"] = {"enable_thinking": False}
        
        # Extract sampling parameters for display
        sampling_params = {}
        for param in ['temperature', 'max_tokens', 'top_p', 'top_k', 'min_p']:
            if param in kwargs:
                sampling_params[param] = kwargs[param]
        
        # Also check extra_body for nested parameters
        if 'extra_body' in kwargs:
            extra_body = kwargs['extra_body']
            for param in ['top_k', 'min_p']:
                if param in extra_body:
                    sampling_params[param] = extra_body[param]
        
        return sampling_params
    
    def call_chat_completions_api(self, messages: List[Dict]) -> Dict:
        """Call the OpenAI Chat Completions API"""
        try:
            kwargs = {
                "model": self.model,
                "messages": messages
            }
            
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            
            # Set temperature (use instance value or default to 1.0)
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            else:
                kwargs["temperature"] = 1.0  # Default temperature
            
            # Add reasoning parameters for OpenRouter
            if self.base_url and "openrouter" in self.base_url.lower():
                reasoning_tokens = {"low": 2000, "medium": 8000, "high": 32000}
                if self.reasoning_effort in reasoning_tokens:
                    if "gemini" in self.model.lower():
                        kwargs["extra_body"] = {"reasoning": {"max_tokens": reasoning_tokens[self.reasoning_effort]}}
                    else:
                        if self.max_tokens is None:
                            kwargs["max_tokens"] = reasoning_tokens[self.reasoning_effort]
            
            # Add sampling parameters based on endpoint type
            # Only apply defaults if not already set by model-specific logic
            if "top_p" not in kwargs and "min_p" not in kwargs:
                # For TCP endpoints, use min_p instead of top_p/top_k
                if self.base_url and ":" in self.base_url:
                    if "extra_body" not in kwargs:
                        kwargs["extra_body"] = {}
                    kwargs["extra_body"]["min_p"] = 0.05
                else:
                    # For most endpoints, use top_p and top_k defaults
                    kwargs["top_p"] = 0.9
                    # Put top_k in extra_body to avoid API errors
                    if "extra_body" not in kwargs:
                        kwargs["extra_body"] = {}
                    if "top_k" not in kwargs["extra_body"]:
                        kwargs["extra_body"]["top_k"] = 50
            
            # Add Qwen-specific parameters (only for no-think flag)
            if "qwen" in self.model.lower() and self.base_url and self.qwen_no_think:
                if "extra_body" not in kwargs:
                    kwargs["extra_body"] = {}
                kwargs["extra_body"]["chat_template_kwargs"] = {"enable_thinking": False}
            
            response = self.client.chat.completions.create(**kwargs)
            return response, kwargs
            
        except Exception as e:
            raise Exception(f"API call failed: {e}")
    
    def extract_code_from_response(self, response) -> str:
        """Extract Python code from the Chat Completions API result"""
        # Get the full text from response
        full_text = ""
        
        if hasattr(response, 'choices') and len(response.choices) > 0:
            message = response.choices[0].message
            if hasattr(message, 'content') and message.content:
                full_text = message.content

        if self.debug and len(full_text) > 0:
            print(f"🔍 Response content: {len(full_text)} chars")
        
        return extract_python_code(full_text, self.debug)
    
    def run_single_attempt(self, task_id: str, task_data: Dict, attempt_num: int, 
                          dataset: str = None, subset: str = None) -> Dict:
        """Run a single attempt for an ARC task"""
        system_content, user_content = self.create_prompt(task_data)
        
        attempt_start_time = datetime.datetime.now()
        conversation_history = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        # Make API call with retries
        response = None
        api_call_successful = False
        error = None
        timed_out = False
        
        # Set timeout based on model configuration
        if self.qwen_no_think:
            api_timeout = 30  # Faster for no-think mode
        else:
            api_timeout = 300  # Longer timeout for reasoning models
        
        for retry_attempt in range(3):
            try:
                result = execute_with_timeout(self.call_chat_completions_api, conversation_history, timeout=api_timeout)
                response, api_kwargs = result
                api_call_successful = True
                break
            except Exception as e:
                error = str(e)
                if retry_attempt < 2:
                    time.sleep(2)
                else:
                    timed_out = True
        
        # Extract sampling parameters for logging
        sampling_params = {}
        if api_call_successful and 'api_kwargs' in locals():
            # Extract sampling parameters from actual API call
            for param in ['temperature', 'max_tokens', 'top_p', 'top_k', 'min_p']:
                if param in api_kwargs:
                    sampling_params[param] = api_kwargs[param]
            # Also check extra_body for nested parameters
            if 'extra_body' in api_kwargs:
                extra_body = api_kwargs['extra_body']
                for param in ['top_k', 'min_p']:
                    if param in extra_body:
                        sampling_params[param] = extra_body[param]
        else:
            # Fallback to instance parameters
            if self.temperature is not None:
                sampling_params["temperature"] = self.temperature
            if self.max_tokens is not None:
                sampling_params["max_tokens"] = self.max_tokens
        
        # Track costs
        usage = getattr(response, 'usage', None)
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        input_rate, output_rate = self.get_model_pricing(self.model)
        attempt_cost = (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate
        total_tokens = usage.total_tokens if usage else 0
        
        # Extract and evaluate program
        program = self.extract_code_from_response(response) if response else ''
        
        # Evaluate on training examples
        train_results = []
        train_correct = 0
        for ex in task_data['train']:
            if not program or program.strip() == '':
                pred, err, tout = None, 'no program', False
            else:
                pred, err, tout = self.executor.execute_program_with_timeout(program, ex['input'])
            is_corr = (pred == ex['output']) if (pred is not None and not err and not tout) else False
            train_results.append({'predicted': pred, 'expected': ex['output'], 'correct': is_corr, 'error': err, 'timed_out': tout})
            if is_corr:
                train_correct += 1
        
        train_accuracy = train_correct / len(task_data['train']) if task_data['train'] else 0.0
        
        # Evaluate on test
        test_input = task_data['test'][0]['input']
        test_expected = task_data['test'][0]['output']
        if not program or program.strip() == '':
            test_pred, test_err, test_tout = None, 'no program', False
        else:
            test_pred, test_err, test_tout = self.executor.execute_program_with_timeout(program, test_input)
        test_correct = (test_pred == test_expected) if (test_pred is not None and not test_err and not test_tout) else False
        
        # Store attempt details
        attempt_detail = {
            'attempt_number': attempt_num + 1,
            'timestamp': attempt_start_time.isoformat(),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'attempt_cost': attempt_cost,
            'program_extracted': bool(program),
            'program': program,
            'train_results': train_results,
            'train_accuracy': train_accuracy,
            'test_predicted': test_pred,
            'test_expected': test_expected,
            'test_correct': test_correct,
            'test_error': test_err,
            'test_timed_out': test_tout,
            'raw_response': serialize_response(response),
            'full_prompt': {'system': system_content, 'user': user_content},
            'sampling_params': sampling_params,
            'api_success': api_call_successful,
            'timed_out': timed_out,
            'error': error
        }
        
        # Update costs
        self._update_costs(attempt_cost, total_tokens)
        
        return {
            'task_id': task_id,
            'attempt_num': attempt_num,
            'attempt_detail': attempt_detail,
            'task_data': task_data,
            'dataset': dataset,
            'subset': subset
        }
    

    
    def run_subset(self, subset_name: str, dataset: str = "arc-agi-1", limit: Optional[int] = None) -> List[Dict]:
        """Run all tasks in a subset with true parallelization at the attempt level"""
        try:
            tasks = self.task_loader.load_tasks_from_subset(subset_name, dataset)
            if limit:
                tasks = tasks[:limit]
            total_tasks = len(tasks)
        except Exception as e:
            print(f"Error loading tasks: {e}")
            return []
        
        total_attempts = total_tasks * self.max_attempts
        
        # Get sampling parameters for display
        sampling_params = self.get_sampling_parameters()
        
        print(f"\nRunning {total_tasks} tasks from {dataset}/{subset_name}")
        print(f"Model: {self.model}")
        print(f"API: All Attempts Mode ({self.max_attempts} attempts per task)")
        print(f"Mode: True parallelization - {total_attempts} total attempts")
        
        if self.max_workers > 1:
            print(f"Parallelization: ENABLED ({self.max_workers} workers)")
        else:
            print("Parallelization: DISABLED (sequential execution)")
        
        # Display sampling parameters
        if sampling_params:
            print(f"Sampling Parameters: {sampling_params}")
        else:
            print("Sampling Parameters: (using model defaults)")
        
        print("-" * 50)
        
        # Create all attempt jobs
        attempt_jobs = []
        for task_idx, (task_id, task_data) in enumerate(tasks):
            for attempt_num in range(self.max_attempts):
                attempt_jobs.append((task_idx, task_id, task_data, attempt_num))
        
        # Track results by task
        task_results = {task_id: {'attempts': [], 'task_data': task_data} for task_id, task_data in tasks}
        completed_attempts = 0
        completed_tasks = 0
        count_lock = threading.Lock()
        
        def attempt_wrapper(task_idx, task_id, task_data, attempt_num):
            nonlocal completed_attempts, completed_tasks
            try:
                result = self.run_single_attempt(task_id, task_data, attempt_num, dataset, subset_name)
                
                with count_lock:
                    # Store attempt result
                    task_results[task_id]['attempts'].append(result['attempt_detail'])
                    completed_attempts += 1
                    
                    # Check if task is complete
                    if len(task_results[task_id]['attempts']) == self.max_attempts:
                        completed_tasks += 1
                        # Calculate and display task summary
                        self._display_task_summary(task_id, task_results[task_id])
                
                return result
            except Exception as e:
                with count_lock:
                    completed_attempts += 1
                    print(f"❌ Attempt {attempt_num + 1} for task {task_id} failed: {e}")
                return None
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(attempt_wrapper, task_idx, task_id, task_data, attempt_num) 
                      for task_idx, task_id, task_data, attempt_num in attempt_jobs]
            
            print(f"🚀 Started {total_attempts} attempts with {self.max_workers} workers")
            
            # Wait for all attempts to complete
            max_wait_time = 600  # 10 minutes total timeout
            start_time = time.time()
            
            try:
                for future in futures:
                    remaining_time = max_wait_time - (time.time() - start_time)
                    if remaining_time <= 0:
                        print("⏰ Timeout reached, some attempts may not have completed")
                        break
                    future.result(timeout=remaining_time)
                print(f"✅ All {total_attempts} attempts completed")
            except Exception as e:
                print(f"⚠️ Some attempts may have failed or timed out: {e}")
            
            # Check final status
            successful_attempts = sum(1 for future in futures if future.done() and not future.exception())
            failed_attempts = sum(1 for future in futures if future.done() and future.exception())
            
            if failed_attempts > 0:
                print(f"❌ {failed_attempts} attempts failed out of {total_attempts} total")
            
            print(f"📊 Final status: {successful_attempts} successful attempts, {failed_attempts} failed")
        
        # Convert task_results to the expected format
        results = []
        for task_id, task_data in tasks:
            if task_id in task_results and len(task_results[task_id]['attempts']) > 0:
                # Sort attempts by attempt number
                attempts = sorted(task_results[task_id]['attempts'], key=lambda x: x['attempt_number'])
                
                result = {
                    'task_id': task_id,
                    'model': self.model,
                    'api_type': 'chat_completions_all_attempts',
                    'dataset': dataset,
                    'subset': subset_name,
                    'attempt_details': attempts,
                    'all_responses': [attempt['raw_response'] for attempt in attempts],
                    'tokens_used': sum(attempt['input_tokens'] + attempt['output_tokens'] for attempt in attempts),
                    'request_cost': sum(attempt['attempt_cost'] for attempt in attempts),
                    'max_attempts': self.max_attempts,
                    'api_success': True,
                    'task_data': task_data
                }
                results.append(result)
                self.save_result(result)
        
        self.save_summary(results, subset_name, dataset)
        return results
    
    def _display_task_summary(self, task_id: str, task_result: Dict):
        """Display a brief summary of a completed task"""
        attempts = task_result['attempts']
        
        # Calculate key stats
        test_correct_attempts = sum(1 for attempt in attempts if attempt['test_correct'])
        train_perfect_attempts = sum(1 for attempt in attempts if attempt['train_accuracy'] == 1.0)
        train_partial_attempts = sum(1 for attempt in attempts if 0 < attempt['train_accuracy'] < 1.0)
        
        # Calculate timeout and error stats
        api_timeouts = sum(1 for attempt in attempts if attempt.get('timed_out', False))
        api_failures = sum(1 for attempt in attempts if not attempt.get('api_success', True))
        execution_timeouts = sum(1 for attempt in attempts if attempt.get('test_timed_out', False))
        max_length_hits = sum(1 for attempt in attempts if attempt.get('raw_response') and 
                             attempt['raw_response'].get('choices') and 
                             attempt['raw_response']['choices'][0].get('finish_reason') == 'length')
        no_program_extracted = sum(1 for attempt in attempts if not attempt.get('program_extracted', False))
        
        # Find best attempt
        best_attempt = max(attempts, key=lambda x: (x['test_correct'], x['train_accuracy']))
        
        # Build summary with detailed stats
        summary = f"✅ {task_id}: {test_correct_attempts}/{len(attempts)} test-correct, {train_perfect_attempts} train-perfect, {train_partial_attempts} train-partial"
        
        # Add timeout/error details if any issues occurred
        issues = []
        if api_timeouts > 0:
            issues.append(f"{api_timeouts} api-timeout")
        if api_failures > 0:
            issues.append(f"{api_failures} api-fail")
        if execution_timeouts > 0:
            issues.append(f"{execution_timeouts} exec-timeout")
        if max_length_hits > 0:
            issues.append(f"{max_length_hits} max-len")
        if no_program_extracted > 0:
            issues.append(f"{no_program_extracted} no-code")
        
        if issues:
            summary += f" | Issues: {', '.join(issues)}"
        
        # Add best attempt info
        if best_attempt['test_correct']:
            summary += f" (best: {best_attempt['train_accuracy']:.1%} train)"
        else:
            summary += f" (best: {best_attempt['train_accuracy']:.1%} train, {best_attempt['test_correct']})"
        
        print(summary)
    
    def save_result(self, result: Dict):
        """Save individual task result"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        thread_id = threading.get_ident()
        
        if self.run_number > 0:
            filename = f"{timestamp}_{thread_id}_{result['task_id']}_simple_run{self.run_number}.json"
        else:
            filename = f"{timestamp}_{thread_id}_{result['task_id']}_simple.json"
        
        filepath = self.logs_dir / filename
        
        try:
            # Use timeout for file I/O to prevent hanging
            def write_file():
                with open(filepath, 'w') as f:
                    json.dump(result, f, indent=2)
            
            execute_with_timeout(write_file, timeout=3)  # 10 second timeout for file write
        except Exception as e:
            if self.debug:
                print(f"Error saving result: {e}")
    
    def save_summary(self, results: List[Dict], subset_name: str, dataset: str):
        """Save summary of all results"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate statistics
        total_tasks = len(results)
        api_successes = [r for r in results if r.get('api_success', True)]
        successful_api_calls = len(api_successes)
        
        # Calculate core metrics using utility functions
        if results:
            metrics = calculate_task_metrics(results, max_tokens=self.max_tokens)
            percentage_metrics = metrics_to_percentages(metrics)
        else:
            percentage_metrics = {
                'weighted_voting_pass2': 0.0,
                'train_majority_pass2': 0.0,
                'oracle_correct': 0.0,
                'all_train_correct': 0.0,
                'min1_train_correct': 0.0,
                'max_length_responses': 0.0,
                'all_timeouts': 0.0
            }
        
        summary = {
            'timestamp': timestamp,
            'dataset': dataset,
            'subset': subset_name,
            'model': self.model,
            'api_type': 'chat_completions_all_attempts',
            'run_number': self.run_number,
            'total_tasks': total_tasks,
            'successful_api_calls': successful_api_calls,
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost,
            'metrics': percentage_metrics,
            'results': results
        }
        
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
        print(f"Total tasks: {total_tasks}")
        print(f"Successful API calls: {successful_api_calls}/{total_tasks} ({successful_api_calls/total_tasks:.1%})")
        print(f"Total tokens used: {self.total_tokens:,}")
        print(f"Total cost: ${self.total_cost:.6f}")
        
        # Print core metrics
        if results:
            print("\n📊 CORE METRICS:")
            print(f"  Pass@2 (Weighted Voting): {percentage_metrics['weighted_voting_pass2']:.1%}")
            print(f"  Pass@2 (Train Majority):  {percentage_metrics['train_majority_pass2']:.1%}")
            print(f"  Oracle (Best Attempt):    {percentage_metrics['oracle_correct']:.1%}")
            print(f"  All Train Correct:        {percentage_metrics['all_train_correct']:.1%}")
            print(f"  Min 1 Train Correct:      {percentage_metrics['min1_train_correct']:.1%}")
            print(f"  Max Length Responses:     {percentage_metrics['max_length_responses']:.1%}")
            print(f"  Timeout Responses:        {percentage_metrics['timeout_responses']:.1%}")
            print(f"  API Failure Responses:    {percentage_metrics['api_failure_responses']:.1%}")
        
        print(f"\nResults saved to: {filepath}")
    
    def run_repeated_subset(self, subset_name: str, dataset: str = "arc-agi-1", limit: Optional[int] = None, repeat_runs: int = 3) -> List[List[Dict]]:
        """Run the same subset multiple times and calculate aggregate statistics"""
        print(f"\nRunning {repeat_runs} repeated tests of {dataset}/{subset_name}")
        print(f"Model: {self.model}")
        print(f"API: All Attempts Mode (max {self.max_attempts} attempts)")
        print("="*70)
        
        all_run_results = []
        
        for run_num in range(1, repeat_runs + 1):
            print(f"\n🚀 STARTING RUN {run_num}/{repeat_runs}")
            print("-" * 50)
            
            runner = ARCTaskRunnerSimple(
                model=self.model,
                max_workers=self.max_workers,
                rate_limit_delay=self.rate_limit_delay,
                max_attempts=self.max_attempts,
                run_number=run_num,
                base_url=self.base_url,
                debug=self.debug,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                reasoning_effort=self.reasoning_effort,
                qwen_no_think=self.qwen_no_think,
                prompt_version=self.prompt_version
            )
            
            results = runner.run_subset(subset_name, dataset, limit)
            all_run_results.append(results)
            
            print(f"\n✅ COMPLETED RUN {run_num}/{repeat_runs}")
        
        # Calculate and display aggregate statistics
        self._calculate_and_display_aggregate_stats(all_run_results, subset_name, dataset, repeat_runs)
        
        return all_run_results
    
    def _calculate_and_display_aggregate_stats(self, all_run_results: List[List[Dict]], subset_name: str, dataset: str, repeat_runs: int):
        """Calculate and display mean and standard deviation across multiple runs"""
        
        # Calculate final layer metrics for each run
        run_stats = []
        
        for run_num, results in enumerate(all_run_results, 1):
            if not results:
                empty_metrics = {
                    'run_number': run_num,
                    'total_tasks': 0,
                    'weighted_voting_pass2': 0.0,
                    'train_majority_pass2': 0.0,
                    'oracle_correct': 0.0,
                    'all_train_correct': 0.0,
                    'min1_train_correct': 0.0,
                    'max_length_responses': 0.0,
                    'timeout_responses': 0.0,
                    'api_failure_responses': 0.0
                }
                run_stats.append(empty_metrics)
                continue
            
            # Calculate metrics for final layer (all attempts) using utility
            metrics = calculate_task_metrics(results, max_tokens=self.max_tokens)
            percentage_metrics = metrics_to_percentages(metrics)
            percentage_metrics['run_number'] = run_num
            run_stats.append(percentage_metrics)
        
        # Calculate aggregate statistics
        if run_stats and any(s['total_tasks'] > 0 for s in run_stats):
            # Extract metrics for valid runs
            valid_runs = [s for s in run_stats if s['total_tasks'] > 0]
            
            metrics = {
                'weighted_voting_pass2': [s['weighted_voting_pass2'] for s in valid_runs],
                'train_majority_pass2': [s['train_majority_pass2'] for s in valid_runs],
                'oracle_correct': [s['oracle_correct'] for s in valid_runs],
                'all_train_correct': [s['all_train_correct'] for s in valid_runs],
                'min1_train_correct': [s['min1_train_correct'] for s in valid_runs],
                'max_length_responses': [s['max_length_responses'] for s in valid_runs],
                'timeout_responses': [s['timeout_responses'] for s in valid_runs],
                'api_failure_responses': [s['api_failure_responses'] for s in valid_runs]
            }
            
            # Calculate means and std devs
            stats = {}
            for metric_name, values in metrics.items():
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
                    stats[metric_name] = {'mean': mean_val, 'std': std_val}
                else:
                    stats[metric_name] = {'mean': 0.0, 'std': 0.0}
            
            # Save aggregate summary
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            aggregate_summary = {
                'timestamp': timestamp,
                'dataset': dataset,
                'subset': subset_name,
                'model': self.model,
                'api_type': 'chat_completions_all_attempts',
                'repeat_runs': repeat_runs,
                'run_statistics': run_stats,
                'aggregate_statistics': stats
            }
            
            filename = f"{timestamp}_aggregate_summary_{dataset}_{subset_name}_all_attempts_{repeat_runs}runs.json"
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
            print(f"Valid runs: {len(valid_runs)}")
            print("")
            
            # Individual run results
            print("INDIVIDUAL RUN RESULTS:")
            print("-" * 70)
            print(f"{'Run':<4} {'Tasks':<6} {'Weighted':<10} {'Train-Maj':<10} {'Oracle':<8} {'All-Train':<10} {'Max-Len':<8}")
            print("-" * 70)
            
            for stats_run in run_stats:
                if stats_run['total_tasks'] > 0:
                    print(f"{stats_run['run_number']:<4} {stats_run['total_tasks']:<6} "
                          f"{stats_run['weighted_voting_pass2']:<10.1%} {stats_run['train_majority_pass2']:<10.1%} "
                          f"{stats_run['oracle_correct']:<8.1%} {stats_run['all_train_correct']:<10.1%} "
                          f"{stats_run['max_length_responses']:<8.1%}")
            
            print("")
            print("AGGREGATE STATISTICS:")
            print("-" * 70)
            for metric_name, stat_data in stats.items():
                mean_val = stat_data['mean']
                std_val = stat_data['std']
                metric_display = metric_name.replace('_', ' ').title()
                print(f"{metric_display}:")
                print(f"  Mean: {mean_val:.1%}")
                print(f"  Std Dev: {std_val:.1%}")
                if len(valid_runs) > 1:
                    ci_lower = max(0, mean_val - 1.96 * std_val)
                    ci_upper = min(1, mean_val + 1.96 * std_val)
                    print(f"  95% CI: [{ci_lower:.1%}, {ci_upper:.1%}]")
                print("")
            
            print(f"Aggregate results saved to: {filepath}")
        else:
            print("\n❌ No valid run statistics to aggregate")

def main():
    parser = argparse.ArgumentParser(description="Run ARC tasks with all-attempts, rolling execution, and voting-based evaluation")
    parser.add_argument("--dataset", default="arc-agi-1", choices=["arc-agi-1", "arc-agi-2"], help="Dataset to use")
    parser.add_argument("--subset", default="shortest_1", help="Subset name")
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model to use")
    parser.add_argument("--limit", type=int, help="Limit number of tasks to run")
    parser.add_argument("--base-url", type=str, help="Base URL for OpenAI-compatible API endpoint")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of parallel workers")
    parser.add_argument("--rate_limit_delay", type=float, default=0.0, help="Delay between API calls in seconds")
    parser.add_argument("--max_attempts", type=int, default=8, help="Maximum number of attempts per task")
    parser.add_argument("--repeat-runs", type=int, default=1, help="Number of times to repeat the entire test")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--max-tokens", type=int, help="Maximum tokens for model responses")
    parser.add_argument("--temperature", type=float, help="Temperature for model responses")
    parser.add_argument("--reasoning_effort", type=str, default="low", help="Reasoning effort for OpenAI models")
    parser.add_argument("--qwen-no-think", action="store_true", help="Disable thinking for Qwen models")
    parser.add_argument("--prompt_version", type=str, default="soar", help="Version of prompts to use")
    
    args = parser.parse_args()
    
    # Validation
    if args.max_workers < 1:
        parser.error("--max_workers must be at least 1")
    if args.repeat_runs < 1:
        parser.error("--repeat-runs must be at least 1")
    if args.temperature is not None and not (0.0 <= args.temperature <= 2.0):
        parser.error("--temperature must be between 0.0 and 2.0")
    
    # Create runner and run tasks
    runner = ARCTaskRunnerSimple(
        model=args.model, 
        max_workers=args.max_workers, 
        rate_limit_delay=args.rate_limit_delay, 
        max_attempts=args.max_attempts, 
        run_number=0,
        base_url=args.base_url,
        debug=args.debug,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        qwen_no_think=args.qwen_no_think,
        prompt_version=args.prompt_version
    )
    
    if args.repeat_runs > 1:
        runner.run_repeated_subset(args.subset, args.dataset, args.limit, args.repeat_runs)
    else:
        runner.run_subset(args.subset, args.dataset, args.limit)

if __name__ == "__main__":
    main() 