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
    from .utils.transduction import is_transduction_cheating
    from .utils.prompt_loader import PromptLoader
except ImportError:
    # Fall back to absolute imports (when run directly)
    from utils.task_loader import TaskLoader
    from utils.scoring import GridScorer, ProgramExecutor
    from utils.prompt_utils import create_arc_prompt, extract_python_code
    from utils.metrics_utils import calculate_task_metrics, format_metrics_display, metrics_to_percentages
    from utils.timeout_utils import execute_with_timeout
    from utils.transduction import is_transduction_cheating
    from utils.prompt_loader import PromptLoader

load_dotenv()

def serialize_response(response):
    """Convert OpenAI response to JSON-serializable format"""
    if not response:
        return None
    
    try:
        choices = []
        for choice in getattr(response, 'choices', []):
            message_data = {
                'role': getattr(choice.message, 'role', None) if hasattr(choice, 'message') else None,
                'content': getattr(choice.message, 'content', None) if hasattr(choice, 'message') else None,
            }
            
            # Capture reasoning content from different model types and standardize to "reasoning" field
            reasoning_content = None
            
            # Check for Qwen reasoning_content field first
            if hasattr(choice, 'message') and hasattr(choice.message, 'reasoning_content'):
                reasoning_content = getattr(choice.message, 'reasoning_content', None)
            
            # Check for Gemini reasoning field
            if hasattr(choice, 'message') and hasattr(choice.message, 'reasoning'):
                reasoning_content = getattr(choice.message, 'reasoning', None)
            
            # Standardize to "reasoning" field
            if reasoning_content:
                message_data['reasoning'] = reasoning_content
            
            # Keep reasoning_details for Gemini (additional structured data)
            if hasattr(choice, 'message') and hasattr(choice.message, 'reasoning_details'):
                message_data['reasoning_details'] = getattr(choice.message, 'reasoning_details', None)
            
            choice_data = {
                'index': getattr(choice, 'index', None),
                'message': message_data,
                'finish_reason': getattr(choice, 'finish_reason', None),
            }
            choices.append(choice_data)
        
        return {
            'id': getattr(response, 'id', None),
            'model': getattr(response, 'model', None),
            'usage': {
                'prompt_tokens': getattr(response.usage, 'prompt_tokens', 0) if hasattr(response, 'usage') and response.usage else 0,
                'completion_tokens': getattr(response.usage, 'completion_tokens', 0) if hasattr(response, 'usage') and response.usage else 0,
                'total_tokens': getattr(response.usage, 'total_tokens', 0) if hasattr(response, 'usage') and response.usage else 0,
            },
            'choices': choices,
        }
    except Exception as e:
        return {'error': f'Failed to serialize response: {str(e)}'}

class ARCTaskRunnerSimple:
    """ARC task runner with all-attempts, rolling execution, and voting-based evaluation"""
    
    def __init__(self, model: str = "gpt-4.1-nano", max_workers: int = 1, rate_limit_delay: float = 0.0, 
                 max_attempts: int = 8, run_number: int = 0, base_url: str = None, debug: bool = False, 
                 max_tokens: int = None, temperature: float = None, reasoning_effort: str = "low", 
                 qwen_no_think: bool = False, prompt_version: str = "soar", unsafe_executor: bool = False):
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
        
        # Set executor type based on safety flag
        self.executor_type = "unrestricted" if unsafe_executor else "docker"
        if unsafe_executor:
            print("⚠️  WARNING: Using unrestricted executor - generated code will run directly on your system!")
        
        # Calculate timeouts based on model configuration
        self.api_timeout = 600 if self.qwen_no_think else 1200
        client_timeout = self.api_timeout + 300  # Buffer for retries/overhead
        self.worker_timeout = 2 * self.api_timeout  # Ample time for parallel execution
        
        # Initialize OpenAI client with calculated timeout
        if base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=base_url, timeout=client_timeout)
            print(f"📝 Using custom endpoint: {base_url} (timeouts: API={self.api_timeout}s, Client={client_timeout}s, Worker={self.worker_timeout}s)")
        else:
            self.client = OpenAI(api_key=self.api_key, timeout=client_timeout)
            print(f"📝 Using OpenAI endpoint (timeouts: API={self.api_timeout}s, Client={client_timeout}s, Worker={self.worker_timeout}s)")
        
        # Initialize fresh instances to prevent state leakage
        self.task_loader = TaskLoader()
        self.scorer = GridScorer()
        self.executor = ProgramExecutor(timeout=0.5, executor_type=self.executor_type)
        self.prompt_loader = PromptLoader()
        
        # Thread-safe cost tracking
        self._cost_lock = threading.Lock()
        self.total_cost = 0.0
        self.total_tokens = 0
        
        # Thread-safe cleanup to prevent race conditions during executor refresh
        self._cleanup_lock = threading.Lock()
        
        # Health monitoring for long runs
        self.health_metrics = {
            'total_attempts': 0,
            'exec_successes': 0,
            'exec_timeouts': 0,
            'exec_errors': 0,
            'exec_times': [],
            'recent_window': 100,  # Rolling window size
            'report_interval': 100  # Report every N attempts (right before cleanup)
        }
        
        # Create logs directory with timestamped subfolder
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logs_dir = Path(__file__).parent / "logs" / timestamp
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Logs will be saved to: {self.logs_dir}")
    
    def _update_costs(self, cost: float, tokens: int):
        """Thread-safe method to update total costs and tokens"""
        with self._cost_lock:
            self.total_cost += cost
            self.total_tokens += tokens
    
    def _update_health_metrics(self, attempt_detail: Dict, exec_time: float):
        """Update health monitoring metrics (thread-safe)"""
        with self._cost_lock:  # Reuse existing lock for simplicity
            self.health_metrics['total_attempts'] += 1
            
            # Track execution success/failure
            if attempt_detail.get('test_exec_error') or attempt_detail.get('train_exec_errors', 0) > 0:
                self.health_metrics['exec_errors'] += 1
            elif attempt_detail.get('test_exec_timeout') or attempt_detail.get('train_exec_timeouts', 0) > 0:
                self.health_metrics['exec_timeouts'] += 1
            else:
                self.health_metrics['exec_successes'] += 1
            
            # Track execution times (keep recent window)
            self.health_metrics['exec_times'].append(exec_time)
            window_size = self.health_metrics['recent_window']
            if len(self.health_metrics['exec_times']) > window_size:
                self.health_metrics['exec_times'] = self.health_metrics['exec_times'][-window_size:]
    
    def _print_health_report(self):
        """Print compact health report"""
        metrics = self.health_metrics
        total = metrics['total_attempts']
        
        if total == 0:
            return
            
        # Overall stats
        success_rate = (metrics['exec_successes'] / total) * 100
        timeout_rate = (metrics['exec_timeouts'] / total) * 100
        error_rate = (metrics['exec_errors'] / total) * 100
        
        # Recent window stats (last N attempts)
        window_size = min(metrics['recent_window'], total)
        recent_successes = 0
        recent_timeouts = 0 
        recent_errors = 0
        
        # Count recent attempts (simplified approach)
        recent_total = min(window_size, total)
        if recent_total > 0:
            # Approximate recent rates (would need more complex tracking for exact)
            recent_success_rate = success_rate  # Simplified for now
            recent_timeout_rate = timeout_rate
            recent_error_rate = error_rate
        
        # Execution time stats
        if metrics['exec_times']:
            avg_time = sum(metrics['exec_times']) / len(metrics['exec_times'])
            recent_times = metrics['exec_times'][-min(50, len(metrics['exec_times'])):]
            recent_avg_time = sum(recent_times) / len(recent_times) if recent_times else avg_time
        else:
            avg_time = recent_avg_time = 0.0
        
        # Compact health report
        print(f"🏥 Health [{total} attempts]: "
              f"Success {success_rate:.0f}% | "
              f"Timeout {timeout_rate:.0f}% | "
              f"ExecErr {error_rate:.0f}% | "
              f"AvgTime {recent_avg_time:.2f}s")
    
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
            if self.base_url and ":" in self.base_url and not self.base_url.startswith("https://"):
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
                if self.base_url and ":" in self.base_url and not self.base_url.startswith("https://"):
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
        exec_start_time = time.time()  # Track execution timing
        conversation_history = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        
        # Make API call with retries
        response = None
        api_call_successful = False
        error = None
        timed_out = False
        
        for retry_attempt in range(3):
            try:
                result = execute_with_timeout(self.call_chat_completions_api, conversation_history, timeout=self.api_timeout)
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
        
        # Check for empty response
        empty_response = False
        if api_call_successful and response:
            if hasattr(response, 'choices') and len(response.choices) > 0:
                message = response.choices[0].message
                content = getattr(message, 'content', '') if hasattr(message, 'content') else ''
                empty_response = not content or content.strip() == ''
            else:
                empty_response = True
        elif api_call_successful:
            empty_response = True
        
        # Check for max tokens hit
        hit_max_tokens = False
        if api_call_successful and response and hasattr(response, 'choices') and len(response.choices) > 0:
            finish_reason = getattr(response.choices[0], 'finish_reason', None)
            hit_max_tokens = (finish_reason == 'length')
        
        # Extract and evaluate program
        program = self.extract_code_from_response(response) if response else ''
        program_extracted = bool(program and program.strip())
        
        # Check for transductive/cheating behavior
        is_transductive = False
        transduction_reason = ""
        if program_extracted:
            is_transductive, transduction_reason = is_transduction_cheating(program, task_data)
        
        # Evaluate on training examples (skip if transductive)
        train_results = []
        train_correct = 0
        train_exec_errors = 0
        train_exec_timeouts = 0
        
        for ex in task_data['train']:
            if not program_extracted:
                pred, err, tout = None, 'no program', False
            elif is_transductive:
                pred, err, tout = None, 'transductive', False
            else:
                pred, err, tout = self.executor.execute_program_with_timeout(program, ex['input'])
            
            # Mark as incorrect if transductive
            is_corr = (pred == ex['output']) if (pred is not None and not err and not tout and not is_transductive) else False
            train_results.append({'predicted': pred, 'expected': ex['output'], 'correct': is_corr, 'error': err, 'timed_out': tout})
            
            if is_corr:
                train_correct += 1
            elif err and err != 'no program' and err != 'transductive':
                train_exec_errors += 1
            elif tout:
                train_exec_timeouts += 1
        
        train_accuracy = train_correct / len(task_data['train']) if task_data['train'] else 0.0
        
        # Evaluate on test
        test_input = task_data['test'][0]['input']
        test_expected = task_data['test'][0]['output']
        test_exec_error = False
        test_exec_timeout = False
        
        if not program_extracted:
            test_pred, test_err, test_tout = None, 'no program', False
        elif is_transductive:
            test_pred, test_err, test_tout = None, 'transductive', False
        else:
            test_pred, test_err, test_tout = self.executor.execute_program_with_timeout(program, test_input)
            if test_err and test_err != 'no program' and test_err != 'transductive':
                test_exec_error = True
            if test_tout:
                test_exec_timeout = True
        
        # Mark as incorrect if transductive
        test_correct = (test_pred == test_expected) if (test_pred is not None and not test_err and not test_tout and not is_transductive) else False
        
        # Store attempt details
        attempt_detail = {
            'attempt_number': attempt_num + 1,
            'timestamp': attempt_start_time.isoformat(),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'attempt_cost': attempt_cost,
            'program_extracted': program_extracted,
            'program': program,
            'is_transductive': is_transductive,
            'transduction_reason': transduction_reason,
            'train_results': train_results,
            'train_accuracy': train_accuracy,
            'train_exec_errors': train_exec_errors,
            'train_exec_timeouts': train_exec_timeouts,
            'test_predicted': test_pred,
            'test_expected': test_expected,
            'test_correct': test_correct,
            'test_error': test_err,
            'test_timed_out': test_tout,
            'test_exec_error': test_exec_error,
            'test_exec_timeout': test_exec_timeout,
            'raw_response': serialize_response(response),
            'full_prompt': {'system': system_content, 'user': user_content},
            'sampling_params': sampling_params,
            'api_success': api_call_successful,
            'api_timeout': timed_out,
            'empty_response': empty_response,
            'hit_max_tokens': hit_max_tokens,
            'error': error
        }
        
        # Update costs
        self._update_costs(attempt_cost, total_tokens)
        
        # Update health metrics and periodic reporting
        exec_time = time.time() - exec_start_time
        self._update_health_metrics(attempt_detail, exec_time)
        
        # Periodic health reports (every N attempts)
        if self.health_metrics['total_attempts'] % self.health_metrics['report_interval'] == 0:
            self._print_health_report()
        
        # Periodic executor cleanup to prevent degradation (every 100 attempts)
        if self.health_metrics['total_attempts'] % 100 == 0:
            # Thread-safe cleanup - only one thread can perform cleanup at a time
            with self._cleanup_lock:
                # Double-check the count after acquiring lock (another thread might have just cleaned up)
                if self.health_metrics['total_attempts'] % 100 == 0:
                    try:
                        print(f"🔄 Periodic executor cleanup at {self.health_metrics['total_attempts']} attempts")
                        ProgramExecutor.cleanup_executor()
                        self.executor = ProgramExecutor(timeout=0.5, executor_type=self.executor_type)
                        print("✅ Executor refreshed")
                    except Exception as cleanup_e:
                        print(f"⚠️ Executor cleanup failed: {cleanup_e}")
        
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
            
            # Validate task data integrity to prevent corruption issues
            print(f"🔍 Validating {total_tasks} tasks...")
            validated_tasks = []
            for task_id, task_data in tasks:
                if not isinstance(task_data, dict):
                    print(f"❌ Invalid task data type for {task_id}: {type(task_data)}")
                    continue
                if 'train' not in task_data or 'test' not in task_data:
                    print(f"❌ Missing train/test data for {task_id}")
                    continue
                if not isinstance(task_data['train'], list) or not isinstance(task_data['test'], list):
                    print(f"❌ Invalid train/test data structure for {task_id}")
                    continue
                if len(task_data['train']) == 0:
                    print(f"❌ No training examples for {task_id}")
                    continue
                if len(task_data['test']) == 0:
                    print(f"❌ No test examples for {task_id}")
                    continue
                validated_tasks.append((task_id, task_data))
            
            if len(validated_tasks) != total_tasks:
                print(f"⚠️ {total_tasks - len(validated_tasks)} tasks failed validation, using {len(validated_tasks)} valid tasks")
                tasks = validated_tasks
                total_tasks = len(tasks)
            
            print(f"✅ Task validation complete: {total_tasks} valid tasks")
            
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
        
        # Display executor type
        executor_info = f"Executor: {self.executor.executor_type} (timeout: {self.executor.timeout}s)"
        if self.executor.executor_type == "unrestricted":
            executor_info += " ⚠️  UNSAFE MODE"
        print(executor_info)
        
        print("-" * 50)
        
        # Create all attempt jobs
        attempt_jobs = []
        for task_idx, (task_id, task_data) in enumerate(tasks):
            for attempt_num in range(self.max_attempts):
                attempt_jobs.append((task_idx, task_id, task_data, attempt_num))
        
        # Track results by task - use thread-safe defaultdict to prevent race conditions
        from collections import defaultdict
        task_results = defaultdict(lambda: {'attempts': [], 'task_data': None})
        
        # Initialize task results with task data
        for task_id, task_data in tasks:
            task_results[task_id]['task_data'] = task_data
        
        completed_attempts = 0
        completed_tasks = 0
        count_lock = threading.Lock()
        
        def attempt_wrapper(task_idx, task_id, task_data, attempt_num):
            nonlocal completed_attempts, completed_tasks
            attempt_start = time.time()
            try:
                result = self.run_single_attempt(task_id, task_data, attempt_num, dataset, subset_name)
                attempt_duration = time.time() - attempt_start
                if attempt_duration > 60:  # Log slow attempts
                    print(f"🐌 Slow attempt: {task_id} attempt {attempt_num + 1} took {attempt_duration:.1f}s (timeout: {self.api_timeout}s)")
                
                with count_lock:
                    # Store attempt result - use thread-safe access
                    if task_id in task_results:
                        task_results[task_id]['attempts'].append(result['attempt_detail'])
                        completed_attempts += 1
                        
                        # Check if task is complete
                        if len(task_results[task_id]['attempts']) == self.max_attempts:
                            completed_tasks += 1
                            # Calculate and display task summary
                            self._display_task_summary(task_id, task_results[task_id])
                            
                            # Save task result immediately when it's complete
                            attempts = sorted(task_results[task_id]['attempts'], key=lambda x: x['attempt_number'])
                            valid_attempts = [attempt for attempt in attempts if isinstance(attempt, dict) and 'attempt_number' in attempt]
                            
                            task_result = {
                                'task_id': task_id,
                                'model': self.model,
                                'api_type': 'chat_completions_all_attempts',
                                'dataset': dataset,
                                'subset': subset_name,
                                'attempt_details': valid_attempts,
                                'all_responses': [attempt.get('raw_response') for attempt in valid_attempts],
                                'tokens_used': sum(attempt.get('input_tokens', 0) + attempt.get('output_tokens', 0) for attempt in valid_attempts),
                                'request_cost': sum(attempt.get('attempt_cost', 0.0) for attempt in valid_attempts),
                                'max_attempts': self.max_attempts,
                                'api_success': True,
                                'task_data': task_data
                            }
                            self.save_result(task_result)
                            # print(f"💾 Saved log for task {task_id}")
                    else:
                        print(f"⚠️ Task {task_id} not found in results dict - possible corruption")
                
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
            start_time = time.time()
            
            try:
                for future_idx, future in enumerate(futures):
                    remaining_time = self.worker_timeout - (time.time() - start_time)
                    if remaining_time <= 0:
                        print("⏰ Timeout reached, some attempts may not have completed")
                        print("🛑 Cancelling all remaining futures and stopping execution")
                        # Cancel all remaining futures
                        for remaining_future in futures[future_idx:]:
                            remaining_future.cancel()
                        break
                    try:
                        future.result(timeout=remaining_time)
                    except Exception as future_e:
                        print(f"🚨 FUTURE TIMEOUT: Future #{future_idx + 1}")
                        print(f"   Error: {future_e}")
                        print(f"   Elapsed time: {time.time() - start_time:.1f}s")
                        print(f"   Remaining time: {remaining_time:.1f}s")
                        print(f"   Future done: {future.done()}")
                        print(f"   Future exception: {future.exception()}")
                        
                        # Get the task details for this future
                        task_idx, task_id, task_data, attempt_num = attempt_jobs[future_idx]
                        print(f"   Task: {task_id}, Attempt: {attempt_num + 1}")
                        print(f"   Task data keys: {list(task_data.keys()) if task_data else 'None'}")
                        
                        # Check if this future has any partial results
                        if future.done() and not future.exception():
                            try:
                                result = future.result(timeout=1)  # Quick check
                                print(f"   Future actually completed successfully")
                                if result and 'attempt_detail' in result:
                                    attempt = result['attempt_detail']
                                    print(f"   API success: {attempt.get('api_success', 'Unknown')}")
                                    print(f"   Timed out: {attempt.get('api_timeout', 'Unknown')}")
                                    print(f"   Error: {attempt.get('error', 'None')}")
                                    print(f"   Program extracted: {attempt.get('program_extracted', 'Unknown')}")
                                    print(f"   Test correct: {attempt.get('test_correct', 'Unknown')}")
                            except Exception as check_e:
                                print(f"   Could not check future result: {check_e}")
                        
                        print(f"   Stopping execution due to future timeout")
                        # Cancel all remaining futures
                        for remaining_future in futures[future_idx:]:
                            remaining_future.cancel()
                        raise future_e
                
                # Check if we completed all futures or hit timeout
                completed_futures = sum(1 for future in futures if future.done())
                if completed_futures == total_attempts:
                    print(f"✅ All {total_attempts} attempts completed")
                else:
                    print(f"⚠️ Only {completed_futures}/{total_attempts} attempts completed (timeout hit)")
                    
            except Exception as e:
                print(f"⚠️ Some attempts may have failed or timed out: {e}")
            
            # Check final status
            successful_attempts = sum(1 for future in futures if future.done() and not future.exception())
            failed_attempts = sum(1 for future in futures if future.done() and future.exception())
            
            if failed_attempts > 0:
                print(f"❌ {failed_attempts} attempts failed out of {total_attempts} total")
            
            print(f"📊 Final status: {successful_attempts} successful attempts, {failed_attempts} failed")
        
        # Convert task_results to the expected format for summary
        results = []
        for task_id, task_data in tasks:
            if task_id in task_results and len(task_results[task_id]['attempts']) > 0:
                # Sort attempts by attempt number
                attempts = sorted(task_results[task_id]['attempts'], key=lambda x: x['attempt_number'])
                
                # Validate attempt data integrity
                valid_attempts = []
                for attempt in attempts:
                    if isinstance(attempt, dict) and 'attempt_number' in attempt:
                        valid_attempts.append(attempt)
                    else:
                        print(f"⚠️ Invalid attempt data for task {task_id}: {type(attempt)}")
                
                if len(valid_attempts) != len(attempts):
                    print(f"⚠️ Task {task_id}: {len(attempts) - len(valid_attempts)} invalid attempts filtered out")
                
                result = {
                    'task_id': task_id,
                    'model': self.model,
                    'api_type': 'chat_completions_all_attempts',
                    'dataset': dataset,
                    'subset': subset_name,
                    'attempt_details': valid_attempts,
                    'all_responses': [attempt.get('raw_response') for attempt in valid_attempts],
                    'tokens_used': sum(attempt.get('input_tokens', 0) + attempt.get('output_tokens', 0) for attempt in valid_attempts),
                    'request_cost': sum(attempt.get('attempt_cost', 0.0) for attempt in valid_attempts),
                    'max_attempts': self.max_attempts,
                    'api_success': True,
                    'task_data': task_data
                }
                results.append(result)
                # Note: save_result() is now called when each task completes, not here
            else:
                print(f"⚠️ Task {task_id} has no valid attempts - skipping")
        
        summary_filepath = self.save_summary(results, subset_name, dataset)
        return results, summary_filepath
    
    def _display_task_summary(self, task_id: str, task_result: Dict):
        """Display a brief summary of a completed task"""
        attempts = task_result['attempts']
        
        # Calculate key stats
        test_correct_attempts = sum(1 for attempt in attempts if attempt['test_correct'])
        train_perfect_attempts = sum(1 for attempt in attempts if attempt['train_accuracy'] == 1.0)
        train_partial_attempts = sum(1 for attempt in attempts if 0 < attempt['train_accuracy'] < 1.0)
        
        # Calculate issues in timeline order
        api_timeouts = sum(1 for attempt in attempts if attempt.get('api_timeout', False))
        api_failures = sum(1 for attempt in attempts if not attempt.get('api_success', True))
        empty_responses = sum(1 for attempt in attempts if attempt.get('empty_response', False))
        max_length_hits = sum(1 for attempt in attempts if attempt.get('hit_max_tokens', False))
        no_code_extracted = sum(1 for attempt in attempts if not attempt.get('program_extracted', False))
        transductive_attempts = sum(1 for attempt in attempts if attempt.get('is_transductive', False))
        train_exec_errors = sum(1 for attempt in attempts if attempt.get('train_exec_errors', 0) > 0)
        train_exec_timeouts = sum(1 for attempt in attempts if attempt.get('train_exec_timeouts', 0) > 0)
        test_exec_errors = sum(1 for attempt in attempts if attempt.get('test_exec_error', False))
        test_exec_timeouts = sum(1 for attempt in attempts if attempt.get('test_exec_timeout', False))
        
        # Find best attempt
        best_attempt = max(attempts, key=lambda x: (x['test_correct'], x['train_accuracy']))
        
        # Build summary
        summary = f"✅ {task_id}: {test_correct_attempts}/{len(attempts)} test-correct, {train_perfect_attempts} train-perfect, {train_partial_attempts} train-partial"
        
        # Add issues in timeline order if any occurred
        issues = []
        if api_timeouts > 0:
            issues.append(f"{api_timeouts} api-timeout")
        if api_failures > 0:
            issues.append(f"{api_failures} api-fail")
        if empty_responses > 0:
            issues.append(f"{empty_responses} empty-response")
        if max_length_hits > 0:
            issues.append(f"{max_length_hits} max-len")
        if no_code_extracted > 0:
            issues.append(f"{no_code_extracted} no-code")
        if transductive_attempts > 0:
            issues.append(f"{transductive_attempts} transductive")
        if train_exec_errors > 0:
            issues.append(f"{train_exec_errors} train-exec-error")
        if train_exec_timeouts > 0:
            issues.append(f"{train_exec_timeouts} train-exec-timeout")
        if test_exec_errors > 0:
            issues.append(f"{test_exec_errors} test-exec-error")
        if test_exec_timeouts > 0:
            issues.append(f"{test_exec_timeouts} test-exec-timeout")
        
        if issues:
            summary += f" | Issues: {', '.join(issues)}"
        
        # Add best attempt performance (separate from issues)
        if best_attempt['test_correct']:
            summary += f" (best: {best_attempt['train_accuracy']:.1%} train)"
        else:
            summary += f" (best: {best_attempt['train_accuracy']:.1%} train, test-failed)"
        
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
            with open(filepath, 'w') as f:
                json.dump(result, f, indent=2)
        except Exception as e:
            print(f"🚨 FILE I/O ERROR: Task {result['task_id']}, Attempt {result['attempt_details'][0]['attempt_number']}")
            print(f"   File: {filepath}")
            print(f"   Error: {e}")
            raise Exception(f"File I/O error for task {result['task_id']}: {e}")
    
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
        
        # Create summary with full results for later aggregation
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
            'results': results,  # Include full results for aggregation
            'task_ids': [result['task_id'] for result in results]  # Also include task IDs for convenience
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
        return filepath
    
    def run_repeated_subset(self, subset_name: str, dataset: str = "arc-agi-1", limit: Optional[int] = None, repeat_runs: int = 3) -> List[List[Dict]]:
        """Run the same subset multiple times with completely independent runs"""
        print(f"\nRunning {repeat_runs} repeated tests of {dataset}/{subset_name}")
        print(f"Model: {self.model}")
        print(f"{self.max_attempts} attempts")
        print("="*70)
        
        # Store run results independently - no shared state
        run_files = []
        
        for run_num in range(1, repeat_runs + 1):
            print(f"\n🚀 STARTING RUN {run_num}/{repeat_runs}")
            print("-" * 50)
            
            # Force garbage collection between runs to clear any shared state
            import gc
            gc.collect()
            
            # Create completely independent runner with no shared state
            runner = ARCTaskRunnerSimple(
                model=self.model,
                max_workers=self.max_workers,
                rate_limit_delay=self.rate_limit_delay,
                max_attempts=self.max_attempts,
                run_number=run_num,  # This ensures unique file names
                base_url=self.base_url,
                debug=self.debug,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                reasoning_effort=self.reasoning_effort,
                qwen_no_think=self.qwen_no_think,
                prompt_version=self.prompt_version,
                unsafe_executor=(self.executor_type == "unrestricted")
            )
            
            try:
                # Run the subset and let it save its own results
                results, summary_filepath = runner.run_subset(subset_name, dataset, limit)
                print(f"\n✅ COMPLETED RUN {run_num}/{repeat_runs}")
                
                # Store the actual summary file path that was created
                run_files.append(summary_filepath)
                
            except Exception as e:
                print(f"\n❌ RUN {run_num} FAILED: {e}")
                run_files.append(None)  # Mark as failed
            finally:
                # Explicit cleanup to prevent state leakage
                # ProgramExecutor uses singleton class variables that persist across instances
                # Without cleanup, second run reuses stale executor context causing systematic failures
                with self._cleanup_lock:  # Thread-safe cleanup
                    try:
                        ProgramExecutor.cleanup_executor()  # Fix: Clean up singleton state
                        if run_num < repeat_runs:  # Only print for non-final runs
                            print(f"🧹 Cleaned up executor state after run {run_num}")
                    except Exception as cleanup_e:
                        print(f"⚠️ Failed to cleanup executor state: {cleanup_e}")
                del runner
                gc.collect()
        
        # Load results from files and calculate aggregate statistics
        all_run_results = self._load_and_aggregate_results(run_files, subset_name, dataset, repeat_runs)
        
        return all_run_results
    
    def _load_and_aggregate_results(self, run_files: List[Optional[Path]], subset_name: str, dataset: str, repeat_runs: int) -> List[List[Dict]]:
        """Load results from individual run files and aggregate statistics"""
        print(f"\n📊 Loading results from {len(run_files)} run files...")
        
        all_run_results = []
        successful_runs = 0
        
        for run_num, filepath in enumerate(run_files, 1):
            if filepath is None:
                print(f"❌ Run {run_num}: Failed run, no results")
                all_run_results.append([])
                continue
                
            try:
                if filepath.exists():
                    with open(filepath, 'r') as f:
                        summary_data = json.load(f)
                    
                    # Extract the results from the summary
                    if 'results' in summary_data:
                        results = summary_data['results']
                        all_run_results.append(results)
                        successful_runs += 1
                        print(f"✅ Run {run_num}: Loaded {len(results)} results from {filepath.name}")
                    else:
                        print(f"⚠️ Run {run_num}: No results found in {filepath.name}")
                        all_run_results.append([])
                else:
                    print(f"❌ Run {run_num}: File not found: {filepath}")
                    all_run_results.append([])
                    
            except Exception as e:
                print(f"❌ Run {run_num}: Error loading {filepath}: {e}")
                all_run_results.append([])
        
        print(f"📊 Successfully loaded {successful_runs}/{repeat_runs} runs")
        
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
            print("-" * 82)
            print(f"{'Run':<4} {'Tasks':<6} {'Weighted':<10} {'Train-Maj':<10} {'Oracle':<8} {'All-Train':<10} {'Min1-Train':<11} {'Max-Len':<8}")
            print("-" * 82)
            
            for stats_run in run_stats:
                if stats_run['total_tasks'] > 0:
                    print(f"{stats_run['run_number']:<4} {stats_run['total_tasks']:<6} "
                          f"{stats_run['weighted_voting_pass2']:<10.1%} {stats_run['train_majority_pass2']:<10.1%} "
                          f"{stats_run['oracle_correct']:<8.1%} {stats_run['all_train_correct']:<10.1%} "
                          f"{stats_run['min1_train_correct']:<11.1%} {stats_run['max_length_responses']:<8.1%}")
            
            print("")
            print("AGGREGATE STATISTICS:")
            print("-" * 82)
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
    parser.add_argument("--unsafe-executor", action="store_true", 
                        help="⚠️  UNSAFE: Use unrestricted executor (no Docker sandboxing). Generated code runs directly on your system. SECURITY RISK!")
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
        prompt_version=args.prompt_version,
        unsafe_executor=args.unsafe_executor
    )
    
    try:
        if args.repeat_runs > 1:
            runner.run_repeated_subset(args.subset, args.dataset, args.limit, args.repeat_runs)
        else:
            results, _ = runner.run_subset(args.subset, args.dataset, args.limit)
    finally:
        # Final cleanup to ensure clean shutdown
        with runner._cleanup_lock:  # Thread-safe cleanup
            try:
                ProgramExecutor.cleanup_executor()
            except Exception as e:
                print(f"⚠️ Final cleanup warning: {e}")

if __name__ == "__main__":
    main() 