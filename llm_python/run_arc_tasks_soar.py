#!/usr/bin/env python3

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

from llm_python.utils.task_loader import TaskData, TaskLoader
from llm_python.utils.arc_tester import ArcTester
from llm_python.utils.prompt_utils import create_arc_prompt, extract_python_code
from llm_python.utils.metrics_utils import calculate_task_metrics, metrics_to_percentages
from llm_python.utils.timeout_utils import execute_with_timeout
from llm_python.utils.transduction import is_transduction_cheating
from llm_python.utils.prompt_loader import PromptLoader
from llm_python.utils.serialization import ResponseSerializer
from llm_python.utils.api_client import ARCAPIClient
from llm_python.utils.result_processor import ResultProcessor
from llm_python.utils.validator import ARCTaskValidator

load_dotenv()


class ARCTaskRunnerSimple:
    """ARC task runner with all-attempts, rolling execution, and voting-based evaluation
    
    Supports multiple API endpoints including:
    - OpenAI/OpenRouter: Standard chat completions with reasoning models
    - DashScope: Alibaba's commercial Qwen models with thinking_budget parameter
    
    Note: DashScope commercial Qwen models always use thinking mode and don't support 
    enable_thinking=False. Use thinking_budget parameter to control reasoning depth.
    """
    
    def __init__(self, model: str = "gpt-4.1-nano", max_workers: int = 1, rate_limit_delay: float = 0.0, 
                 max_attempts: int = 8, run_number: int = 0, base_url: str = None, debug: bool = False, 
                 max_tokens: int = None, temperature: float = None, reasoning_effort: str = "low", 
                 qwen_no_think: bool = False, prompt_version: str = "soar", unsafe_executor: bool = False, 
                 lora_adapter: str = None):
        # Core configuration
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        self.max_attempts = max_attempts
        self.run_number = run_number
        self.debug = debug
        self.prompt_version = prompt_version
        
        # Standard API timeout for network safety, no infrastructure timeouts
        api_timeout = 1800  # 30 minutes for network safety only
        
        # Warn about safety settings
        executor_type = "unrestricted" if unsafe_executor else "docker"
        if unsafe_executor:
            print("⚠️  WARNING: Using unrestricted executor - generated code will run directly on your system!")
        
        # Store references needed by other methods
        self.large_file_errors = []
        
        # Initialize services
        self.api_client = ARCAPIClient(
            model=model, base_url=base_url, max_tokens=max_tokens, 
            temperature=temperature, reasoning_effort=reasoning_effort,
            qwen_no_think=qwen_no_think, lora_adapter=lora_adapter,
            api_timeout=api_timeout  # Standard timeout for network safety
        )
        
        self.result_processor = ResultProcessor(
            model=model, run_number=run_number, max_tokens=max_tokens
        )
        
        # Initialize remaining components
        self.task_loader = TaskLoader()
        self.executor = ArcTester(
            timeout=0.5,
            executor_type=executor_type,
            max_output_chars=10_000,
            max_output_cells=1_800,
        )
        self.prompt_loader = PromptLoader()
        
        # Thread-safe cost tracking
        self._cost_lock = threading.Lock()
        self.total_cost = 0.0
        self.total_tokens = 0
        
        # Thread-safe state management
        self._cleanup_lock = threading.Lock()
        
        # Health monitoring for long runs
        self.health_metrics = {
            'total_attempts': 0,
            'exec_successes': 0,
            'exec_timeouts': 0,
            'exec_errors': 0,
            'exec_times': [],
            'recent_window': 100,
            'report_interval': 100
        }
        
        print(f"⏰ API timeout: {api_timeout}s (network safety only, no infrastructure timeouts)")
        print(f"📁 Logs will be saved to: {self.result_processor.logs_dir}")
    
    @property
    def model(self):
        """Get the model name from the API client"""
        return self.api_client.model
    
    @property
    def logs_dir(self):
        """Get the logs directory from the result processor"""
        return self.result_processor.logs_dir
    
    @property
    def base_url(self):
        """Get the base URL from the API client"""
        return self.api_client.base_url
    
    @property
    def max_tokens(self):
        """Get max tokens from the API client"""
        return self.api_client.max_tokens
    
    @property
    def temperature(self):
        """Get temperature from the API client"""
        return self.api_client.temperature
    
    @property
    def reasoning_effort(self):
        """Get reasoning effort from the API client"""
        return self.api_client.reasoning_effort
    
    @property
    def qwen_no_think(self):
        """Get qwen no think setting from the API client"""
        return self.api_client.qwen_no_think
    
    @property
    def lora_adapter(self):
        """Get LORA adapter from the API client"""
        return self.api_client.lora_adapter
    
    @property
    def executor_type(self):
        """Get executor type from the executor"""
        return self.executor.executor_type
    
    @property
    def api_timeout(self):
        """Get API timeout from the API client"""
        return self.api_client.api_timeout
    
    # HF token handling moved to ARCAPIClient
    
    def _check_models_endpoint(self):
        """Check what models are available at the /models endpoint and validate arguments"""
        try:
            print("🔍 Checking available models...")
            models_response = self.api_client.client.models.list()
            
            if hasattr(models_response, 'data') and models_response.data:
                available_models = []
                available_lora_adapters = []
                
                print(f"📋 Available models ({len(models_response.data)}):")
                for model in models_response.data:
                    model_id = getattr(model, 'id', 'unknown')
                    owned_by = getattr(model, 'owned_by', 'unknown')
                    available_models.append(model_id)
                    
                    # Show LORA adapters if present
                    if hasattr(model, 'lora_adapters') and model.lora_adapters:
                        lora_list = model.lora_adapters
                        available_lora_adapters.extend(lora_list)
                        print(f"   • {model_id} (owner: {owned_by}) [LORA: {', '.join(lora_list)}]")
                    elif 'lora' in model_id.lower() or 'adapter' in model_id.lower():
                        available_lora_adapters.append(model_id)
                        print(f"   • {model_id} (owner: {owned_by}) [LORA adapter]")
                    else:
                        print(f"   • {model_id} (owner: {owned_by})")
                
                # Validate model argument
                if self.model not in available_models:
                    print(f"⚠️  WARNING: Specified model '{self.model}' not found in available models")
                    print(f"   Available models: {', '.join(available_models)}")
                    print("   This may cause API errors during execution")
                else:
                    print(f"✅ Model '{self.model}' found in endpoint")
                
                # LORA adapter info (no validation since not exposed via /models)
                if self.lora_adapter:
                    print(f"🎯 Will use LORA adapter: {self.lora_adapter}")
            else:
                print("   No models found in response")
                
        except Exception as e:
            print(f"⚠️  Could not check /models endpoint: {e}")
            print("   This might be normal for some endpoints (OpenAI, etc.)")
            print("   Skipping validation - will attempt to use specified model/LORA")
    
    def _update_costs(self, cost: float, tokens: int):
        """Thread-safe method to update total costs and tokens"""
        with self._cost_lock:
            self.total_cost += cost
            self.total_tokens += tokens
    
    def _check_file_size_before_save(self, data: Dict, filename: str, max_size_gb: float = 1.0) -> tuple[bool, str]:
        """Check if JSON data would exceed size limit before saving
        
        Returns:
            tuple: (should_save, error_message)
        """
        try:
            # Estimate JSON size by serializing to string
            json_str = json.dumps(ResponseSerializer.make_json_safe(data), indent=2)
            size_bytes = len(json_str.encode('utf-8'))
            size_gb = size_bytes / (1024**3)
            
            if size_gb > max_size_gb:
                # Analyze what's making the file large
                size_details = self._analyze_large_file_components(data, size_bytes)
                
                error_msg = f"File {filename} would be {size_gb:.2f}GB (>{max_size_gb}GB limit). {size_details}"
                
                # Thread-safe error tracking
                with self._cost_lock:
                    self.large_file_errors.append({
                        'filename': filename,
                        'size_gb': size_gb,
                        'size_bytes': size_bytes,
                        'details': size_details,
                        'timestamp': datetime.datetime.now().isoformat()
                    })
                
                return False, error_msg
            
            return True, ""
            
        except Exception as e:
            # If we can't estimate size, err on the side of caution
            error_msg = f"Could not estimate size for {filename}: {e}"
            return True, error_msg  # Allow saving if we can't check
    
    def _analyze_large_file_components(self, data: Dict, total_size: int) -> str:
        """Analyze what components are making a file large"""
        details = []
        
        try:
            # Check major components
            if 'attempt_details' in data:
                attempt_details_size = len(json.dumps(data['attempt_details']).encode('utf-8'))
                if attempt_details_size > total_size * 0.1:  # >10% of total
                    details.append(f"attempt_details: {attempt_details_size / (1024**2):.1f}MB")
                    
                    # Check individual attempts for large components
                    for i, attempt in enumerate(data['attempt_details'][:3]):  # Check first 3
                        if 'raw_response' in attempt:
                            resp_size = len(json.dumps(attempt['raw_response']).encode('utf-8'))
                            if resp_size > 50 * 1024 * 1024:  # >50MB
                                details.append(f"attempt {i+1} response: {resp_size / (1024**2):.1f}MB")
                        
                        if 'program' in attempt:
                            prog_size = len(str(attempt['program']).encode('utf-8'))
                            if prog_size > 1024 * 1024:  # >1MB
                                details.append(f"attempt {i+1} program: {prog_size / 1024:.1f}KB")
            
            if 'results' in data:
                results_size = len(json.dumps(data['results']).encode('utf-8'))
                if results_size > total_size * 0.1:  # >10% of total
                    details.append(f"results array: {results_size / (1024**2):.1f}MB")
            
            if 'all_responses' in data:
                responses_size = len(json.dumps(data['all_responses']).encode('utf-8'))
                if responses_size > total_size * 0.1:  # >10% of total
                    details.append(f"all_responses: {responses_size / (1024**2):.1f}MB")
            
            return "Large components: " + ", ".join(details) if details else "Unknown large components"
            
        except Exception as e:
            return f"Error analyzing file size: {e}"
    
    def _report_large_file_errors(self):
        """Report any large file errors that occurred during the run"""
        if not self.large_file_errors:
            return
        
        print("")
        print("🚨" * 30)
        print("LARGE FILE ERRORS SUMMARY")
        print("🚨" * 30)
        print(f"Number of files that exceeded 1GB limit: {len(self.large_file_errors)}")
        print("")
        
        # Sort by size for easier analysis
        sorted_errors = sorted(self.large_file_errors, key=lambda x: x['size_gb'], reverse=True)
        
        print("Files blocked from saving (largest first):")
        print("-" * 80)
        
        for i, error in enumerate(sorted_errors[:10], 1):  # Show top 10
            print(f"{i:2d}. {error['filename']}")
            print(f"    Size: {error['size_gb']:.2f}GB ({error['size_bytes']:,} bytes)")
            print(f"    {error['details']}")
            print(f"    Time: {error['timestamp']}")
            print("")
        
        if len(sorted_errors) > 10:
            print(f"... and {len(sorted_errors) - 10} more files blocked")
            print("")
        
        # Provide guidance
        total_blocked_size = sum(e['size_gb'] for e in self.large_file_errors)
        print(f"Total data blocked: {total_blocked_size:.2f}GB")
        print("")
        print("💡 RECOMMENDATIONS:")
        print("   • Large responses often indicate very verbose model outputs")
        print("   • Consider using lower max_tokens to reduce response size")
        print("   • Check if model is generating extremely long programs")
        print("   • Monitor raw_response fields which may contain large reasoning traces")
        print("   • For debugging, you can increase the size limit parameter")
        print("🚨" * 30)
    
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
    
    
    def create_prompt(self, task_data: Dict) -> tuple[str, str]:
        """Create a prompt for the model to solve an ARC task"""
        return create_arc_prompt(task_data, self.prompt_loader, self.prompt_version)
    
    def get_sampling_parameters(self) -> Dict:
        """Get the sampling parameters that will be used for API calls"""
        return self.api_client.get_sampling_parameters()
    
    def call_chat_completions_api(self, messages: List[Dict]) -> tuple:
        """Call the OpenAI Chat Completions API"""
        result = self.api_client.call_chat_completions_api(messages)
        if result['success']:
            return result['response'], result['sampling_params']
        else:
            # Return None response with error info
            return None, result['sampling_params']
    
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
    
    def run_single_attempt(self, task_id: str, task_data: TaskData, attempt_num: int, 
                          dataset: str = None, subset: str = None, full_prompt: Dict = None) -> Dict:
        """Run a single attempt for an ARC task"""
        system_content = full_prompt['system']
        user_content = full_prompt['user']
        
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
                response, api_kwargs = self.call_chat_completions_api(conversation_history)
                api_call_successful = True
                break
            except Exception as e:
                error = str(e)
                # Check if this is actually a timeout error vs other API errors
                is_timeout_error = (
                    "timeout" in str(e).lower() or 
                    "TimeoutError" in str(type(e).__name__) or
                    "concurrent.futures._base.TimeoutError" in str(type(e))
                )
                # Don't retry timeout errors - they indicate the request took too long
                if is_timeout_error:
                    timed_out = True
                    break
                # Only retry non-timeout errors (API errors, network issues, etc.)
                elif retry_attempt < 2:
                    time.sleep(2)
                else:
                    # Final non-timeout error
                    timed_out = False
        
        # Extract sampling parameters for logging
        sampling_params = {}
        if api_call_successful and 'api_kwargs' in locals():
            # Extract sampling parameters from actual API call
            for param in ['temperature', 'max_tokens', 'top_p', 'top_k', 'min_p', 'thinking_budget']:
                if param in api_kwargs:
                    sampling_params[param] = api_kwargs[param]
            # Also check extra_body for nested parameters
            if 'extra_body' in api_kwargs:
                extra_body = api_kwargs['extra_body']
                for param in ['top_k', 'min_p', 'thinking_budget']:
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
        input_rate, output_rate = self.api_client.get_model_pricing()
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
        is_train_transductive = False
        train_transduction_reason = ""
        is_test_transductive = False
        test_transduction_reason = ""
        if program_extracted:
            is_train_transductive, train_transduction_reason, is_test_transductive, test_transduction_reason = is_transduction_cheating(program, task_data)
        
        # Evaluate on training examples (skip if transductive)
        train_results = []
        train_correct = 0
        train_exec_errors = 0
        train_exec_timeouts = 0
        
        for ex in task_data['train']:
            if not program_extracted:
                pred, err, tout = None, 'no program', False
            elif is_train_transductive:
                pred, err, tout = None, 'train_transductive', False
            else:
                pred, err, tout = self.executor.execute_program_with_timeout(program, ex['input'])
            
            # Mark as incorrect if train transductive
            is_corr = (pred == ex['output']) if (pred is not None and not err and not tout and not is_train_transductive) else False
            train_results.append({'predicted': pred, 'expected': ex['output'], 'correct': is_corr, 'error': err, 'timed_out': tout})
            
            if is_corr:
                train_correct += 1
            elif err and err != 'no program' and err != 'train_transductive':
                train_exec_errors += 1
            elif tout:
                train_exec_timeouts += 1
        
        train_accuracy = train_correct / len(task_data['train']) if task_data['train'] else 0.0
        
        # Evaluate on all test examples
        test_results = []
        test_predictions = []
        test_correct_count = 0
        any_test_exec_error = False
        any_test_exec_timeout = False
        
        for test_idx, test_example in enumerate(task_data['test']):
            test_input = test_example['input']
            test_expected = test_example['output']
            
            if not program_extracted:
                test_pred, test_err, test_tout = None, 'no program', False
            elif is_train_transductive:
                test_pred, test_err, test_tout = None, 'train_transductive', False
            else:
                # Allow test execution even if test_transductive (for metadata logging)
                test_pred, test_err, test_tout = self.executor.execute_program_with_timeout(program, test_input)
                if test_err and test_err != 'no program' and test_err != 'train_transductive':
                    any_test_exec_error = True
                if test_tout:
                    any_test_exec_timeout = True
            
            # Mark as correct/incorrect normally (test transduction doesn't block execution)
            is_correct = (test_pred == test_expected) if (test_pred is not None and not test_err and not test_tout and not is_train_transductive) else False
            
            if is_correct:
                test_correct_count += 1
            
            test_results.append({
                'test_idx': test_idx,
                'predicted': test_pred,
                'expected': test_expected,
                'correct': is_correct,
                'error': test_err,
                'timed_out': test_tout
            })
            test_predictions.append(test_pred)
        
        # Overall test correctness (all test cases must be correct)
        test_correct = (test_correct_count == len(task_data['test'])) if len(task_data['test']) > 0 else False
        
        # Store attempt details
        attempt_detail = {
            'attempt_number': attempt_num + 1,
            'timestamp': attempt_start_time.isoformat(),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'attempt_cost': attempt_cost,
            'program_extracted': program_extracted,
            'program': program,
            'is_train_transductive': is_train_transductive,
            'train_transduction_reason': train_transduction_reason,
            'is_test_transductive': is_test_transductive,
            'test_transduction_reason': test_transduction_reason,
            # Legacy field for backwards compatibility
            'is_transductive': is_train_transductive,
            'transduction_reason': train_transduction_reason,
            'train_results': train_results,
            'train_accuracy': train_accuracy,
            'train_exec_errors': train_exec_errors,
            'train_exec_timeouts': train_exec_timeouts,
            # Multi-test support: store all predictions and detailed results  
            'test_predicted': tuple(test_predictions) if len(task_data['test']) > 1 else (test_predictions[0] if test_predictions else None),  # Tuple for multiple tests, raw prediction for single test
            'test_results': test_results,        # Detailed results for each test case
            'test_correct': test_correct,        # True if ALL test cases are correct
            'test_correct_count': test_correct_count,  # Number of correct test cases
            'test_exec_error': any_test_exec_error,
            'test_exec_timeout': any_test_exec_timeout,
            # Legacy fields for backwards compatibility (using first test case)
            'test_error': test_results[0]['error'] if test_results else 'no program',
            'test_timed_out': test_results[0]['timed_out'] if test_results else False,
            'raw_response': ResponseSerializer.serialize_response(response),
            'sampling_params': sampling_params,
            'api_success': api_call_successful,
            'api_timeout': timed_out,
            'empty_response': empty_response,
            'hit_max_tokens': hit_max_tokens,
            'error': error,
            # Add fields expected by metrics_utils
            'all_test_correct': test_correct,  # True if ALL test cases are correct
            'code_ran': program_extracted      # Alias for program_extracted
        }
        
        # Update costs
        self._update_costs(attempt_cost, total_tokens)
        
        # Update health metrics and periodic reporting
        exec_time = time.time() - exec_start_time
        self._update_health_metrics(attempt_detail, exec_time)
        
        # Periodic health reports (every N attempts)
        if self.health_metrics['total_attempts'] % self.health_metrics['report_interval'] == 0:
            self._print_health_report()
        
        return {
            'task_id': task_id,
            'attempt_num': attempt_num,
            'attempt_detail': attempt_detail,
            'task_data': task_data,
            'dataset': dataset,
            'subset': subset,
            'full_prompt': full_prompt or {'system': system_content, 'user': user_content}
        }
    

    
    def run_subset(self, subset_name: str, dataset: str = "arc-agi-1", limit: Optional[int] = None) -> List[Dict]:
        """Run all tasks in a subset with true parallelization at the attempt level"""
        try:
            tasks = self.task_loader.load_tasks_from_subset(subset_name, dataset)
            if limit:
                tasks = tasks[:limit]
            total_tasks = len(tasks)
            
            # Validate task data integrity to prevent corruption issues
            validated_tasks = ARCTaskValidator.validate_tasks(tasks)
            
            if len(validated_tasks) != total_tasks:
                print(f"⚠️ {total_tasks - len(validated_tasks)} tasks failed validation, using {len(validated_tasks)} valid tasks")
                tasks = validated_tasks
                total_tasks = len(tasks)
            
            print(f"✅ Task validation complete: {total_tasks} valid tasks")
            
        except Exception as e:
            print(f"Error loading tasks: {e}")
            return [], None
        
        total_attempts = total_tasks * self.max_attempts
        
        # Get sampling parameters for display
        sampling_params = self.get_sampling_parameters()
        
        print(f"\nRunning {total_tasks} tasks from {dataset}/{subset_name}")
        print(f"Model: {self.model}")
        print(f"API: All Attempts Mode ({self.max_attempts} attempts per task)")
        print(f"Mode: True parallelization - {total_attempts} total attempts")
        
        if self.max_workers > 1:
            print(f"Parallelization: ENABLED ({self.max_workers} workers)")
            # Show the new scheduling strategy
            concurrent_tasks = max(1, self.max_workers // self.max_attempts)
            if concurrent_tasks == 1:
                print(f"Scheduling: Task-by-task (1 task × {self.max_attempts} attempts = {self.max_attempts} workers used)")
            else:
                print(f"Scheduling: Batched ({concurrent_tasks} tasks × {self.max_attempts} attempts = {concurrent_tasks * self.max_attempts} workers used)")
                if self.max_workers > concurrent_tasks * self.max_attempts:
                    unused_workers = self.max_workers - (concurrent_tasks * self.max_attempts)
                    print(f"Note: {unused_workers} workers will be idle due to batching strategy")
        else:
            print("Parallelization: DISABLED (sequential execution)")
        
        # No infrastructure timeouts to avoid GPU overload
        print("")
        print("✅ No infrastructure timeouts - requests complete naturally to avoid GPU overload")
        print("")
        
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
        
        # Create attempt jobs with task-by-task prioritization for better GPU caching
        # This approach prioritizes completing tasks while still utilizing all workers
        attempt_jobs = []
        
        # Calculate how many tasks can have all attempts running simultaneously
        concurrent_tasks = max(1, self.max_workers // self.max_attempts)
        
        # Group tasks into batches that can run concurrently
        for batch_start in range(0, len(tasks), concurrent_tasks):
            batch_end = min(batch_start + concurrent_tasks, len(tasks))
            batch_tasks = tasks[batch_start:batch_end]
            
            # For this batch, interleave attempts to maximize prefix caching
            # while still allowing parallelization across tasks in the batch
            for attempt_num in range(self.max_attempts):
                for task_idx_in_batch, (task_id, task_data) in enumerate(batch_tasks):
                    task_idx = batch_start + task_idx_in_batch
                    attempt_jobs.append((task_idx, task_id, task_data, attempt_num))
        
        # Track results by task - use thread-safe defaultdict to prevent race conditions
        from collections import defaultdict
        task_results = defaultdict(lambda: {'attempts': [], 'task_data': None})
        
        # Initialize task results with task data and prompts (create once per task)
        for task_id, task_data in tasks:
            task_results[task_id]['task_data'] = task_data
            system_content, user_content = self.create_prompt(task_data)
            task_results[task_id]['full_prompt'] = {'system': system_content, 'user': user_content}
        
        completed_attempts = 0
        completed_tasks = 0
        count_lock = threading.Lock()
        
        def attempt_wrapper(task_idx, task_id, task_data, attempt_num):
            nonlocal completed_attempts, completed_tasks
            attempt_start = time.time()
            try:
                # Get the pre-created prompt for this task
                full_prompt = task_results[task_id]['full_prompt']
                # No infrastructure timeout - let requests complete to avoid GPU overload
                result = self.run_single_attempt(
                    task_id,
                    task_data,
                    attempt_num,
                    dataset,
                    subset_name,
                    full_prompt
                )
                attempt_duration = time.time() - attempt_start
                if attempt_duration > 60:  # Log slow attempts
                    print(f"🐌 Slow attempt: {task_id} attempt {attempt_num + 1} took {attempt_duration:.1f}s")
                
                with count_lock:
                    # Store attempt result - use thread-safe access
                    if task_id in task_results:
                        task_results[task_id]['attempts'].append(result['attempt_detail'])
                        # Prompt is already stored at task level during initialization
                        completed_attempts += 1
                        
                        # Check if task is complete
                        if len(task_results[task_id]['attempts']) == self.max_attempts:
                            completed_tasks += 1
                            # Calculate and display task summary
                            self._display_task_summary(task_id, task_results[task_id])
                            
                            # Save task result immediately when it's complete
                            attempts = sorted(task_results[task_id]['attempts'], key=lambda x: x['attempt_number'])
                            valid_attempts = [attempt for attempt in attempts if isinstance(attempt, dict) and 'attempt_number' in attempt]
                            
                            # Trim failed attempts to reduce file size
                            trimmed_attempts = [self._trim_failed_attempt(attempt) for attempt in valid_attempts]
                            
                            # Only include raw_responses for non-trimmed attempts
                            all_responses = []
                            for i, attempt in enumerate(trimmed_attempts):
                                if attempt.get('data_trimmed', False):
                                    all_responses.append(None)  # No raw response for trimmed attempts
                                else:
                                    all_responses.append(valid_attempts[i].get('raw_response'))
                            
                            task_result = {
                                'task_id': task_id,
                                'model': self.model,
                                'api_type': 'chat_completions_all_attempts',
                                'dataset': dataset,
                                'subset': subset_name,
                                'attempt_details': trimmed_attempts,
                                'all_responses': all_responses,
                                'tokens_used': sum(attempt.get('input_tokens', 0) + attempt.get('output_tokens', 0) for attempt in trimmed_attempts),
                                'request_cost': sum(attempt.get('attempt_cost', 0.0) for attempt in trimmed_attempts),
                                'max_attempts': self.max_attempts,
                                'api_success': True,
                                'task_data': task_data,
                                'full_prompt': task_results[task_id].get('full_prompt')  # Store prompt once per task
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
            
            # Wait for all attempts to complete, reporting progress periodically
            start_time = time.time()
            
            from concurrent.futures import as_completed
            completed_count = 0
            remaining = set(futures)
            progress_interval = 15.0
            
            while remaining:
                time_elapsed = time.time() - start_time
                try:
                    # Use a short timeout so we can log periodic progress
                    for future in as_completed(list(remaining), timeout=progress_interval):
                        remaining.discard(future)
                        completed_count += 1
                        try:
                            _ = future.result()
                        except Exception as future_e:
                            print(f"🚨 Future #{completed_count} error: {future_e}")
                    # Periodic progress log
                    done_now = total_attempts - len(remaining)
                    print(f"⏳ Progress: {done_now}/{total_attempts} attempts done; {len(remaining)} remaining")
                except Exception:
                    # No futures completed in this window; print a heartbeat
                    done_now = total_attempts - len(remaining)
                    print(f"⏳ No completions in last {progress_interval:.0f}s — {done_now}/{total_attempts} done; {len(remaining)} remaining")
                    continue
            
            print(f"✅ All {total_attempts} attempts completed")
            
            # Check final status - handle cancelled futures properly
            successful_attempts = 0
            failed_attempts = 0
            cancelled_attempts = 0
            
            for future in futures:
                if not future.done():
                    continue
                elif future.cancelled():
                    cancelled_attempts += 1
                else:
                    try:
                        exception = future.exception()
                        if exception is None:
                            successful_attempts += 1
                        else:
                            failed_attempts += 1
                    except Exception:
                        # Handle any other edge cases
                        failed_attempts += 1
            
            if failed_attempts > 0:
                print(f"❌ {failed_attempts} attempts failed out of {total_attempts} total")
            
            if cancelled_attempts > 0:
                print(f"🛑 {cancelled_attempts} attempts were cancelled due to timeout")
            
            print(f"📊 Final status: {successful_attempts} successful, {failed_attempts} failed, {cancelled_attempts} cancelled")
        
        # All attempts should complete now without infrastructure timeouts
        
        # Convert task_results to the expected format for summary (including partial tasks)
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
                
                # All tasks should have complete attempts now
                api_type = 'chat_completions_all_attempts'
                
                # Trim failed attempts to reduce file size for summary
                trimmed_attempts = [self._trim_failed_attempt(attempt) for attempt in valid_attempts]
                
                # Only include raw_responses for non-trimmed attempts
                all_responses = []
                for i, attempt in enumerate(trimmed_attempts):
                    if attempt.get('data_trimmed', False):
                        all_responses.append(None)  # No raw response for trimmed attempts
                    else:
                        all_responses.append(valid_attempts[i].get('raw_response'))
                
                result = {
                    'task_id': task_id,
                    'model': self.model,
                    'api_type': api_type,
                    'dataset': dataset,
                    'subset': subset_name,
                    'attempt_details': trimmed_attempts,
                    'all_responses': all_responses,
                    'tokens_used': sum(attempt.get('input_tokens', 0) + attempt.get('output_tokens', 0) for attempt in trimmed_attempts),
                    'request_cost': sum(attempt.get('attempt_cost', 0.0) for attempt in valid_attempts),
                    'max_attempts': self.max_attempts,
                    'actual_attempts': len(valid_attempts),  # Track how many were actually completed
                    'api_success': True,
                    'task_data': task_data
                }
                results.append(result)
                # Note: save_result() is now called when each task completes, not here
            else:
                print(f"⚠️ Task {task_id} has no valid attempts - skipping")
        
        summary_filepath = self.save_summary(results, subset_name, dataset)
        
        # Report any large file errors at the end of the run
        self._report_large_file_errors()
        
        return results, summary_filepath
    
    def _display_task_summary(self, task_id: str, task_result: Dict):
        """Display a brief summary of a completed task"""
        attempts = task_result['attempts']
        
        # Calculate key stats
        test_correct_attempts = sum(1 for attempt in attempts if attempt.get('test_correct', False))
        train_perfect_attempts = sum(1 for attempt in attempts if attempt.get('train_accuracy', 0.0) == 1.0)
        # Align with Min 1 Train logic: task-level, has partial but not perfect training
        has_perfect_train = any(attempt.get('train_accuracy', 0.0) == 1.0 for attempt in attempts)
        has_partial_train = any(0 < attempt.get('train_accuracy', 0.0) < 1.0 for attempt in attempts)
        task_has_partial_train = has_partial_train and not has_perfect_train
        
        # Calculate issues in timeline order
        api_timeouts = sum(1 for attempt in attempts if attempt.get('api_timeout', False))
        api_failures = sum(1 for attempt in attempts if not attempt.get('api_success', True))
        empty_responses = sum(1 for attempt in attempts if attempt.get('empty_response', False))
        max_length_hits = sum(1 for attempt in attempts if attempt.get('hit_max_tokens', False))
        no_code_extracted = sum(1 for attempt in attempts if not attempt.get('program_extracted', False))
        train_transductive_attempts = sum(1 for attempt in attempts if attempt.get('is_train_transductive', False))
        test_transductive_attempts = sum(1 for attempt in attempts if attempt.get('is_test_transductive', False))
        train_exec_errors = sum(1 for attempt in attempts if attempt.get('train_exec_errors', 0) > 0)
        train_exec_timeouts = sum(1 for attempt in attempts if attempt.get('train_exec_timeouts', 0) > 0)
        test_exec_errors = sum(1 for attempt in attempts if attempt.get('test_exec_error', False))
        test_exec_timeouts = sum(1 for attempt in attempts if attempt.get('test_exec_timeout', False))
        
        # Find best attempt
        best_attempt = max(attempts, key=lambda x: (x.get('test_correct', False), x.get('train_accuracy', 0.0)))
        
        # Build summary
        partial_indicator = "train-partial" if task_has_partial_train else "no-partial"
        summary = f"✅ {task_id}: {test_correct_attempts}/{len(attempts)} test-correct, {train_perfect_attempts} train-perfect, {partial_indicator}"
        
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
        if train_transductive_attempts > 0:
            issues.append(f"{train_transductive_attempts} train-transductive")
        if test_transductive_attempts > 0:
            issues.append(f"{test_transductive_attempts} test-transductive")
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
        if best_attempt.get('test_correct', False):
            summary += f" (best: {best_attempt.get('train_accuracy', 0.0):.1%} train)"
        else:
            summary += f" (best: {best_attempt.get('train_accuracy', 0.0):.1%} train, test-failed)"
        
        print(summary)
    
    def _trim_failed_attempt(self, attempt: Dict) -> Dict:
        """Trim data from attempts that failed to execute (exec errors, no code, API failures)
        
        Keep only essential metadata for failed attempts to reduce file size.
        """
        # Check if this attempt should be trimmed (execution failures)
        should_trim = (
            # Execution errors
            attempt.get('train_exec_errors', 0) > 0 or
            attempt.get('test_exec_error', False) or
            # No code extracted
            not attempt.get('program_extracted', False) or
            # API failures
            attempt.get('api_timeout', False) or
            not attempt.get('api_success', True) or
            attempt.get('empty_response', False)
        )
        
        if not should_trim:
            # Keep everything for attempts that ran successfully
            return attempt
        
        # Create trimmed version for failed attempts
        trimmed = {
            # Essential metadata
            'attempt_number': attempt.get('attempt_number'),
            'timestamp': attempt.get('timestamp'),
            'input_tokens': attempt.get('input_tokens', 0),
            'output_tokens': attempt.get('output_tokens', 0),
            'attempt_cost': attempt.get('attempt_cost', 0.0),
            
            # Key flags
            'program_extracted': attempt.get('program_extracted', False),
            'api_success': attempt.get('api_success', True),
            'api_timeout': attempt.get('api_timeout', False),
            'empty_response': attempt.get('empty_response', False),
            'hit_max_tokens': attempt.get('hit_max_tokens', False),
            
            # Transduction flags
            'is_train_transductive': attempt.get('is_train_transductive', False),
            'is_test_transductive': attempt.get('is_test_transductive', False),
            
            # Error counts instead of full arrays
            'train_exec_errors': attempt.get('train_exec_errors', 0),
            'train_exec_timeouts': attempt.get('train_exec_timeouts', 0),
            'test_exec_error': attempt.get('test_exec_error', False),
            'test_exec_timeout': attempt.get('test_exec_timeout', False),
            
            # Accuracy metrics (zeros for failed attempts)
            'train_accuracy': attempt.get('train_accuracy', 0.0),
            'test_correct': attempt.get('test_correct', False),
            'test_correct_count': attempt.get('test_correct_count', 0),
            
            # Truncated error message if present
            'error_summary': str(attempt.get('error', ''))[:200] if attempt.get('error') else None,
            
            # Keep empty fields that downstream code expects (for compatibility)
            'program': '',  # Empty instead of missing to avoid KeyErrors
            'raw_response': None,  # Explicit None instead of missing
            'sampling_params': {},  # Empty dict instead of missing
            'train_results': [],  # Empty list instead of missing  
            'test_results': [],  # Empty list instead of missing
            'test_predicted': None,  # Explicit None instead of missing
            
            # Mark as trimmed for transparency
            'data_trimmed': True,
            'trim_reason': 'execution_failure'
        }
        
        # Keep required fields as empty/None instead of dropping them entirely
        # This maintains compatibility with downstream code while still reducing size
        
        return trimmed
    
    def save_result(self, result: Dict):
        """Save individual task result"""
        self.result_processor.save_result(result)
    
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
                'all_test_correct': 0.0,
                'all_train_correct': 0.0,
                'min1_train_correct': 0.0,
                'min1_code_success': 0.0,
                'passes_examples_all': 0.0,
                'max_length_responses': 0.0,
                'timeout_responses': 0.0,
                'api_failure_responses': 0.0,
            }
        
        # All tasks should complete now
        api_type_summary = 'chat_completions_all_attempts'
        complete_tasks = total_tasks
        
        # Build compact results for summary (strip heavy fields like raw_response/program/all_responses)
        results_for_summary = []
        for r in results:
            try:
                compact = {k: v for k, v in r.items() if k != 'all_responses'}
                compact_attempts = []
                for att in r.get('attempt_details', []):
                    if isinstance(att, dict):
                        compact_att = {k: v for k, v in att.items() if k != 'raw_response'}
                        # Keep 'program' field but ensure it's empty string for failed attempts (avoid KeyError)
                        if 'program' not in compact_att:
                            compact_att['program'] = ''
                        compact_attempts.append(compact_att)
                    else:
                        compact_attempts.append(att)
                compact['attempt_details'] = compact_attempts
                results_for_summary.append(compact)
            except Exception:
                # Fallback to original entry on unexpected structure
                results_for_summary.append(r)

        # Create summary with compact results for later aggregation
        summary = {
            'timestamp': timestamp,
            'dataset': dataset,
            'subset': subset_name,
            'model': self.model,
            'api_type': api_type_summary,
            'run_number': self.run_number,
            'total_tasks': total_tasks,
            'complete_tasks': complete_tasks,
            'successful_api_calls': successful_api_calls,
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost,
            'metrics': percentage_metrics,
            'results': results_for_summary,  # Compact results to keep summary small
            'task_ids': [result['task_id'] for result in results]  # Also include task IDs for convenience
        }
        
        # Simple filename without timeout indicator
        if self.run_number > 0:
            filename = f"{timestamp}_summary_{dataset}_{subset_name}_simple_run{self.run_number}.json"
        else:
            filename = f"{timestamp}_summary_{dataset}_{subset_name}_simple.json"
        
        filepath = self.logs_dir / filename
        
        # Sanitize for JSON and check file size before saving summary
        safe_summary = ResponseSerializer.make_json_safe(summary)
        should_save, size_error = self._check_file_size_before_save(safe_summary, filename)
        
        if not should_save:
            print(f"🚨 LARGE SUMMARY FILE ERROR: {size_error}")
            print(f"   Summary will not be saved to prevent disk issues")
            print(f"   Consider reducing data collection or increasing size limit")
            # Still continue with printing the summary statistics
        else:
            try:
                with open(filepath, 'w') as f:
                    json.dump(safe_summary, f, indent=2)
            except Exception as e:
                print(f"🚨 SUMMARY FILE I/O ERROR: {e}")
                should_save = False  # Mark as unsaved
        
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
            print(f"\n📊 CORE METRICS:")
            print(f"  Pass@2 (Weighted Voting): {percentage_metrics['weighted_voting_pass2']:.1%}")
            print(f"  Pass@2 (Train Majority):  {percentage_metrics['train_majority_pass2']:.1%}")
            print(f"  Oracle (Best Attempt):    {percentage_metrics['all_test_correct']:.1%}")
            print(f"  All Train Correct:        {percentage_metrics['all_train_correct']:.1%}")
            print(f"  Min 1 Train Correct:      {percentage_metrics['min1_train_correct']:.1%}")
            print(f"  Min 1 Code Success:       {percentage_metrics['min1_code_success']:.1%}")
            print(f"  Max Length Responses:     {percentage_metrics['max_length_responses']:.1%}")
            print(f"  Timeout Responses:        {percentage_metrics['timeout_responses']:.1%}")
            print(f"  API Failure Responses:    {percentage_metrics['api_failure_responses']:.1%}")
        
        # Update final message based on whether file was actually saved
        if should_save:
            print(f"\nResults saved to: {filepath}")
            return filepath
        else:
            final_message = "⚠️ Results NOT SAVED due to size limit - see errors above"
            print(f"\n{final_message}")
            return None
    
    def run_repeated_subset(self, subset_name: str, dataset: str = "arc-agi-1", limit: Optional[int] = None, repeat_runs: int = 3) -> List[List[Dict]]:
        """Run the same subset multiple times with completely independent runs"""
        print(f"\nRunning {repeat_runs} repeated tests of {dataset}/{subset_name}")
        print(f"Model: {self.model}")
        print(f"{self.max_attempts} attempts")
        print("="*70)
        
        # No infrastructure timeouts to avoid GPU overload
        print("")
        print("✅ No infrastructure timeouts - requests complete naturally to avoid GPU overload")
        print("")
        
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
                unsafe_executor=(self.executor_type == "unrestricted"),
                lora_adapter=self.lora_adapter
            )
            
            try:
                # Run the subset and let it save its own results
                results, summary_filepath = runner.run_subset(subset_name, dataset, limit)
                print(f"\n✅ COMPLETED RUN {run_num}/{repeat_runs}")
                
                # Store the actual summary file path that was created
                run_files.append(summary_filepath)
                
            except Exception as e:
                print(f"\n❌ RUN {run_num} FAILED: {e}")
                try:
                    import traceback as _tb
                    _tb.print_exc()
                except Exception:
                    pass
                run_files.append(None)  # Mark as failed
            finally:
                # Explicit cleanup to prevent state leakage
                # ArcTester uses singleton class variables that persist across instances
                # Without cleanup, second run reuses stale executor context causing systematic failures
                with self._cleanup_lock:  # Thread-safe cleanup
                    try:
                        ArcTester.cleanup_executor()  # Fix: Clean up singleton state
                        if run_num < repeat_runs:  # Only print for non-final runs
                            print(f"🧹 Cleaned up executor state after run {run_num}")
                    except Exception as cleanup_e:
                        print(f"⚠️ Failed to cleanup executor state: {cleanup_e}")
                del runner
                gc.collect()
        
        # Load results from files and calculate aggregate statistics
        all_run_results = self._load_and_aggregate_results(run_files, subset_name, dataset, repeat_runs)
        
        # Report any large file errors that occurred across all runs
        # Note: Each individual run reports its own errors, but this provides a summary
        self._report_large_file_errors()
        
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
                    'all_test_correct': 0.0,
                    'all_train_correct': 0.0,
                    'min1_train_correct': 0.0,
                    'min1_code_success': 0.0,
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
                'all_test_correct': [s['all_test_correct'] for s in valid_runs],
                'all_train_correct': [s['all_train_correct'] for s in valid_runs],
                'min1_train_correct': [s['min1_train_correct'] for s in valid_runs],
                'min1_code_success': [s['min1_code_success'] for s in valid_runs],
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
            
            # Check file size before saving aggregate summary
            should_save, size_error = self._check_file_size_before_save(aggregate_summary, filename)
            
            if not should_save:
                print(f"🚨 LARGE AGGREGATE SUMMARY ERROR: {size_error}")
                print(f"   Aggregate summary will not be saved to prevent disk issues")
                print(f"   Consider reducing data collection or increasing size limit")
                filepath = None  # Mark as unsaved
            else:
                try:
                    with open(filepath, 'w') as f:
                        json.dump(aggregate_summary, f, indent=2)
                except Exception as e:
                    print(f"🚨 AGGREGATE SUMMARY FILE I/O ERROR: {e}")
                    filepath = None  # Mark as unsaved
            
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
            print("-" * 98)
            print(f"{'Run':<4} {'Tasks':<6} {'Weighted':<10} {'Train-Maj':<10} {'Oracle':<8} {'All-Train':<10} {'Min1-Train':<11} {'Code-Success':<12} {'Max-Len':<8}")
            print("-" * 98)
            
            for stats_run in run_stats:
                if stats_run['total_tasks'] > 0:
                    print(f"{stats_run['run_number']:<4} {stats_run['total_tasks']:<6} "
                          f"{stats_run['weighted_voting_pass2']:<10.1%} {stats_run['train_majority_pass2']:<10.1%} "
                          f"{stats_run['all_test_correct']:<8.1%} {stats_run['all_train_correct']:<10.1%} "
                          f"{stats_run['min1_train_correct']:<11.1%} "
                          f"{stats_run['min1_code_success']:<12.1%} {stats_run['max_length_responses']:<8.1%}")
            
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
            
            if filepath:
                print(f"Aggregate results saved to: {filepath}")
            else:
                print("⚠️ Aggregate results NOT SAVED due to size limit - see errors above")
        else:
            print("\n❌ No valid run statistics to aggregate")

def main():
    parser = argparse.ArgumentParser(description="Run ARC tasks with all-attempts, rolling execution, and voting-based evaluation")
    parser.add_argument("--dataset", default="arc-agi-1", choices=["arc-agi-1", "arc-agi-1r", "arc-agi-2"], help="Dataset to use")
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
    parser.add_argument("--qwen-no-think", action="store_true", help="Disable thinking for Qwen models (Note: Not supported by DashScope commercial models)")
    parser.add_argument("--unsafe-executor", action="store_true", 
                        help="⚠️  UNSAFE: Use unrestricted executor (no Docker sandboxing). Generated code runs directly on your system. SECURITY RISK!")
    parser.add_argument("--prompt_version", type=str, default="soar", help="Version of prompts to use")
    parser.add_argument("--lora-adapter", type=str, help="LORA adapter name to load on sglang server (e.g., 'ckpt-1057')")
    
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
        unsafe_executor=args.unsafe_executor,
        lora_adapter=args.lora_adapter
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
                ArcTester.cleanup_executor()
            except Exception as e:
                print(f"⚠️ Final cleanup warning: {e}")

if __name__ == "__main__":
    main() 