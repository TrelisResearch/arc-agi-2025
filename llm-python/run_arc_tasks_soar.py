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
        self.logs_dir = Path("llm-python/logs")
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
    
    def call_chat_completions_api(self, messages: List[Dict]) -> Dict:
        """Call the OpenAI Chat Completions API"""
        try:
            kwargs = {
                "model": self.model,
                "messages": messages
            }
            
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            
            # Add reasoning parameters for OpenRouter
            if self.base_url and "openrouter" in self.base_url.lower():
                reasoning_tokens = {"low": 2000, "medium": 8000, "high": 32000}
                if self.reasoning_effort in reasoning_tokens:
                    if "gemini" in self.model.lower():
                        kwargs["extra_body"] = {"reasoning": {"max_tokens": reasoning_tokens[self.reasoning_effort]}}
                    else:
                        if self.max_tokens is None:
                            kwargs["max_tokens"] = reasoning_tokens[self.reasoning_effort]
            
            # Add Qwen-specific parameters
            if "qwen" in self.model.lower() and self.base_url:
                if self.qwen_no_think:
                    qwen_params = {
                        "top_p": 0.8,
                        "extra_body": {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}
                    }
                    if self.temperature is None:
                        qwen_params["temperature"] = 0.7
                    kwargs.update(qwen_params)
                else:
                    qwen_params = {
                        "top_p": 0.95,
                        "extra_body": {"top_k": 20}
                    }
                    if self.temperature is None:
                        qwen_params["temperature"] = 0.6
                    kwargs.update(qwen_params)
            
            response = self.client.chat.completions.create(**kwargs)
            return response
            
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
    
    def run_task_all_attempts(self, task_id: str, task_data: Dict, 
                            dataset: str = None, subset: str = None) -> Dict:
        """Run all attempts for a single ARC task"""
        total_cost = 0.0
        total_tokens = 0
        all_responses = []
        attempt_details = []
        system_content, user_content = self.create_prompt(task_data)
        
        for attempt in range(self.max_attempts):
            if self.debug:
                print(f"🔍 Task {task_id} attempt {attempt + 1}/{self.max_attempts}")
            
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
            
            for retry_attempt in range(3):
                try:
                    response = execute_with_timeout(self.call_chat_completions_api, conversation_history, timeout=30)
                    api_call_successful = True
                    break
                except Exception as e:
                    error = str(e)
                    if retry_attempt < 2:
                        time.sleep(2)
                    else:
                        timed_out = True
            
            # Track costs
            usage = getattr(response, 'usage', None)
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            input_rate, output_rate = self.get_model_pricing(self.model)
            attempt_cost = (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate
            total_cost += attempt_cost
            total_tokens += usage.total_tokens if usage else 0
            
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
            
            if self.debug:
                print(f"🔍 Attempt {attempt + 1}: Program={len(program)} chars, Train={train_accuracy:.2f}, Test={test_correct}")
            
            # Store attempt details
            attempt_details.append({
                'attempt_number': attempt + 1,
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
                'api_success': api_call_successful,
                'timed_out': timed_out,
                'error': error
            })
            all_responses.append(serialize_response(response))
        
        self._update_costs(total_cost, total_tokens)
        return {
            'task_id': task_id,
            'model': self.model,
            'api_type': 'chat_completions_all_attempts',
            'dataset': dataset,
            'subset': subset,
            'attempt_details': attempt_details,
            'all_responses': all_responses,
            'tokens_used': total_tokens,
            'request_cost': total_cost,
            'max_attempts': self.max_attempts,
            'api_success': True,
            'task_data': task_data
        }
    
    def report_metrics(self, results, upto_attempt):
        """Calculate and report metrics for completed attempts"""
        metrics = calculate_task_metrics(results, upto_attempt, self.max_tokens)
        print(format_metrics_display(metrics, upto_attempt))
    
    def run_subset(self, subset_name: str, dataset: str = "arc-agi-1", limit: Optional[int] = None) -> List[Dict]:
        """Run all tasks in a subset with rolling execution and layerwise reporting"""
        try:
            tasks = self.task_loader.load_tasks_from_subset(subset_name, dataset)
            if limit:
                tasks = tasks[:limit]
            total_tasks = len(tasks)
        except Exception as e:
            print(f"Error loading tasks: {e}")
            return []
        
        print(f"\nRunning {total_tasks} tasks from {dataset}/{subset_name}")
        print(f"Model: {self.model}")
        print(f"API: All Attempts Mode (max {self.max_attempts} attempts)")
        print("Mode: All attempts, rolling execution, voting-based evaluation")
        
        if self.max_workers > 1:
            print(f"Parallelization: ENABLED ({self.max_workers} workers)")
        else:
            print("Parallelization: DISABLED (sequential execution)")
        print("-" * 50)
        
        results = [None] * total_tasks
        completed_count = 0
        count_lock = threading.Lock()
        
        def task_wrapper(idx, task_id, task_data):
            nonlocal completed_count
            try:
                result = self.run_task_all_attempts(task_id, task_data, dataset, subset_name)
                self.save_result(result)
                results[idx] = result
                
                # Simple progress indicator
                with count_lock:
                    completed_count += 1
                    print(f"✅ {completed_count}/{total_tasks} tasks completed ({task_id})")
            except Exception as e:
                # Handle any unexpected errors in the worker
                error_result = {
                    'task_id': task_id,
                    'model': self.model,
                    'api_type': 'chat_completions_all_attempts',
                    'dataset': dataset,
                    'subset': subset_name,
                    'error': str(e),
                    'api_success': False,
                    'attempt_details': [],
                    'all_responses': [],
                    'tokens_used': 0,
                    'request_cost': 0.0,
                    'max_attempts': self.max_attempts,
                    'task_data': task_data
                }
                results[idx] = error_result
                
                with count_lock:
                    completed_count += 1
                    print(f"❌ {completed_count}/{total_tasks} tasks failed ({task_id}): {e}")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(task_wrapper, idx, task_id, task_data) for idx, (task_id, task_data) in enumerate(tasks)]
            
            print(f"🚀 Started {total_tasks} tasks with {self.max_workers} workers")
            
            # Simple approach: just wait for all futures with timeout
            max_wait_time = 600  # 10 minutes total timeout
            start_time = time.time()
            
            try:
                # Wait for all futures to complete
                for future in futures:
                    remaining_time = max_wait_time - (time.time() - start_time)
                    if remaining_time <= 0:
                        print("⏰ Timeout reached, some tasks may not have completed")
                        break
                    future.result(timeout=remaining_time)
                print(f"✅ All {total_tasks} tasks completed")
            except Exception as e:
                print(f"⚠️ Some tasks may have failed or timed out: {e}")
            
            # Check final status
            completed_count = sum(1 for future in futures if future.done() and not future.exception())
            failed_count = sum(1 for future in futures if future.done() and future.exception())
            
            if failed_count > 0:
                print(f"❌ {failed_count} tasks failed:")
                for i, future in enumerate(futures):
                    if future.done() and future.exception():
                        print(f"   Task {i}: {future.exception()}")
            
            print(f"📊 Final status: {completed_count} successful, {failed_count} failed out of {total_tasks} total")
        
        self.save_summary(results, subset_name, dataset)
        return results
    
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
        correct_tasks = sum(1 for r in results if r.get('score', {}).get('correct', False))
        api_successes = [r for r in results if r.get('api_success', True)]
        successful_api_calls = len(api_successes)
        
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
        print(f"Results saved to: {filepath}")
    
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
                    'all_timeouts': 0.0
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
                'all_timeouts': [s['all_timeouts'] for s in valid_runs]
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