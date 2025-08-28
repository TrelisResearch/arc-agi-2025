#!/usr/bin/env python3

import json
import argparse
import datetime
import time
import threading
import traceback
import numpy as np
import os
import re
from pathlib import Path
from typing import Dict, List, NotRequired, Optional, TypedDict, Union, Tuple, Any
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import requests

from llm_python.datasets.collector import SoarDatasetCollector
from llm_python.utils.code import is_random
from llm_python.utils.task_loader import TaskData, get_task_loader
from llm_python.utils.arc_tester import ArcTester
from llm_python.utils.prompt_utils import create_arc_prompt, extract_python_code
from llm_python.utils.metrics_utils import (
    calculate_task_metrics,
    metrics_to_percentages,
)
from llm_python.transduction.code_classifier import CodeTransductionClassifier
from llm_python.utils.prompt_loader import PromptLoader
from llm_python.utils.serialization import ResponseSerializer
from llm_python.utils.api_client import ARCAPIClient
from llm_python.utils.validator import ARCTaskValidator
from llm_python.datasets.schema import ProgramSample

load_dotenv()


# Type definitions for attempt results
class TrainResult(TypedDict):
    """Result of running program on a single training example"""

    predicted: Optional[List[List[int]]]  # Grid output or None if error/timeout
    expected: List[List[int]]  # Expected output grid
    correct: bool  # Whether prediction matches expected
    error: Optional[str]  # Error message if execution failed
    timed_out: bool  # Whether execution timed out


class TestResult(TypedDict):
    """Result of running program on a single test example"""

    test_idx: int  # Index of test case
    predicted: Optional[List[List[int]]]  # Grid output or None if error/timeout
    expected: List[List[int]]  # Expected output grid
    correct: bool  # Whether prediction matches expected
    error: Optional[str]  # Error message if execution failed
    timed_out: bool  # Whether execution timed out


class AttemptDetail(TypedDict):
    """Detailed result from a single attempt to solve an ARC task"""

    # Basic metadata
    attempt_number: int  # 1-indexed attempt number
    timestamp: str  # ISO format timestamp

    # Token and cost tracking
    input_tokens: int  # Number of input tokens
    output_tokens: int  # Number of output tokens
    attempt_cost: float  # Cost in dollars for this attempt

    # Program extraction and validation
    program_extracted: bool  # Whether code was extracted from response
    program: str  # Extracted Python code
    outputs_valid: bool  # Whether all outputs pass ARC validation (30x30, proper format)

    # Transduction detection
    is_transductive: bool  # Whether program hardcodes outputs
    transduction_confidence: float  # Confidence score for transduction classification
    transduction_reason: str  # Why it's considered transductive

    # Training results
    train_results: List[TrainResult]  # Results for each training example
    train_accuracy: float  # Fraction of training examples correct
    train_exec_errors: int  # Number of training examples with execution errors
    train_exec_timeouts: int  # Number of training examples that timed out

    # Test results (supports multiple test cases)
    test_predicted: Union[
        None, List[List[int]], Tuple[Optional[List[List[int]]], ...]
    ]  # Predictions: None, single grid, or list of grids
    test_results: List[TestResult]  # Detailed results for each test case
    test_correct: bool  # True if ALL test cases are correct
    test_correct_count: int  # Number of correct test cases
    test_exec_error: bool  # Whether any test case had execution error
    test_exec_timeout: bool  # Whether any test case timed out

    # Legacy test fields (backwards compatibility - use first test case)
    test_error: Optional[str]  # Error from first test case
    test_timed_out: bool  # Timeout status from first test case

    # API response and metadata
    raw_response: Optional[Dict[str, Any]]  # Serialized API response
    sampling_params: Dict[str, Any]  # Parameters used for API call
    api_success: bool  # Whether API call succeeded
    api_timeout: bool  # Whether API call timed out
    empty_response: bool  # Whether response was empty
    hit_max_tokens: bool  # Whether response hit token limit
    error: Optional[str]  # Error message if API call failed

    # Metrics aliases (for compatibility with metrics_utils)
    all_test_correct: bool  # Alias for test_correct
    code_ran: bool  # Alias for program_extracted

    # Optional trimming metadata (for reducing file sizes)
    error_summary: NotRequired[
        Optional[str]
    ]  # Truncated error message for failed attempts
    data_trimmed: NotRequired[bool]  # Whether attempt data was trimmed
    trim_reason: NotRequired[str]  # Reason for trimming data


class SingleAttemptResult(TypedDict):
    """Complete result from run_single_attempt method"""

    task_id: str  # Unique task identifier
    attempt_num: int  # 0-indexed attempt number
    attempt_detail: AttemptDetail  # Detailed attempt results
    task_data: TaskData  # Original task data
    dataset: Optional[str]  # Dataset name (e.g., "arc-agi-1")
    subset: Optional[str]  # Subset name (e.g., "training")
    full_prompt: Dict[str, str]  # System and user prompts used


class ARCTaskRunnerSimple:
    """ARC task runner with all-attempts, rolling execution, and voting-based evaluation

    Supports multiple API endpoints including:
    - OpenAI/OpenRouter: Standard chat completions with reasoning models
    - DashScope: Alibaba's commercial Qwen models with thinking_budget parameter

    Note: DashScope commercial Qwen models always use thinking mode and don't support
    enable_thinking=False. Use thinking_budget parameter to control reasoning depth.
    """

    def __init__(
        self,
        model: str = "gpt-4.1-nano",
        max_workers: int = 1,
        rate_limit_delay: float = 0.0,
        max_attempts: int = 8,
        run_number: int = 0,
        base_url: Optional[str] = None,
        debug: bool = False,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        reasoning_effort: str = "low",
        qwen_no_think: bool = False,
        prompt_version: str = "soar",
        unsafe_executor: bool = False,
        lora_adapter: Optional[str] = None,
        sample_name: Optional[str] = None,
        no_transductive_penalty: bool = False,
        parquet_output_dir: Optional[str] = None,
    ):
        # Core configuration
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        self.max_attempts = max_attempts
        self.run_number = run_number
        self.debug = debug
        self.prompt_version = prompt_version
        # Use custom output directory if provided, otherwise use default
        output_dir = Path(parquet_output_dir) if parquet_output_dir else None
        self.dataset_collector = SoarDatasetCollector(sample_name, output_dir=output_dir)
        self.no_transductive_penalty = no_transductive_penalty
        
        # Standard API timeout for network safety, no infrastructure timeouts
        api_timeout = 600  # 10 minutes for network safety only

        # Optional global timeout from environment variable
        global_timeout = None
        if "GLOBAL_TIMEOUT" in os.environ:
            try:
                global_timeout = int(os.environ["GLOBAL_TIMEOUT"])
                print(
                    f"⏰ Global timeout set to {global_timeout}s via GLOBAL_TIMEOUT environment variable"
                )
            except ValueError:
                print(
                    "⚠️ Invalid GLOBAL_TIMEOUT value - must be integer seconds. Ignoring."
                )

        self.global_timeout = global_timeout
        self.inactivity_timeout = 600  # Default inactivity timeout of 5 minutes

        # Warn about safety settings
        executor_type = "unrestricted" if unsafe_executor else "docker"
        if unsafe_executor:
            print(
                "⚠️  WARNING: Using unrestricted executor - generated code will run directly on your system!"
            )

        # Initialize services
        self.api_client = ARCAPIClient(
            model=model,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            qwen_no_think=qwen_no_think,
            lora_adapter=lora_adapter,
            api_timeout=api_timeout,  # Standard timeout for network safety
        )

        # Initialize remaining components
        self.task_loader = get_task_loader()
        self.executor = ArcTester(
            timeout=1,
            executor_type=executor_type,
            max_output_chars=10_000,
            max_output_cells=1_800,
        )
        self.prompt_loader = PromptLoader()
        self.transduction_classifier = CodeTransductionClassifier()

        # Thread-safe cost tracking
        self._cost_lock = threading.Lock()
        self.total_cost = 0.0
        self.total_tokens = 0

        # Health monitoring for long runs
        self.health_metrics = {
            "total_attempts": 0,
            "exec_successes": 0,
            "exec_timeouts": 0,
            "exec_errors": 0,
            "exec_times": [],
            "recent_window": 100,
            "report_interval": 100,
        }

        # vLLM metrics monitoring
        self.metrics_url = None
        self._metrics_thread = None
        self._stop_metrics = False
        self._extract_metrics_url_from_base_url()

        print(
            f"⏰ API timeout: {api_timeout}s (network safety only, no infrastructure timeouts)"
        )
        print(f"🗄️ Sampled programs will be logged to {self.dataset_collector.output_path()}")
        
        if self.metrics_url:
            print(f"📊 vLLM metrics monitoring enabled ({self.metrics_url})", flush=True)
        else:
            print(f"📊 vLLM metrics monitoring disabled (base_url: {self.base_url})", flush=True)

    @property
    def model(self):
        """Get the model name from the API client"""
        return self.api_client.model

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
    
    def _extract_metrics_url_from_base_url(self):
        """Extract the metrics URL from the base URL if it's a vLLM server"""
        if not self.base_url:
            return
        
        try:
            # vLLM serves metrics on the same port as the API server
            # Just add /metrics to the base URL
            base_url = self.base_url.rstrip('/')
            if base_url.endswith('/v1'):
                base_url = base_url[:-3]  # Remove /v1 suffix
            potential_metrics_url = f"{base_url}/metrics"
            
            # Always enable metrics URL construction for potential vLLM servers
            # We'll test connectivity later during actual monitoring
            self.metrics_url = potential_metrics_url
            
            # Optional: Test if metrics endpoint is immediately available
            try:
                response = requests.get(potential_metrics_url, timeout=2)
                if response.status_code == 200 and 'vllm:' in response.text:
                    if self.debug:
                        print(f"🔍 vLLM metrics confirmed at initialization: {potential_metrics_url}")
                elif self.debug:
                    print(f"🔍 Metrics endpoint found but no vLLM metrics detected (may not be ready yet)")
            except Exception as e:
                if self.debug:
                    print(f"🔍 Metrics endpoint test failed during init (server may not be ready): {e}")
                # Don't disable - server might not be ready yet
                
        except Exception:
            # If extraction fails, disable metrics monitoring
            self.metrics_url = None
    
    def _get_llm_metrics(self):
        """Fetch metrics from LLM server (vLLM or SGLang)"""
        if not self.metrics_url:
            return None
        
        try:
            response = requests.get(self.metrics_url, timeout=5)
            response.raise_for_status()
            
            metrics_text = response.text
            
            # Detect server type based on metric prefixes
            if 'sglang:' in metrics_text:
                return self._parse_sglang_metrics(metrics_text)
            elif 'vllm:' in metrics_text:
                return self._parse_vllm_metrics(metrics_text)
            else:
                if self.debug:
                    print(f"🔍 Unknown LLM server type (no vllm: or sglang: metrics found)")
                return None
            
        except Exception as e:
            if self.debug:
                print(f"🔍 Failed to fetch LLM metrics: {e}")
            return None
    
    def _parse_vllm_metrics(self, metrics_text):
        """Parse vLLM-specific metrics"""
        # Parse vLLM metrics using regex (handle Prometheus labels)
        running_match = re.search(r'vllm:num_requests_running\{[^}]*\}\s+(\d+(?:\.\d+)?)', metrics_text)
        waiting_match = re.search(r'vllm:num_requests_waiting\{[^}]*\}\s+(\d+(?:\.\d+)?)', metrics_text)
        cache_usage_match = re.search(r'vllm:gpu_cache_usage_perc\{[^}]*\}\s+(\d+(?:\.\d+)?)', metrics_text)
        success_stop_match = re.search(r'vllm:request_success_total\{[^}]*finished_reason="stop"[^}]*\}\s+(\d+(?:\.\d+)?)', metrics_text)
        success_length_match = re.search(r'vllm:request_success_total\{[^}]*finished_reason="length"[^}]*\}\s+(\d+(?:\.\d+)?)', metrics_text)
        
        running = int(float(running_match.group(1))) if running_match else 0
        waiting = int(float(waiting_match.group(1))) if waiting_match else 0
        cache_usage = float(cache_usage_match.group(1)) if cache_usage_match else 0.0
        success_stop = int(float(success_stop_match.group(1))) if success_stop_match else 0
        success_length = int(float(success_length_match.group(1))) if success_length_match else 0
        
        return {
            'server_type': 'vLLM',
            'num_requests_running': running,
            'num_requests_waiting': waiting,
            'cache_usage_perc': cache_usage,
            'success_stop': success_stop,
            'success_length': success_length,
            'throughput': None  # vLLM doesn't expose real-time throughput
        }
    
    def _parse_sglang_metrics(self, metrics_text):
        """Parse SGLang-specific metrics"""
        # Sum across all ranks for multi-GPU setups
        running_matches = re.findall(r'sglang:num_running_reqs\{[^}]*\}\s+(\d+(?:\.\d+)?)', metrics_text)
        queue_matches = re.findall(r'sglang:num_queue_reqs\{[^}]*\}\s+(\d+(?:\.\d+)?)', metrics_text)
        usage_matches = re.findall(r'sglang:token_usage\{[^}]*\}\s+(\d+(?:\.\d+)?)', metrics_text)
        throughput_matches = re.findall(r'sglang:gen_throughput\{[^}]*\}\s+(\d+(?:\.\d+)?)', metrics_text)
        
        # Get total requests counter
        total_requests_match = re.search(r'sglang:num_requests_total\{[^}]*\}\s+(\d+(?:\.\d+)?)', metrics_text)
        
        # Sum values across all ranks
        running = sum(int(float(x)) for x in running_matches) if running_matches else 0
        waiting = sum(int(float(x)) for x in queue_matches) if queue_matches else 0
        avg_cache_usage = sum(float(x) for x in usage_matches) / len(usage_matches) if usage_matches else 0.0
        total_throughput = sum(float(x) for x in throughput_matches) if throughput_matches else 0.0
        total_requests = int(float(total_requests_match.group(1))) if total_requests_match else 0
        
        return {
            'server_type': 'SGLang',
            'num_requests_running': running,
            'num_requests_waiting': waiting,
            'cache_usage_perc': avg_cache_usage,  # Already a percentage (0.01 = 1%)
            'success_stop': total_requests,  # SGLang doesn't separate success types
            'success_length': 0,
            'throughput': total_throughput  # Real-time generation throughput
        }
    
    def _start_metrics_monitoring(self):
        """Start the metrics monitoring thread"""
        if not self.metrics_url:
            print(f"📊 Skipping metrics monitoring: no metrics URL")
            return
        if self._metrics_thread:
            print(f"📊 Metrics monitoring already running")
            return
        
        print(f"📊 Starting metrics monitoring thread for {self.metrics_url}", flush=True)
        
        def monitor_metrics():
            print(f"📊 Metrics thread started, polling {self.metrics_url} every 30s", flush=True)
            while not self._stop_metrics:
                try:
                    metrics = self._get_llm_metrics()
                    if metrics:
                        server_type = metrics['server_type']
                        running = metrics['num_requests_running']
                        pending = metrics['num_requests_waiting']
                        cache_pct = metrics['cache_usage_perc']
                        success_stop = metrics['success_stop']
                        success_length = metrics['success_length']
                        throughput = metrics.get('throughput')
                        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                        
                        # Format display based on server type
                        if server_type == 'SGLang':
                            # SGLang: cache_usage is already percentage, show throughput
                            throughput_str = f", {throughput:.1f}tok/s" if throughput and throughput > 0 else ""
                            print(f"📊 [{timestamp}] SGLang: {running}R/{pending}Q, Cache {cache_pct*100:.1f}%, Reqs {success_stop}{throughput_str}", flush=True)
                        else:  # vLLM
                            # vLLM: convert cache from fraction to percentage, show success breakdown
                            print(f"📊 [{timestamp}] vLLM: {running}R/{pending}P, Cache {cache_pct*100:.1f}%, Success {success_stop}+{success_length}", flush=True)
                    else:
                        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                        print(f"📊 [{timestamp}] LLM metrics fetch failed", flush=True)
                except Exception as e:
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                    print(f"📊 [{timestamp}] LLM metrics error: {e}", flush=True)
                
                # Sleep for 30 seconds, but check stop flag every second
                for _ in range(30):
                    if self._stop_metrics:
                        break
                    time.sleep(1)
        
        self._metrics_thread = threading.Thread(target=monitor_metrics, daemon=True)
        self._metrics_thread.start()
    
    def _stop_metrics_monitoring(self):
        """Stop the metrics monitoring thread"""
        self._stop_metrics = True
        if self._metrics_thread:
            self._metrics_thread.join(timeout=2)

    def _check_models_endpoint(self):
        """Check what models are available at the /models endpoint and validate arguments"""
        try:
            print("🔍 Checking available models...")
            models_response = self.api_client.client.models.list()

            if hasattr(models_response, "data") and models_response.data:
                available_models = []
                available_lora_adapters = []

                print(f"📋 Available models ({len(models_response.data)}):")
                for model in models_response.data:
                    model_id = getattr(model, "id", "unknown")
                    owned_by = getattr(model, "owned_by", "unknown")
                    available_models.append(model_id)

                    # Show LORA adapters if present
                    if hasattr(model, "lora_adapters") and model.lora_adapters:
                        lora_list = model.lora_adapters
                        available_lora_adapters.extend(lora_list)
                        print(
                            f"   • {model_id} (owner: {owned_by}) [LORA: {', '.join(lora_list)}]"
                        )
                    elif "lora" in model_id.lower() or "adapter" in model_id.lower():
                        available_lora_adapters.append(model_id)
                        print(f"   • {model_id} (owner: {owned_by}) [LORA adapter]")
                    else:
                        print(f"   • {model_id} (owner: {owned_by})")

                # Validate model argument
                if self.model not in available_models:
                    print(
                        f"⚠️  WARNING: Specified model '{self.model}' not found in available models"
                    )
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

    def _update_health_metrics(self, attempt_detail: AttemptDetail, exec_time: float):
        """Update health monitoring metrics (thread-safe)"""
        with self._cost_lock:  # Reuse existing lock for simplicity
            self.health_metrics["total_attempts"] += 1

            # Track execution success/failure
            if (
                attempt_detail.get("test_exec_error")
                or attempt_detail.get("train_exec_errors", 0) > 0
            ):
                self.health_metrics["exec_errors"] += 1
            elif (
                attempt_detail.get("test_exec_timeout")
                or attempt_detail.get("train_exec_timeouts", 0) > 0
            ):
                self.health_metrics["exec_timeouts"] += 1
            else:
                self.health_metrics["exec_successes"] += 1

            # Track execution times (keep recent window)
            self.health_metrics["exec_times"].append(exec_time)
            window_size = self.health_metrics["recent_window"]
            if len(self.health_metrics["exec_times"]) > window_size:
                self.health_metrics["exec_times"] = self.health_metrics["exec_times"][
                    -window_size:
                ]

    def _print_health_report(self):
        """Print compact health report"""
        metrics = self.health_metrics
        total = metrics["total_attempts"]

        if total == 0:
            return

        # Overall stats
        success_rate = (metrics["exec_successes"] / total) * 100
        timeout_rate = (metrics["exec_timeouts"] / total) * 100
        error_rate = (metrics["exec_errors"] / total) * 100

        # Recent window stats (last N attempts)
        window_size = min(metrics["recent_window"], total)
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
        if metrics["exec_times"]:
            avg_time = sum(metrics["exec_times"]) / len(metrics["exec_times"])
            recent_times = metrics["exec_times"][-min(50, len(metrics["exec_times"])) :]
            recent_avg_time = (
                sum(recent_times) / len(recent_times) if recent_times else avg_time
            )
        else:
            avg_time = recent_avg_time = 0.0

        # Compact health report
        print(
            f"🏥 Health [{total} attempts]: "
            f"Success {success_rate:.0f}% | "
            f"Timeout {timeout_rate:.0f}% | "
            f"ExecErr {error_rate:.0f}% | "
            f"AvgTime {recent_avg_time:.2f}s"
        )

    def _maybe_log_program_to_database(
        self, task_id: str, attempt_detail: AttemptDetail
    ):
        """Log successful programs to the local database"""

        try:
            # Only log programs that extracted successfully and have valid outputs. Possibly this first check is redundant?
            if not attempt_detail.get("program_extracted", False):
                return  # No program to log
            
            if not attempt_detail.get("outputs_valid", False):
                return  # Don't log programs with invalid outputs
            
            program = attempt_detail.get("program", "").strip()
            if not program:
                return

            # Skip logging for random programs
            if is_random(program):
                return

            # Extract correctness arrays
            train_results = attempt_detail.get("train_results", [])
            test_results = attempt_detail.get("test_results", [])

            correct_train_input = [
                result.get("correct", False) for result in train_results
            ]
            correct_test_input = [
                result.get("correct", False) for result in test_results
            ]

            # Extract predicted outputs (convert to lists if they're numpy arrays)
            predicted_train_output = [result.get("predicted", None) for result in train_results]
            predicted_test_output = [result.get("predicted", None) for result in test_results]
           
            # Extract reasoning from raw response if available
            reasoning = ""
            raw_response = attempt_detail.get("raw_response", {})
            if isinstance(raw_response, dict) and "choices" in raw_response:
                choices = raw_response.get("choices", [])
                if choices and isinstance(choices[0], dict):
                    message = choices[0].get("message", {})
                    reasoning = message.get("reasoning", "") or message.get(
                        "content", ""
                    )

            # Create ProgramSample
            program_sample = ProgramSample(
                task_id=task_id,
                reasoning=reasoning,
                code=program,
                correct_train_input=correct_train_input,
                correct_test_input=correct_test_input,
                predicted_train_output=predicted_train_output,
                predicted_test_output=predicted_test_output,
                model=self.model,
                is_transductive=attempt_detail.get("is_transductive", False),
            )

            self.dataset_collector.collect(program_sample)

        except Exception:
            # Don't let database logging errors crash the main execution
            traceback.print_exc()

    def create_prompt(self, task_data: Dict) -> tuple[str, str]:
        """Create a prompt for the model to solve an ARC task"""
        return create_arc_prompt(task_data, self.prompt_loader, self.prompt_version)

    def get_sampling_parameters(self) -> Dict:
        """Get the sampling parameters that will be used for API calls"""
        return self.api_client.get_sampling_parameters()

    def call_chat_completions_api(self, messages: List[Dict]) -> tuple:
        """Call the OpenAI Chat Completions API"""
        result = self.api_client.call_chat_completions_api(messages)
        if result["success"]:
            return result["response"], result["sampling_params"]
        else:
            # Return None response with error info
            return None, result["sampling_params"]

    def extract_code_from_response(self, response) -> str:
        """Extract Python code from the Chat Completions API result"""
        # Get the full text from response
        full_text = ""

        if hasattr(response, "choices") and len(response.choices) > 0:
            message = response.choices[0].message
            if hasattr(message, "content") and message.content:
                full_text = message.content

        if self.debug and len(full_text) > 0:
            print(f"🔍 Response content: {len(full_text)} chars")

        return extract_python_code(full_text, self.debug)

    def run_single_attempt(
        self,
        task_id: str,
        task_data: TaskData,
        attempt_num: int,
        dataset: Optional[str] = None,
        subset: Optional[str] = None,
        full_prompt: Optional[Dict] = None,
    ) -> SingleAttemptResult:
        """Run a single attempt for an ARC task"""
        system_content = full_prompt["system"]
        user_content = full_prompt["user"]

        attempt_start_time = datetime.datetime.now()
        exec_start_time = time.time()  # Track execution timing
        conversation_history = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        # Make API call with retries
        response = None
        api_call_successful = False
        error = None
        timed_out = False

        for retry_attempt in range(3):
            try:
                response, api_kwargs = self.call_chat_completions_api(
                    conversation_history
                )
                api_call_successful = True
                break
            except Exception as e:
                error = str(e)
                # Check if this is actually a timeout error vs other API errors
                is_timeout_error = (
                    "timeout" in str(e).lower()
                    or "TimeoutError" in str(type(e).__name__)
                    or "concurrent.futures._base.TimeoutError" in str(type(e))
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
        if api_call_successful and "api_kwargs" in locals():
            # Extract sampling parameters from actual API call
            for param in [
                "temperature",
                "max_tokens",
                "top_p",
                "top_k",
                "min_p",
                "thinking_budget",
            ]:
                if param in api_kwargs:
                    sampling_params[param] = api_kwargs[param]
            # Also check extra_body for nested parameters
            if "extra_body" in api_kwargs:
                extra_body = api_kwargs["extra_body"]
                for param in ["top_k", "min_p", "thinking_budget"]:
                    if param in extra_body:
                        sampling_params[param] = extra_body[param]
        else:
            # Fallback to instance parameters
            if self.temperature is not None:
                sampling_params["temperature"] = self.temperature
            if self.max_tokens is not None:
                sampling_params["max_tokens"] = self.max_tokens

        # Track costs
        usage = getattr(response, "usage", None)
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        input_rate, output_rate = self.api_client.get_model_pricing()
        attempt_cost = (input_tokens / 1_000_000) * input_rate + (
            output_tokens / 1_000_000
        ) * output_rate
        total_tokens = usage.total_tokens if usage else 0

        # Check for empty response
        empty_response = False
        if api_call_successful and response:
            if hasattr(response, "choices") and len(response.choices) > 0:
                message = response.choices[0].message
                content = (
                    getattr(message, "content", "")
                    if hasattr(message, "content")
                    else ""
                )
                empty_response = not content or content.strip() == ""
            else:
                empty_response = True
        elif api_call_successful:
            empty_response = True

        # Check for max tokens hit
        hit_max_tokens = False
        if (
            api_call_successful
            and response
            and hasattr(response, "choices")
            and len(response.choices) > 0
        ):
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            hit_max_tokens = finish_reason == "length"

        # Extract and evaluate program
        program = self.extract_code_from_response(response) if response else ""
        program_extracted = bool(program and program.strip())

        # Evaluate on training examples
        train_results: List[TrainResult] = []
        train_correct = 0
        train_exec_errors = 0
        train_exec_timeouts = 0

        for ex in task_data["train"]:
            if not program_extracted:
                pred, err, tout = None, "no program", False
            else:
                # Always execute, even if marked as transductive (for analysis)
                pred, err, tout = self.executor.execute_program_with_timeout(
                    program, ex["input"]
                )

            # Check correctness based on actual execution results
            is_corr = (
                (pred == ex["output"])
                if (pred is not None and not err and not tout)
                else False
            )
            train_results.append(
                {
                    "predicted": pred,
                    "expected": ex["output"],
                    "correct": is_corr,
                    "error": err,
                    "timed_out": tout,
                }
            )

            if is_corr:
                train_correct += 1
            elif err and err != "no program":
                train_exec_errors += 1
            elif tout:
                train_exec_timeouts += 1

        train_accuracy = (
            train_correct / len(task_data["train"]) if task_data["train"] else 0.0
        )

        # Evaluate on all test examples
        test_results: List[TestResult] = []
        test_predictions = []
        test_correct_count = 0
        any_test_exec_error = False
        any_test_exec_timeout = False

        for test_idx, test_example in enumerate(task_data["test"]):
            test_input = test_example["input"]
            # Check if we're in SUBMIT mode - test outputs may not be available
            submit_mode = os.getenv("SUBMIT", "").lower() == "true"

            # Only access test_expected if not in SUBMIT mode
            if submit_mode:
                test_expected = None  # No ground truth available in SUBMIT mode
            else:
                test_expected = test_example.get("output", None)

            if not program_extracted:
                test_pred, test_err, test_tout = None, "no program", False
            else:
                # Always execute, even if marked as transductive (for analysis)
                test_pred, test_err, test_tout = (
                    self.executor.execute_program_with_timeout(program, test_input)
                )
                if test_err and test_err != "no program":
                    any_test_exec_error = True
                if test_tout:
                    any_test_exec_timeout = True

            # Mark as correct/incorrect only when we have ground truth (not in SUBMIT mode)
            is_correct = (
                (test_pred == test_expected)
                if (
                    test_pred is not None
                    and test_expected is not None  # Only score if we have ground truth
                    and not test_err
                    and not test_tout
                    and not submit_mode  # Never score in SUBMIT mode
                )
                else False
            )

            if is_correct:
                test_correct_count += 1

            test_results.append(
                {
                    "test_idx": test_idx,
                    "predicted": test_pred,
                    "expected": test_expected,  # Will be None in SUBMIT mode
                    "correct": is_correct,  # Will be False in SUBMIT mode
                    "error": test_err,
                    "timed_out": test_tout,
                }
            )
            # Store raw prediction - don't clean here
            test_predictions.append(test_pred)

        # Overall test correctness (all test cases must be correct)
        # In SUBMIT mode, we cannot determine correctness without ground truth
        submit_mode = os.getenv("SUBMIT", "").lower() == "true"
        test_correct = (
            (test_correct_count == len(task_data["test"]))
            if len(task_data["test"]) > 0 and not submit_mode
            else False
        )

        # Validate all outputs meet ARC specification - collect all predictions
        all_predictions = [
            result.get("predicted")
            for results_list in [train_results, test_results]
            for result in results_list
        ]
        
        # Use common validator (rejects None predictions and invalid grids)
        outputs_valid = program_extracted and all_predictions and \
            ARCTaskValidator.validate_prediction_list(all_predictions, "outputs")[0]

        # Run transduction detection ONLY if outputs are valid
        is_transductive = False
        transduction_confidence = 0.0
        transduction_reason = ""
        if program_extracted and outputs_valid:
            try:
                is_transductive, transduction_confidence = (
                    self.transduction_classifier.is_transductive(program, task_data)
                )
                if is_transductive:
                    transduction_reason = f"Code-based transduction detected (confidence: {transduction_confidence:.3f})"
                else:
                    transduction_reason = f"Not transductive (confidence: {1 - transduction_confidence:.3f})"
            except Exception as e:
                transduction_reason = f"Transduction detection failed: {e}"
                transduction_confidence = 0.5  # Default to uncertain if detection fails
        elif not program_extracted:
            transduction_reason = "No program extracted"
            transduction_confidence = 0.0
        else:
            transduction_reason = "Invalid outputs - skipping transduction detection"
            transduction_confidence = 0.0

        # Store attempt details
        # Note: This dictionary structure matches the AttemptDetail TypedDict defined above
        attempt_detail = AttemptDetail(
            attempt_number=attempt_num + 1,
            timestamp=attempt_start_time.isoformat(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            attempt_cost=attempt_cost,
            program_extracted=program_extracted,
            program=program,
            outputs_valid=outputs_valid,
            is_transductive=is_transductive,
            transduction_confidence=transduction_confidence,
            transduction_reason=transduction_reason,
            train_results=train_results,
            train_accuracy=train_accuracy,
            train_exec_errors=train_exec_errors,
            train_exec_timeouts=train_exec_timeouts,
            # Store predictions as list for consistency
            test_predicted=test_predictions,
            test_results=test_results,
            test_correct=test_correct,
            test_correct_count=test_correct_count,
            test_exec_error=any_test_exec_error,
            test_exec_timeout=any_test_exec_timeout,
            # Legacy fields for backwards compatibility (using first test case)
            test_error=test_results[0]["error"] if test_results else "no program",
            test_timed_out=test_results[0]["timed_out"] if test_results else False,
            raw_response=ResponseSerializer.serialize_response(response),
            sampling_params=sampling_params,
            api_success=api_call_successful,
            api_timeout=timed_out,
            empty_response=empty_response,
            hit_max_tokens=hit_max_tokens,
            error=error,
            # Add fields expected by metrics_utils
            all_test_correct=test_correct,
            code_ran=program_extracted,
        )

        # Update costs
        self._update_costs(attempt_cost, total_tokens)

        # Update health metrics and periodic reporting
        exec_time = time.time() - exec_start_time
        self._update_health_metrics(attempt_detail, exec_time)

        # Periodic health reports (every N attempts)
        if (
            self.health_metrics["total_attempts"]
            % self.health_metrics["report_interval"]
            == 0
        ):
            self._print_health_report()

        # Log successful programs to database
        self._maybe_log_program_to_database(task_id, attempt_detail)

        result: SingleAttemptResult = {
            "task_id": task_id,
            "attempt_num": attempt_num,
            "attempt_detail": attempt_detail,
            "task_data": task_data,
            "dataset": dataset,
            "subset": subset,
            "full_prompt": full_prompt
            or {"system": system_content, "user": user_content},
        }
        return result

    def run_subset(
        self,
        subset_name: str,
        dataset: str = "arc-prize-2025",
        limit: Optional[int] = None,
    ) -> List[Dict]:
        """Run all tasks in a subset with true parallelization at the attempt level"""
        try:
            print(f"Loading subset: {dataset}/{subset_name}")
            tasks = self.task_loader.get_subset_tasks(f"{dataset}/{subset_name}")
            if limit:
                tasks = tasks[:limit]
            total_tasks = len(tasks)

            # Validate task data integrity to prevent corruption issues
            validated_tasks = ARCTaskValidator.validate_tasks(tasks)

            if len(validated_tasks) != total_tasks:
                print(
                    f"⚠️ {total_tasks - len(validated_tasks)} tasks failed validation, using {len(validated_tasks)} valid tasks"
                )
                tasks = validated_tasks
                total_tasks = len(tasks)

            print(f"✅ Task validation complete: {total_tasks} valid tasks")

            # Sort tasks by total length (training + test examples)
            def calculate_task_length(task_tuple):
                task_id, task_data = task_tuple
                total_length = 0
                # Add training examples length
                if "train" in task_data:
                    for example in task_data["train"]:
                        if "input" in example:
                            total_length += len(str(example["input"]))
                        if "output" in example:
                            total_length += len(str(example["output"]))
                # Add test examples length (only inputs since we don't have outputs)
                if "test" in task_data:
                    for example in task_data["test"]:
                        if "input" in example:
                            total_length += len(str(example["input"]))
                return total_length

            # Sort tasks from shortest to longest
            tasks = sorted(tasks, key=calculate_task_length)
            print(f"📏 Tasks sorted by length (shortest to longest)")

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
                print(
                    f"Scheduling: Task-by-task (1 task × {self.max_attempts} attempts = {self.max_attempts} workers used)"
                )
            else:
                print(
                    f"Scheduling: Batched ({concurrent_tasks} tasks × {self.max_attempts} attempts = {concurrent_tasks * self.max_attempts} workers used)"
                )
                if self.max_workers > concurrent_tasks * self.max_attempts:
                    unused_workers = self.max_workers - (
                        concurrent_tasks * self.max_attempts
                    )
                    print(
                        f"Note: {unused_workers} workers will be idle due to batching strategy"
                    )
        else:
            print("Parallelization: DISABLED (sequential execution)")

        # No infrastructure timeouts to avoid GPU overload
        print("")
        print(
            "✅ No infrastructure timeouts - requests complete naturally to avoid GPU overload"
        )
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
        
        # Start vLLM metrics monitoring if available
        self._start_metrics_monitoring()

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
        import threading

        task_results = defaultdict(lambda: {"attempts": [], "task_data": None})

        # Initialize task results with task data and prompts (create once per task)
        first_prompt_shown = False
        for task_id, task_data in tasks:
            task_results[task_id]["task_data"] = task_data
            system_content, user_content = self.create_prompt(task_data)
            task_results[task_id]["full_prompt"] = {
                "system": system_content,
                "user": user_content,
            }

            # Show the first task's prompt for debugging
            if not first_prompt_shown:
                print(f"\n📝 FIRST TASK PROMPT ({task_id}):")
                print("=" * 80)
                print("SYSTEM:")
                print(system_content)
                print("\nUSER:")
                print(user_content)
                print("=" * 80)
                first_prompt_shown = True

        completed_attempts = 0
        completed_tasks = 0
        count_lock = threading.Lock()

        def attempt_wrapper(task_idx, task_id, task_data, attempt_num):
            nonlocal completed_attempts, completed_tasks
            attempt_start = time.time()

            try:
                # Get the pre-created prompt for this task
                full_prompt = task_results[task_id]["full_prompt"]
                # No infrastructure timeout - let requests complete to avoid GPU overload
                result = self.run_single_attempt(
                    task_id, task_data, attempt_num, dataset, subset_name, full_prompt
                )
                attempt_duration = time.time() - attempt_start
                if attempt_duration > 60:  # Log slow attempts
                    print(
                        f"🐌 Slow attempt: {task_id} attempt {attempt_num + 1} took {attempt_duration:.1f}s"
                    )

                with count_lock:
                    # Store attempt result - use thread-safe access
                    if task_id in task_results:
                        task_results[task_id]["attempts"].append(
                            result["attempt_detail"]
                        )
                        # Prompt is already stored at task level during initialization
                        completed_attempts += 1

                        # Check if task is complete
                        if len(task_results[task_id]["attempts"]) == self.max_attempts:
                            completed_tasks += 1
                            # Calculate and display task summary
                            self._display_task_summary(task_id, task_results[task_id])

                            # Task completed - only database logging remains
                    else:
                        print(
                            f"⚠️ Task {task_id} not found in results dict - possible corruption"
                        )

                return result
            except Exception as e:
                with count_lock:
                    completed_attempts += 1
                    print(
                        f"❌ Attempt {attempt_num + 1} for task {task_id} failed: {e}"
                    )
                return None

        executor = ThreadPoolExecutor(self.max_workers)
        try:
            futures = [
                executor.submit(
                    attempt_wrapper, task_idx, task_id, task_data, attempt_num
                )
                for task_idx, task_id, task_data, attempt_num in attempt_jobs
            ]

            print(
                f"🚀 Started {total_attempts} attempts with {self.max_workers} workers"
            )

            # Wait for all attempts to complete, reporting progress periodically
            start_time = time.time()

            from concurrent.futures import as_completed

            completed_count = 0
            remaining = set(futures)
            progress_interval = 15.0
            last_progress_time = time.time()

            while remaining:
                time_elapsed = time.time() - start_time

                # Check global timeout or inactivity
                if (self.global_timeout and time_elapsed > self.global_timeout) or (time.time() - last_progress_time > self.inactivity_timeout):
                    print(
                        f"⏰ Global timeout reached ({self.global_timeout}s). Cancelling remaining attempts..."
                    )
                    # Cancel remaining futures
                    cancelled_count = 0
                    for future in remaining:
                        if future.cancel():
                            cancelled_count += 1

                    completed_naturally = total_attempts - len(remaining)
                    print(
                        f"⏰ Timeout: {completed_naturally} attempts completed, {cancelled_count} cancelled, {len(remaining) - cancelled_count} already running"
                    )
                    break

                try:
                    # Use a short timeout so we can log periodic progress
                    for future in as_completed(
                        list(remaining), timeout=progress_interval
                    ):
                        remaining.discard(future)
                        completed_count += 1
                        last_progress_time = time.time()
                        try:
                            _ = future.result()
                        except Exception as future_e:
                            print(f"🚨 Future #{completed_count} error: {future_e}")
                    # Periodic progress log
                    done_now = total_attempts - len(remaining)
                    timeout_info = (
                        f" (timeout in {self.global_timeout - time_elapsed:.0f}s)"
                        if self.global_timeout
                        else ""
                    )
                    print(
                        f"⏳ Progress: {done_now}/{total_attempts} attempts done; {len(remaining)} remaining{timeout_info}"
                    )
                except KeyboardInterrupt:
                    print(
                        f"\n🛑 Cancellation requested - cancelling queued requests, waiting for in-flight requests..."
                    )

                    # Cancel futures that haven't started yet
                    cancelled_count = 0
                    for future in list(remaining):
                        if (
                            future.cancel()
                        ):  # Returns True if successfully cancelled (hadn't started)
                            cancelled_count += 1
                            remaining.discard(future)

                    in_flight_count = len(remaining)
                    print(
                        f"   Cancelled {cancelled_count} queued requests, waiting for {in_flight_count} in-flight requests"
                    )
                    if in_flight_count > 0:
                        print(
                            f"   (In-flight requests may take up to {self.api_client.api_timeout}s each to complete)"
                        )
                    # Continue waiting for the actually running requests
                except TimeoutError:
                    # No futures completed in this window; print a heartbeat
                    done_now = total_attempts - len(remaining)
                    timeout_info = (
                        f" (timeout in {self.global_timeout - time_elapsed:.0f}s)"
                        if self.global_timeout
                        else ""
                    )
                    print(
                        f"⏳ No completions in last {progress_interval:.0f}s — {done_now}/{total_attempts} done; {len(remaining)} remaining{timeout_info}"
                    )
                    continue
                except Exception as e:
                    print(f"🚨 Unexpected error: {e}")
                    traceback.print_exc()

            elapsed_time = time.time() - start_time
            if self.global_timeout and elapsed_time > self.global_timeout:
                print(
                    f"⏰ Execution stopped after {elapsed_time:.1f}s due to global timeout"
                )
            else:
                print(
                    f"✅ All {total_attempts} attempts completed in {elapsed_time:.1f}s"
                )

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
                print(
                    f"❌ {failed_attempts} attempts failed out of {total_attempts} total"
                )

            if cancelled_attempts > 0:
                print(f"🛑 {cancelled_attempts} attempts were cancelled due to timeout")

            print(
                f"📊 Final status: {successful_attempts} successful, {failed_attempts} failed, {cancelled_attempts} cancelled"
            )

        finally:
            # We don't want to block execution if we have one stuck thread.
            executor.shutdown(wait=False)
        
        # Stop vLLM metrics monitoring
        self._stop_metrics_monitoring()

        # Convert task_results to the expected format for summary (including partial tasks)
        print("Converting task results to summary format...")
        results = []
        for task_id, task_data in tasks:
            if task_id in task_results and len(task_results[task_id]["attempts"]) > 0:
                # Sort attempts by attempt number
                attempts = sorted(
                    task_results[task_id]["attempts"], key=lambda x: x["attempt_number"]
                )

                # Validate attempt data integrity
                valid_attempts = []
                for attempt in attempts:
                    if isinstance(attempt, dict) and "attempt_number" in attempt:
                        valid_attempts.append(attempt)
                    else:
                        print(
                            f"⚠️ Invalid attempt data for task {task_id}: {type(attempt)}"
                        )

                if len(valid_attempts) != len(attempts):
                    print(
                        f"⚠️ Task {task_id}: {len(attempts) - len(valid_attempts)} invalid attempts filtered out"
                    )

                # Transductive detection is now done during program execution

                # All tasks should have complete attempts now
                api_type = "chat_completions_all_attempts"

                # Trim failed attempts to reduce file size for summary
                trimmed_attempts = [
                    self._trim_failed_attempt(attempt) for attempt in valid_attempts
                ]

                result = {
                    "task_id": task_id,
                    "model": self.model,
                    "api_type": api_type,
                    "dataset": dataset,
                    "subset": subset_name,
                    "attempt_details": trimmed_attempts,
                    "tokens_used": sum(
                        attempt.get("input_tokens", 0) + attempt.get("output_tokens", 0)
                        for attempt in trimmed_attempts
                    ),
                    "request_cost": sum(
                        attempt.get("attempt_cost", 0.0) for attempt in valid_attempts
                    ),
                    "max_attempts": self.max_attempts,
                    "actual_attempts": len(
                        valid_attempts
                    ),  # Track how many were actually completed
                    "api_success": True,
                    "task_data": task_data,
                }
                results.append(result)
            else:
                print(f"⚠️ Task {task_id} has no valid attempts - skipping")

        print(f"✅ Converted results for {len(results)}/{total_tasks} tasks")

        # Check if we're in SUBMIT mode
        submit_mode = os.getenv("SUBMIT", "").lower() == "true"

        if submit_mode:
            # In SUBMIT mode: show limited summary and suggest submission generation
            self._print_submit_mode_summary(results, subset_name, dataset, elapsed_time)
            print(f"\n🎯 To generate submission file, run:")
            print(f"   uv run python llm_python/generate_submission.py --dataset {dataset} --subset {subset_name}")
        else:
            # Normal mode: print full summary to console only (no file saving)
            self._print_summary(results, subset_name, dataset, elapsed_time)

        return results

    def _display_task_summary(self, task_id: str, task_result: Dict):
        """Display a brief summary of a completed task"""
        attempts = task_result["attempts"]

        # Check if we're in SUBMIT mode (test outputs available for scoring)
        submit_mode = os.getenv("SUBMIT", "").lower() == "true"


        # 1. OUTPUT VALIDITY CATEGORIES
        total_attempts = len(attempts)
        valid_outputs = sum(1 for att in attempts if att.get("outputs_valid", False))
        no_programs = sum(1 for att in attempts if not att.get("program_extracted", False))
        exec_failures = sum(1 for att in attempts if att.get("train_exec_errors", 0) > 0 or att.get("test_exec_error", False))
        max_length = sum(1 for att in attempts if att.get("hit_max_tokens", False))
        api_timeout = sum(1 for att in attempts if att.get("api_timeout", False))
        # Invalid outputs: program extracted, no execution failure, but outputs invalid
        invalid_outputs = sum(1 for att in attempts 
                             if att.get("program_extracted", False) 
                             and not att.get("outputs_valid", False)
                             and not (att.get("train_exec_errors", 0) > 0 or att.get("test_exec_error", False)))
        
        # 2. TEST CATEGORIES (if not submit mode)
        test_perfect_total = test_perfect_trans = 0
        test_partial_total = test_partial_trans = 0  
        test_incorrect_total = test_incorrect_trans = 0
        
        if not submit_mode:
            for att in attempts:
                if not att.get("outputs_valid", False):
                    continue
                is_trans = att.get("is_transductive", False)
                test_correct_count = att.get("test_correct_count", 0)
                total_test = len(task_result["task_data"]["test"]) if "task_data" in task_result else 0
                
                if test_correct_count == total_test and total_test > 0:
                    test_perfect_total += 1
                    if is_trans:
                        test_perfect_trans += 1
                elif test_correct_count > 0:
                    test_partial_total += 1
                    if is_trans:
                        test_partial_trans += 1
                else:
                    test_incorrect_total += 1
                    if is_trans:
                        test_incorrect_trans += 1

        # 3. TRAIN CATEGORIES (includes transductive)
        train_perfect_total = train_perfect_trans = 0
        train_partial_total = train_partial_trans = 0
        train_incorrect_total = train_incorrect_trans = 0
        
        for att in attempts:
            if not att.get("outputs_valid", False):
                continue
            is_trans = att.get("is_transductive", False)
            train_acc = att.get("train_accuracy", 0.0)
            
            if train_acc == 1.0:
                train_perfect_total += 1
                if is_trans:
                    train_perfect_trans += 1
            elif train_acc > 0:
                train_partial_total += 1
                if is_trans:
                    train_partial_trans += 1
            else:
                train_incorrect_total += 1
                if is_trans:
                    train_incorrect_trans += 1

        # Find best attempt for additional context
        best_attempt = max(
            attempts,
            key=lambda x: (x.get("test_correct", False), x.get("train_accuracy", 0.0)),
        )

        # Build the summary line with new structure
        
        # Helper function to format category with transductive breakdown
        def format_with_trans(total, trans, category_name):
            if total == 0:
                return ""
            if trans > 0:
                return f"{total} {category_name} (of which {trans} trans)"
            else:
                return f"{total} {category_name}"
        
        # 1. OUTPUT VALIDITY section
        validity_parts = []
        if valid_outputs > 0:
            validity_parts.append(f"{valid_outputs} valid outputs")
        if no_programs > 0:
            validity_parts.append(f"{no_programs} no programs")
        if exec_failures > 0:
            validity_parts.append(f"{exec_failures} execution failures")
        if max_length > 0:
            validity_parts.append(f"{max_length} max length")
        if api_timeout > 0:
            validity_parts.append(f"{api_timeout} api timeout")
        if invalid_outputs > 0:
            validity_parts.append(f"{invalid_outputs} invalid outputs")
        
        validity_section = ", ".join(validity_parts)
        
        # 2. TEST section (if not submit mode)
        test_section = ""
        if not submit_mode:
            test_parts = []
            test_perf = format_with_trans(test_perfect_total, test_perfect_trans, "test-perfect")
            if test_perf:
                test_parts.append(test_perf)
            test_part = format_with_trans(test_partial_total, test_partial_trans, "test-partial")
            if test_part:
                test_parts.append(test_part)
            test_incorr = format_with_trans(test_incorrect_total, test_incorrect_trans, "test-incorrect")
            if test_incorr:
                test_parts.append(test_incorr)
            test_section = ", ".join(test_parts)
        
        # 3. TRAIN section
        train_parts = []
        train_perf = format_with_trans(train_perfect_total, train_perfect_trans, "train-perfect")
        if train_perf:
            train_parts.append(train_perf)
        train_part = format_with_trans(train_partial_total, train_partial_trans, "train-partial")
        if train_part:
            train_parts.append(train_part)
        train_incorr = format_with_trans(train_incorrect_total, train_incorrect_trans, "train-incorrect")
        if train_incorr:
            train_parts.append(train_incorr)
        train_section = ", ".join(train_parts)
        
        # Assemble final summary
        summary = f"✅ {task_id}: {total_attempts} attempts"
        if validity_section:
            summary += f" | {validity_section}"
        if test_section:
            summary += f" | {test_section}"
        if train_section:
            summary += f" | {train_section}"

        # Add best attempt performance
        if submit_mode:
            summary += f" (best: {best_attempt.get('train_accuracy', 0.0):.1%} train)"
        else:
            if best_attempt.get("test_correct", False):
                summary += f" (best: {best_attempt.get('train_accuracy', 0.0):.1%} train)"
            else:
                summary += f" (best: {best_attempt.get('train_accuracy', 0.0):.1%} train, test-failed)"

        print(summary)

    def _trim_failed_attempt(self, attempt: AttemptDetail) -> AttemptDetail:
        """Trim data from attempts that failed to execute (exec errors, no code, API failures)

        Keep only essential metadata for failed attempts to reduce file size.
        """
        # Check if this attempt should be trimmed (execution failures)
        should_trim = (
            # Execution errors
            attempt.get("train_exec_errors", 0) > 0
            or attempt.get("test_exec_error", False)
            or
            # No code extracted
            not attempt.get("program_extracted", False)
            or
            # API failures
            attempt.get("api_timeout", False)
            or not attempt.get("api_success", True)
            or attempt.get("empty_response", False)
        )

        if not should_trim:
            # Keep everything for attempts that ran successfully
            return attempt

        # Create trimmed version for failed attempts
        # Start by copying the attempt dict (shallow copy)
        # Create a new AttemptDetail with only the required fields, using type-safe assignment
        trimmed: AttemptDetail = AttemptDetail(
            attempt_number=attempt.get("attempt_number", 0),
            timestamp=attempt.get("timestamp", ""),
            input_tokens=attempt.get("input_tokens", 0),
            output_tokens=attempt.get("output_tokens", 0),
            attempt_cost=attempt.get("attempt_cost", 0.0),
            program_extracted=attempt.get("program_extracted", False),
            program="",  # Always empty for trimmed
            is_transductive=attempt.get("is_transductive", False),
            transduction_confidence=attempt.get("transduction_confidence", 0.0),
            transduction_reason=attempt.get("transduction_reason", ""),
            train_results=[],  # Always empty for trimmed
            train_accuracy=attempt.get("train_accuracy", 0.0),
            train_exec_errors=attempt.get("train_exec_errors", 0),
            train_exec_timeouts=attempt.get("train_exec_timeouts", 0),
            test_predicted=None,
            test_results=[],  # Always empty for trimmed
            test_correct=attempt.get("test_correct", False),
            test_correct_count=attempt.get("test_correct_count", 0),
            test_exec_error=attempt.get("test_exec_error", False),
            test_exec_timeout=attempt.get("test_exec_timeout", False),
            test_error=attempt.get("test_error", None),
            test_timed_out=attempt.get("test_timed_out", False),
            raw_response=None,
            sampling_params=attempt.get("sampling_params", {}),
            api_success=attempt.get("api_success", False),
            api_timeout=attempt.get("api_timeout", False),
            empty_response=attempt.get("empty_response", False),
            hit_max_tokens=attempt.get("hit_max_tokens", False),
            error=attempt.get("error", None),
            all_test_correct=attempt.get("all_test_correct", False),
            code_ran=attempt.get("code_ran", False),
            # Extra fields for trimming metadata
            error_summary=str(attempt.get("error", ""))[:200]
            if attempt.get("error")
            else None,
            data_trimmed=True,
            trim_reason="execution_failure",
        )
        return trimmed

    def _print_summary(
        self,
        results: List[Dict],
        subset_name: str,
        dataset: str,
        elapsed_time: Optional[float] = None,
    ):
        """Print summary of all results to console"""
        # Calculate statistics
        total_tasks = len(results)
        api_successes = [r for r in results if r.get("api_success", True)]
        successful_api_calls = len(api_successes)

        # Calculate core metrics using utility functions
        if results:
            metrics = calculate_task_metrics(results, max_tokens=self.max_tokens)
            percentage_metrics = metrics_to_percentages(metrics)
        else:
            percentage_metrics = {
                "weighted_voting_pass2": 0.0,
                "train_majority_pass2": 0.0,
                "all_test_correct": 0.0,
                "all_train_correct": 0.0,
                "min1_train_correct": 0.0,
                "min1_code_success": 0.0,
                "passes_examples_all": 0.0,
                "max_length_responses": 0.0,
                "timeout_responses": 0.0,
                "api_failure_responses": 0.0,
            }

        # Print summary
        run_info = f" (Run {self.run_number})" if self.run_number > 0 else ""
        print("\n" + "=" * 50)
        print(f"SUMMARY{run_info}")
        print("=" * 50)
        print(f"Dataset: {dataset}")
        print(f"Subset: {subset_name}")
        print(f"Model: {self.model}")

        print(f"Total tasks: {total_tasks}")
        if elapsed_time:
            print(f"Total time: {elapsed_time:.1f}s")

        print(
            f"Successful API calls: {successful_api_calls}/{total_tasks} ({(successful_api_calls / total_tasks):.1%})"
            if total_tasks > 0
            else f"Successful API calls: {successful_api_calls}/{total_tasks} (0.0%)"
        )
        print(f"Total tokens used: {self.total_tokens:,}")
        print(f"Total cost: ${self.total_cost:.6f}")

        # Print core metrics
        if results:
            print(f"\n📊 CORE METRICS:")
            print(
                f"  Pass@2 (Weighted Voting): {percentage_metrics['weighted_voting_pass2']:.1%} ({percentage_metrics['weighted_voting_pass2_excl']:.1%} excl. trans)"
            )
            print(
                f"  Pass@2 (Train Majority):  {percentage_metrics['train_majority_pass2']:.1%} ({percentage_metrics['train_majority_pass2_excl']:.1%} excl. trans)"
            )
            print(
                f"  Oracle (Best Attempt):    {percentage_metrics['all_test_correct']:.1%} ({percentage_metrics['all_test_correct_excl']:.1%} excl. trans)"
            )
            print(
                f"  All Train Correct:        {percentage_metrics['all_train_correct_incl']:.1%} ({percentage_metrics['all_train_correct']:.1%} excl. trans)"
            )
            print(
                f"  Min 1 Train Correct:      {percentage_metrics['min1_train_correct_incl']:.1%} ({percentage_metrics['min1_train_correct']:.1%} excl. trans)"
            )
            print(
                f"  Min 1 Code Success:       {percentage_metrics['min1_code_success']:.1%}"
            )
            print(
                f"  Max Length Responses:     {percentage_metrics['max_length_responses']:.1%}"
            )
            print(
                f"  Timeout Responses:        {percentage_metrics['timeout_responses']:.1%}"
            )
            print(
                f"  API Failure Responses:    {percentage_metrics['api_failure_responses']:.1%}"
            )

    def _print_submit_mode_summary(
        self,
        results: List[Dict],
        subset_name: str,
        dataset: str,
        elapsed_time: Optional[float] = None,
    ):
        """Print limited summary for SUBMIT mode (no scoring metrics available)"""
        # Calculate basic statistics
        total_tasks = len(results)
        api_successes = [r for r in results if r.get("api_success", True)]
        successful_api_calls = len(api_successes)

        # Calculate response-level and train metrics (test metrics unavailable without test outputs)
        total_responses = sum(
            len(result.get("attempt_details", [])) for result in results
        )
        max_length_responses = sum(
            sum(
                1
                for att in result.get("attempt_details", [])
                if att.get("hit_max_tokens", False)
            )
            for result in results
        )
        timeout_responses = sum(
            sum(
                1
                for att in result.get("attempt_details", [])
                if att.get("api_timeout", False)
            )
            for result in results
        )
        api_failure_responses = sum(
            sum(
                1
                for att in result.get("attempt_details", [])
                if not att.get("api_success", True)
                and not att.get("api_timeout", False)
            )
            for result in results
        )
        code_success_responses = sum(
            sum(
                1
                for att in result.get("attempt_details", [])
                if att.get("program_extracted", False)
            )
            for result in results
        )

        # Calculate train accuracy metrics (possible since we have train outputs)
        all_train_correct = sum(
            1
            for result in results
            if any(
                att.get("train_accuracy", 0.0) == 1.0
                for att in result.get("attempt_details", [])
            )
        )
        min1_train_correct = sum(
            1
            for result in results
            if any(
                att.get("train_accuracy", 0.0) > 0.0
                for att in result.get("attempt_details", [])
            )
        )

        # Print summary
        run_info = f" (Run {self.run_number})" if self.run_number > 0 else ""
        print("\n" + "=" * 50)
        print(f"SUBMIT MODE SUMMARY{run_info}")
        print("=" * 50)
        print(f"Dataset: {dataset}")
        print(f"Subset: {subset_name}")
        print(f"Model: {self.model}")

        print(f"Total tasks processed: {total_tasks}")
        if elapsed_time:
            print(f"Total time: {elapsed_time:.1f}s")

        print(
            f"Successful API calls: {successful_api_calls}/{total_tasks} ({(successful_api_calls / total_tasks):.1%})"
            if total_tasks > 0
            else f"Successful API calls: {successful_api_calls}/{total_tasks} (0.0%)"
        )
        print(f"Total tokens used: {self.total_tokens:,}")
        print(f"Total cost: ${self.total_cost:.6f}")

        # Print response-level and train metrics (test metrics unavailable without test outputs)
        if total_responses > 0:
            print(f"\n📊 RESPONSE METRICS:")
            print(f"  Total responses: {total_responses}")
            print(
                f"  Code extracted: {code_success_responses}/{total_responses} ({code_success_responses / total_responses:.1%})"
            )
            print(
                f"  Max length responses: {max_length_responses}/{total_responses} ({max_length_responses / total_responses:.1%})"
            )
            print(
                f"  Timeout responses: {timeout_responses}/{total_responses} ({timeout_responses / total_responses:.1%})"
            )
            print(
                f"  API failure responses: {api_failure_responses}/{total_responses} ({api_failure_responses / total_responses:.1%})"
            )

            print(f"\n📊 TRAIN METRICS:")
            print(
                f"  All train correct: {all_train_correct}/{total_tasks} ({all_train_correct / total_tasks:.1%})"
            )
            print(
                f"  Min 1 train correct: {min1_train_correct}/{total_tasks} ({min1_train_correct / total_tasks:.1%})"
            )
            print(
                f"\n⚠️  Note: Test accuracy metrics unavailable in SUBMIT mode (no test outputs)"
            )





def main():
    parser = argparse.ArgumentParser(
        description="Run ARC tasks with all-attempts, rolling execution, and voting-based evaluation"
    )
    parser.add_argument(
        "--dataset",
        default="arc-prize-2025",
        help="Dataset to use (e.g., arc-prize-2025, arc-agi-1, arc-agi-2)",
    )
    parser.add_argument(
        "--subset",
        default="training",
        help="Dataset subset to use (e.g., training, evaluation, unique_training_tasks)",
    )
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model to use")
    parser.add_argument("--limit", type=int, help="Limit number of tasks to run")
    parser.add_argument(
        "--base-url", type=str, help="Base URL for OpenAI-compatible API endpoint"
    )
    parser.add_argument(
        "--max_workers", type=int, default=1, help="Maximum number of parallel workers"
    )
    parser.add_argument(
        "--rate_limit_delay",
        type=float,
        default=0.0,
        help="Delay between API calls in seconds",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=8,
        help="Maximum number of attempts per task",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument(
        "--log-to-db",
        action="store_true",
        default=True,
        help="Log successful programs to local database (default: True)",
    )
    parser.add_argument(
        "--no-log-to-db",
        dest="log_to_db",
        action="store_false",
        help="Disable logging programs to local database",
    )
    parser.add_argument(
        "--max-tokens", type=int, help="Maximum tokens for model responses"
    )
    parser.add_argument(
        "--temperature", type=float, help="Temperature for model responses"
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="low",
        help="Reasoning effort for OpenAI models",
    )
    parser.add_argument(
        "--qwen-no-think",
        action="store_true",
        help="Disable thinking for Qwen models (Note: Not supported by DashScope commercial models)",
    )
    parser.add_argument(
        "--no-transductive-penalty",
        action="store_true",
        help="Disable transductive penalty in voting (default: apply penalty)",
    )
    parser.add_argument(
        "--unsafe-executor",
        action="store_true",
        help="⚠️  UNSAFE: Use unrestricted executor (no Docker sandboxing). Generated code runs directly on your system. SECURITY RISK!",
    )
    parser.add_argument(
        "--prompt_version", type=str, default="soar", help="Version of prompts to use"
    )
    parser.add_argument(
        "--lora-adapter",
        type=str,
        help="LORA adapter name to load on sglang server (e.g., 'ckpt-1057')",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        help="Path to the local database file for storing programs",
    )
    parser.add_argument(
        "--parquet-output-dir",
        type=str,
        help="Directory where parquet files should be saved (overrides default location)",
    )

    args = parser.parse_args()

    # Validation
    if args.max_workers < 1:
        parser.error("--max_workers must be at least 1")
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
        no_transductive_penalty=args.no_transductive_penalty,
        qwen_no_think=args.qwen_no_think,
        prompt_version=args.prompt_version,
        unsafe_executor=args.unsafe_executor,
        lora_adapter=args.lora_adapter,
        sample_name=f"{args.model.replace('/', '_').replace(':', '_')}_{args.dataset}_{args.subset}",
        parquet_output_dir=args.parquet_output_dir,
    )

    # Run the task subset
    results = runner.run_subset(args.subset, args.dataset, args.limit)
    runner.dataset_collector.flush()
    print(f"All sampled programs saved to {runner.dataset_collector.output_path()}")


if __name__ == "__main__":
    main()
