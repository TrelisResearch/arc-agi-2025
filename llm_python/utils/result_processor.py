#!/usr/bin/env python3

import json
import datetime
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .serialization import ResponseSerializer
from .metrics_utils import calculate_task_metrics, metrics_to_percentages


class ResultProcessor:
    """Handles saving and processing of ARC task results"""
    
    def __init__(self, model: str, run_number: int = 0, max_tokens: Optional[int] = None):
        self.model = model
        self.run_number = run_number
        self.max_tokens = max_tokens
        
        # Create run-specific subdirectory
        run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logs_dir = Path("llm_python/logs") / run_timestamp
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread-safe tracking
        self.large_file_errors = []
        self._results_lock = threading.Lock()
    
    def _check_file_size_before_save(self, data: Dict, filename: str) -> Tuple[bool, str]:
        """Check if the data would create a problematically large file"""
        try:
            # Estimate size by serializing to JSON string
            json_str = ResponseSerializer.safe_json_dumps(data)
            size_bytes = len(json_str.encode('utf-8'))
            size_mb = size_bytes / (1024 * 1024)
            
            # Define size limits
            max_size_mb = 100  # 100MB limit per file
            warn_size_mb = 10  # Warn at 10MB
            
            if size_mb > max_size_mb:
                error_msg = f"File {filename} would be {size_mb:.1f}MB (exceeds {max_size_mb}MB limit)"
                # Thread-safe error tracking
                with self._results_lock:
                    self.large_file_errors.append({
                        'filename': filename,
                        'size_mb': size_mb,
                        'timestamp': datetime.datetime.now().isoformat()
                    })
                return False, error_msg
            elif size_mb > warn_size_mb:
                print(f"⚠️  Large file warning: {filename} will be {size_mb:.1f}MB")
            
            return True, ""
            
        except Exception as e:
            print(f"⚠️  Could not estimate file size for {filename}: {e}")
            return True, ""  # Allow save on error to avoid blocking execution
    
    def save_result(self, result: Dict):
        """Save individual task result"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        thread_id = threading.get_ident()
        
        if self.run_number > 0:
            filename = f"{timestamp}_{thread_id}_{result['task_id']}_simple_run{self.run_number}.json"
        else:
            filename = f"{timestamp}_{thread_id}_{result['task_id']}_simple.json"
        
        filepath = self.logs_dir / filename
        
        # Sanitize for JSON and check size before saving
        safe_result = ResponseSerializer.make_json_safe(result)
        should_save, size_error = self._check_file_size_before_save(safe_result, filename)
        
        if not should_save:
            print(f"🚨 LARGE FILE ERROR: {size_error}")
            print(f"   Task: {result['task_id']}")
            print(f"   Skipping file save to prevent disk issues")
            return  # Skip saving but continue execution
        
        try:
            with open(filepath, 'w') as f:
                json.dump(safe_result, f, indent=2)
        except Exception as e:
            print(f"🚨 FILE I/O ERROR: Task {result['task_id']}, Attempt {result.get('attempt_details', [{}])[0].get('attempt_number', '?')}")
            print(f"   File: {filepath}")
            print(f"   Error: {e}")
            raise Exception(f"File I/O error for task {result['task_id']}: {e}")
    
    def save_summary(self, results: List[Dict], subset_name: str, dataset: str, 
                    timeout_occurred: bool = False, total_tokens: int = 0, total_cost: float = 0.0):
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
                'max_length_responses': 0.0,
                'timeout_responses': 0.0,
                'api_failure_responses': 0.0
            }
        
        # Determine summary type and calculate additional timeout stats
        if timeout_occurred:
            api_type_summary = 'chat_completions_timeout_partial'
            partial_tasks = sum(1 for r in results if r.get('is_partial', False))
            complete_tasks = total_tasks - partial_tasks
        else:
            api_type_summary = 'chat_completions_all_attempts'
            partial_tasks = 0
            complete_tasks = total_tasks
        
        # Build compact results for summary (strip heavy fields like raw_response/program/all_responses)
        results_for_summary = self._create_compact_results(results)
        
        # Create summary
        summary = {
            'timestamp': timestamp,
            'dataset': dataset,
            'subset': subset_name,
            'model': self.model,
            'api_type': api_type_summary,
            'run_number': self.run_number,
            'total_tasks': total_tasks,
            'complete_tasks': complete_tasks,
            'partial_tasks': partial_tasks,
            'timeout_occurred': timeout_occurred,
            'successful_api_calls': successful_api_calls,
            'total_tokens': total_tokens,
            'total_cost': total_cost,
            'metrics': percentage_metrics,
            'results': results_for_summary
        }
        
        # Add large file error summary if any occurred
        if self.large_file_errors:
            summary['large_file_errors'] = self.large_file_errors
            summary['large_file_count'] = len(self.large_file_errors)
        
        # Save summary
        if self.run_number > 0:
            summary_filename = f"{timestamp}_summary_{dataset}_{subset_name}_{self.model}_run{self.run_number}.json"
        else:
            summary_filename = f"{timestamp}_summary_{dataset}_{subset_name}_{self.model}.json"
        
        summary_filepath = self.logs_dir / summary_filename
        
        # Check size before saving summary
        safe_summary = ResponseSerializer.make_json_safe(summary)
        should_save_summary, size_error = self._check_file_size_before_save(safe_summary, summary_filename)
        
        if should_save_summary:
            try:
                with open(summary_filepath, 'w') as f:
                    json.dump(safe_summary, f, indent=2)
                print(f"📄 Summary saved: {summary_filepath}")
            except Exception as e:
                print(f"🚨 SUMMARY I/O ERROR: {e}")
                print(f"   File: {summary_filepath}")
        else:
            print(f"🚨 LARGE SUMMARY ERROR: {size_error}")
            print("   Summary not saved due to size constraints")
        
        return summary
    
    def _create_compact_results(self, results: List[Dict]) -> List[Dict]:
        """Create compact version of results by removing heavy fields"""
        results_for_summary = []
        for r in results:
            try:
                compact = {k: v for k, v in r.items() if k != 'all_responses'}
                compact_attempts = []
                for att in r.get('attempt_details', []):
                    if isinstance(att, dict):
                        compact_att = {k: v for k, v in att.items() if k not in ('raw_response', 'program')}
                        compact_attempts.append(compact_att)
                    else:
                        compact_attempts.append(att)
                compact['attempt_details'] = compact_attempts
                results_for_summary.append(compact)
            except Exception:
                # Fallback to original entry on unexpected structure
                results_for_summary.append(r)
        
        return results_for_summary
    
    def get_large_file_errors(self) -> List[Dict]:
        """Get list of large file errors that occurred"""
        with self._results_lock:
            return self.large_file_errors.copy()