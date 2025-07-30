"""
Utility functions for calculating metrics from task results.
"""

from typing import Dict, List, Optional
from .voting_utils import (
    filter_non_transductive_attempts,
    compute_weighted_majority_voting,
    compute_train_majority_voting
)

def calculate_task_metrics(
    results: List[Dict], 
    upto_attempt: Optional[int] = None,
    max_tokens: Optional[int] = None
) -> Dict:
    """
    Calculate comprehensive metrics from a list of task results.
    
    Args:
        results: List of task result dictionaries
        upto_attempt: If specified, only consider attempts up to this index (1-based)
        max_tokens: Maximum tokens setting for max_length calculation
    
    Returns:
        Dictionary containing all calculated metrics
    """
    test_correct = 0
    train_majority_correct = 0
    oracle_test_correct = 0
    all_train_correct = 0
    min1_train_correct = 0
    min1_transductive = 0
    min1_code_runs = 0
    
    # Response-level metrics (across all attempts/layers)
    total_responses = 0
    max_length_responses = 0
    timeout_responses = 0
    api_failure_responses = 0
    
    total = 0
    
    for result in results:
        if not result:
            continue
        
        # Filter attempts if upto_attempt is specified
        attempt_details = result['attempt_details']
        if upto_attempt is not None:
            attempt_details = attempt_details[:upto_attempt]
        
        # Count response-level metrics across ALL attempts (including transductive)
        for att in attempt_details:
            total_responses += 1
            
            # Max-length responses
            if max_tokens and att.get('output_tokens', 0) >= max_tokens:
                max_length_responses += 1
            
            # Timeout responses
            if att.get('api_timeout', False):
                timeout_responses += 1
            
            # API failure responses (failures after retries, but not timeouts)
            if not att.get('api_success', True) and not att.get('api_timeout', False):
                api_failure_responses += 1
        
        # Filter out transductive attempts using optimized function
        non_transductive = filter_non_transductive_attempts({
            'attempt_details': attempt_details,
            'task_data': result['task_data']
        })
        
        # Check if any attempts were transductive
        if len(attempt_details) > len(non_transductive):
            min1_transductive += 1
        
        if not non_transductive:
            # Even if no non-transductive attempts, still count this task
            total += 1
            continue
        
        # Get ground truth from task_data (avoiding redundant test_expected field)
        gt = result['task_data']['test'][0]['output']
        
        # Weighted majority voting (pass@2)
        try:
            top2_weighted = compute_weighted_majority_voting(non_transductive)
            if any(t == gt for t in top2_weighted):
                test_correct += 1
        except Exception:
            pass  # Skip if voting fails
        
        # Train-majority voting (pass@2)
        try:
            top2_train_majority = compute_train_majority_voting(non_transductive)
            if any(t == gt for t in top2_train_majority):
                train_majority_correct += 1
        except Exception:
            pass  # Skip if voting fails
        
        # Oracle test correct (any attempt got test correct)
        if any(att['test_correct'] for att in non_transductive):
            oracle_test_correct += 1
        
        # Oracle all-train correct (any attempt got all training examples correct)
        if any(all(tr['correct'] for tr in att['train_results']) for att in non_transductive):
            all_train_correct += 1
        
        # Min-1-train correct (any attempt got at least one training example correct)
        if any(any(tr['correct'] for tr in att['train_results']) for att in non_transductive):
            min1_train_correct += 1
        
        # Min-1-code runs (any attempt had code that executed successfully)
        # Check across ALL attempts (not just non-transductive) for code execution
        if any(
            att.get('program', '').strip() and  # Has non-empty code
            (att.get('test_correct', False) or  # Either test executed
             (att.get('train_results', []) and len(att['train_results']) > 0))  # Or training executed
            for att in attempt_details
        ):
            min1_code_runs += 1
        
        total += 1
    
    return {
        'test_correct': test_correct,
        'train_majority_correct': train_majority_correct,
        'oracle_test_correct': oracle_test_correct,
        'all_train_correct': all_train_correct,
        'min1_train_correct': min1_train_correct,
        'min1_transductive': min1_transductive,
        'min1_code_runs': min1_code_runs,
        'total_responses': total_responses,
        'max_length_responses': max_length_responses,
        'timeout_responses': timeout_responses,
        'api_failure_responses': api_failure_responses,
        'total': total
    }


def format_metrics_display(metrics: Dict, layer: Optional[int] = None) -> str:
    """
    Format metrics dictionary into a human-readable string.
    
    Args:
        metrics: Dictionary from calculate_task_metrics()
        layer: Optional layer number for display
    
    Returns:
        Formatted string for display
    """
    total = metrics['total']
    total_responses = metrics['total_responses']
    
    if total == 0:
        return "No valid tasks found."
    
    header = f"[Layer {layer}] Metrics:" if layer else "Metrics:"
    
    lines = [
        f"\n{header}",
        f"  Test correct (oracle): {metrics['oracle_test_correct']}/{total} ({metrics['oracle_test_correct']/total:.1%})",
        f"  Test correct (pass@2, train-majority): {metrics['train_majority_correct']}/{total} ({metrics['train_majority_correct']/total:.1%})",
        f"  Test correct (pass@2, weighted voting): {metrics['test_correct']}/{total} ({metrics['test_correct']/total:.1%})",
        f"  All-train correct (oracle): {metrics['all_train_correct']}/{total} ({metrics['all_train_correct']/total:.1%})",
        f"  Min-1-train correct (oracle): {metrics['min1_train_correct']}/{total} ({metrics['min1_train_correct']/total:.1%})",
        f"  Min-1-code runs: {metrics['min1_code_runs']}/{total} ({metrics['min1_code_runs']/total:.1%})",
        f"  Min-1-transductive: {metrics['min1_transductive']}/{total} ({metrics['min1_transductive']/total:.1%})",
    ]
    
    # Response-level metrics (as percentages of total responses)
    if total_responses > 0:
        lines.extend([
            f"  Max-length responses: {metrics['max_length_responses']}/{total_responses} ({metrics['max_length_responses']/total_responses:.1%})",
            f"  Timeout responses: {metrics['timeout_responses']}/{total_responses} ({metrics['timeout_responses']/total_responses:.1%})",
            f"  API failure responses: {metrics['api_failure_responses']}/{total_responses} ({metrics['api_failure_responses']/total_responses:.1%})"
        ])
    else:
        lines.extend([
            f"  Max-length responses: 0/0 (0.0%)",
            f"  Timeout responses: 0/0 (0.0%)",
            f"  API failure responses: 0/0 (0.0%)"
        ])
    
    return "\n".join(lines)


def metrics_to_percentages(metrics: Dict) -> Dict:
    """
    Convert raw metrics counts to percentage format.
    
    Args:
        metrics: Dictionary from calculate_task_metrics()
    
    Returns:
        Dictionary with percentage values instead of counts
    """
    total = metrics['total']
    total_responses = metrics['total_responses']
    
    if total == 0:
        return {key: 0.0 for key in metrics.keys() if key not in ['total', 'total_responses']}
    
    result = {
        'weighted_voting_pass2': metrics['test_correct'] / total,
        'train_majority_pass2': metrics['train_majority_correct'] / total,
        'oracle_correct': metrics['oracle_test_correct'] / total,
        'all_train_correct': metrics['all_train_correct'] / total,
        'min1_train_correct': metrics['min1_train_correct'] / total,
        'min1_transductive': metrics['min1_transductive'] / total,
        'min1_code_runs': metrics['min1_code_runs'] / total,
        'total_tasks': total,
        'total_responses': total_responses
    }
    
    # Response-level percentages
    if total_responses > 0:
        result.update({
            'max_length_responses': metrics['max_length_responses'] / total_responses,
            'timeout_responses': metrics['timeout_responses'] / total_responses,
            'api_failure_responses': metrics['api_failure_responses'] / total_responses,
        })
    else:
        result.update({
            'max_length_responses': 0.0,
            'timeout_responses': 0.0,
            'api_failure_responses': 0.0,
        })
    
    return result 