"""
Utility functions for voting and result aggregation.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Callable
from collections import defaultdict, Counter
from .transduction import is_transduction_cheating


def serialize_prediction_for_voting(test_pred) -> str:
    """Convert test prediction to JSON-serializable string key for voting"""
    if test_pred is None:
        return "None"
    
    try:
        # Convert sets to lists and other non-serializable types
        if isinstance(test_pred, set):
            test_pred = list(test_pred)
        elif hasattr(test_pred, 'tolist'):  # numpy arrays
            test_pred = test_pred.tolist()
        return json.dumps(test_pred, sort_keys=True)
    except (TypeError, ValueError):
        # If still not serializable, convert to string representation
        return str(test_pred)


def deserialize_prediction_from_voting(key: str):
    """Convert JSON string key back to prediction for voting results"""
    if key == "None":
        return None
    
    try:
        return json.loads(key)
    except (json.JSONDecodeError, ValueError):
        # If we can't deserialize, this was a string representation
        return None  # Skip invalid entries


def filter_non_transductive_attempts(result: Dict) -> List[Dict]:
    """Filter out transductive attempts from a result's attempt_details"""
    non_transductive = []
    for att in result['attempt_details']:
        # Use stored flag if available (performance optimization), otherwise re-calculate
        if 'is_transductive' in att:
            is_cheat = att['is_transductive']
        else:
            # Fallback for backwards compatibility with older logs
            is_cheat, _, _, _ = is_transduction_cheating(att['program'], result['task_data'])
        
        if not is_cheat:
            non_transductive.append(att)
    return non_transductive


def generic_voting(
    attempts: List[Dict], 
    weighting_func: Callable[[Dict], float], 
    top_k: int = 2
) -> List:
    """
    Generic voting function that can handle different weighting strategies.
    
    Args:
        attempts: List of attempt dictionaries with 'test_predicted' and other fields
        weighting_func: Function that takes an attempt dict and returns a weight
        top_k: Number of top predictions to return
    
    Returns:
        List of top k predictions
    """
    if not attempts:
        return []
    
    # Collect votes with weights
    pattern_stats = defaultdict(lambda: {'total_weight': 0.0, 'attempts': []})
    
    for att in attempts:
        key = serialize_prediction_for_voting(att['test_predicted'])
        weight = weighting_func(att)
        pattern_stats[key]['total_weight'] += weight
        pattern_stats[key]['attempts'].append(att)
    
    # Sort by total weight (descending)
    weighted_patterns = sorted(
        pattern_stats.items(), 
        key=lambda x: x[1]['total_weight'], 
        reverse=True
    )
    
    # Return top k predictions
    top_k_predictions = []
    for key, _ in weighted_patterns[:top_k]:
        prediction = deserialize_prediction_from_voting(key)
        if prediction is not None:  # Skip invalid entries
            top_k_predictions.append(prediction)
    
    return top_k_predictions


def compute_weighted_majority_voting(attempts: List[Dict], top_k: int = 2) -> List:
    """Compute weighted majority voting based on count + 1000 * train_accuracy"""
    def weight_func(att: Dict) -> float:
        return 1.0 + 1000.0 * att.get('train_accuracy', 0.0)
    
    return generic_voting(attempts, weight_func, top_k)


def compute_train_majority_voting(attempts: List[Dict], top_k: int = 2) -> List:
    """Compute train-majority voting for test outputs"""
    if not attempts:
        return []
    
    # Find attempts with most train correct
    best_train_score = max(
        sum(tr['correct'] for tr in att['train_results']) 
        for att in attempts
    )
    
    # Filter to best group and do simple majority voting
    best_group = [
        att for att in attempts 
        if sum(tr['correct'] for tr in att['train_results']) == best_train_score
    ]
    
    def weight_func(att: Dict) -> float:
        return 1.0  # Equal weight for all in best group
    
    return generic_voting(best_group, weight_func, top_k) 