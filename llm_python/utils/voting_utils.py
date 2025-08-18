"""
Utility functions for voting and result aggregation.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Callable
from collections import defaultdict, Counter
from .validator import ARCTaskValidator


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
    """Filter out train-transductive attempts from a result's attempt_details"""
    non_transductive = []
    for att in result['attempt_details']:
        # Use stored flag if available, otherwise re-calculate
        if 'is_train_transductive' in att:
            is_train_transductive = att['is_train_transductive']
        else:
            # Re-calculate if not stored (for older data)
            from .transduction import detect_transduction
            is_train_transductive, _, _, _ = detect_transduction(att.get('program', ''), result['task_data'])
        
        if not is_train_transductive:
            non_transductive.append(att)
    return non_transductive


def filter_valid_predictions(attempts: List[Dict]) -> List[Dict]:
    """Filter out attempts with invalid ARC grid predictions"""
    valid_attempts = []
    for att in attempts:
        test_predicted = att.get('test_predicted')
        
        # Handle different prediction formats
        if test_predicted is None:
            continue
            
        # Check if prediction is valid (test_predicted should be a list of grids)
        is_valid = True
        if isinstance(test_predicted, list):
            # Validate each grid in the list
            for grid in test_predicted:
                if grid is not None and not ARCTaskValidator.validate_prediction(grid, "voting_filter"):
                    is_valid = False
                    break
        else:
            # Invalid format - test_predicted should always be a list
            is_valid = False
        
        if is_valid:
            valid_attempts.append(att)
    
    return valid_attempts


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
        key = serialize_prediction_for_voting(att.get('test_predicted'))
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
        sum(tr.get('correct', False) for tr in att.get('train_results', [])) 
        for att in attempts
    )
    
    # Filter to best group and do simple majority voting
    best_group = [
        att for att in attempts 
        if sum(tr.get('correct', False) for tr in att.get('train_results', [])) == best_train_score
    ]
    
    def weight_func(att: Dict) -> float:
        return 1.0  # Equal weight for all in best group
    
    return generic_voting(best_group, weight_func, top_k) 