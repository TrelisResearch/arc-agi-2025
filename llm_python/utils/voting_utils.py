"""
Utility functions for voting and result aggregation.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Callable
from collections import defaultdict, Counter


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


def compute_weighted_majority_voting(attempts: List[Dict], top_k: int = 2, no_transductive_penalty: bool = False) -> List:
    """Compute weighted majority voting based on count + 1000 * train_accuracy, with optional transductive confidence penalty"""
    # Filter out attempts with invalid outputs
    valid_attempts = [att for att in attempts if att.get('outputs_valid', True)]
    
    def weight_func(att: Dict) -> float:
        train_acc = att.get('train_accuracy', 0.0)
        base_weight = 1.0 + 1000.0 * train_acc
        
        # Apply transductive penalty using (1 - confidence)² unless disabled
        if not no_transductive_penalty:
            confidence = att.get("transduction_confidence")
            if confidence is None:
                # Log missing confidence and default to no penalty
                print(f"⚠️ Missing transduction_confidence, defaulting to 0.0 (no penalty)")
                confidence = 0.0
            penalty = (1 - confidence) ** 2
            return base_weight * penalty
        
        return base_weight
    
    return generic_voting(valid_attempts, weight_func, top_k)


def compute_train_majority_voting(attempts: List[Dict], top_k: int = 2) -> List:
    """Compute train-majority voting for test outputs with transductive confidence penalty"""
    # Filter out attempts with invalid outputs
    valid_attempts = [att for att in attempts if att.get('outputs_valid', True)]
    
    if not valid_attempts:
        return []
    
    # Calculate effective train score with transductive penalty
    def effective_train_score(att: Dict) -> float:
        train_correct = sum(tr.get('correct', False) for tr in att.get('train_results', []))
        if att.get("is_transductive", False):
            confidence = att.get("transduction_confidence", 1.0)
            penalty = (1 - confidence) ** 2
            return train_correct * penalty
        return train_correct
    
    # Find best effective score
    best_train_score = max(effective_train_score(att) for att in valid_attempts)
    
    # Filter to best group
    best_group = [
        att for att in valid_attempts 
        if abs(effective_train_score(att) - best_train_score) < 0.001  # Small epsilon for float comparison
    ]
    
    def weight_func(att: Dict) -> float:
        # Apply same transductive penalty in final weighting
        if att.get("is_transductive", False):
            confidence = att.get("transduction_confidence", 1.0)
            return (1 - confidence) ** 2
        return 1.0
    
    return generic_voting(best_group, weight_func, top_k) 