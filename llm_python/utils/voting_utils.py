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
    except (json.JSONDecodeError, ValueError) as e:
        # If we can't deserialize, try to return the string representation
        print(f"    Warning: Failed to deserialize key as JSON: {e}")
        print(f"    Attempting to eval as Python literal...")
        try:
            import ast
            return ast.literal_eval(key)
        except (ValueError, SyntaxError) as e2:
            print(f"    Failed to eval as Python literal: {e2}")
            print(f"    Returning key as-is: {repr(key)}")
            return key  # Return the string itself rather than None






def generic_voting(
    attempts: List[Dict],
    weighting_func: Callable[[Dict], float],
    top_k: int = 2,
    task_id: str = None
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

    # print("\n" + "="*50)
    # print(f"VOTING DEBUG - Starting task: {task_id or 'UNKNOWN'}")
    # print("="*50)

    # Collect votes with weights
    pattern_stats = defaultdict(lambda: {'total_weight': 0.0, 'attempts': []})

    # Assign a stable index to each key as it is first seen
    key_to_index = {}

    # print(f"\nProcessing {len(attempts)} attempts:")
    # print("-" * 30)

    for i, att in enumerate(attempts):
        key = serialize_prediction_for_voting(att.get('test_predicted'))
        if key not in key_to_index:
            key_to_index[key] = len(key_to_index)

        weight = weighting_func(att)
        pattern_stats[key]['total_weight'] += weight
        pattern_stats[key]['attempts'].append(att)

        # Debug print for each attempt
        train_acc = att.get('train_accuracy', 0.0)
        transductive = att.get('is_transductive', False)
        trans_conf = att.get('transduction_confidence', 'N/A')
        # print(f"Attempt {i+1}: Key #{key_to_index[key]} | Weight: {weight:.4f} | Train acc: {train_acc:.3f} | Transductive: {transductive} | Trans conf: {trans_conf}")

    # Sort by total weight (descending)
    weighted_patterns = sorted(
        pattern_stats.items(),
        key=lambda x: x[1]['total_weight'],
        reverse=True
    )

    # print(f"\nPattern/Key weights summary:")
    # print("-" * 30)
    # for key, stats in weighted_patterns:
    #     print(f"Key #{key_to_index[key]} | Total Weight: {stats['total_weight']:.4f} | From {len(stats['attempts'])} attempts")

    # Return top k predictions
    top_k_predictions = []
    # print(f"\nDeserializing top {top_k} predictions:")
    # print("-" * 30)
    for i, (key, stats) in enumerate(weighted_patterns[:top_k]):
        # print(f"Key #{key_to_index[key]}: '{key[:100]}{'...' if len(key) > 100 else ''}'")
        prediction = deserialize_prediction_from_voting(key)
        if prediction is not None:
            top_k_predictions.append(prediction)
            # print(f"  -> Successfully deserialized")
        # else:
            # print(f"  -> Failed to deserialize, skipping")

    # print(f"\nReturning top {len(top_k_predictions)} predictions")
    # print("="*50 + "\n")

    return top_k_predictions


def compute_weighted_majority_voting(attempts: List[Dict], top_k: int = 2, no_transductive_penalty: bool = False, task_id: str = None) -> List:
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
    
    return generic_voting(valid_attempts, weight_func, top_k, task_id)


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