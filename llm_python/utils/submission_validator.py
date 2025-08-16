"""
Submission file validation utility for ARC competitions.
Validates submission.json files against competition guidelines.
"""

import json
import math
from typing import List, Union


def ensure_2d_grid(grid: Union[List, None], task_id: str = "", attempt_name: str = "") -> List[List[int]]:
    """
    Ensure grid is a proper 2D list, fix if needed.
    
    Args:
        grid: Input grid that may be None, 1D flat list, or proper 2D list
        task_id: Task identifier for logging
        attempt_name: Attempt name for logging
        
    Returns:
        Valid 2D grid as list of lists
    """
    if grid is None:
        return [[0, 0], [0, 0]]
    
    # If it's a flat list, try to reshape it
    if isinstance(grid, list) and len(grid) > 0 and not isinstance(grid[0], list):
        if task_id and attempt_name:
            print(f"⚠️  {task_id} {attempt_name}: Found flattened grid with {len(grid)} elements, attempting to reshape")
        
        # Try common square shapes first
        sqrt_len = int(math.sqrt(len(grid)))
        if sqrt_len * sqrt_len == len(grid):
            # Perfect square, reshape to square grid
            reshaped = []
            for i in range(sqrt_len):
                reshaped.append(grid[i*sqrt_len:(i+1)*sqrt_len])
            if task_id and attempt_name:
                print(f"   Reshaped to {sqrt_len}x{sqrt_len} grid")
            return reshaped
        else:
            if task_id and attempt_name:
                print(f"   Cannot reshape {len(grid)} elements to square grid, using fallback")
            return [[0, 0], [0, 0]]
    
    # If it's already 2D, validate it
    if isinstance(grid, list) and len(grid) > 0 and isinstance(grid[0], list):
        return grid
    
    # Fallback for any other case
    return [[0, 0], [0, 0]]


def validate_submission_file(submission_path: str, expected_task_ids: List[str]) -> None:
    """Validate submission file format against competition guidelines"""
    print(f"\n🔍 VALIDATING SUBMISSION: {submission_path}")
    
    errors = []
    warnings = []
    
    try:
        # Load the submission file
        with open(submission_path, 'r') as f:
            submission = json.load(f)
        
        # Check 1: File is a dictionary
        if not isinstance(submission, dict):
            errors.append("Submission file must be a JSON dictionary")
            return
        
        # Check 2: All expected task IDs are present
        submission_task_ids = set(submission.keys())
        expected_task_ids_set = set(expected_task_ids)
        
        missing_tasks = expected_task_ids_set - submission_task_ids
        extra_tasks = submission_task_ids - expected_task_ids_set
        
        if missing_tasks:
            errors.append(f"Missing task IDs: {sorted(list(missing_tasks))[:10]}{'...' if len(missing_tasks) > 10 else ''} ({len(missing_tasks)} total)")
        
        if extra_tasks:
            warnings.append(f"Extra task IDs not in dataset: {sorted(list(extra_tasks))[:10]}{'...' if len(extra_tasks) > 10 else ''} ({len(extra_tasks)} total)")
        
        # Check 3: Validate structure for each task
        invalid_structure_count = 0
        empty_predictions_count = 0
        
        for task_id, predictions in submission.items():
            # Must be a list
            if not isinstance(predictions, list):
                errors.append(f"Task {task_id}: predictions must be a list")
                continue
            
            # Must have at least one prediction
            if len(predictions) == 0:
                errors.append(f"Task {task_id}: must have at least one prediction")
                continue
            
            # Check each prediction
            for i, pred in enumerate(predictions):
                if not isinstance(pred, dict):
                    errors.append(f"Task {task_id}[{i}]: prediction must be a dictionary")
                    continue
                
                # Must have both attempt_1 and attempt_2
                if "attempt_1" not in pred:
                    errors.append(f"Task {task_id}[{i}]: missing 'attempt_1'")
                if "attempt_2" not in pred:
                    errors.append(f"Task {task_id}[{i}]: missing 'attempt_2'")
                
                # Check if attempts are grids (list of lists)
                for attempt_key in ["attempt_1", "attempt_2"]:
                    if attempt_key in pred:
                        attempt = pred[attempt_key]
                        if not isinstance(attempt, list):
                            errors.append(f"Task {task_id}[{i}].{attempt_key}: must be a list (grid)")
                        elif len(attempt) > 0 and not isinstance(attempt[0], list):
                            errors.append(f"Task {task_id}[{i}].{attempt_key}: must be a list of lists (2D grid)")
                        elif attempt == [[0, 0], [0, 0]]:
                            empty_predictions_count += 1
        
        # Summary statistics
        total_predictions = sum(len(predictions) for predictions in submission.values())
        unique_tasks = len(submission)
        
        print(f"📊 Validation Results:")
        print(f"  Total tasks: {unique_tasks}")
        print(f"  Total predictions: {total_predictions}")
        print(f"  Empty predictions ([[0,0],[0,0]]): {empty_predictions_count}")
        
        # Report errors and warnings
        if errors:
            print(f"❌ VALIDATION FAILED - {len(errors)} errors:")
            for error in errors[:10]:  # Show first 10 errors
                print(f"   • {error}")
            if len(errors) > 10:
                print(f"   ... and {len(errors) - 10} more errors")
        else:
            print(f"✅ VALIDATION PASSED - No structural errors found")
        
        if warnings:
            print(f"⚠️  {len(warnings)} warnings:")
            for warning in warnings[:5]:  # Show first 5 warnings
                print(f"   • {warning}")
            if len(warnings) > 5:
                print(f"   ... and {len(warnings) - 5} more warnings")
        
        # Final status
        if not errors:
            print(f"🎯 Submission file is ready for competition!")
        else:
            print(f"🚨 Fix errors before submitting!")
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
    except Exception as e:
        print(f"❌ Validation error: {e}")