#!/usr/bin/env python3

import math
from typing import Dict, List, Tuple, Any, Union


class ARCTaskValidator:
    """Validates ARC task data integrity and structure"""
    
    @staticmethod
    def validate_tasks(tasks: List[Tuple[str, Dict]]) -> List[Tuple[str, Dict]]:
        """Validate a list of tasks and return only valid ones"""
        total_tasks = len(tasks)
        print(f"🔍 Validating {total_tasks} tasks...")
        
        validated_tasks = []
        for task_id, task_data in tasks:
            if ARCTaskValidator.validate_single_task(task_id, task_data):
                validated_tasks.append((task_id, task_data))
        
        if len(validated_tasks) != total_tasks:
            print(f"⚠️ {total_tasks - len(validated_tasks)} tasks failed validation, using {len(validated_tasks)} valid tasks")
        
        return validated_tasks
    
    @staticmethod
    def validate_single_task(task_id: str, task_data: Any) -> bool:
        """Validate a single task's data structure"""
        # Check if task_data is a dictionary
        if not isinstance(task_data, dict):
            print(f"❌ Invalid task data type for {task_id}: {type(task_data)}")
            return False
        
        # Check for required keys
        if 'train' not in task_data or 'test' not in task_data:
            print(f"❌ Missing train/test data for {task_id}")
            return False
        
        # Check if train and test are lists
        if not isinstance(task_data['train'], list) or not isinstance(task_data['test'], list):
            print(f"❌ Invalid train/test data structure for {task_id}")
            return False
        
        # Check if there are examples
        if len(task_data['train']) == 0:
            print(f"❌ No training examples for {task_id}")
            return False
        
        if len(task_data['test']) == 0:
            print(f"❌ No test examples for {task_id}")
            return False
        
        # Validate structure of train/test examples
        if not ARCTaskValidator._validate_examples(task_id, task_data['train'], 'train'):
            return False
        
        if not ARCTaskValidator._validate_examples(task_id, task_data['test'], 'test'):
            return False
        
        return True
    
    @staticmethod
    def _validate_examples(task_id: str, examples: List[Dict], example_type: str) -> bool:
        """Validate the structure of train or test examples"""
        for i, example in enumerate(examples):
            if not isinstance(example, dict):
                print(f"❌ Invalid {example_type} example {i} for {task_id}: not a dict")
                return False
            
            # Train examples should have both input and output
            # Test examples should have input, and may or may not have output
            if 'input' not in example:
                print(f"❌ Missing input in {example_type} example {i} for {task_id}")
                return False
            
            if example_type == 'train' and 'output' not in example:
                print(f"❌ Missing output in {example_type} example {i} for {task_id}")
                return False
            
            # Validate grid structure (should be list of lists)
            if not ARCTaskValidator._validate_grid(example['input'], f"{task_id} {example_type}[{i}] input"):
                return False
            
            if 'output' in example and example['output'] is not None:
                if not ARCTaskValidator._validate_grid(example['output'], f"{task_id} {example_type}[{i}] output"):
                    return False
        
        return True
    
    @staticmethod
    def _validate_grid(grid: Any, description: str, verbose: bool = False) -> bool:
        """Validate that a grid is a list of lists with consistent dimensions"""
        if not isinstance(grid, list):
            if verbose:
                print(f"❌ Invalid grid for {description}: not a list")
            return False
        
        if len(grid) == 0:
            if verbose:
                print(f"❌ Empty grid for {description}")
            return False
        
        # Check first row to get expected width
        if not isinstance(grid[0], list):
            if verbose:
                print(f"❌ Invalid grid for {description}: first row not a list")
            return False
        
        expected_width = len(grid[0])
        if expected_width == 0:
            if verbose:
                print(f"❌ Empty first row in grid for {description}")
            return False
        
        # Check grid size limits (1x1 to 30x30 per ARC competition requirements)
        height = len(grid)
        width = expected_width
        if not (1 <= height <= 30):
            if verbose:
                print(f"❌ Invalid grid height for {description}: {height} (must be 1-30)")
            return False
        if not (1 <= width <= 30):
            if verbose:
                print(f"❌ Invalid grid width for {description}: {width} (must be 1-30)")
            return False
        
        # Validate all rows
        for i, row in enumerate(grid):
            if not isinstance(row, list):
                if verbose:
                    print(f"❌ Invalid grid for {description}: row {i} not a list")
                return False
            
            if len(row) != expected_width:
                if verbose:
                    print(f"❌ Inconsistent grid width for {description}: row {i} has {len(row)} items, expected {expected_width}")
                return False
            
            # Check that all values are integers in range 0-9 (accept numpy types)
            for j, cell in enumerate(row):
                # Accept both Python int and numpy integer types
                try:
                    # Convert to Python int if it's a numeric type
                    cell_int = int(cell)
                except (ValueError, TypeError):
                    if verbose:
                        print(f"❌ Invalid cell value for {description} at [{i}][{j}]: {cell} (type: {type(cell)}, cannot convert to int)")
                    return False
                
                if not (0 <= cell_int <= 9):
                    if verbose:
                        print(f"❌ Cell value out of range for {description} at [{i}][{j}]: {cell_int} (should be 0-9)")
                    return False
                
                # Convert non-integer types (like bool, numpy types) to standard Python int
                if not isinstance(cell, int) or cell != cell_int:
                    print(f"⚠️ Converting cell value in {description} at [{i}][{j}]: {cell} (type: {type(cell)}) -> {cell_int}")
                    row[j] = cell_int
        
        return True
    
    @staticmethod
    def validate_prediction(prediction: Any, description: str = "prediction", verbose: bool = False) -> bool:
        """Validate that a prediction matches ARC grid format"""
        return ARCTaskValidator._validate_grid(prediction, description, verbose)
    
    @staticmethod
    def validate_prediction_list(predictions: List[Any], description: str = "predictions") -> Tuple[bool, List[str]]:
        """
        Validate a list of predictions for ARC tasks.
        
        Args:
            predictions: List of predicted grids (may contain None for execution errors)
            description: Description for error messages
            
        Returns:
            (is_valid, errors): Tuple of validation status and list of error messages
        """
        errors = []
        
        if not predictions:
            errors.append(f"{description}: No predictions provided")
            return False, errors
            
        for i, pred in enumerate(predictions):
            if pred is None:
                errors.append(f"{description}[{i}]: Execution error (None prediction)")
            elif not ARCTaskValidator.validate_prediction(pred, f"{description}[{i}]"):
                errors.append(f"{description}[{i}]: Invalid ARC grid format")
                
        return len(errors) == 0, errors


def replace_invalid_grid(grid: Union[List, None], task_id: str = "", attempt_name: str = "") -> List[List[int]]:
    """
    Replace invalid grids with ARC competition-compliant fallback.
    
    Args:
        grid: Input grid that may be invalid
        task_id: Task identifier for logging
        attempt_name: Attempt name for logging
        
    Returns:
        Valid ARC grid (2x2 fallback if input is invalid)
    """
    fallback_grid = [[0, 0], [0, 0]]
    
    if grid is None:
        return fallback_grid
    
    # If it's a flat list, we cannot safely reshape without knowing intended dimensions
    if isinstance(grid, list) and len(grid) > 0 and not isinstance(grid[0], list):
        if task_id and attempt_name:
            print(f"⚠️  {task_id} {attempt_name}: Found flat list with {len(grid)} elements, cannot determine intended grid shape, using fallback")
        return fallback_grid
    
    # If it's a 2D list, validate it meets ARC requirements
    if isinstance(grid, list) and len(grid) > 0 and isinstance(grid[0], list):
        if ARCTaskValidator.validate_prediction(grid, f"{task_id}_{attempt_name}" if task_id and attempt_name else "grid"):
            return grid
        elif task_id and attempt_name:
            print(f"⚠️  {task_id} {attempt_name}: Grid failed ARC validation, using fallback")
        return fallback_grid
    
    # Fallback for any other case
    return fallback_grid