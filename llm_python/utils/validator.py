#!/usr/bin/env python3

from typing import Dict, List, Tuple, Any


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
        
        print(f"✅ Task validation complete: {len(validated_tasks)} valid tasks")
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
            
            if 'output' in example:
                if not ARCTaskValidator._validate_grid(example['output'], f"{task_id} {example_type}[{i}] output"):
                    return False
        
        return True
    
    @staticmethod
    def _validate_grid(grid: Any, description: str) -> bool:
        """Validate that a grid is a list of lists with consistent dimensions"""
        if not isinstance(grid, list):
            print(f"❌ Invalid grid for {description}: not a list")
            return False
        
        if len(grid) == 0:
            print(f"❌ Empty grid for {description}")
            return False
        
        # Check first row to get expected width
        if not isinstance(grid[0], list):
            print(f"❌ Invalid grid for {description}: first row not a list")
            return False
        
        expected_width = len(grid[0])
        if expected_width == 0:
            print(f"❌ Empty first row in grid for {description}")
            return False
        
        # Validate all rows
        for i, row in enumerate(grid):
            if not isinstance(row, list):
                print(f"❌ Invalid grid for {description}: row {i} not a list")
                return False
            
            if len(row) != expected_width:
                print(f"❌ Inconsistent grid width for {description}: row {i} has {len(row)} items, expected {expected_width}")
                return False
            
            # Check that all values are integers (ARC grids contain integers 0-9)
            for j, cell in enumerate(row):
                if not isinstance(cell, int):
                    print(f"❌ Invalid cell value for {description} at [{i}][{j}]: {cell} (type: {type(cell)})")
                    return False
                
                if not (0 <= cell <= 9):
                    print(f"❌ Cell value out of range for {description} at [{i}][{j}]: {cell} (should be 0-9)")
                    return False
        
        return True
    
    @staticmethod
    def validate_prediction(prediction: Any, description: str = "prediction") -> bool:
        """Validate that a prediction matches ARC grid format"""
        return ARCTaskValidator._validate_grid(prediction, description)