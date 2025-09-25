#!/usr/bin/env python3
"""
Program validation for augmented tasks using existing ArcTester infrastructure.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from llm_python.utils.arc_tester import ArcTester
from llm_python.datasets.io import read_soar_parquet


class ProgramValidator:
    """Validates programs against augmented tasks."""

    def __init__(self, timeout: int = 15):
        """Initialize validator with ArcTester."""
        self.tester = ArcTester(timeout=timeout, executor_type="unrestricted")

    def validate_program_on_task(self, program_code: str, task_data: Dict[str, Any]) -> Tuple[bool, List[bool], List[bool]]:
        """
        Test a program against a task and return correctness results.

        Args:
            program_code: Python code containing transform function
            task_data: Task data with 'train' and 'test' examples

        Returns:
            Tuple of (all_correct, train_correct_list, test_correct_list)
        """
        train_correct = []
        test_correct = []

        # Test on training examples
        for example in task_data['train']:
            pred, err, timeout = self.tester.execute_program_with_timeout(program_code, example['input'])

            if pred is not None and not err and not timeout:
                is_correct = (pred == example['output'])
            else:
                is_correct = False

            train_correct.append(is_correct)

        # Test on test examples (if we have expected outputs)
        for example in task_data['test']:
            if 'output' in example and example['output'] is not None:
                pred, err, timeout = self.tester.execute_program_with_timeout(program_code, example['input'])

                if pred is not None and not err and not timeout:
                    is_correct = (pred == example['output'])
                else:
                    is_correct = False

                test_correct.append(is_correct)
            else:
                # No expected output available, just check if execution succeeds
                pred, err, timeout = self.tester.execute_program_with_timeout(program_code, example['input'])
                is_correct = (pred is not None and not err and not timeout)
                test_correct.append(is_correct)

        # All correct if all train and test examples pass
        all_correct = all(train_correct) and all(test_correct) and len(train_correct) > 0

        return all_correct, train_correct, test_correct

    def get_predicted_outputs(self, program_code: str, task_data: Dict[str, Any]) -> Tuple[List[List[List[int]]], List[List[List[int]]]]:
        """
        Get predicted outputs for all examples in a task.

        Args:
            program_code: Python code containing transform function
            task_data: Task data with 'train' and 'test' examples

        Returns:
            Tuple of (train_predictions, test_predictions)
        """
        train_predictions = []
        test_predictions = []

        # Get predictions for training examples
        for example in task_data['train']:
            pred, err, timeout = self.tester.execute_program_with_timeout(program_code, example['input'])
            train_predictions.append(pred)

        # Get predictions for test examples
        for example in task_data['test']:
            pred, err, timeout = self.tester.execute_program_with_timeout(program_code, example['input'])
            test_predictions.append(pred)

        return train_predictions, test_predictions


def load_fully_correct_programs(parquet_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load programs that are fully correct (all train + test correct) from parquet file.

    Args:
        parquet_path: Path to parquet file with program data

    Returns:
        Dictionary mapping task_id -> list of fully correct program records
    """
    df = read_soar_parquet(parquet_path)

    fully_correct_programs = {}

    for _, record in df.iterrows():
        task_id = record['task_id']
        correct_train = record['correct_train_input']
        correct_test = record['correct_test_input']
        is_transductive = record.get('is_transductive', False)

        # Convert numpy arrays to lists if needed
        if hasattr(correct_train, 'tolist'):
            correct_train = correct_train.tolist()
        if hasattr(correct_test, 'tolist'):
            correct_test = correct_test.tolist()

        # Check if all train and test are correct and not transductive
        all_train_correct = isinstance(correct_train, list) and len(correct_train) > 0 and all(correct_train)
        all_test_correct = isinstance(correct_test, list) and len(correct_test) > 0 and all(correct_test)

        if all_train_correct and all_test_correct and not is_transductive:
            if task_id not in fully_correct_programs:
                fully_correct_programs[task_id] = []

            fully_correct_programs[task_id].append({
                'row_id': record['row_id'],
                'task_id': task_id,
                'code': record['code'],
                'reasoning': record.get('reasoning', ''),
                'model': record.get('model', ''),
                'correct_train_input': correct_train,
                'correct_test_input': correct_test,
                'predicted_train_output': record.get('predicted_train_output', []),
                'predicted_test_output': record.get('predicted_test_output', []),
                'is_transductive': is_transductive,
                'refined_from_id': record.get('refined_from_id', None)
            })

    return fully_correct_programs