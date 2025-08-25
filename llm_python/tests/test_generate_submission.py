#!/usr/bin/env python3
"""
Tests for the submission generator script.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

# Add the parent directory to sys.path to allow imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm_python.generate_submission import SubmissionGenerator
from llm_python.datasets.io import write_soar_parquet


class TestSubmissionGenerator(unittest.TestCase):
    """Test the SubmissionGenerator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.generator = SubmissionGenerator(
            no_transductive_penalty=False,
            output_dir=self.temp_dir,
            debug=False
        )
        
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_test_parquet_data(self, task_ids=None, num_attempts_per_task=2):
        """Create test parquet data"""
        if task_ids is None:
            task_ids = ["test_task_1", "test_task_2"]
        
        data = []
        for task_id in task_ids:
            for i in range(num_attempts_per_task):
                # Create realistic test data
                correct_train = [True, False, True]  # Mixed training accuracy
                predicted_test = [[[1, 2], [3, 4]]]  # Single test output to match mock
                
                data.append({
                    "task_id": task_id,
                    "reasoning": f"Test reasoning for {task_id} attempt {i}",
                    "code": f"# Test code for {task_id}\nprint('hello')",
                    "correct_train_input": correct_train,
                    "correct_test_input": [True, False],  # 2 test cases
                    "predicted_train_output": [[[1, 1]], [[2, 2]], [[3, 3]]],
                    "predicted_test_output": predicted_test,
                    "model": "test-model",
                    "is_transductive": i == 0,  # First attempt is transductive
                })
        
        return pd.DataFrame(data)
    
    def create_test_parquet_file(self, filename="test.parquet", task_ids=None, num_attempts=2):
        """Create a test parquet file and return its path"""
        df = self.create_test_parquet_data(task_ids, num_attempts)
        filepath = Path(self.temp_dir) / filename
        
        # Use the proper parquet writing function
        write_soar_parquet(df, filepath)
        return filepath
    
    def test_load_parquet_data(self):
        """Test loading parquet data"""
        # Create test parquet files
        file1 = self.create_test_parquet_file("test1.parquet", ["task1", "task2"])
        file2 = self.create_test_parquet_file("test2.parquet", ["task3"])
        
        # Load data
        df = self.generator.load_parquet_data([file1, file2])
        
        # Verify data was loaded correctly
        self.assertEqual(len(df), 6)  # 2 tasks * 2 attempts + 1 task * 2 attempts
        self.assertIn("task1", df["task_id"].values)
        self.assertIn("task2", df["task_id"].values)
        self.assertIn("task3", df["task_id"].values)
    
    def test_convert_parquet_to_attempts(self):
        """Test conversion from parquet to attempt format"""
        df = self.create_test_parquet_data(["task1"], 2)
        
        results_by_task = self.generator.convert_parquet_to_attempts(df)
        
        # Verify structure
        self.assertIn("task1", results_by_task)
        self.assertEqual(len(results_by_task["task1"]), 2)
        
        # Verify attempt structure
        attempt = results_by_task["task1"][0]
        self.assertIn("test_predicted", attempt)
        self.assertIn("train_accuracy", attempt)
        self.assertIn("is_transductive", attempt)
        self.assertIn("outputs_valid", attempt)
        self.assertIn("program_extracted", attempt)
        
        # Verify train accuracy calculation
        expected_acc = 2/3  # [True, False, True] = 2 out of 3 correct
        self.assertAlmostEqual(attempt["train_accuracy"], expected_acc)
    
    def test_find_recent_parquets_single_file(self):
        """Test finding parquets with single file input"""
        test_file = self.create_test_parquet_file("single.parquet")
        
        files = self.generator.find_recent_parquets(test_file)
        
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0], test_file)
    
    def test_find_recent_parquets_directory(self):
        """Test finding parquets in directory"""
        # Create multiple parquet files with different timestamps
        import time
        
        file1 = self.create_test_parquet_file("older.parquet")
        time.sleep(0.1)
        file2 = self.create_test_parquet_file("newer.parquet")
        
        files = self.generator.find_recent_parquets(self.temp_dir, n_files=2)
        
        self.assertEqual(len(files), 2)
        # Should be sorted by most recent first
        self.assertEqual(files[0], file2)  # newer file first
        self.assertEqual(files[1], file1)  # older file second
    
    @patch('llm_python.generate_submission.get_task_loader')
    def test_generate_submission_basic(self, mock_get_task_loader):
        """Test basic submission generation"""
        # Mock task loader
        mock_loader = MagicMock()
        mock_loader.get_subset_tasks.return_value = [
            ("task1", {"test": [{"input": [[1]], "output": [[2]]}]}),
            ("task2", {"test": [{"input": [[3]], "output": [[4]]}]}),
        ]
        mock_get_task_loader.return_value = mock_loader
        
        # Create test parquet data
        parquet_file = self.create_test_parquet_file("test.parquet", ["task1", "task2"], 3)
        
        # Generate submission
        submission_path = self.generator.generate_submission(
            parquet_paths=[parquet_file],
            dataset="test-dataset",
            subset="test-subset",
            model_name="test-model"
        )
        
        # Verify submission file was created
        self.assertTrue(Path(submission_path).exists())
        
        # Load and verify submission format
        with open(submission_path, 'r') as f:
            submission = json.load(f)
        
        self.assertIn("task1", submission)
        self.assertIn("task2", submission)
        
        # Verify submission structure
        task1_submission = submission["task1"]
        self.assertEqual(len(task1_submission), 1)  # 1 test case
        self.assertIn("attempt_1", task1_submission[0])
        self.assertIn("attempt_2", task1_submission[0])
        
        # Verify attempts are valid grids
        attempt_1 = task1_submission[0]["attempt_1"]
        attempt_2 = task1_submission[0]["attempt_2"]
        self.assertIsInstance(attempt_1, list)
        self.assertIsInstance(attempt_2, list)
    
    def test_generate_submission_with_missing_tasks(self):
        """Test submission generation when some tasks have no attempts"""
        # Create parquet data with only task1 and task2
        parquet_file = self.create_test_parquet_file("test.parquet", ["task1", "task2"], 2)
        
        # Mock the task loader to simulate the fallback behavior
        with patch('llm_python.generate_submission.get_task_loader') as mock_get_task_loader:
            mock_loader = MagicMock()
            # Make it raise an exception to trigger fallback mode
            mock_loader.get_subset_tasks.side_effect = Exception("Dataset not found")
            mock_get_task_loader.return_value = mock_loader
            
            # Generate submission (should use fallback behavior)
            submission_path = self.generator.generate_submission(
                parquet_paths=[parquet_file],
                dataset="test-dataset",
                subset="test-subset"
            )
        
        # Load submission
        with open(submission_path, 'r') as f:
            submission = json.load(f)
        
        # Verify tasks from parquet are present
        self.assertIn("task1", submission)
        self.assertIn("task2", submission)
        
        # In fallback mode, only tasks from parquet data are included
        self.assertEqual(len(submission), 2)
    
    @patch('llm_python.generate_submission.compute_weighted_majority_voting')
    def test_voting_integration(self, mock_voting):
        """Test integration with weighted voting"""
        # Mock voting to return specific predictions
        mock_voting.return_value = [
            [[[1, 1], [1, 1]]],  # Prediction 1
            [[[2, 2], [2, 2]]],  # Prediction 2
        ]
        
        # Create test data
        df = self.create_test_parquet_data(["task1"], 3)
        results_by_task = self.generator.convert_parquet_to_attempts(df)
        
        # Mock task loader
        with patch('llm_python.generate_submission.get_task_loader') as mock_get_task_loader:
            mock_loader = MagicMock()
            mock_loader.get_subset_tasks.return_value = [
                ("task1", {"test": [{"input": [[1]], "output": [[2]]}]}),
            ]
            mock_get_task_loader.return_value = mock_loader
            
            parquet_file = self.create_test_parquet_file("test.parquet", ["task1"], 3)
            
            # Generate submission
            submission_path = self.generator.generate_submission(
                parquet_paths=[parquet_file],
                dataset="test-dataset",
                subset="test-subset"
            )
        
        # Verify voting was called
        mock_voting.assert_called_once()
        
        # Verify submission uses voting results
        with open(submission_path, 'r') as f:
            submission = json.load(f)
        
        task1_submission = submission["task1"][0]
        self.assertEqual(task1_submission["attempt_1"], [[1, 1], [1, 1]])
        self.assertEqual(task1_submission["attempt_2"], [[2, 2], [2, 2]])
    
    def test_transductive_penalty_flag(self):
        """Test that transductive penalty flag is passed correctly"""
        # Test with penalty enabled
        generator_with_penalty = SubmissionGenerator(
            no_transductive_penalty=False,
            output_dir=self.temp_dir
        )
        self.assertFalse(generator_with_penalty.no_transductive_penalty)
        
        # Test with penalty disabled
        generator_no_penalty = SubmissionGenerator(
            no_transductive_penalty=True,
            output_dir=self.temp_dir
        )
        self.assertTrue(generator_no_penalty.no_transductive_penalty)
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Test with no parquet files
        with self.assertRaises(ValueError):
            self.generator.load_parquet_data([])
        
        # Test with non-existent parquet file
        non_existent = Path(self.temp_dir) / "nonexistent.parquet"
        with self.assertRaises(ValueError):
            self.generator.load_parquet_data([non_existent])
        
        # Test with invalid path
        with self.assertRaises(ValueError):
            self.generator.find_recent_parquets("/nonexistent/path")


class TestSubmissionGeneratorCLI(unittest.TestCase):
    """Test the CLI interface of generate_submission.py"""
    
    def test_cli_help(self):
        """Test that CLI help works"""
        import subprocess
        result = subprocess.run([
            'uv', 'run', 'python', 'llm_python/generate_submission.py', '--help'
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("Generate ARC submission files from parquet data", result.stdout)


class TestLogicDuplication(unittest.TestCase):
    """Test that there's no duplication of logic between task runner and submission generator"""
    
    def test_no_submission_logic_in_task_runner(self):
        """Verify submission logic was removed from task runner"""
        task_runner_path = Path(__file__).parent.parent / "run_arc_tasks_soar.py"
        
        with open(task_runner_path, 'r') as f:
            content = f.read()
        
        # Check that submission creation method was removed
        self.assertNotIn("_create_submission_file", content)
        
        # Check that submission-related imports were removed
        self.assertNotIn("validate_submission_file", content)
        
        # Voting should NOT be imported directly since it's used via metrics_utils
        # The task runner imports metrics_utils, which imports voting functions
        self.assertNotIn("from llm_python.utils.voting_utils import", content)
    
    def test_submission_generator_has_voting_logic(self):
        """Verify submission generator has its own voting logic"""
        generator_path = Path(__file__).parent.parent / "generate_submission.py"
        
        with open(generator_path, 'r') as f:
            content = f.read()
        
        # Should import voting functionality
        self.assertIn("compute_weighted_majority_voting", content)
        
        # Should use voting in generation
        self.assertIn("compute_weighted_majority_voting(", content)


if __name__ == "__main__":
    # Run the tests
    unittest.main()