"""
Integration tests for the task runner with new dataset loading functionality.
"""

import pytest
from unittest.mock import patch, MagicMock
from llm_python.run_arc_tasks_soar import ARCTaskRunnerSimple
from llm_python.utils.task_loader import get_task_loader


class TestTaskRunnerDatasetIntegration:
    """Integration tests for task runner with HF/parquet dataset support"""

    @pytest.fixture
    def runner(self):
        """Create a minimal task runner for testing"""
        return ARCTaskRunnerSimple(
            model="gpt-4o-mini",
            max_workers=1,
            max_attempts=1,
            debug=False,
            unsafe_executor=True,  # Use unsafe executor for testing to avoid Docker dependency
        )

    def test_task_runner_traditional_subset(self, runner):
        """Test that task runner still works with traditional subsets"""
        # Find a small traditional subset
        loader = get_task_loader()
        subsets = loader.get_available_subsets()
        traditional_subsets = [s for s in subsets if s.startswith("arc-prize-")]
        
        if traditional_subsets:
            subset_name = traditional_subsets[0].split('/')[-1]  # Just the subset part
            
            # Mock the actual task running to avoid making API calls
            with patch.object(runner, 'run_single_attempt') as mock_run:
                mock_run.return_value = {
                    "task_id": "test_task",
                    "attempt_num": 0,
                    "attempt_detail": {
                        "attempt_number": 1,
                        "program_extracted": True,
                        "test_correct": False,
                        "train_accuracy": 0.5,
                        "attempt_cost": 0.001,
                        "input_tokens": 100,
                        "output_tokens": 50,
                    },
                    "task_data": {"train": [], "test": []},
                    "dataset": "arc-prize-2025",
                    "subset": subset_name,
                    "full_prompt": {"system": "test", "user": "test"}
                }
                
                # This should not raise an exception
                results = runner.run_subset(subset_name, limit=1)
                assert isinstance(results, list)

    @patch('datasets.load_dataset')
    def test_task_runner_huggingface_dataset(self, mock_load_dataset, runner):
        """Test task runner with mocked HuggingFace dataset"""
        # Setup mock dataset
        mock_dataset = MagicMock()
        mock_dataset.column_names = ['task_id', 'code', 'reasoning']
        mock_dataset.__len__.return_value = 5
        mock_dataset.select.return_value = mock_dataset
        mock_dataset.__getitem__.side_effect = lambda key: ['00576224', '007bbfb7']  # Real task IDs
        mock_load_dataset.return_value = mock_dataset
        
        # Mock the actual task running to avoid making API calls  
        with patch.object(runner, 'run_single_attempt') as mock_run:
            mock_run.return_value = {
                "task_id": "00576224",
                "attempt_num": 0,
                "attempt_detail": {
                    "attempt_number": 1,
                    "program_extracted": True,
                    "test_correct": False,
                    "train_accuracy": 0.5,
                    "attempt_cost": 0.001,
                    "input_tokens": 100,
                    "output_tokens": 50,
                },
                "task_data": {"train": [], "test": []},
                "dataset": None,
                "subset": "username/test-dataset",
                "full_prompt": {"system": "test", "user": "test"}
            }
            
            # Test loading HF dataset
            results = runner.run_subset("username/test-dataset", limit=2)
            
            # Verify it tried to load HF dataset
            mock_load_dataset.assert_called_once_with("username/test-dataset", split="train")
            assert isinstance(results, list)

    @patch('llm_python.datasets.io.read_soar_parquet')
    def test_task_runner_parquet_dataset(self, mock_read_parquet, runner):
        """Test task runner with mocked parquet dataset"""
        import pandas as pd
        
        # Setup mock parquet data with real task IDs
        mock_df = pd.DataFrame({
            'task_id': ['00576224', '007bbfb7', '00d62c1b'],
            'code': ['def solve(grid): return grid'] * 3,
            'is_transductive': [False, False, True]
        })
        mock_read_parquet.return_value = mock_df
        
        # Mock the actual task running
        with patch.object(runner, 'run_single_attempt') as mock_run:
            mock_run.return_value = {
                "task_id": "00576224",
                "attempt_num": 0,
                "attempt_detail": {
                    "attempt_number": 1,
                    "program_extracted": True,
                    "test_correct": False,
                    "train_accuracy": 0.5,
                    "attempt_cost": 0.001,
                    "input_tokens": 100,
                    "output_tokens": 50,
                },
                "task_data": {"train": [], "test": []},
                "dataset": None,
                "subset": "/path/to/data.parquet",
                "full_prompt": {"system": "test", "user": "test"}
            }
            
            # Test loading parquet dataset
            results = runner.run_subset("/path/to/data.parquet", limit=2)
            
            # Verify it tried to load parquet
            mock_read_parquet.assert_called_once_with("/path/to/data.parquet")
            assert isinstance(results, list)

    def test_task_runner_fallback_behavior(self, runner):
        """Test that task runner falls back to traditional loading when new method fails"""
        # Mock the new method to fail, but traditional to succeed
        with patch.object(runner.task_loader, 'get_dataset_subset', side_effect=ValueError("Mock failure")):
            with patch.object(runner.task_loader, 'get_subset_tasks') as mock_traditional:
                # Mock successful traditional loading
                mock_traditional.return_value = [
                    ('00576224', {'train': [{'input': [[1]], 'output': [[2]]}], 'test': [{'input': [[3]], 'output': [[4]]}]})
                ]
                
                with patch.object(runner, 'run_single_attempt') as mock_run:
                    mock_run.return_value = {
                        "task_id": "00576224",
                        "attempt_num": 0,
                        "attempt_detail": {
                            "attempt_number": 1,
                            "program_extracted": True,
                            "test_correct": False,
                            "train_accuracy": 0.5,
                            "attempt_cost": 0.001,
                            "input_tokens": 100,
                            "output_tokens": 50,
                        },
                        "task_data": {"train": [], "test": []},
                        "dataset": "arc-prize-2025",
                        "subset": "training",
                        "full_prompt": {"system": "test", "user": "test"}
                    }
                    
                    # Should fall back to traditional method
                    results = runner.run_subset("training", dataset="arc-prize-2025", limit=1)
                    
                    # Verify fallback was used
                    mock_traditional.assert_called_once_with("arc-prize-2025/training")
                    assert isinstance(results, list)

    def test_task_runner_error_handling(self, runner):
        """Test error handling when both new and traditional methods fail"""
        with patch.object(runner.task_loader, 'get_dataset_subset', side_effect=ValueError("New method failed")):
            with patch.object(runner.task_loader, 'get_subset_tasks', side_effect=ValueError("Traditional method failed")):
                
                # Both methods fail, should return error tuple
                results = runner.run_subset("nonexistent-dataset", limit=1)
                assert results == ([], None)


class TestTaskRunnerRealDataset:
    """Tests with real datasets that may require network access"""
    
    @pytest.mark.integration
    def test_trelis_dataset_real(self):
        """Test with real Trelis dataset if accessible"""
        runner = ARCTaskRunnerSimple(
            model="gpt-4o-mini",
            max_workers=1,
            max_attempts=1,
            debug=False,
            unsafe_executor=True,
        )
        
        try:
            # This will only work if user has access to the dataset
            with patch.object(runner, 'run_single_attempt') as mock_run:
                mock_run.return_value = {
                    "task_id": "test_task",
                    "attempt_num": 0,
                    "attempt_detail": {
                        "attempt_number": 1,
                        "program_extracted": True,
                        "test_correct": False,
                        "train_accuracy": 0.5,
                        "attempt_cost": 0.001,
                        "input_tokens": 100,
                        "output_tokens": 50,
                    },
                    "task_data": {"train": [], "test": []},
                    "dataset": None,
                    "subset": "Trelis/arc-agi-partials-for-refinement",
                    "full_prompt": {"system": "test", "user": "test"}
                }
                
                results = runner.run_subset("Trelis/arc-agi-partials-for-refinement", limit=1)
                assert isinstance(results, list)
                
        except Exception as e:
            pytest.skip(f"Real dataset test skipped: {e}")

    @pytest.mark.integration 
    def test_dataset_type_detection_integration(self):
        """Test dataset type detection with real data"""
        runner = ARCTaskRunnerSimple(
            model="gpt-4o-mini", 
            max_workers=1,
            max_attempts=1,
            debug=False,
            unsafe_executor=True,
        )
        
        loader = runner.task_loader
        
        # Test detection on real subset names
        traditional_subsets = [s for s in loader.get_available_subsets() if s.startswith("arc-prize-")][:3]
        
        for subset in traditional_subsets:
            detected_type = loader._detect_dataset_type(subset)
            assert detected_type == "traditional"
        
        # Test HF detection
        assert loader._detect_dataset_type("Trelis/arc-agi-partials-for-refinement") == "huggingface"
        
        # Test parquet detection
        assert loader._detect_dataset_type("/tmp/data.parquet") == "parquet"


class TestCommandLineInterface:
    """Test command line interface with new dataset support"""
    
    def test_help_message_includes_new_formats(self):
        """Test that help message mentions new dataset formats"""
        from llm_python.run_arc_tasks_soar import main
        import sys
        from io import StringIO
        
        # Capture help output
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            # This should trigger help and exit
            sys.argv = ['run_arc_tasks_soar.py', '--help']
            main()
        except SystemExit:
            # Expected for --help
            pass
        finally:
            sys.stdout = old_stdout
        
        help_text = captured_output.getvalue()
        
        # Check that help mentions the new formats
        assert "HuggingFace datasets" in help_text or "username/dataset-name" in help_text
        assert "Parquet files" in help_text or ".parquet" in help_text