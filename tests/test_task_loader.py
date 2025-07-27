#!/usr/bin/env python3

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, mock_open
import sys
import os

# Add the llm-python directory to the path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'llm-python'))
from utils.task_loader import TaskLoader, TaskData, TaskExample


class TestTaskLoader:
    """Test suite for TaskLoader class"""
    
    @pytest.fixture
    def sample_task_data(self) -> TaskData:
        """Sample task data for testing"""
        return {
            "train": [
                {
                    "input": [[1, 2], [3, 4]],
                    "output": [[2, 4], [6, 8]]
                },
                {
                    "input": [[0, 1], [1, 0]],
                    "output": [[0, 2], [2, 0]]
                }
            ],
            "test": [
                {
                    "input": [[5, 6], [7, 8]],
                    "output": [[10, 12], [14, 16]]
                }
            ]
        }
    
    @pytest.fixture
    def temp_data_structure(self, sample_task_data):
        """Create a temporary directory structure mimicking the data folder"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create directory structure
            arc_agi_1_train = temp_path / "arc-agi-1" / "training"
            arc_agi_1_eval = temp_path / "arc-agi-1" / "evaluation"
            arc_agi_2_train = temp_path / "arc-agi-2" / "training"
            subsets_dir = temp_path / "subsets" / "arc-agi-1"
            
            arc_agi_1_train.mkdir(parents=True)
            arc_agi_1_eval.mkdir(parents=True)
            arc_agi_2_train.mkdir(parents=True)
            subsets_dir.mkdir(parents=True)
            
            # Create sample task files
            with open(arc_agi_1_train / "6150a2bd.json", 'w') as f:
                json.dump(sample_task_data, f)
            
            with open(arc_agi_1_eval / "test_eval.json", 'w') as f:
                json.dump(sample_task_data, f)
            
            # Create subset files
            with open(subsets_dir / "shortest_training_1.txt", 'w') as f:
                f.write("6150a2bd\n")
            
            with open(subsets_dir / "test_subset.txt", 'w') as f:
                f.write("6150a2bd\ntest_eval\n")
            
            yield temp_path
    
    def test_init_valid_data_root(self, temp_data_structure):
        """Test TaskLoader initialization with valid data root"""
        loader = TaskLoader(str(temp_data_structure))
        assert loader.data_root == Path(temp_data_structure)
    
    def test_init_invalid_data_root(self):
        """Test TaskLoader initialization with invalid data root"""
        with pytest.raises(ValueError, match="Data root directory not found"):
            TaskLoader("/nonexistent/path")
    
    def test_load_task_from_training(self, temp_data_structure, sample_task_data):
        """Test loading a task from training directory"""
        loader = TaskLoader(str(temp_data_structure))
        task_data = loader.load_task("6150a2bd", "arc-agi-1")
        
        assert task_data == sample_task_data
        assert len(task_data["train"]) == 2
        assert len(task_data["test"]) == 1
    
    def test_load_task_from_evaluation(self, temp_data_structure, sample_task_data):
        """Test loading a task from evaluation directory"""
        loader = TaskLoader(str(temp_data_structure))
        task_data = loader.load_task("test_eval", "arc-agi-1")
        
        assert task_data == sample_task_data
    
    def test_load_task_not_found(self, temp_data_structure):
        """Test loading a non-existent task"""
        loader = TaskLoader(str(temp_data_structure))
        
        with pytest.raises(FileNotFoundError, match="Task nonexistent not found"):
            loader.load_task("nonexistent", "arc-agi-1")
    
    def test_load_subset(self, temp_data_structure):
        """Test loading task IDs from a subset file"""
        loader = TaskLoader(str(temp_data_structure))
        task_ids = loader.load_subset("shortest_training_1", "arc-agi-1")
        
        assert task_ids == ["6150a2bd"]
    
    def test_load_subset_not_found(self, temp_data_structure):
        """Test loading a non-existent subset"""
        loader = TaskLoader(str(temp_data_structure))
        
        with pytest.raises(FileNotFoundError, match="Subset file not found"):
            loader.load_subset("nonexistent", "arc-agi-1")
    
    def test_load_tasks_from_subset(self, temp_data_structure, sample_task_data):
        """Test loading all tasks from a subset"""
        loader = TaskLoader(str(temp_data_structure))
        tasks = loader.load_tasks_from_subset("test_subset", "arc-agi-1")
        
        assert len(tasks) == 2
        task_ids = [task_id for task_id, _ in tasks]
        assert "6150a2bd" in task_ids
        assert "test_eval" in task_ids
        
        # Check task data
        for task_id, task_data in tasks:
            assert task_data == sample_task_data
    
    def test_load_tasks_from_subset_with_missing_task(self, temp_data_structure, capsys):
        """Test loading tasks when some tasks in subset don't exist"""
        # Create a subset with a missing task
        subsets_dir = temp_data_structure / "subsets" / "arc-agi-1"
        with open(subsets_dir / "mixed_subset.txt", 'w') as f:
            f.write("6150a2bd\nmissing_task\n")
        
        loader = TaskLoader(str(temp_data_structure))
        tasks = loader.load_tasks_from_subset("mixed_subset", "arc-agi-1")
        
        # Should only return the found task
        assert len(tasks) == 1
        assert tasks[0][0] == "6150a2bd"
        
        # Should print warning for missing task
        captured = capsys.readouterr()
        assert "Warning: Task missing_task not found" in captured.out
    
    def test_get_available_subsets(self, temp_data_structure):
        """Test getting available subset files"""
        loader = TaskLoader(str(temp_data_structure))
        subsets = loader.get_available_subsets("arc-agi-1")
        
        expected_subsets = ["shortest_training_1", "test_subset"]
        assert sorted(subsets) == sorted(expected_subsets)
    
    def test_get_available_subsets_nonexistent_dataset(self, temp_data_structure):
        """Test getting subsets for non-existent dataset"""
        loader = TaskLoader(str(temp_data_structure))
        subsets = loader.get_available_subsets("nonexistent")
        
        assert subsets == []
    
    def test_format_task_for_prompt(self, sample_task_data):
        """Test formatting task data for prompting"""
        loader = TaskLoader()
        formatted = loader.format_task_for_prompt(sample_task_data)
        
        assert "Training Examples:" in formatted
        assert "Example 1:" in formatted
        assert "Example 2:" in formatted
        assert "Input:" in formatted
        assert "Output:" in formatted
        assert "1 2" in formatted  # Grid formatting
        assert "3 4" in formatted
        
        # Should not include test by default
        assert "Test Input:" not in formatted
    
    def test_format_task_for_prompt_with_test(self, sample_task_data):
        """Test formatting task data with test input included"""
        loader = TaskLoader()
        formatted = loader.format_task_for_prompt(sample_task_data, include_test=True)
        
        assert "Training Examples:" in formatted
        assert "Test Input:" in formatted
        assert "5 6" in formatted  # Test input
        assert "7 8" in formatted
    
    def test_format_task_for_prompt_no_test(self, sample_task_data):
        """Test formatting task data with no test cases"""
        task_data_no_test = sample_task_data.copy()
        task_data_no_test["test"] = []
        
        loader = TaskLoader()
        formatted = loader.format_task_for_prompt(task_data_no_test, include_test=True)
        
        assert "Training Examples:" in formatted
        assert "Test Input:" not in formatted
    
    def test_format_grid(self):
        """Test grid formatting"""
        loader = TaskLoader()
        grid = [[1, 2, 3], [4, 5, 6]]
        formatted = loader._format_grid(grid)
        
        expected = "1 2 3\n4 5 6"
        assert formatted == expected
    
    def test_format_grid_empty(self):
        """Test formatting empty grid"""
        loader = TaskLoader()
        grid = []
        formatted = loader._format_grid(grid)
        
        assert formatted == ""
    
    def test_get_test_outputs(self, sample_task_data):
        """Test extracting test outputs"""
        loader = TaskLoader()
        outputs = loader.get_test_outputs(sample_task_data)
        
        assert len(outputs) == 1
        assert outputs[0] == [[10, 12], [14, 16]]
    
    def test_get_test_outputs_no_test(self):
        """Test extracting test outputs when no test cases exist"""
        loader = TaskLoader()
        task_data = {"train": [], "test": []}
        outputs = loader.get_test_outputs(task_data)
        
        assert outputs == []


class TestTaskLoaderIntegration:
    """Integration tests using real data (marked as integration)"""
    
    @pytest.mark.integration
    def test_load_real_shortest_task(self):
        """Test loading a real task from the shortest subset"""
        # This test requires actual data to be present
        try:
            loader = TaskLoader()
            task_ids = loader.load_subset("shortest_training_1", "arc-agi-1")
            
            if task_ids:
                task_id = task_ids[0]
                task_data = loader.load_task(task_id, "arc-agi-1")
                
                # Verify basic structure
                assert "train" in task_data
                assert "test" in task_data
                assert isinstance(task_data["train"], list)
                assert isinstance(task_data["test"], list)
                
                # Verify each training example has input/output
                for example in task_data["train"]:
                    assert "input" in example
                    assert "output" in example
                    assert isinstance(example["input"], list)
                    assert isinstance(example["output"], list)
        
        except FileNotFoundError:
            pytest.skip("Real ARC data not available")
    
    @pytest.mark.integration
    def test_get_real_available_subsets(self):
        """Test getting real available subsets"""
        try:
            loader = TaskLoader()
            subsets = loader.get_available_subsets("arc-agi-1")
            
            # Should have at least some basic subsets
            assert len(subsets) > 0
            assert any("shortest" in subset for subset in subsets)
            
        except:
            pytest.skip("Real ARC data not available")


if __name__ == "__main__":
    pytest.main([__file__])
