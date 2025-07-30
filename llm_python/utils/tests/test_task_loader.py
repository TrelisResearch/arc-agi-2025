#!/usr/bin/env python3

import pytest
import sys
import os

# Add the llm_python directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.task_loader import TaskLoader


class TestTaskLoader:
    """Basic tests for TaskLoader using real data"""
    
    def test_load_task(self):
        """Test loading a real task"""
        loader = TaskLoader()
        try:
            task_ids = loader.load_subset("shortest_training_1", "arc-agi-1")
            if task_ids:
                task_data = loader.load_task(task_ids[0], "arc-agi-1")
                assert "train" in task_data
                assert "test" in task_data
                assert len(task_data["train"]) > 0
        except FileNotFoundError:
            pytest.skip("Real ARC data not available")
    
    def test_load_subset(self):
        """Test loading a subset"""
        loader = TaskLoader()
        try:
            task_ids = loader.load_subset("shortest_training_1", "arc-agi-1")
            assert isinstance(task_ids, list)
            assert len(task_ids) > 0
        except FileNotFoundError:
            pytest.skip("Real ARC data not available")
    
    def test_format_task_for_prompt(self):
        """Test task formatting"""
        loader = TaskLoader()
        try:
            task_ids = loader.load_subset("shortest_training_1", "arc-agi-1")
            if task_ids:
                task_data = loader.load_task(task_ids[0], "arc-agi-1")
                
                formatted = loader.format_task_for_prompt(task_data)
                assert "Training Examples:" in formatted
                assert "Test Input:" not in formatted
                
                # With test
                formatted_with_test = loader.format_task_for_prompt(task_data, include_test=True)
                assert "Test Input:" in formatted_with_test
        except FileNotFoundError:
            pytest.skip("Real ARC data not available")
    
    def test_get_test_outputs(self):
        """Test getting test outputs"""
        loader = TaskLoader()
        try:
            task_ids = loader.load_subset("shortest_training_1", "arc-agi-1")
            if task_ids:
                task_data = loader.load_task(task_ids[0], "arc-agi-1")
                outputs = loader.get_test_outputs(task_data)
                assert isinstance(outputs, list)
                if len(outputs) > 0:
                    assert isinstance(outputs[0], list)  # Should be a grid
        except FileNotFoundError:
            pytest.skip("Real ARC data not available")
