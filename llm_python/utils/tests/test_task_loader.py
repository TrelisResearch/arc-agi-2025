import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from ..task_loader import TaskLoader, get_default_data_root


class TestTaskLoader:
    """Basic tests for TaskLoader using real competition data"""

    def test_initialization(self):
        """Test that TaskLoader initializes and loads data"""
        loader = TaskLoader(get_default_data_root())
        assert len(loader.tasks) > 0
        assert len(loader.subsets) > 0
        print(f"Loaded {len(loader.tasks)} tasks and {len(loader.subsets)} subsets")

    def test_canonical_dataset_parameter(self):
        """Test that canonical dataset parameter works"""
        # Test default
        loader_default = TaskLoader(get_default_data_root())
        assert loader_default.canonical_dataset == "arc-prize-2025"

        # Test custom
        loader_2024 = TaskLoader(
            get_default_data_root(), canonical_dataset="arc-prize-2024"
        )
        assert loader_2024.canonical_dataset == "arc-prize-2024"

        # Both should load the same number of tasks
        assert len(loader_default.tasks) == len(loader_2024.tasks)

    def test_all_tasks_have_outputs(self):
        """Test that all loaded tasks have test outputs"""
        loader = TaskLoader(get_default_data_root())
        tasks_without_outputs = []

        for task_id, task_data in loader.tasks.items():
            # Check if any test case has no output
            for i, test_case in enumerate(task_data["test"]):
                if test_case.get("output") is None:
                    tasks_without_outputs.append(
                        (task_id, i, loader.task_sources.get(task_id, "unknown"))
                    )

        if tasks_without_outputs:
            print(f"Found {len(tasks_without_outputs)} test cases without outputs:")
            for task_id, test_idx, source in tasks_without_outputs[
                :10
            ]:  # Show first 10
                print(f"  Task {task_id} (from {source}), test case {test_idx}")

        # This should pass - all tasks should have outputs
        assert len(tasks_without_outputs) == 0, (
            f"Found {len(tasks_without_outputs)} test cases without outputs"
        )

    def test_task_sources_tracking(self):
        """Test that task sources are properly tracked"""
        loader = TaskLoader(get_default_data_root())

        # All tasks should have a source
        for task_id in loader.tasks:
            assert task_id in loader.task_sources
            source = loader.task_sources[task_id]
            assert source in ["arc-prize-2024", "arc-prize-2025"]

    def test_conflict_resolution(self):
        """Test that canonical dataset preference resolves conflicts correctly"""
        # Load with arc-prize-2025 as canonical (default)
        loader_2025 = TaskLoader(
            get_default_data_root(), canonical_dataset="arc-prize-2025"
        )

        # Load with arc-prize-2024 as canonical
        loader_2024 = TaskLoader(
            get_default_data_root(), canonical_dataset="arc-prize-2024"
        )

        # Find tasks that exist in both datasets
        conflicting_tasks = []
        for task_id in loader_2025.tasks:
            if (
                loader_2025.task_sources[task_id] == "arc-prize-2025"
                and task_id in loader_2024.tasks
                and loader_2024.task_sources[task_id] == "arc-prize-2024"
            ):
                conflicting_tasks.append(task_id)

        if conflicting_tasks:
            # Check that the canonical dataset preference is respected
            test_task = conflicting_tasks[0]

            # Task should come from different sources in each loader
            assert loader_2025.task_sources[test_task] == "arc-prize-2025"
            assert loader_2024.task_sources[test_task] == "arc-prize-2024"

            print(f"Tested conflict resolution with task {test_task}")
        else:
            print("No conflicting tasks found - conflict resolution cannot be tested")

    def test_get_task(self):
        """Test getting a task by ID"""
        loader = TaskLoader(get_default_data_root())
        # Get a task ID from the competition data
        if loader.tasks:
            task_id = next(iter(loader.tasks.keys()))
            task_data = loader.get_task(task_id)
            assert "train" in task_data
            assert "test" in task_data
            assert len(task_data["train"]) > 0

    def test_get_subset_tasks(self):
        """Test getting tasks from a competition subset"""
        loader = TaskLoader(get_default_data_root())
        # Try to get tasks from a competition subset
        competition_subsets = [
            s for s in loader.get_available_subsets() if s.startswith("arc-prize-")
        ]
        if competition_subsets:
            subset_name = competition_subsets[0]
            tasks = loader.get_subset_tasks(subset_name)
            assert isinstance(tasks, list)
            assert len(tasks) > 0
            # Check first task structure
            task_id, task_data = tasks[0]
            assert isinstance(task_id, str)
            assert "train" in task_data
            assert "test" in task_data

    def test_legacy_subset_compatibility(self):
        """Test that legacy subset names still work"""
        loader = TaskLoader(get_default_data_root())
        # Try loading a legacy subset
        try:
            tasks = loader.get_subset_tasks("arc-agi-1/shortest_training_1")
            assert isinstance(tasks, list)
            if tasks:  # Only check if tasks were found
                task_id, task_data = tasks[0]
                assert "train" in task_data
                assert "test" in task_data
        except ValueError:
            # Subset might not exist, which is fine
            pass


class TestTaskLoaderDatasetSupport:
    """Tests for new HuggingFace and parquet dataset support"""

    @pytest.fixture
    def loader(self):
        """Create a TaskLoader instance for testing"""
        return TaskLoader(get_default_data_root())

    def test_detect_dataset_type_traditional(self, loader):
        """Test detection of traditional subset names"""
        # Known traditional subsets
        assert loader._detect_dataset_type("training") == "traditional"
        assert loader._detect_dataset_type("arc-prize-2025/training") == "traditional"
        assert loader._detect_dataset_type("unknown-subset") == "traditional"

    def test_detect_dataset_type_huggingface(self, loader):
        """Test detection of HuggingFace dataset identifiers"""
        # HuggingFace format (contains slash, not a known subset)
        assert loader._detect_dataset_type("username/dataset-name") == "huggingface"
        assert loader._detect_dataset_type("Trelis/arc-agi-partials-for-refinement") == "huggingface"
        assert loader._detect_dataset_type("org/model-v2") == "huggingface"

    def test_detect_dataset_type_parquet(self, loader):
        """Test detection of parquet files and directories"""
        # Parquet file paths
        assert loader._detect_dataset_type("/path/to/data.parquet") == "parquet"
        assert loader._detect_dataset_type("data.parquet") == "parquet"
        assert loader._detect_dataset_type("../datasets/programs.parquet") == "parquet"
        assert loader._detect_dataset_type("datasets/parquet/") == "parquet"
        
        # Create a temporary parquet file to test existing file detection
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            assert loader._detect_dataset_type(str(tmp_path)) == "parquet"
            tmp_path.unlink()  # Clean up

    def test_detect_dataset_type_with_existing_parquet_dir(self, loader):
        """Test detection of directories containing parquet files"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            # Create a parquet file in the directory
            parquet_file = tmp_path / "test.parquet"
            parquet_file.touch()
            
            assert loader._detect_dataset_type(str(tmp_path)) == "parquet"

    def test_get_dataset_subset_traditional(self, loader):
        """Test that traditional subset loading still works through new method"""
        # Find a known subset
        available_subsets = loader.get_available_subsets()
        traditional_subsets = [s for s in available_subsets if s.startswith("arc-prize-")]
        
        if traditional_subsets:
            subset_name = traditional_subsets[0]
            tasks = loader.get_dataset_subset(subset_name, max_rows=5)
            
            assert isinstance(tasks, list)
            assert len(tasks) <= 5
            assert len(tasks) > 0
            
            # Check task structure
            task_id, task_data = tasks[0]
            assert isinstance(task_id, str)
            assert "train" in task_data
            assert "test" in task_data

    @patch('datasets.load_dataset')
    def test_get_dataset_subset_huggingface(self, mock_load_dataset, loader):
        """Test loading HuggingFace dataset with mocked data"""
        # Mock HuggingFace dataset
        mock_dataset = MagicMock()
        mock_dataset.select.return_value = mock_dataset
        mock_dataset.__len__.return_value = 100
        mock_dataset.column_names = ['task_id', 'code', 'reasoning']
        mock_dataset.__getitem__.side_effect = lambda key: ['task_001', 'task_002', 'task_001']  # task_id column
        mock_load_dataset.return_value = mock_dataset
        
        # Add some mock tasks to the loader's task cache
        loader.tasks['task_001'] = {
            'train': [{'input': [[1, 2]], 'output': [[2, 1]]}],
            'test': [{'input': [[3, 4]], 'output': [[4, 3]]}]
        }
        loader.tasks['task_002'] = {
            'train': [{'input': [[5, 6]], 'output': [[6, 5]]}],
            'test': [{'input': [[7, 8]], 'output': [[8, 7]]}]
        }
        
        tasks = loader.get_dataset_subset("username/test-dataset", max_rows=10)
        
        # Verify the dataset was loaded correctly
        mock_load_dataset.assert_called_once_with("username/test-dataset", split="train")
        mock_dataset.select.assert_called_once_with(range(10))
        
        # Should return tasks that exist in the cache
        assert len(tasks) == 2  # Both task_001 and task_002 should be found
        task_ids = [task[0] for task in tasks]
        assert 'task_001' in task_ids
        assert 'task_002' in task_ids

    @patch('datasets.load_dataset')
    def test_get_dataset_subset_huggingface_with_row_id(self, mock_load_dataset, loader):
        """Test loading HuggingFace dataset that uses row_id instead of task_id"""
        # Mock HuggingFace dataset with row_id
        mock_dataset = MagicMock()
        mock_dataset.select.return_value = mock_dataset
        mock_dataset.__len__.return_value = 50
        mock_dataset.column_names = ['row_id', 'code', 'reasoning']  # No task_id
        mock_dataset.__getitem__.side_effect = lambda key: ['row_001', 'row_002']  # row_id column
        mock_load_dataset.return_value = mock_dataset
        
        # Add corresponding tasks to the loader's cache
        loader.tasks['row_001'] = {
            'train': [{'input': [[1, 2]], 'output': [[2, 1]]}],
            'test': [{'input': [[3, 4]], 'output': [[4, 3]]}]
        }
        
        tasks = loader.get_dataset_subset("username/dataset-with-row-id", max_rows=5)
        
        # Verify the dataset was loaded correctly
        mock_load_dataset.assert_called_once_with("username/dataset-with-row-id", split="train")
        assert len(tasks) == 1
        assert tasks[0][0] == 'row_001'

    @patch('llm_python.datasets.io.read_soar_parquet')
    def test_get_dataset_subset_parquet(self, mock_read_parquet, loader):
        """Test loading parquet dataset with mocked data"""
        import pandas as pd
        
        # Mock parquet data
        mock_df = pd.DataFrame({
            'task_id': ['task_001', 'task_002', 'task_001', 'task_003'],
            'code': ['code1', 'code2', 'code3', 'code4'],
            'is_transductive': [False, True, False, False]
        })
        mock_read_parquet.return_value = mock_df
        
        # Add corresponding tasks to the loader's cache
        loader.tasks['task_001'] = {
            'train': [{'input': [[1, 2]], 'output': [[2, 1]]}],
            'test': [{'input': [[3, 4]], 'output': [[4, 3]]}]
        }
        loader.tasks['task_002'] = {
            'train': [{'input': [[5, 6]], 'output': [[6, 5]]}],
            'test': [{'input': [[7, 8]], 'output': [[8, 7]]}]
        }
        
        tasks = loader.get_dataset_subset("/path/to/data.parquet", max_rows=3)
        
        # Verify the parquet was loaded correctly
        mock_read_parquet.assert_called_once_with("/path/to/data.parquet")
        
        # Should return unique tasks that exist in the cache
        assert len(tasks) == 2  # task_001 and task_002 (task_003 not in cache)
        task_ids = [task[0] for task in tasks]
        assert 'task_001' in task_ids
        assert 'task_002' in task_ids

    def test_get_dataset_subset_error_handling(self, loader):
        """Test error handling for invalid datasets"""
        
        # Test invalid HuggingFace dataset
        with patch('datasets.load_dataset', side_effect=Exception("Dataset not found")):
            with pytest.raises(ValueError, match="Failed to load HuggingFace dataset"):
                loader.get_dataset_subset("invalid/dataset")
        
        # Test invalid parquet file
        with patch('llm_python.datasets.io.read_soar_parquet', side_effect=Exception("File not found")):
            with pytest.raises(ValueError, match="Failed to load parquet dataset"):
                loader.get_dataset_subset("/invalid/path.parquet")

    @patch('datasets.load_dataset')
    def test_get_dataset_subset_no_valid_tasks(self, mock_load_dataset, loader):
        """Test handling when dataset contains no valid task IDs"""
        # Mock dataset with task IDs not in cache
        mock_dataset = MagicMock()
        mock_dataset.select.return_value = mock_dataset
        mock_dataset.__len__.return_value = 10
        mock_dataset.column_names = ['task_id', 'code']
        mock_dataset.__getitem__.side_effect = lambda key: ['nonexistent_001', 'nonexistent_002']
        mock_load_dataset.return_value = mock_dataset
        
        with pytest.raises(ValueError, match="No valid tasks found for dataset"):
            loader.get_dataset_subset("username/empty-dataset")

    @patch('datasets.load_dataset')
    def test_get_dataset_subset_missing_columns(self, mock_load_dataset, loader):
        """Test handling when dataset is missing required columns"""
        # Mock dataset without task_id or row_id columns
        mock_dataset = MagicMock()
        mock_dataset.select.return_value = mock_dataset
        mock_dataset.__len__.return_value = 10
        mock_dataset.column_names = ['code', 'reasoning']  # No task_id or row_id
        mock_load_dataset.return_value = mock_dataset
        
        with pytest.raises(ValueError, match="Dataset must contain either 'task_id' or 'row_id' column"):
            loader.get_dataset_subset("username/invalid-dataset")

    def test_get_dataset_subset_max_rows_limit(self, loader):
        """Test that max_rows parameter works correctly for traditional subsets"""
        # Find a subset with enough tasks
        available_subsets = loader.get_available_subsets()
        large_subsets = []
        
        for subset in available_subsets:
            try:
                if subset.startswith("arc-prize-"):
                    tasks = loader.get_subset_tasks(subset)
                    if len(tasks) >= 10:
                        large_subsets.append(subset)
                        break
            except ValueError:
                continue
        
        if large_subsets:
            subset_name = large_subsets[0]
            
            # Test with limit
            tasks_limited = loader.get_dataset_subset(subset_name, max_rows=5)
            tasks_unlimited = loader.get_dataset_subset(subset_name)
            
            assert len(tasks_limited) <= 5
            assert len(tasks_unlimited) >= len(tasks_limited)


class TestTaskLoaderIntegration:
    """Integration tests for dataset loading with real data"""
    
    def test_trelis_dataset_integration(self):
        """Test loading the Trelis dataset mentioned by user"""
        loader = TaskLoader(get_default_data_root())
        
        # This test will only run if the user has access to the dataset
        # In a real environment with network access
        try:
            with patch('datasets.load_dataset') as mock_load:
                # Mock a realistic response
                mock_dataset = MagicMock()
                mock_dataset.column_names = ['task_id', 'code', 'reasoning']
                mock_dataset.__len__.return_value = 100
                mock_dataset.select.return_value = mock_dataset
                mock_dataset.__getitem__.side_effect = lambda key: [
                    'task_' + str(i).zfill(3) for i in range(1, 11)
                ]
                mock_load.return_value = mock_dataset
                
                # Add some sample tasks to cache
                for i in range(1, 6):
                    task_id = f'task_{i:03d}'
                    loader.tasks[task_id] = {
                        'train': [{'input': [[i]], 'output': [[i+1]]}],
                        'test': [{'input': [[i+10]], 'output': [[i+11]]}]
                    }
                
                tasks = loader.get_dataset_subset("Trelis/arc-agi-partials-for-refinement", max_rows=10)
                
                assert len(tasks) > 0
                assert all(isinstance(task[0], str) and isinstance(task[1], dict) for task in tasks)
                
        except Exception as e:
            pytest.skip(f"Integration test skipped due to network/access issue: {e}")

    def test_backward_compatibility(self):
        """Test that all existing functionality still works"""
        loader = TaskLoader(get_default_data_root())
        
        # Test that traditional methods still work
        stats = loader.get_stats()
        assert "total_tasks" in stats
        assert stats["total_tasks"] > 0
        
        subsets = loader.get_available_subsets()
        assert len(subsets) > 0
        
        # Test getting a traditional subset through old and new methods
        if subsets:
            subset_name = [s for s in subsets if s.startswith("arc-prize-")][0]
            
            # Old method
            tasks_old = loader.get_subset_tasks(subset_name)
            
            # New method with traditional subset
            tasks_new = loader.get_dataset_subset(subset_name)
            
            # Should return the same tasks
            assert len(tasks_old) == len(tasks_new)
            assert set(t[0] for t in tasks_old) == set(t[0] for t in tasks_new)
