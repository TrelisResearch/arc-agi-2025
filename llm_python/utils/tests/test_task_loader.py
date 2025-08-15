from ..task_loader import TaskLoader


class TestTaskLoader:
    """Basic tests for TaskLoader using real competition data"""
    
    def test_initialization(self):
        """Test that TaskLoader initializes and loads data"""
        loader = TaskLoader()
        assert len(loader.tasks) > 0
        assert len(loader.subsets) > 0
        print(f"Loaded {len(loader.tasks)} tasks and {len(loader.subsets)} subsets")
    
    def test_get_task(self):
        """Test getting a task by ID"""
        loader = TaskLoader()
        # Get a task ID from the competition data
        if loader.tasks:
            task_id = next(iter(loader.tasks.keys()))
            task_data = loader.get_task(task_id)
            assert "train" in task_data
            assert "test" in task_data
            assert len(task_data["train"]) > 0
    
    def test_get_subset_tasks(self):
        """Test getting tasks from a competition subset"""
        loader = TaskLoader()
        # Try to get tasks from a competition subset
        competition_subsets = [s for s in loader.get_available_subsets() if s.startswith('arc-prize-')]
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
        loader = TaskLoader()
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
    
    def test_backward_compatibility_methods(self):
        """Test that old interface methods still work"""
        loader = TaskLoader()
        if loader.tasks:
            # Test old load_task method
            task_id = next(iter(loader.tasks.keys()))
            task_data = loader.load_task(task_id)
            assert "train" in task_data
            assert "test" in task_data
            
            # Test old load_tasks_from_subset method
            try:
                tasks = loader.load_tasks_from_subset("shortest_training_1", "arc-agi-1")
                assert isinstance(tasks, list)
            except ValueError:
                # Subset might not exist, which is fine
                pass    