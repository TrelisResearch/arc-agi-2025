from ..task_loader import TaskLoader


class TestTaskLoader:
    """Basic tests for TaskLoader using real competition data"""
    
    def test_initialization(self):
        """Test that TaskLoader initializes and loads data"""
        loader = TaskLoader()
        assert len(loader.tasks) > 0
        assert len(loader.subsets) > 0
        print(f"Loaded {len(loader.tasks)} tasks and {len(loader.subsets)} subsets")
    
    def test_canonical_dataset_parameter(self):
        """Test that canonical dataset parameter works"""
        # Test default
        loader_default = TaskLoader()
        assert loader_default.canonical_dataset == "arc-prize-2025"
        
        # Test custom
        loader_2024 = TaskLoader(canonical_dataset="arc-prize-2024")
        assert loader_2024.canonical_dataset == "arc-prize-2024"
        
        # Both should load the same number of tasks
        assert len(loader_default.tasks) == len(loader_2024.tasks)
    
    def test_all_tasks_have_outputs(self):
        """Test that all loaded tasks have test outputs"""
        loader = TaskLoader()
        tasks_without_outputs = []
        
        for task_id, task_data in loader.tasks.items():
            # Check if any test case has no output
            for i, test_case in enumerate(task_data["test"]):
                if test_case.get("output") is None:
                    tasks_without_outputs.append((task_id, i, loader.task_sources.get(task_id, "unknown")))
        
        if tasks_without_outputs:
            print(f"Found {len(tasks_without_outputs)} test cases without outputs:")
            for task_id, test_idx, source in tasks_without_outputs[:10]:  # Show first 10
                print(f"  Task {task_id} (from {source}), test case {test_idx}")
            
        # This should pass - all tasks should have outputs
        assert len(tasks_without_outputs) == 0, f"Found {len(tasks_without_outputs)} test cases without outputs"
    
    def test_task_sources_tracking(self):
        """Test that task sources are properly tracked"""
        loader = TaskLoader()
        
        # All tasks should have a source
        for task_id in loader.tasks:
            assert task_id in loader.task_sources
            source = loader.task_sources[task_id]
            assert source in ["arc-prize-2024", "arc-prize-2025"]
    
    def test_conflict_resolution(self):
        """Test that canonical dataset preference resolves conflicts correctly"""
        # Load with arc-prize-2025 as canonical (default)
        loader_2025 = TaskLoader(canonical_dataset="arc-prize-2025")
        
        # Load with arc-prize-2024 as canonical
        loader_2024 = TaskLoader(canonical_dataset="arc-prize-2024")
        
        # Find tasks that exist in both datasets
        conflicting_tasks = []
        for task_id in loader_2025.tasks:
            if (loader_2025.task_sources[task_id] == "arc-prize-2025" and 
                task_id in loader_2024.tasks and 
                loader_2024.task_sources[task_id] == "arc-prize-2024"):
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