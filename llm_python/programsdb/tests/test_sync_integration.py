"""
Integration tests for the sync functionality.

Tests the full round-trip of:
1. Creating a database and adding programs
2. Exporting to parquet with schema validation
3. Importing from parquet into a new database
4. Verifying data integrity
"""

import pytest
from typing import List
from pathlib import Path

from ..localdb import get_localdb
from ..schema import ProgramSample
from ..sync import import_from_parquet, sync_database_to_cloud


@pytest.fixture
def test_programs():
    """Fixture that provides test programs for testing."""
    return create_test_programs()


@pytest.fixture
def cleanup_db():
    """Fixture to track and clean up only test database instances."""
    created_dbs = []
    
    def track_db(db_path):
        """Track a database path for cleanup."""
        created_dbs.append(db_path)
        return get_localdb(db_path)
    
    yield track_db
    
    # Clean up using the class method that handles thread-local instances
    from ..localdb import LocalProgramsDB, _thread_local
    for db_path in created_dbs:
        if hasattr(_thread_local, 'db_instances') and db_path in _thread_local.db_instances:
            # Close the database connection if it has one
            if hasattr(_thread_local.db_instances[db_path], 'close'):
                _thread_local.db_instances[db_path].close()
            del _thread_local.db_instances[db_path]


def create_test_programs() -> List[ProgramSample]:
    """Create a set of test programs with various data patterns."""
    programs = []
    
    # Create diverse test data to stress-test the schema
    test_cases: List[ProgramSample] = [
        {
            'task_id': 'task_001',
            'reasoning': 'Simple pattern recognition',
            'code': 'def generate(input_grid):\n    return [[1, 2], [3, 4]]',
            'correct_train_input': [True, False, True],
            'correct_test_input': [False, True],
            'predicted_train_output': [
                [[1, 2], [3, 4]], 
                [[5, 6], [7, 8]], 
                [[9, 10], [11, 12]]
            ],
            'predicted_test_output': [
                [[13, 14], [15, 16]], 
                [[17, 18], [19, 20]]
            ],
            'model': 'gpt-4-test'
        },
        {
            'task_id': 'task_002',
            'reasoning': 'Complex transformation with multiple steps',
            'code': 'def generate(input_grid):\n    # Multi-line\n    # Complex logic\n    return transform(input_grid)',
            'correct_train_input': [False, False, False, True],
            'correct_test_input': [True],
            'predicted_train_output': [
                [[0]], 
                [[1, 1], [1, 1]], 
                [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
                [[3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]]
            ],
            'predicted_test_output': [
                [[4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4], [4, 4, 4, 4, 4]]
            ],
            'model': 'claude-3-test'
        },
        {
            'task_id': 'task_003', 
            'reasoning': None,  # Test optional field
            'code': 'def generate(input_grid):\n    return input_grid',
            'correct_train_input': [],  # Test empty arrays
            'correct_test_input': [],
            'predicted_train_output': [],
            'predicted_test_output': [],
            'model': 'local-model'
        },
        {
            'task_id': 'task_001',  # Duplicate task_id with different code
            'reasoning': 'Alternative approach for same task',
            'code': 'def generate(input_grid):\n    return [[x + 1 for x in row] for row in input_grid]',
            'correct_train_input': [True, True, False],
            'correct_test_input': [True, False],
            'predicted_train_output': [
                [[2, 3], [4, 5]], 
                [[6, 7], [8, 9]], 
                [[10, 11], [12, 13]]
            ],
            'predicted_test_output': [
                [[14, 15], [16, 17]], 
                [[18, 19], [20, 21]]
            ],
            'model': 'gpt-4-test'
        }
    ]
    
    for case in test_cases:
        programs.append(ProgramSample(**case))
    
    return programs


class TestSyncIntegration:
    """Test suite for sync functionality integration."""
    
    def test_schema_validation(self, tmp_path, cleanup_db):
        """Test that schema validation catches mismatched parquet files."""
        
        parquet_path = tmp_path / "invalid_schema.parquet"
        db_path = tmp_path / "schema_test.db"
        
        # Create a parquet file with wrong schema
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        wrong_data = {
            'task_id': ['task1'],
            'code': ['def test(): pass'],
            'wrong_field': ['this should not be here'],
            # Missing required fields
        }
        
        df = pd.DataFrame(wrong_data)
        # Use a different schema - this should fail validation
        wrong_schema = pa.schema([
            ("task_id", pa.string()),
            ("code", pa.string()),
            ("wrong_field", pa.string()),
        ])
        
        table = pa.Table.from_pandas(df, schema=wrong_schema)
        pq.write_table(table, str(parquet_path))
        
        # Try to import - should fail with schema validation error
        cleanup_db(str(db_path))
        
        with pytest.raises(ValueError, match="Missing required field"):
            import_from_parquet(str(parquet_path), str(db_path))

    def test_full_roundtrip_with_sync_function(self, test_programs, tmp_path, cleanup_db):
        """Test complete round-trip using sync_database_to_cloud with local destination."""
        # Step 1: Create source database with test programs
        source_db_path = str(tmp_path / "source.db")
        source_db = cleanup_db(source_db_path)
        
        # Add test programs to source
        for program in test_programs:
            source_db.add_program(program)
        
        print(f"Added {len(test_programs)} programs to source database")
        
        # Step 2: Use sync function to export to local parquet file
        parquet_path = str(tmp_path / "sync_export.parquet")
        result_path = sync_database_to_cloud(source_db_path, destination=parquet_path)
        
        # Verify the sync worked and returned correct path
        assert result_path == parquet_path, f"Expected sync to return {parquet_path}, got {result_path}"
        assert Path(parquet_path).exists(), "Parquet file should exist after sync"
        
        print(f"Synced to local parquet file: {parquet_path}")
        
        # Step 3: Create target database and import from the synced parquet
        target_db_path = str(tmp_path / "target.db")
        target_db = cleanup_db(target_db_path)
        
        # Verify target is initially empty
        assert target_db.count_programs() == 0, "Target DB should be initially empty"
        
        # Import from the synced parquet file
        imported_count = import_from_parquet(parquet_path, target_db_path)
        
        # Step 4: Verify target database matches source
        assert imported_count == len(test_programs), f"Should import all {len(test_programs)} programs"
        assert target_db.count_programs() == source_db.count_programs(), "Target and source should have same program count"
        assert target_db.get_task_count() == source_db.get_task_count(), "Target and source should have same task count"
        
        # Step 5: Verify data integrity by comparing specific programs
        source_programs = source_db.get_all_programs()
        target_programs = target_db.get_all_programs()
        
        # Sort both lists by key for comparison
        source_programs.sort(key=lambda x: x['key'])
        target_programs.sort(key=lambda x: x['key'])
        
        assert len(source_programs) == len(target_programs), "Program counts should match"
        
        for source_prog, target_prog in zip(source_programs, target_programs):
            assert source_prog == target_prog, f"Programs should be identical: {source_prog['key']}"
        
        print("✓ Full round-trip with sync function completed successfully")
        print(f"✓ Source DB: {source_db.count_programs()} programs")
        print(f"✓ Target DB: {target_db.count_programs()} programs")
        print(f"✓ Parquet file: {imported_count} programs imported")


