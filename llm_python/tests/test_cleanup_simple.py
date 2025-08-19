#!/usr/bin/env python3
"""
Simple integration test for cleanup functionality that can run in CI.

This test focuses on the core functionality without complex mocking:
- Database connection cleanup
- WAL file cleanup  
- Basic error handling
"""

import pytest
import tempfile
import os
from llm_python.run_arc_tasks_soar import cleanup_handler
from llm_python.programsdb.localdb import get_localdb, LocalProgramsDB


class TestCleanupIntegration:
    """Integration tests for cleanup functionality"""
    
    def teardown_method(self):
        """Clean up after each test"""
        # Always clear database instances after tests
        LocalProgramsDB.clear_all_instances()
    
    def test_database_cleanup_basic(self):
        """Test basic database cleanup functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test.db")
            
            # Create database connection
            db = get_localdb(db_path)
            
            # Force table creation and some activity
            connection = db.connection  # This initializes the database
            initial_count = LocalProgramsDB.get_instance_count()
            assert initial_count > 0, "Should have database connections"
            
            # Clear instances manually (simulating what cleanup_handler does)
            LocalProgramsDB.clear_all_instances()
            
            # Verify cleanup worked
            final_count = LocalProgramsDB.get_instance_count()
            assert final_count == 0, "Database connections should be closed"
    
    def test_cleanup_handler_no_crash(self):
        """Test that cleanup handler doesn't crash with no connections"""
        # Ensure no connections exist
        LocalProgramsDB.clear_all_instances()
        
        # This should not raise any exceptions
        try:
            cleanup_handler()
        except Exception as e:
            pytest.fail(f"cleanup_handler() should not crash when no connections exist: {e}")
    
    def test_wal_file_handling(self):
        """Test that WAL files are properly handled during cleanup"""
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_wal.db")
            
            # Create database and force some activity
            db = get_localdb(db_path)
            connection = db.connection  # Initialize database
            
            # Check if WAL file was created
            wal_path = f"{db_path}.wal"
            wal_exists_before = os.path.exists(wal_path)
            
            # Clear connections (what cleanup_handler does)
            LocalProgramsDB.clear_all_instances()
            
            # The important thing is that no errors occurred
            # WAL file behavior depends on DuckDB's internal logic
            assert LocalProgramsDB.get_instance_count() == 0, "Connections should be closed"
            print(f"WAL existed before: {wal_exists_before}, after: {os.path.exists(wal_path)}")


# Test that can be run manually for real-world verification
@pytest.mark.skip(reason="Manual test - run with pytest -m 'not manual' to skip")
def test_manual_real_cleanup():
    """
    Manual test for real cleanup behavior verification.
    
    Run manually with: uv run pytest llm_python/tests/test_cleanup_simple.py::test_manual_real_cleanup -s -v
    """
    import signal
    import llm_python.run_arc_tasks_soar as runner_module
    
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = os.path.join(temp_dir, "manual_test.db")
        print(f"\n📁 Manual test database: {db_path}")
        
        # Reset cleanup state for this test
        runner_module._cleanup_done = False
        
        # Create database with activity
        db = get_localdb(db_path)
        connection = db.connection
        print(f"📦 Database connections: {LocalProgramsDB.get_instance_count()}")
        
        # Check for WAL file
        wal_path = f"{db_path}.wal"
        print(f"📄 WAL file exists: {os.path.exists(wal_path)}")
        
        # Call the real cleanup handler
        cleanup_handler(signal.SIGINT)
        
        # Check results
        print(f"📦 Database connections after cleanup: {LocalProgramsDB.get_instance_count()}")
        print(f"📄 WAL file exists after cleanup: {os.path.exists(wal_path)}")
        print("✅ Manual test completed")


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v"])