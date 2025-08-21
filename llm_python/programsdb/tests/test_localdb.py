import pytest
import tempfile
import os

from ..localdb import get_localdb
from ..schema import ProgramSample


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing"""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=True) as f:
        db_path = f.name
    yield db_path
    # Cleanup - remove file if it exists
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def sample_program() -> ProgramSample:
    """Create a sample program for testing"""
    return {
        'task_id': 'test_task_001',
        'reasoning': 'This is a test reasoning trace',
        'code': '''def generate(input_grid):
    return [[1, 2], [3, 4]]''',
        'correct_train_input': [True, False],
        'correct_test_input': [True],
        'predicted_train_output': [[[1, 2], [3, 4]], [[0, 0], [0, 0]]],
        'predicted_test_output': [[[1, 2], [3, 4]]],
        'model': 'test_model',
        'is_transductive': False
    }


class TestLocalProgramsDB:
    """Test cases for LocalProgramsDB class"""
    
    def test_init_creates_database_and_table(self, temp_db_path):
        """Test that initializing LocalProgramsDB creates database and table"""
        # Make sure temp file doesn't exist
        if os.path.exists(temp_db_path):
            os.unlink(temp_db_path)
        
        db = get_localdb(temp_db_path)
        
        # Access connection to trigger table creation
        _ = db.connection
        
        # Check that database file was created
        assert os.path.exists(temp_db_path)
        
        # Check that table exists
        result = db.connection.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_name = 'programs'
        """).fetchone()
        assert result is not None
    
    def test_add_program_success(self, temp_db_path, sample_program):
        """Test successfully adding a valid program"""
        db = get_localdb(temp_db_path)
        
        # Should not raise any exception
        db.add_program(sample_program)
        
        # Verify program was added
        count = db.connection.execute("SELECT COUNT(*) FROM programs").fetchone()[0]
        assert count == 1
        
        # Check specific data
        row = db.connection.execute("SELECT task_id, model, key FROM programs").fetchone()
        assert row[0] == 'test_task_001'
        assert row[1] == 'test_model'
        assert len(row[2]) == 64  # SHA-256 hash is 64 hex characters
    
    def test_add_program_missing_required_field(self, temp_db_path):
        """Test that adding program with missing required field raises error"""
        db = get_localdb(temp_db_path)
        
        incomplete_program = {
            'task_id': 'test_task_001',
            # Missing other required fields
        }
        
        with pytest.raises(ValueError, match="Missing required fields"):
            db.add_program(incomplete_program)
    
    def test_add_program_invalid_types(self, temp_db_path, sample_program):
        """Test that adding program with invalid types raises error"""
        db = get_localdb(temp_db_path)
        
        # Test invalid correct_train_input (should be list of bools)
        invalid_program = sample_program.copy()
        invalid_program['correct_train_input'] = "not a list"
        
        with pytest.raises(ValueError, match="correct_train_input must be a list"):
            db.add_program(invalid_program)
        
        # Test invalid predicted_train_output (should be 3D list)
        invalid_program = sample_program.copy()
        invalid_program['predicted_train_output'] = [[1, 2, 3]]  # 2D instead of 3D
        
        with pytest.raises(ValueError, match="must be a list \\(row\\)"):
            db.add_program(invalid_program)
    
    def test_add_multiple_programs(self, temp_db_path, sample_program):
        """Test adding multiple programs"""
        db = get_localdb(temp_db_path)
        
        # Add first program
        db.add_program(sample_program)
        
        # Add second program with different task_id
        second_program = sample_program.copy()
        second_program['task_id'] = 'test_task_002'
        second_program['model'] = 'different_model'
        db.add_program(second_program)
        
        # Verify both programs were added
        count = db.connection.execute("SELECT COUNT(*) FROM programs").fetchone()[0]
        assert count == 2
        
        # Check we have both task_ids
        task_ids = db.connection.execute("SELECT DISTINCT task_id FROM programs ORDER BY task_id").fetchall()
        assert len(task_ids) == 2
        assert task_ids[0][0] == 'test_task_001'
        assert task_ids[1][0] == 'test_task_002'
    
    def test_lazy_connection(self, temp_db_path):
        """Test that database connection is created lazily"""
        # Make sure temp file doesn't exist
        if os.path.exists(temp_db_path):
            os.unlink(temp_db_path)
        
        # Create LocalProgramsDB instance - should not create file yet
        db = get_localdb(temp_db_path)
        assert not os.path.exists(temp_db_path)
        
        # Only when we actually use it should the file be created
        _ = db.connection
        assert os.path.exists(temp_db_path)
