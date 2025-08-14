import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import duckdb

from .schema import ProgramSample


# Global registry for singleton instances
_db_instances: Dict[str, 'LocalProgramsDB'] = {}


def get_localdb(db_path: Optional[str] = None) -> 'LocalProgramsDB':
    """
    Get or create a LocalProgramsDB instance using singleton pattern.
    
    Args:
        db_path: Path to the database file. If None, uses default location.
        
    Returns:
        LocalProgramsDB instance for the specified path
    """
    # Normalize the path
    if db_path is None:
        current_dir = Path(__file__).parent
        db_path = str(current_dir / "local.db")
    else:
        db_path = str(Path(db_path).resolve())
    
    # Return existing instance or create new one
    if db_path not in _db_instances:
        _db_instances[db_path] = LocalProgramsDB(db_path)
    return _db_instances[db_path]


class LocalProgramsDB:
    """
    A simple DuckDB-based database for storing SOAR program examples.
    
    Features:
    - Lazy connection initialization
    - Automatic table creation based on SoarProgramExample schema
    - Basic validation when adding programs
    - JSON serialization for complex fields
    """
    
    def __init__(self, db_path: str):
        """
        Initialize the database connection manager.
        
        Note: This class should not be instantiated directly. Use get_localdb() 
        function instead to ensure singleton behavior per database path.
        
        Args:
            db_path: Path to the database file.
        """
        # Only initialize if this is a fresh instance
        if not hasattr(self, '_initialized'):
            self.db_path = db_path
            self._connection: Optional[duckdb.DuckDBPyConnection] = None
            self._initialized = True
    
    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Lazy initialization of DuckDB connection."""
        if self._connection is None:
            self._connection = duckdb.connect(self.db_path)
            self._ensure_table_exists()
        return self._connection
    
    def _ensure_table_exists(self) -> None:
        """Create the programs table and metadata table if they don't exist."""
        # Create metadata table for storing database ID
        create_metadata_sql = """
        CREATE TABLE IF NOT EXISTS metadata (
            key VARCHAR PRIMARY KEY,
            value VARCHAR NOT NULL
        )
        """
        self.connection.execute(create_metadata_sql)
        
        # Create main programs table
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS programs (
            key VARCHAR UNIQUE NOT NULL,
            task_id VARCHAR NOT NULL,
            reasoning TEXT,
            code TEXT NOT NULL,
            correct_train_input BOOLEAN[] NOT NULL,
            correct_test_input BOOLEAN[] NOT NULL,
            predicted_train_output INTEGER[][][] NOT NULL,
            predicted_test_output INTEGER[][][] NOT NULL,
            model VARCHAR NOT NULL
        )
        """
        self.connection.execute(create_table_sql)
        
        # Ensure database has a unique ID
        self._ensure_database_id()
    
    def _ensure_database_id(self) -> None:
        """Ensure the database has a unique ID, creating one if it doesn't exist."""
        result = self.connection.execute("SELECT value FROM metadata WHERE key = 'db_id'").fetchone()
        if not result:
            # Generate a new unique ID using uuid
            import uuid
            db_id = uuid.uuid4().hex[:16]  # Use first 16 characters for shorter ID
            self.connection.execute("INSERT INTO metadata (key, value) VALUES ('db_id', ?)", [db_id])
    
    def get_database_id(self) -> str:
        """Get the unique database ID."""
        result = self.connection.execute("SELECT value FROM metadata WHERE key = 'db_id'").fetchone()
        if not result:
            raise ValueError("Database ID not found - database may not be properly initialized")
        return result[0]
    
    def get_all_programs(self) -> List[Dict[str, Any]]:
        """
        Get all programs in the database.
        
        Returns:
            List of all program dictionaries
        """
        query_sql = "SELECT * FROM programs ORDER BY task_id, model"
        
        cursor = self.connection.execute(query_sql)
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        
        programs = []
        for result in results:
            program_dict = dict(zip(columns, result))
            programs.append(program_dict)
        
        return programs
    
    def sync_to_cloud(self) -> str:
        """
        Sync this database to Google Cloud Storage.
        
        Returns:
            The GCS path where the file was uploaded
        """
        # Import here to avoid circular imports
        from .sync import sync_database_to_cloud
        return sync_database_to_cloud(self.db_path)
    
    def _validate_program(self, program: Dict[str, Any]) -> None:
        """
        Validate a program dictionary against the SoarProgramExample schema.
        
        Args:
            program: Program data to validate
            
        Raises:
            ValueError: If validation fails
        """
        required_fields = [
            'task_id', 'code', 'correct_train_input', 'correct_test_input',
            'predicted_train_output', 'predicted_test_output', 'model'
        ]
        
        # Check all required fields are present
        missing_fields = [field for field in required_fields if field not in program]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Check field types
        if not isinstance(program['task_id'], str) or not program['task_id'].strip():
            raise ValueError("task_id must be a non-empty string")
        
        if not isinstance(program['code'], str) or not program['code'].strip():
            raise ValueError("code must be a non-empty string")
        
        if not isinstance(program['model'], str) or not program['model'].strip():
            raise ValueError("model must be a non-empty string")
        
        # Validate boolean arrays
        for field in ['correct_train_input', 'correct_test_input']:
            if not isinstance(program[field], list):
                raise ValueError(f"{field} must be a list")
            if not all(isinstance(x, bool) for x in program[field]):
                raise ValueError(f"{field} must be a list of booleans")
        
        # Validate 3D arrays (grids)
        for field in ['predicted_train_output', 'predicted_test_output']:
            if not isinstance(program[field], list):
                raise ValueError(f"{field} must be a list")
            
            for i, grid in enumerate(program[field]):
                if not isinstance(grid, list):
                    raise ValueError(f"{field}[{i}] must be a list (grid)")
                
                for j, row in enumerate(grid):
                    if not isinstance(row, list):
                        raise ValueError(f"{field}[{i}][{j}] must be a list (row)")
                    
                    if not all(isinstance(cell, int) for cell in row):
                        raise ValueError(f"{field}[{i}][{j}] must contain only integers")
        
        # Validate optional fields if present
        if 'reasoning' in program and program['reasoning'] is not None:
            if not isinstance(program['reasoning'], str):
                raise ValueError("reasoning must be a string if provided")
    
    def generate_key(self, task_id: str, code: str) -> str:
        """
        Generate a unique key for a program based on task_id and code.
        
        Args:
            task_id: The task identifier
            code: The program code
            
        Returns:
            A SHA-256 hash of the task_id and code
        """
        combined = f"{task_id}:{code}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()
    
    def _generate_key(self, task_id: str, code: str) -> str:
        """Legacy private method - use generate_key instead."""
        return self.generate_key(task_id, code)
    
    def add_program(self, program: Union[ProgramSample, Dict[str, Any]]) -> None:
        """
        Add a program to the database with validation.
        
        Args:
            program: Program data conforming to SoarProgramExample schema
            
        Raises:
            ValueError: If program validation fails
        """
        # Convert to dict - TypedDict is already a dict in runtime
        program_dict = dict(program)
        
        # Validate the program
        self._validate_program(program_dict)
        
        # Generate unique key from task_id and code
        key = self.generate_key(program_dict['task_id'], program_dict['code'])
        
        # Insert into database (ON CONFLICT IGNORE to avoid duplicates)
        insert_sql = """
        INSERT OR IGNORE INTO programs 
        (key, task_id, reasoning, code, correct_train_input, correct_test_input,
         predicted_train_output, predicted_test_output, model)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        self.connection.execute(insert_sql, [
            key,
            program_dict['task_id'],
            program_dict.get('reasoning'),
            program_dict['code'],
            program_dict['correct_train_input'],
            program_dict['correct_test_input'],
            program_dict['predicted_train_output'],
            program_dict['predicted_test_output'],
            program_dict['model']
        ])
    
    def get_programs_by_task(self, task_id: str) -> List[Dict[str, Any]]:
        """
        Get all programs for a specific task.
        
        Args:
            task_id: Task ID
            
        Returns:
            List of program dictionaries
        """
        query_sql = "SELECT * FROM programs WHERE task_id = ? ORDER BY model"
        
        cursor = self.connection.execute(query_sql, [task_id])
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        
        programs = []
        for result in results:
            program_dict = dict(zip(columns, result))
            programs.append(program_dict)
        
        return programs
    
    def count_programs(self) -> int:
        """Get total number of programs in the database."""
        result = self.connection.execute("SELECT COUNT(*) FROM programs").fetchone()
        return result[0] if result else 0
    
    def get_task_count(self) -> int:
        """Get number of unique tasks in the database."""
        result = self.connection.execute("SELECT COUNT(DISTINCT task_id) FROM programs").fetchone()
        return result[0] if result else 0
    
    def close(self) -> None:
        """Close the database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
    
    @classmethod
    def clear_all_instances(cls) -> None:
        """
        Close all database connections and clear singleton instances.
        Useful for testing or when you want to force recreation of instances.
        """
        for instance in _db_instances.values():
            instance.close()
        _db_instances.clear()
    
    @classmethod
    def get_instance_count(cls) -> int:
        """Get the number of active singleton instances."""
        return len(_db_instances)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
