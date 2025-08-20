"""
Tests for llm_python.datasets.validation module.

Tests validation functionality including schema checks, business logic, and result objects.
"""

import pandas as pd

from llm_python.datasets.validation import validate_soar_dataframe, ValidationResult
from llm_python.datasets.tests.test_io import create_valid_sample_data


class TestValidationResult:
    """Test the ValidationResult dataclass and its methods."""
    
    def test_is_valid_all_pass(self):
        """Test is_valid returns True when all validations pass."""
        result = ValidationResult(
            total_rows=10,
            schema_valid=True,
            schema_error=None,
            business_logic_valid=True,
            business_logic_issues=[],
            correctness_sample_size=5,
            correctness_valid=True,
            correctness_errors=[]
        )
        assert result.is_valid() is True
    
    def test_is_valid_schema_fail(self):
        """Test is_valid returns False when schema validation fails."""
        result = ValidationResult(
            total_rows=10,
            schema_valid=False,
            schema_error="Missing column",
            business_logic_valid=True,
            business_logic_issues=[],
            correctness_sample_size=5,
            correctness_valid=True,
            correctness_errors=[]
        )
        assert result.is_valid() is False
    
    def test_is_valid_business_logic_fail(self):
        """Test is_valid returns False when business logic validation fails."""
        result = ValidationResult(
            total_rows=10,
            schema_valid=True,
            schema_error=None,
            business_logic_valid=False,
            business_logic_issues=["Empty lists found"],
            correctness_sample_size=5,
            correctness_valid=True,
            correctness_errors=[]
        )
        assert result.is_valid() is False
    
    def test_is_valid_correctness_fail(self):
        """Test is_valid returns False when correctness validation fails."""
        result = ValidationResult(
            total_rows=10,
            schema_valid=True,
            schema_error=None,
            business_logic_valid=True,
            business_logic_issues=[],
            correctness_sample_size=5,
            correctness_valid=False,
            correctness_errors=["Program failed"]
        )
        assert result.is_valid() is False


class TestValidateSoarDataframe:
    """Test the main validation function."""
    
    def test_validate_valid_data(self):
        """Test validation of valid data."""
        df = create_valid_sample_data()
        
        result = validate_soar_dataframe(df, correctness_samples=0)
        
        assert result.total_rows == 2
        assert result.schema_valid is True
        assert result.schema_error is None
        assert result.business_logic_valid is True
        assert result.business_logic_issues == []
        assert result.correctness_valid is True
        assert result.correctness_errors == []
        assert result.is_valid() is True
    
    def test_validate_empty_dataframe(self):
        """Test validation of empty DataFrame."""
        df = pd.DataFrame()
        
        result = validate_soar_dataframe(df, correctness_samples=0)
        
        assert result.total_rows == 0
        assert result.schema_valid is False
        assert "Schema validation failed" in result.schema_error
        assert result.business_logic_valid is True  # No business logic issues for empty
        assert result.correctness_valid is True  # No correctness issues for empty
        assert result.is_valid() is False
    
    def test_validate_missing_columns(self):
        """Test validation with missing required columns."""
        df = pd.DataFrame({
            'task_id': ['task_001'],
            'code': ['def generate(): pass']
            # Missing other required columns
        })
        
        result = validate_soar_dataframe(df, correctness_samples=0)
        
        assert result.schema_valid is False
        assert "Schema validation failed" in result.schema_error
        assert result.is_valid() is False
    
    def test_business_logic_empty_lists(self):
        """Test business logic validation catches empty lists."""
        df = pd.DataFrame({
            'task_id': ['task_001'],
            'reasoning': [None],
            'code': ['def generate(): return []'],
            'correct_train_input': [[]],  # Empty list
            'correct_test_input': [[True]],
            'predicted_train_output': [[]],  # Empty list
            'predicted_test_output': [[[]]],
            'model': ['test_model']
        })
        
        result = validate_soar_dataframe(df, correctness_samples=0)
        
        assert result.schema_valid is True  # Schema is valid
        assert result.business_logic_valid is False  # But business logic has issues
        assert len(result.business_logic_issues) > 0
        assert any("empty lists" in issue for issue in result.business_logic_issues)
        assert result.is_valid() is False
    
    def test_business_logic_empty_code(self):
        """Test business logic validation catches empty code."""
        df = pd.DataFrame({
            'task_id': ['task_001'],
            'reasoning': [None],
            'code': ['   '],  # Whitespace only
            'correct_train_input': [[True]],
            'correct_test_input': [[True]],
            'predicted_train_output': [[[]]],
            'predicted_test_output': [[[]]],
            'model': ['test_model']
        })
        
        result = validate_soar_dataframe(df, correctness_samples=0)
        
        assert result.business_logic_valid is False
        assert any("empty or whitespace-only code" in issue for issue in result.business_logic_issues)
    
    def test_correctness_sampling(self):
        """Test that correctness sampling works correctly."""
        df = create_valid_sample_data()
        
        # Test with sample size larger than dataframe
        result = validate_soar_dataframe(df, correctness_samples=10)
        assert result.correctness_sample_size == 2  # Should be limited to df size
        
        # Test with sample size smaller than dataframe
        result = validate_soar_dataframe(df, correctness_samples=1)
        assert result.correctness_sample_size == 1
        
        # Test with zero samples
        result = validate_soar_dataframe(df, correctness_samples=0)
        assert result.correctness_sample_size == 0
        assert result.correctness_valid is True  # Should pass with no samples
    
    def test_seed_consistency(self):
        """Test that using the same seed produces consistent results."""
        df = create_valid_sample_data()
        
        result1 = validate_soar_dataframe(df, correctness_samples=1, seed=42)
        result2 = validate_soar_dataframe(df, correctness_samples=1, seed=42)
        
        # Results should be identical with same seed
        assert result1.correctness_sample_size == result2.correctness_sample_size
        assert result1.correctness_valid == result2.correctness_valid
    
    def test_program_correctness_with_real_program(self):
        """Test program correctness validation with a real Python program."""
        
        # Create a simple program that adds 1 to all values in the input grid
        program_code = """
import numpy as np

def generate(input_grid):
    # Add 1 to all values in the input grid
    return (np.array(input_grid) + 1).tolist()
"""
        
        # Create test data with known inputs and expected outputs
        # We'll create a fake task ID that probably doesn't exist to test error handling
        expected_output = [[1, 2], [3, 4]]  # If input was [[0, 1], [2, 3]]
        expected_output2 = [[6]]  # If input was [[5]]
        
        df = pd.DataFrame({
            'task_id': ['fake_task_123'],  # Non-existent task ID
            'reasoning': ['Adds 1 to all values'],
            'code': [program_code],
            'correct_train_input': [[True]],  # Assume the program works on training
            'correct_test_input': [[True]],   # Assume the program works on test
            'predicted_train_output': [[expected_output]],  # What we expect for training
            'predicted_test_output': [[expected_output2]],   # What we expect for test
            'model': ['test_model']
        })
        
        # Test with actual program execution (should fail due to non-existent task)
        result = validate_soar_dataframe(df, correctness_samples=1)
        
        # Should pass schema and business logic validation
        assert result.schema_valid is True
        assert result.business_logic_valid is True
        
        # Should fail correctness validation due to non-existent task or other execution issues
        assert result.correctness_valid is False
        assert len(result.correctness_errors) > 0
        assert result.correctness_sample_size == 1
        
        # The error should mention the program execution failure
        error_message = ' '.join(result.correctness_errors)
        assert 'execution failed' in error_message.lower() or 'row 0' in error_message.lower()
    
    

