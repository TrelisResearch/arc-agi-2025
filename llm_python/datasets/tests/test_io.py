"""
Tests for llm_python.datasets.io module.

Tests parquet reading/writing, schema validation, and error handling.
"""

import pytest
import pandas as pd
import pyarrow as pa
import tempfile
from pathlib import Path

from llm_python.datasets.io import (
    read_soar_parquet, 
    write_soar_parquet, 
    validate_soar_dataframe_schema
)


def create_valid_sample_data() -> pd.DataFrame:
    """Create a valid DataFrame matching the ProgramSample schema."""
    return pd.DataFrame({
        'row_id': ["abcdef", "123456"],
        'task_id': ['task_001', 'task_002'],
        'reasoning': ['Some reasoning', None],  # Test nullable field
        'program': ['def generate():\n    pass', 'def generate():\n    return []'],
        'correct_train_input': [[True, False], [True, True]],
        'correct_test_input': [[False], [True]],
        'predicted_train_output': [
            [[[1, 2], [3, 4]], [[5, 6]]],
            [[[7, 8]], [[9, 0]]]
        ],
        'predicted_test_output': [
            [[[1, 1]]],
            [[[2, 2]]]
        ],
        'model': ['gpt-4o-mini', 'claude-3-5-sonnet'],
    })


def create_invalid_schema_data() -> pd.DataFrame:
    """Create DataFrame with schema violations."""
    return pd.DataFrame({
        'row_id': ["abcdef"],
        'task_id': ['task_001'],
        'reasoning': ['Some reasoning'],
        'program': ['def generate(): pass'],
        # Missing required columns - this should definitely fail
        'extra_column': ['should not be here'],  # Extra column
    })


def create_missing_columns_data() -> pd.DataFrame:
    """Create DataFrame missing required columns."""
    return pd.DataFrame({
        'task_id': ['task_001'],
        'reasoning': ['Some reasoning'],
        'program': ['def generate(): pass'],
        # Missing other required columns
    })


def create_null_required_field_data() -> pd.DataFrame:
    """Create DataFrame with null in required field."""
    return pd.DataFrame({
        'row_id': ["abcdef"],
        'task_id': [None],  # Null in required field
        'reasoning': ['Some reasoning'],
        'program': ['def generate(): pass'],
        'correct_train_input': [[True, False]],
        'correct_test_input': [[False]],
        'predicted_train_output': [
            [[[1, 2], [3, 4]]]
        ],
        'predicted_test_output': [
            [[[1, 1]]]
        ],
        'model': ['gpt-4']
    })


def create_wrong_type_data() -> pd.DataFrame:
    """Create DataFrame with wrong data types."""
    return pd.DataFrame({
        'row_id': ["abcdef"],
        'task_id': ['task_001'],
        'reasoning': ['Some reasoning'],
        'program': ['def generate(): pass'],
        'correct_train_input': ['not_a_list'],  # Wrong type - should be list[bool]
        'correct_test_input': [[False]],
        'predicted_train_output': [
            [[[1, 2], [3, 4]]]
        ],
        'predicted_test_output': [
            [[[1, 1]]]
        ],
        'model': ['gpt-4']
    })


class TestParquetReadWrite:
    """Test parquet file reading and writing functionality."""
    
    def test_roundtrip_valid_data(self):
        """Test writing and reading back valid data preserves it correctly."""
        df_original = create_valid_sample_data()
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
        try:
            # Write the data
            write_soar_parquet(df_original, tmp_path)
            
            # Read it back
            df_read = read_soar_parquet(tmp_path)
            
            # Check structure is preserved
            assert len(df_read) == len(df_original)
            
            # Check specific values
            assert df_read['task_id'].tolist() == ['task_001', 'task_002']
            # Note: PyArrow uses pd.NA instead of None for nulls
            assert df_read['reasoning'].tolist() == ['Some reasoning', pd.NA]
            assert df_read['correct_train_input'].tolist() == [[True, False], [True, True]]
            
        finally:
            tmp_path.unlink(missing_ok=True)
    
    def test_write_invalid_schema_fails(self):
        """Test that writing invalid data raises appropriate error."""
        df_invalid = create_invalid_schema_data()
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
        try:
            with pytest.raises((pa.ArrowInvalid, pa.ArrowTypeError, ValueError, KeyError)):
                write_soar_parquet(df_invalid, tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
    
    def test_read_nonexistent_file_fails(self):
        """Test that reading non-existent file raises appropriate error."""
        with pytest.raises(FileNotFoundError):
            read_soar_parquet('/nonexistent/path/file.parquet')
    
    def test_preserves_data_types(self):
        """Test that PyArrow data types are preserved through roundtrip."""
        df_original = create_valid_sample_data()
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
        try:
            write_soar_parquet(df_original, tmp_path)
            df_read = read_soar_parquet(tmp_path)
            
            # Check that list columns remain as proper Python lists
            assert isinstance(df_read['correct_train_input'].iloc[0], list)
            assert isinstance(df_read['predicted_train_output'].iloc[0], list)
            assert isinstance(df_read['predicted_train_output'].iloc[0][0], list)
            assert isinstance(df_read['predicted_train_output'].iloc[0][0][0], list)
            
        finally:
            tmp_path.unlink(missing_ok=True)


class TestSchemaValidation:
    """Test standalone schema validation functionality."""
    
    def test_validate_valid_data_passes(self):
        """Test that valid data passes validation."""
        df_valid = create_valid_sample_data()
        
        # Should not raise any exception
        validate_soar_dataframe_schema(df_valid)
    
    def test_validate_missing_columns_fails(self):
        """Test that missing required columns fail validation."""
        df_missing = create_missing_columns_data()
        
        with pytest.raises(ValueError, match="Schema validation failed"):
            validate_soar_dataframe_schema(df_missing)
    
    def test_validate_extra_columns_fails(self):
        """Test that extra columns fail validation."""
        df_extra = create_invalid_schema_data()
        
        with pytest.raises(ValueError, match="Schema validation failed"):
            validate_soar_dataframe_schema(df_extra)
    
    def test_validate_null_required_field_fails(self):
        """Test that null values in required fields fail validation."""
        df_null = create_null_required_field_data()
        
        with pytest.raises(ValueError, match="Schema validation failed"):
            validate_soar_dataframe_schema(df_null)
    
    def test_validate_wrong_types_fails(self):
        """Test that wrong data types fail validation."""
        df_wrong_type = create_wrong_type_data()
        
        with pytest.raises(ValueError, match="Schema validation failed"):
            validate_soar_dataframe_schema(df_wrong_type)
    
    def test_validate_nullable_reasoning_field(self):
        """Test that null reasoning field is allowed."""
        df = create_valid_sample_data()
        df['reasoning'] = [None, None]  # All null reasoning should be fine
        
        # Should not raise any exception
        validate_soar_dataframe_schema(df)
    
    def test_validate_empty_dataframe(self):
        """Test validation of empty DataFrame with correct schema."""
        # Create empty DataFrame with correct columns and types
        df_empty = pd.DataFrame({
            'row_id': pd.Series([], dtype='string'),
            'task_id': pd.Series([], dtype='string'),
            'reasoning': pd.Series([], dtype='string'),
            'program': pd.Series([], dtype='string'),
            'correct_train_input': pd.Series([], dtype='object'),
            'correct_test_input': pd.Series([], dtype='object'),
            'predicted_train_output': pd.Series([], dtype='object'),
            'predicted_test_output': pd.Series([], dtype='object'),
            'model': pd.Series([], dtype='string'),
            'is_transductive': pd.Series([], dtype='bool'),
        })
        
        # Should not raise any exception
        validate_soar_dataframe_schema(df_empty)
    
    def test_validate_null_in_later_rows(self):
        """Test that validation catches null values in non-first rows."""
        df = pd.DataFrame({
            'row_id': ["abcdef", "123456"],
            'task_id': ['task_001', None],  # Null in second row
            'reasoning': ['Some reasoning', 'More reasoning'],
            'program': ['def generate(): pass', 'def generate(): return []'],
            'correct_train_input': [[True, False], [True]],
            'correct_test_input': [[False], [True]],
            'predicted_train_output': [
                [[[1, 2]]],
                [[[3, 4]]]
            ],
            'predicted_test_output': [
                [[[5, 6]]],
                [[[7, 8]]]
            ],
            'model': ['gpt-4', 'claude-3']
        })
        
        # Should catch the null in the second row
        with pytest.raises(ValueError, match="Schema validation failed"):
            validate_soar_dataframe_schema(df)


class TestEdgeCases:
    """Test edge cases and corner scenarios."""
    
    def test_large_nested_lists(self):
        """Test handling of large nested list structures."""
        # Create data with larger nested structures
        large_output = [[[i, i+1] for i in range(10)] for _ in range(5)]
        
        df = pd.DataFrame({
            'row_id': ["abcdef"],
            'task_id': ['large_task'],
            'reasoning': ['Large structure test'],
            'program': ['def generate(): pass'],
            'correct_train_input': [[True] * 20],
            'correct_test_input': [[False] * 15],
            'predicted_train_output': [large_output],
            'predicted_test_output': [large_output[:2]],
            'model': ['gpt-4o-mini'],
        })
        
        # Should validate successfully
        validate_soar_dataframe_schema(df)
        
        # Should roundtrip successfully
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
        try:
            write_soar_parquet(df, tmp_path)
            df_read = read_soar_parquet(tmp_path)
            
            assert df_read['predicted_train_output'].iloc[0] == large_output
            
        finally:
            tmp_path.unlink(missing_ok=True)
    
    def test_unicode_strings(self):
        """Test handling of unicode strings in text fields."""
        df = pd.DataFrame({
            'row_id': ["abcdef"],
            'task_id': ['测试_task_🔥'],
            'reasoning': ['Unicode reasoning: 数学 🧮 πŸ"'],
            'program': ['def generate():\n    # Comment with émojis 🐍\n    return []'],
            'correct_train_input': [[True]],
            'correct_test_input': [[False]],
            'predicted_train_output': [
                [[[1, 2]]]
            ],
            'predicted_test_output': [
                [[[3, 4]]]
            ],
            'model': ['claude-3-5-sonnet'],
        })
        
        # Should validate and roundtrip successfully
        validate_soar_dataframe_schema(df)
        
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
        try:
            write_soar_parquet(df, tmp_path)
            df_read = read_soar_parquet(tmp_path)
            
            assert df_read['task_id'].iloc[0] == '测试_task_🔥'
            assert '🧮' in df_read['reasoning'].iloc[0]
            
        finally:
            tmp_path.unlink(missing_ok=True)
    
    def test_empty_lists_in_list_columns(self):
        """Test that empty lists in list columns are handled correctly."""
        df = pd.DataFrame({
            'row_id': ["abcdef"],
            'task_id': ['empty_lists_task'],
            'reasoning': [None],
            'program': ['def generate(): return []'],
            'correct_train_input': [[]],  # Empty list
            'correct_test_input': [[]],   # Empty list
            'predicted_train_output': [[]],  # Empty list
            'predicted_test_output': [[]],   # Empty list
            'model': ['gpt-4o-mini'],
        })
        
        # Should validate successfully (empty lists are valid)
        validate_soar_dataframe_schema(df)
        
        # Should roundtrip successfully
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            tmp_path = Path(tmp.name)
            
        try:
            write_soar_parquet(df, tmp_path)
            df_read = read_soar_parquet(tmp_path)
            
            assert df_read['correct_train_input'].iloc[0] == []
            assert df_read['predicted_train_output'].iloc[0] == []
            
        finally:
            tmp_path.unlink(missing_ok=True)
