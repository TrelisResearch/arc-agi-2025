"""
Tests for llm_python.datasets.validation module.

Tests validation functionality including schema checks, business logic, and result objects.
"""

import pandas as pd

from llm_python.datasets.validation import (
    validate_soar_dataframe,
    validate_soar_dataframe_correctness,
    validate_soar_row,
    ValidationResult,
    CorrectnessValidationResult,
)
from llm_python.datasets.tests.test_io import create_valid_sample_data


class TestValidationResult:
    def test_format_result_all_pass(self):
        result = ValidationResult(
            total_rows=2,
            schema_valid=True,
            schema_error=None,
            business_logic_valid=True,
            business_logic_issues=[],
        )
        assert result.is_valid() is True
        assert "PASS" in result.summary()

    def test_format_result_schema_fail(self):
        result = ValidationResult(
            total_rows=2,
            schema_valid=False,
            schema_error="Missing column",
            business_logic_valid=True,
            business_logic_issues=[],
        )
        assert result.is_valid() is False
        assert "FAIL" in result.summary()

    def test_format_result_business_logic_fail(self):
        result = ValidationResult(
            total_rows=2,
            schema_valid=True,
            schema_error=None,
            business_logic_valid=False,
            business_logic_issues=["Empty lists found"],
        )
        assert result.is_valid() is False
        assert "FAIL" in result.summary()


class TestCorrectnessValidationResult:
    def test_correctness_result_all_pass(self):
        result = CorrectnessValidationResult(
            total_rows=2, sample_size=2, correctness_valid=True, correctness_errors=[]
        )
        assert result.is_valid() is True
        assert "PASS" in result.summary()

    def test_correctness_result_fail(self):
        result = CorrectnessValidationResult(
            total_rows=2,
            sample_size=2,
            correctness_valid=False,
            correctness_errors=["Row 0: execution failed"],
        )
        assert result.is_valid() is False
        assert "FAIL" in result.summary()


class TestValidateSoarRow:
    def test_row_valid(self):
        row = {
            "task_id": "task_001",
            "reasoning": None,
            "code": "def generate(): return [[1]]",
            "correct_train_input": [True],
            "correct_test_input": [True],
            "predicted_train_output": [[[1]]],
            "predicted_test_output": [[[1]]],
            "model": "test_model",
            "is_transductive": False,
        }
        errors = validate_soar_row(row).errors
        assert errors == []

    def test_row_missing_fields(self):
        row = {"task_id": "task_001", "code": "def generate(): pass"}
        errors = validate_soar_row(row).errors
        assert any("Missing required fields" in e for e in errors)

    def test_row_empty_code(self):
        row = {
            "task_id": "task_001",
            "reasoning": None,
            "code": "   ",
            "correct_train_input": [True],
            "correct_test_input": [True],
            "predicted_train_output": [[[1]]],
            "predicted_test_output": [[[1]]],
            "model": "test_model",
            "is_transductive": False,
        }
        errors = validate_soar_row(row).errors
        assert any("code must be a non-empty string" in e for e in errors)

    """Test the ValidationResult dataclass and its methods."""

    def test_is_valid_all_pass(self):
        """Test is_valid returns True when all validations pass."""
        result = ValidationResult(
            total_rows=10,
            schema_valid=True,
            schema_error=None,
            business_logic_valid=True,
            business_logic_issues=[],
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
        )
        assert result.is_valid() is False


class TestValidateSoarDataframe:
    """Test the main validation function."""

    def test_validate_valid_data(self):
        """Test validation of valid data."""
        df = create_valid_sample_data()
        format_result = validate_soar_dataframe(df)
        correctness_result = validate_soar_dataframe_correctness(
            df, correctness_samples=0
        )
        assert format_result.total_rows == 2
        assert format_result.schema_valid is True
        assert format_result.schema_error is None
        assert format_result.business_logic_valid is True
        assert format_result.business_logic_issues == []
        assert correctness_result.correctness_valid is True
        assert correctness_result.correctness_errors == []
        assert format_result.is_valid() is True
        assert correctness_result.is_valid() is True

    def test_validate_empty_dataframe(self):
        """Test validation of empty DataFrame."""
        df = pd.DataFrame()
        format_result = validate_soar_dataframe(df)
        correctness_result = validate_soar_dataframe_correctness(
            df, correctness_samples=0
        )
        assert format_result.total_rows == 0
        assert format_result.schema_valid is False
        assert (
            format_result.schema_error is not None
            and "Schema validation failed" in format_result.schema_error
        )
        assert (
            format_result.business_logic_valid is True
        )  # No business logic issues for empty
        assert (
            correctness_result.correctness_valid is True
        )  # No correctness issues for empty
        assert format_result.is_valid() is False

    def test_validate_missing_columns(self):
        """Test validation with missing required columns."""
        df = pd.DataFrame(
            {
                "task_id": ["task_001"],
                "code": ["def generate(): pass"],
                # Missing other required columns
            }
        )
        format_result = validate_soar_dataframe(df)
        correctness_result = validate_soar_dataframe_correctness(
            df, correctness_samples=0
        )
        assert format_result.schema_valid is False
        assert (
            format_result.schema_error is not None
            and "Schema validation failed" in format_result.schema_error
        )
        assert format_result.is_valid() is False

    def test_business_logic_empty_lists(self):
        """Test business logic validation catches empty lists."""
        df = pd.DataFrame(
            {
                "task_id": ["task_001"],
                "reasoning": [None],
                "code": ["def generate(): return []"],
                "correct_train_input": [[]],  # Empty list
                "correct_test_input": [[True]],
                "predicted_train_output": [[]],  # Empty list
                "predicted_test_output": [[[]]],
                "model": ["gpt-4o-mini"],
                "is_transductive": [False],
            }
        )
        format_result = validate_soar_dataframe(df)
        correctness_result = validate_soar_dataframe_correctness(
            df, correctness_samples=0
        )
        assert format_result.schema_valid is True  # Schema is valid
        assert (
            format_result.business_logic_valid is False
        )  # But business logic has issues
        assert len(format_result.business_logic_issues) > 0
        assert any(
            "empty list" in issue for issue in format_result.business_logic_issues
        )
        assert format_result.is_valid() is False

    def test_business_logic_empty_code(self):
        """Test business logic validation catches empty code."""
        df = pd.DataFrame(
            {
                "task_id": ["task_001"],
                "reasoning": [None],
                "code": ["   "],  # Whitespace only
                "correct_train_input": [[True]],
                "correct_test_input": [[True]],
                "predicted_train_output": [[[]]],
                "predicted_test_output": [[[]]],
                "model": ["test_model"],
            }
        )
        format_result = validate_soar_dataframe(df)
        assert format_result.business_logic_valid is False
        assert any(
            "code must be a non-empty string" in issue
            for issue in format_result.business_logic_issues
        )

    def test_correctness_sampling(self):
        """Test that correctness sampling works correctly."""
        df = create_valid_sample_data()
        # Test with sample size larger than dataframe
        correctness_result = validate_soar_dataframe_correctness(
            df, correctness_samples=10
        )
        assert correctness_result.sample_size == 2  # Should be limited to df size
        correctness_result = validate_soar_dataframe_correctness(
            df, correctness_samples=1
        )
        assert correctness_result.sample_size == 1
        correctness_result = validate_soar_dataframe_correctness(
            df, correctness_samples=0
        )
        assert correctness_result.sample_size == 0
        assert (
            correctness_result.correctness_valid is True
        )  # Should pass with no samples

    def test_seed_consistency(self):
        """Test that using the same seed produces consistent results."""
        df = create_valid_sample_data()
        correctness_result1 = validate_soar_dataframe_correctness(
            df, correctness_samples=1, seed=42
        )
        correctness_result2 = validate_soar_dataframe_correctness(
            df, correctness_samples=1, seed=42
        )
        assert correctness_result1.sample_size == correctness_result2.sample_size
        assert (
            correctness_result1.correctness_valid
            == correctness_result2.correctness_valid
        )

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
        df = pd.DataFrame(
            {
                "task_id": ["fake_task_123"],  # Non-existent task ID
                "reasoning": ["Adds 1 to all values"],
                "code": [program_code],
                "correct_train_input": [[True]],  # Assume the program works on training
                "correct_test_input": [[True]],  # Assume the program works on test
                "predicted_train_output": [
                    [expected_output]
                ],  # What we expect for training
                "predicted_test_output": [
                    [expected_output2]
                ],  # What we expect for test
                "model": ["gpt-4o-mini"],
                "is_transductive": [False],
            }
        )
        # Test with actual program execution (should fail due to non-existent task)
        format_result = validate_soar_dataframe(df)
        correctness_result = validate_soar_dataframe_correctness(
            df, correctness_samples=1
        )
        assert format_result.schema_valid is True
        assert format_result.business_logic_valid is True
        assert correctness_result.correctness_valid is False
        assert len(correctness_result.correctness_errors) > 0
        assert correctness_result.sample_size == 1
        error_message = " ".join(correctness_result.correctness_errors)
        assert (
            "execution failed" in error_message.lower()
            or "row 0" in error_message.lower()
        )
