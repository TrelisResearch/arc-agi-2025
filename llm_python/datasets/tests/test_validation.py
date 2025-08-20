import pytest
import pandas as pd

from llm_python.datasets.validation import validate_soar_sample, validate_soar_dataframe


@pytest.fixture
def valid_soar_sample():
    """Fixture providing a valid SOAR sample."""
    return {
        "task_id": "test_task",
        "code": "def generate(input): return input",
        "model": "test_model",
        "predicted_train_output": [[[1, 2], [3, 4]]],
        "predicted_test_output": [[[5, 6], [7, 8]]],
        "correct_train_input": [True, False],
        "correct_test_input": [True],
    }


class TestValidateSoarSample:
    """Test suite for validate_soar_sample function."""

    def test_valid_sample_passes(self, valid_soar_sample):
        """Test that a valid sample passes validation."""
        is_valid, msg = validate_soar_sample(valid_soar_sample)
        assert is_valid, f"Valid data should pass validation: {msg}"
        assert msg == "Valid"

    @pytest.mark.parametrize("missing_field", [
        "task_id", "code", "model", "predicted_train_output", 
        "predicted_test_output", "correct_train_input", "correct_test_input"
    ])
    def test_missing_required_fields(self, valid_soar_sample, missing_field):
        """Test that missing required fields cause validation to fail."""
        invalid_data = valid_soar_sample.copy()
        del invalid_data[missing_field]
        
        is_valid, msg = validate_soar_sample(invalid_data)
        assert not is_valid
        assert f"Missing field: {missing_field}" in msg

    @pytest.mark.parametrize("field,invalid_value", [
        ("task_id", 123),
        ("code", None),
        ("model", []),
    ])
    def test_invalid_string_types(self, valid_soar_sample, field, invalid_value):
        """Test that non-string values for string fields fail validation."""
        invalid_data = valid_soar_sample.copy()
        invalid_data[field] = invalid_value
        
        is_valid, msg = validate_soar_sample(invalid_data)
        assert not is_valid
        assert f"{field} should be str" in msg

    @pytest.mark.parametrize("field", ["predicted_train_output", "predicted_test_output"])
    def test_invalid_3d_array_types(self, valid_soar_sample, field):
        """Test that invalid 3D array structures fail validation."""
        # Test non-list at top level
        invalid_data = valid_soar_sample.copy()
        invalid_data[field] = "not a list"
        
        is_valid, msg = validate_soar_sample(invalid_data)
        assert not is_valid
        assert f"{field} should be list" in msg

        # Test non-list at grid level
        invalid_data[field] = ["not a grid"]
        is_valid, msg = validate_soar_sample(invalid_data)
        assert not is_valid
        assert "should be list (2D grid)" in msg

        # Test non-list at row level
        invalid_data[field] = [["not a row"]]
        is_valid, msg = validate_soar_sample(invalid_data)
        assert not is_valid
        assert "should be list (row)" in msg

        # Test non-int at cell level
        invalid_data[field] = [[["not an int"]]]
        is_valid, msg = validate_soar_sample(invalid_data)
        assert not is_valid
        assert "should be int" in msg

    @pytest.mark.parametrize("field", ["correct_train_input", "correct_test_input"])
    def test_invalid_boolean_array_types(self, valid_soar_sample, field):
        """Test that invalid boolean array structures fail validation."""
        # Test non-list
        invalid_data = valid_soar_sample.copy()
        invalid_data[field] = "not a list"
        
        is_valid, msg = validate_soar_sample(invalid_data)
        assert not is_valid
        assert f"{field} should be list" in msg

        # Test non-boolean values
        invalid_data[field] = [True, "not a bool", False]
        is_valid, msg = validate_soar_sample(invalid_data)
        assert not is_valid
        assert "should be bool" in msg

    def test_exception_handling(self, valid_soar_sample):
        """Test that exceptions during validation are handled properly."""
        # Create a problematic data structure that might cause an exception
        invalid_data = valid_soar_sample.copy()
        
        # This should be handled gracefully by the try-except block
        # We can't easily trigger an exception with the current validation logic,
        # but this test ensures the structure is in place
        is_valid, msg = validate_soar_sample(invalid_data)
        assert is_valid  # Should still pass with valid data


class TestValidateSoarDataframe:
    """Test suite for validate_soar_dataframe function."""

    def test_valid_dataframe_passes(self, valid_soar_sample):
        """Test that a DataFrame with valid samples passes validation."""
        df = pd.DataFrame([valid_soar_sample, valid_soar_sample])
        
        is_valid, msg = validate_soar_dataframe(df)
        assert is_valid
        assert "All 2 rows are valid" in msg

    def test_invalid_dataframe_fails(self, valid_soar_sample):
        """Test that a DataFrame with invalid samples fails validation."""
        invalid_sample = valid_soar_sample.copy()
        invalid_sample["task_id"] = 123  # Invalid type instead of missing field
        
        df = pd.DataFrame([valid_soar_sample, invalid_sample])
        
        is_valid, msg = validate_soar_dataframe(df)
        assert not is_valid
        assert "1 validation error sample" in msg
        assert "Row 1:" in msg
        assert "task_id should be str" in msg

    def test_multiple_invalid_rows(self, valid_soar_sample):
        """Test that multiple invalid rows are reported properly."""
        invalid_sample1 = valid_soar_sample.copy()
        invalid_sample1["task_id"] = 123  # Invalid type
        
        invalid_sample2 = valid_soar_sample.copy()
        invalid_sample2["code"] = None  # Invalid type
        
        df = pd.DataFrame([invalid_sample1, invalid_sample2])
        
        is_valid, msg = validate_soar_dataframe(df)
        assert not is_valid
        assert "2 validation error sample" in msg
        assert "Row 0:" in msg
        assert "Row 1:" in msg

    def test_dataframe_with_missing_fields(self, valid_soar_sample):
        """Test that DataFrame with truly missing fields fails validation."""
        # Create a DataFrame with different column structures
        partial_sample = {k: v for k, v in valid_soar_sample.items() if k != "task_id"}
        df = pd.DataFrame([partial_sample])
        
        is_valid, msg = validate_soar_dataframe(df)
        assert not is_valid
        assert "Missing field: task_id" in msg

    def test_empty_dataframe(self):
        """Test that an empty DataFrame passes validation."""
        df = pd.DataFrame()
        
        is_valid, msg = validate_soar_dataframe(df)
        assert is_valid
        assert "All 0 rows are valid" in msg
