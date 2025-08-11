"""
Simple test for the bigquery converter functions.
"""

import pandas as pd
from llm_python.datasets.bigquery_converter import (
    convert_bq_nested_structure,
    extract_boolean_values,
    validate_soar_data,
    convert_bigquery_to_soar
)


def test_convert_bq_nested_structure():
    """Test the BigQuery nested structure conversion."""
    # Test simple list
    simple_list = [1, 2, 3]
    assert convert_bq_nested_structure(simple_list) == [1, 2, 3]
    
    # Test BigQuery format
    bq_format = {
        "list": [
            {"element": [
                {"element": [{"element": 1}, {"element": 2}]},
                {"element": [{"element": 3}, {"element": 4}]}
            ]},
            {"element": [
                {"element": [{"element": 5}, {"element": 6}]},
                {"element": [{"element": 7}, {"element": 8}]}
            ]}
        ]
    }
    
    result = convert_bq_nested_structure(bq_format)
    expected = [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
    ]
    assert result == expected, f"Expected {expected}, got {result}"
    
    print("✓ convert_bq_nested_structure tests passed")


def test_extract_boolean_values():
    """Test boolean value extraction."""
    # Test simple boolean list
    simple_bools = [True, False, True]
    assert extract_boolean_values(simple_bools) == [True, False, True]
    
    # Test BigQuery format
    bq_bools = [
        {"element": True},
        {"element": False},
        {"element": True}
    ]
    assert extract_boolean_values(bq_bools) == [True, False, True]
    
    print("✓ extract_boolean_values tests passed")


def test_validate_soar_data():
    """Test SOAR data validation."""
    # Valid data
    valid_data = {
        'task_id': 'test_task',
        'code': 'def generate(input): return input',
        'model': 'test_model',
        'predicted_train_output': [[[1, 2], [3, 4]]],
        'predicted_test_output': [[[5, 6], [7, 8]]],
        'correct_train_input': [True, False],
        'correct_test_input': [True]
    }
    
    is_valid, msg = validate_soar_data(valid_data)
    assert is_valid, f"Valid data should pass validation: {msg}"
    
    # Invalid data - missing field
    invalid_data = valid_data.copy()
    del invalid_data['task_id']
    
    is_valid, msg = validate_soar_data(invalid_data)
    assert not is_valid, "Data missing task_id should fail validation"
    assert "Missing field: task_id" in msg
    
    print("✓ validate_soar_data tests passed")


def test_convert_bigquery_to_soar():
    """Test full BigQuery to SOAR conversion."""
    # Create mock BigQuery data
    bq_data = pd.DataFrame([
        {
            'task_id': 'test_task_1',
            'code': 'def generate(input): return input',
            'model': 'test_model',
            'predicted_train_output': {
                "list": [
                    {"element": {"list": [
                        {"element": {"list": [{"element": 1}, {"element": 2}]}},
                        {"element": {"list": [{"element": 3}, {"element": 4}]}}
                    ]}}
                ]
            },
            'predicted_test_output': {
                "list": [
                    {"element": {"list": [
                        {"element": {"list": [{"element": 5}, {"element": 6}]}}
                    ]}}
                ]
            },
            'correct_train_input': {"list": [{"element": True}, {"element": False}]},
            'correct_test_input': {"list": [{"element": True}]}
        }
    ])
    
    result_df = convert_bigquery_to_soar(bq_data, show_progress=False)
    
    assert len(result_df) == 1, f"Expected 1 row, got {len(result_df)}"
    assert result_df.iloc[0]['task_id'] == 'test_task_1'
    assert result_df.iloc[0]['predicted_train_output'] == [[[1, 2], [3, 4]]]
    assert result_df.iloc[0]['predicted_test_output'] == [[[5, 6]]]
    assert result_df.iloc[0]['correct_train_input'] == [True, False]
    assert result_df.iloc[0]['correct_test_input'] == [True]
    
    print("✓ convert_bigquery_to_soar tests passed")


if __name__ == "__main__":
    test_convert_bq_nested_structure()
    test_extract_boolean_values()
    test_validate_soar_data()
    test_convert_bigquery_to_soar()
    print("\n✅ All tests passed!")
