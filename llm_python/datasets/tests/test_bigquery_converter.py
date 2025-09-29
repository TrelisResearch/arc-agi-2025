"""
Simple test for the bigquery converter functions.
"""

import pandas as pd
from llm_python.datasets.bigquery_converter import (
    _convert_bq_nested_structure,
    _extract_boolean_values,
    convert_bigquery_to_soar,
)


def test_convert_bq_nested_structure():
    """Test the BigQuery nested structure conversion."""
    # Test simple list
    simple_list = [1, 2, 3]
    assert _convert_bq_nested_structure(simple_list) == [1, 2, 3]

    # Test BigQuery format
    bq_format = {
        "list": [
            {
                "element": [
                    {"element": [{"element": 1}, {"element": 2}]},
                    {"element": [{"element": 3}, {"element": 4}]},
                ]
            },
            {
                "element": [
                    {"element": [{"element": 5}, {"element": 6}]},
                    {"element": [{"element": 7}, {"element": 8}]},
                ]
            },
        ]
    }

    result = _convert_bq_nested_structure(bq_format)
    expected = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    assert result == expected


def test_extract_boolean_values():
    """Test boolean value extraction."""
    # Test simple boolean list
    simple_bools = [True, False, True]
    assert _extract_boolean_values(simple_bools) == [True, False, True]

    # Test BigQuery format
    bq_bools = [{"element": True}, {"element": False}, {"element": True}]
    assert _extract_boolean_values(bq_bools) == [True, False, True]


def test_convert_bigquery_to_soar():
    """Test full BigQuery to SOAR conversion."""
    # Create mock BigQuery data
    bq_data = pd.DataFrame(
        [
            {
                "row_id": "abcdef",
                "task_id": "test_task_1",
                "code": "def generate(input): return input",
                "model": "test_model",
                "predicted_train_output": {
                    "list": [
                        {
                            "element": {
                                "list": [
                                    {
                                        "element": {
                                            "list": [{"element": 1}, {"element": 2}]
                                        }
                                    },
                                    {
                                        "element": {
                                            "list": [{"element": 3}, {"element": 4}]
                                        }
                                    },
                                ]
                            }
                        }
                    ]
                },
                "predicted_test_output": {
                    "list": [
                        {
                            "element": {
                                "list": [
                                    {
                                        "element": {
                                            "list": [{"element": 5}, {"element": 6}]
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                },
                "correct_train_input": {
                    "list": [{"element": True}, {"element": False}]
                },
                "correct_test_input": {"list": [{"element": True}]},
                "reasoning": "This is a test reasoning",
                "is_transductive": True
            }
        ]
    )

    result_df = convert_bigquery_to_soar(bq_data, show_progress=False)

    assert len(result_df) == 1
    assert result_df.iloc[0]["row_id"] == "abcdef"
    assert result_df.iloc[0]["task_id"] == "test_task_1"
    assert result_df.iloc[0]["predicted_train_output"] == [[[1, 2], [3, 4]]]
    assert result_df.iloc[0]["predicted_test_output"] == [[[5, 6]]]
    assert result_df.iloc[0]["correct_train_input"] == [True, False]
    assert result_df.iloc[0]["correct_test_input"] == [True]
    assert result_df.iloc[0]["reasoning"] == "This is a test reasoning"
    assert result_df.iloc[0]["is_transductive"]

