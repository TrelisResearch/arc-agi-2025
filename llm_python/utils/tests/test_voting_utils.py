from ..voting_utils import (
    compute_train_majority_voting,
    compute_weighted_majority_voting,
    deserialize_prediction_from_voting,
    serialize_prediction_for_voting,
)


class TestVotingUtils:
    """Tests for voting utility functions"""

    def test_serialize_deserialize_prediction(self):
        """Test prediction serialization and deserialization"""
        # Test with simple list
        test_pred = [[1, 2], [3, 4]]
        serialized = serialize_prediction_for_voting(test_pred)
        deserialized = deserialize_prediction_from_voting(serialized)
        assert deserialized == test_pred

        # Test with None
        serialized_none = serialize_prediction_for_voting(None)
        deserialized_none = deserialize_prediction_from_voting(serialized_none)
        assert deserialized_none is None

        # Test with different values
        test_pred2 = [[0, 0, 5], [0, 7, 3], [8, 3, 3]]
        serialized2 = serialize_prediction_for_voting(test_pred2)
        deserialized2 = deserialize_prediction_from_voting(serialized2)
        assert deserialized2 == test_pred2

    def test_compute_weighted_majority_voting(self):
        """Test weighted majority voting"""
        # Create test attempts with different training accuracies
        attempts = [
            {"test_predicted": [[[1, 2], [3, 4]]], "train_accuracy": 0.5},
            {
                "test_predicted": [
                    [[1, 2], [3, 4]]
                ],  # Same prediction, higher accuracy
                "train_accuracy": 1.0,
            },
            {
                "test_predicted": [
                    [[5, 6], [7, 8]]
                ],  # Different prediction, lower accuracy
                "train_accuracy": 0.0,
            },
        ]

        result = compute_weighted_majority_voting(attempts, top_k=2)

        # Should return top 2 predictions by weight
        # [[[1, 2], [3, 4]]] should have highest weight (1.0 + 1000*1.0 = 1001.0)
        # [[[5, 6], [7, 8]]] should have lowest weight (1.0 + 1000*0.0 = 1.0)
        assert len(result) == 2
        assert [[[1, 2], [3, 4]]] in result
        assert [[[5, 6], [7, 8]]] in result

    def test_compute_train_majority_voting(self):
        """Test train-majority voting"""
        # Create test attempts with different training results
        attempts = [
            {
                "test_predicted": [[[1, 2], [3, 4]]],
                "train_results": [{"correct": True}, {"correct": False}],  # 1/2 correct
            },
            {
                "test_predicted": [
                    [[1, 2], [3, 4]]
                ],  # Same prediction, better training
                "train_results": [{"correct": True}, {"correct": True}],  # 2/2 correct
            },
            {
                "test_predicted": [
                    [[5, 6], [7, 8]]
                ],  # Different prediction, same best training
                "train_results": [{"correct": True}, {"correct": True}],  # 2/2 correct
            },
        ]

        result = compute_train_majority_voting(attempts, top_k=2)

        # Should return predictions from attempts with best training performance (2/2 correct)
        assert len(result) == 2
        assert [[[1, 2], [3, 4]]] in result
        assert [[[5, 6], [7, 8]]] in result

    def test_empty_attempts(self):
        """Test voting with empty attempts list"""
        empty_attempts = []

        weighted_result = compute_weighted_majority_voting(empty_attempts)
        assert weighted_result == []

        train_result = compute_train_majority_voting(empty_attempts)
        assert train_result == []

    def test_single_attempt(self):
        """Test voting with single attempt"""
        single_attempt = [
            {
                "test_predicted": [[[1, 2], [3, 4]]],
                "train_accuracy": 0.8,
                "train_results": [{"correct": True}, {"correct": False}],
            }
        ]

        weighted_result = compute_weighted_majority_voting(single_attempt, top_k=1)
        assert weighted_result == [[[[1, 2], [3, 4]]]]

        train_result = compute_train_majority_voting(single_attempt, top_k=1)
        assert train_result == [[[[1, 2], [3, 4]]]]
