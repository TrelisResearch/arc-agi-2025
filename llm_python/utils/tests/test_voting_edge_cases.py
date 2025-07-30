"""
Tests for voting utilities edge cases and format handling.
"""

import pytest
from llm_python.utils.voting_utils import (
    serialize_prediction_for_voting, 
    deserialize_prediction_from_voting,
    compute_weighted_majority_voting,
    compute_train_majority_voting
)


class TestVotingEdgeCases:
    """Test voting utilities handle various prediction formats correctly"""
    
    def test_serialization_deserialization(self):
        """Test prediction serialization/deserialization preserves data"""
        
        # 2D grid (most common ARC case)
        grid_2d = [[1, 2], [3, 4]]
        serialized = serialize_prediction_for_voting(grid_2d)
        deserialized = deserialize_prediction_from_voting(serialized)
        assert deserialized == grid_2d, "2D grid should round-trip correctly"
        
        # Tuple of grids (multi-test case)
        multi_grids = ([[1, 2]], [[3, 4]])
        serialized = serialize_prediction_for_voting(multi_grids)
        deserialized = deserialize_prediction_from_voting(serialized)
        assert deserialized == list(multi_grids), "Tuple should convert to list but preserve content"
        
        # None case
        serialized = serialize_prediction_for_voting(None)
        deserialized = deserialize_prediction_from_voting(serialized)
        assert deserialized is None, "None should round-trip correctly"
        
        # Empty list
        empty = []
        serialized = serialize_prediction_for_voting(empty)
        deserialized = deserialize_prediction_from_voting(serialized)
        assert deserialized == empty, "Empty list should round-trip correctly"
    
    def test_weighted_voting_accuracy_preference(self):
        """Test weighted voting prefers higher train accuracy"""
        
        # Create attempts with different train accuracies
        attempts = [
            {
                "test_predicted": [[0, 0]],  # Wrong answer
                "train_accuracy": 0.9,
                "train_results": [{"correct": True}, {"correct": True}, {"correct": False}]
            },
            {
                "test_predicted": [[1, 2]],  # Right answer  
                "train_accuracy": 1.0,
                "train_results": [{"correct": True}, {"correct": True}]
            },
            {
                "test_predicted": [[3, 3]],  # Another wrong answer
                "train_accuracy": 0.5,
                "train_results": [{"correct": True}, {"correct": False}]
            }
        ]
        
        results = compute_weighted_majority_voting(attempts, top_k=3)
        
        # Highest train accuracy should be first
        assert results[0] == [[1, 2]], "Highest train accuracy prediction should be ranked first"
        assert len(results) == 3, "Should return all unique predictions"
    
    def test_train_majority_voting_best_group(self):
        """Test train majority voting filters to best train performance group"""
        
        attempts = [
            {
                "test_predicted": [[1, 1]],  # 2/2 train correct
                "train_accuracy": 1.0,
                "train_results": [{"correct": True}, {"correct": True}]
            },
            {
                "test_predicted": [[2, 2]],  # 2/2 train correct (same group)
                "train_accuracy": 1.0,
                "train_results": [{"correct": True}, {"correct": True}]
            },
            {
                "test_predicted": [[3, 3]],  # 1/2 train correct (worse group)
                "train_accuracy": 0.5,
                "train_results": [{"correct": True}, {"correct": False}]
            }
        ]
        
        results = compute_train_majority_voting(attempts, top_k=3)
        
        # Should only include predictions from best group (2/2 correct)
        assert [[1, 1]] in results, "Best group prediction should be included"
        assert [[2, 2]] in results, "Best group prediction should be included"
        assert [[3, 3]] not in results, "Worse group prediction should be excluded"
        assert len(results) == 2, "Should only include best group predictions"
    
    def test_voting_with_identical_predictions(self):
        """Test voting handles identical predictions correctly"""
        
        # Multiple attempts with same prediction
        attempts = [
            {
                "test_predicted": [[1, 2], [3, 4]],
                "train_accuracy": 1.0,
                "train_results": [{"correct": True}]
            },
            {
                "test_predicted": [[1, 2], [3, 4]],  # Same prediction
                "train_accuracy": 0.8,
                "train_results": [{"correct": True}]
            },
            {
                "test_predicted": [[5, 6], [7, 8]],  # Different prediction
                "train_accuracy": 0.6,
                "train_results": [{"correct": True}]
            }
        ]
        
        results = compute_weighted_majority_voting(attempts, top_k=2)
        
        # Identical predictions should be combined with higher total weight
        assert [[1, 2], [3, 4]] in results, "Combined identical predictions should win"
        assert results[0] == [[1, 2], [3, 4]], "Combined weight should make it first"
        assert len(results) == 2, "Should return unique predictions only"
    
    def test_empty_attempts_handling(self):
        """Test voting handles empty or invalid attempts gracefully"""
        
        # Empty attempts list
        results = compute_weighted_majority_voting([], top_k=2)
        assert results == [], "Empty attempts should return empty results"
        
        results = compute_train_majority_voting([], top_k=2)
        assert results == [], "Empty attempts should return empty results"
        
        # Attempts with None predictions
        attempts = [
            {
                "test_predicted": None,
                "train_accuracy": 1.0,
                "train_results": [{"correct": True}]
            }
        ]
        
        results = compute_weighted_majority_voting(attempts, top_k=2)
        assert results == [], "None predictions should be filtered out"
    
    def test_single_vs_multi_test_format_consistency(self):
        """Test that single and multi-test predictions are handled consistently"""
        
        # Single test case prediction
        single_attempts = [{
            "test_predicted": [[1, 2], [3, 4]],  # Raw 2D grid
            "train_accuracy": 1.0,
            "train_results": [{"correct": True}]
        }]
        
        # Multi test case prediction  
        multi_attempts = [{
            "test_predicted": ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),  # Tuple of grids
            "train_accuracy": 1.0, 
            "train_results": [{"correct": True}]
        }]
        
        single_results = compute_weighted_majority_voting(single_attempts, top_k=1)
        multi_results = compute_weighted_majority_voting(multi_attempts, top_k=1)
        
        # Both should return valid results
        assert len(single_results) == 1, "Single test should return result"
        assert len(multi_results) == 1, "Multi test should return result" 
        assert single_results[0] == [[1, 2], [3, 4]], "Single test should preserve grid format"
        assert multi_results[0] == [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], "Multi test should preserve content (tuple->list conversion is expected)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])