"""
Integration tests for metrics calculation with voting utilities.
These tests ensure voting metrics are calculated correctly for both single and multi-test ARC tasks.
"""

import pytest
from llm_python.utils.metrics_utils import calculate_task_metrics
from llm_python.utils.voting_utils import compute_weighted_majority_voting, compute_train_majority_voting


class TestMetricsVotingIntegration:
    """Test voting metrics calculation with realistic ARC task data"""
    
    def create_task_data(self, test_outputs):
        """Helper to create ARC task data structure"""
        return {
            "task_data": {
                "test": [{"output": output} for output in test_outputs]
            }
        }
    
    def create_attempt(self, test_predicted, train_accuracy=1.0, train_results=None):
        """Helper to create attempt data structure"""
        if train_results is None:
            train_results = [{"correct": True}]  # Default: perfect train
            
        return {
            "test_predicted": test_predicted,
            "train_accuracy": train_accuracy,
            "train_results": train_results,
            "all_test_correct": True,  # Assume correct for oracle test
            "code_ran": True,
            "test_exec_error": False,
            "test_exec_timeout": False,
            "train_exec_errors": 0,
            "train_exec_timeouts": 0,
            "is_transductive": False,
            "outputs_valid": True  # Required for voting logic
        }
    
    def test_single_test_case_voting_success(self):
        """Test voting works correctly for single test case (most common ARC scenario)"""
        # Single test case with 2x2 grid
        test_output = [[1, 2], [3, 4]]
        task_data = self.create_task_data([test_output])
        
        # Multiple attempts with same correct prediction - test_predicted should be list of grids
        attempts = [
            self.create_attempt([test_output]),  # Correct prediction (wrapped in list)
            self.create_attempt([test_output]),  # Same correct prediction (wrapped in list)
            self.create_attempt([[[0, 0], [0, 0]]]),  # Wrong prediction (wrapped in list)
        ]
        
        results = [{
            "task_data": task_data["task_data"],
            "attempt_details": attempts
        }]
        
        metrics = calculate_task_metrics(results)
        
        # All voting metrics should be 100% (1/1 task)
        assert metrics["weighted_pass2"] == 1, "Weighted voting should be 100%"
        assert metrics["train_majority_pass2"] == 1, "Train majority voting should be 100%"
        assert metrics["all_test_correct"] == 1, "Oracle should be 100%"
    
    def test_single_test_case_voting_failure(self):
        """Test voting correctly identifies failures for single test case"""
        # Single test case with 2x2 grid
        test_output = [[1, 2], [3, 4]]
        task_data = self.create_task_data([test_output])
        
        # All attempts have wrong predictions
        attempts = [
            self.create_attempt([[[0, 0], [0, 0]]]),  # Wrong prediction (wrapped in list)
            self.create_attempt([[[5, 5], [5, 5]]]),  # Different wrong prediction (wrapped in list)
        ]
        
        # Update to mark as incorrect
        for att in attempts:
            att["all_test_correct"] = False
        
        results = [{
            "task_data": task_data["task_data"],
            "attempt_details": attempts
        }]
        
        metrics = calculate_task_metrics(results)
        
        # All voting metrics should be 0% (0/1 tasks)
        assert metrics["weighted_pass2"] == 0, "Weighted voting should be 0%"
        assert metrics["train_majority_pass2"] == 0, "Train majority voting should be 0%"
        assert metrics["all_test_correct"] == 0, "Oracle should be 0%"
    
    def test_multi_test_case_voting_success(self):
        """Test voting works correctly for multiple test cases"""
        # Two test cases
        test_outputs = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        task_data = self.create_task_data(test_outputs)
        
        # Attempts with correct predictions (as list)
        correct_prediction = test_outputs
        attempts = [
            self.create_attempt(correct_prediction),
            self.create_attempt(correct_prediction),
        ]
        
        results = [{
            "task_data": task_data["task_data"],
            "attempt_details": attempts
        }]
        
        metrics = calculate_task_metrics(results)
        
        # All voting metrics should be 100%
        assert metrics["weighted_pass2"] == 1, "Weighted voting should be 100%"
        assert metrics["train_majority_pass2"] == 1, "Train majority voting should be 100%"
        assert metrics["all_test_correct"] == 1, "Oracle should be 100%"
    
    def test_voting_with_different_train_accuracies(self):
        """Test weighted voting considers train accuracy correctly"""
        test_output = [[1, 2], [3, 4]]
        task_data = self.create_task_data([test_output])
        
        # High train accuracy attempt with correct prediction should win
        attempts = [
            self.create_attempt([[[0, 0], [0, 0]]], train_accuracy=0.5),  # Wrong, low train (wrapped in list)
            self.create_attempt([test_output], train_accuracy=1.0),  # Correct, high train (wrapped in list)
        ]
        
        results = [{
            "task_data": task_data["task_data"],
            "attempt_details": attempts
        }]
        
        metrics = calculate_task_metrics(results)
        
        # Weighted voting should pick the high-accuracy correct prediction
        assert metrics["weighted_pass2"] == 1, "Weighted voting should favor high train accuracy"
    
    def test_voting_utilities_directly(self):
        """Test voting utilities return expected formats"""
        test_output = [[1, 2], [3, 4]]
        
        attempts = [
            self.create_attempt([test_output], train_accuracy=1.0),  # Wrapped in list
            self.create_attempt([[[0, 0], [0, 0]]], train_accuracy=0.5),  # Wrapped in list
        ]
        
        # Test weighted voting
        weighted_results = compute_weighted_majority_voting(attempts, top_k=2)
        assert len(weighted_results) <= 2, "Should return at most top_k results"
        assert [test_output] in weighted_results, "Correct prediction list should be in results"
        
        # Test train majority voting
        train_results = compute_train_majority_voting(attempts, top_k=2)
        assert len(train_results) <= 2, "Should return at most top_k results"
        assert [test_output] in train_results, "Correct prediction list should be in results"
    
    def test_edge_cases(self):
        """Test edge cases that previously caused bugs"""
        
        # Empty attempts
        results = [{"task_data": {"test": [{"output": [[1, 2]]}]}, "attempt_details": []}]
        metrics = calculate_task_metrics(results)
        assert metrics["weighted_pass2"] == 0
        assert metrics["train_majority_pass2"] == 0
        
        # None predictions
        attempts = [self.create_attempt(None)]
        results = [{"task_data": {"test": [{"output": [[1, 2]]}]}, "attempt_details": attempts}]
        metrics = calculate_task_metrics(results)
        assert metrics["weighted_pass2"] == 0
        assert metrics["train_majority_pass2"] == 0
        
        # All transductive attempts (should be filtered out)
        attempts = [self.create_attempt([[[1, 2]]])]  # Wrapped in list
        attempts[0]["is_transductive"] = True
        results = [{"task_data": {"test": [{"output": [[1, 2]]}]}, "attempt_details": attempts}]
        metrics = calculate_task_metrics(results)
        # Should skip task entirely due to no non-transductive attempts
        assert metrics["total"] == 1  # Task still counted
        
    def test_bug_regression_first_function(self):
        """Specific test for the _first() function bug that was corrupting predictions"""
        # This test specifically catches the bug where _first() was taking the first row 
        # instead of the full grid for single test cases
        
        test_output = [[1, 2], [3, 4]]  # 2x2 grid
        task_data = self.create_task_data([test_output])
        
        # Create attempt with correct prediction
        attempts = [self.create_attempt([test_output])]  # Wrapped in list
        
        results = [{
            "task_data": task_data["task_data"],
            "attempt_details": attempts
        }]
        
        metrics = calculate_task_metrics(results)
        
        # If the _first() bug exists, this would fail because it would compare
        # [1, 2] (first row) vs [[1, 2], [3, 4]] (full grid)
        assert metrics["weighted_pass2"] == 1, "Should handle 2D grids correctly without taking first row"
        assert metrics["train_majority_pass2"] == 1, "Should handle 2D grids correctly without taking first row"
    
    def test_realistic_arc_task_structure(self):
        """Test voting with realistic ARC task structure: multiple train examples, single test case"""
        # Typical ARC task: single test case with 2x2 output
        test_output = [[1, 2], [3, 4]]
        task_data = self.create_task_data([test_output])
        
        # Create attempts with different training performance on multiple examples
        attempts = [
            # Attempt 1: Perfect on all 4 training examples
            self.create_attempt(
                [test_output], 
                train_accuracy=1.0,
                train_results=[
                    {"correct": True}, {"correct": True}, 
                    {"correct": True}, {"correct": True}
                ]
            ),
            # Attempt 2: Good but not perfect (3/4 correct) 
            self.create_attempt(
                [[[0, 0], [0, 0]]], 
                train_accuracy=0.75,
                train_results=[
                    {"correct": True}, {"correct": True}, 
                    {"correct": True}, {"correct": False}
                ]
            ),
            # Attempt 3: Poor performance (1/4 correct)
            self.create_attempt(
                [[[9, 9], [9, 9]]], 
                train_accuracy=0.25,
                train_results=[
                    {"correct": True}, {"correct": False}, 
                    {"correct": False}, {"correct": False}
                ]
            ),
        ]
        
        results = [{
            "task_data": task_data["task_data"],
            "attempt_details": attempts
        }]
        
        metrics = calculate_task_metrics(results)
        
        # The high-accuracy attempt with correct prediction should win
        assert metrics["weighted_pass2"] == 1, "Voting should favor high train accuracy with correct prediction"
        assert metrics["train_majority_pass2"] == 1, "Train majority voting should also succeed"
        assert metrics["all_test_correct"] == 1, "Oracle should detect the correct attempt"
        
    def test_multiple_training_examples_edge_cases(self):
        """Test edge cases with multiple training examples"""
        test_output = [[1, 2], [3, 4]]
        task_data = self.create_task_data([test_output])
        
        # Edge case: All attempts have same train accuracy but different train result patterns
        attempts = [
            # Same accuracy (0.6) but different patterns
            self.create_attempt(
                [test_output], 
                train_accuracy=0.6,
                train_results=[
                    {"correct": True}, {"correct": True}, {"correct": True}, 
                    {"correct": False}, {"correct": False}  # 3/5 correct
                ]
            ),
            self.create_attempt(
                [[[0, 0], [0, 0]]], 
                train_accuracy=0.6,
                train_results=[
                    {"correct": False}, {"correct": True}, {"correct": True}, 
                    {"correct": True}, {"correct": False}  # 3/5 correct
                ]
            ),
        ]
        
        results = [{
            "task_data": task_data["task_data"],
            "attempt_details": attempts
        }]
        
        metrics = calculate_task_metrics(results)
        
        # With same train accuracy, should still pick the correct prediction
        assert metrics["weighted_pass2"] == 1, "Should pick correct prediction even with same train accuracy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])