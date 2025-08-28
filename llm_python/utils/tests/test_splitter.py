"""
Test suite for the splitter functionality in SOAR task runner.
"""

import pytest
import random
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.prompt_utils import create_arc_prompt, _format_grid_for_prompt, _get_grid_shape_string


class MockPromptLoader:
    """Mock prompt loader for testing"""
    
    def get_system_message(self, version: str) -> str:
        return f"System message for {version}"
    
    def get_initial_turn_prompt(self, version: str) -> str:
        return "Task:\n{task_content}"


def create_sample_task_data(num_train: int = 3, num_test: int = 1) -> Dict:
    """Create sample task data for testing"""
    task_data = {
        'train': [],
        'test': []
    }
    
    # Create training examples with distinct patterns
    for i in range(num_train):
        task_data['train'].append({
            'input': [[i, i+1], [i+2, i+3]],  # 2x2 grid with unique values
            'output': [[i*10, (i+1)*10], [(i+2)*10, (i+3)*10]]  # Scaled by 10
        })
    
    # Create test examples
    for i in range(num_test):
        task_data['test'].append({
            'input': [[100+i, 101+i], [102+i, 103+i]],
            'output': [[200+i, 201+i], [202+i, 203+i]]
        })
    
    return task_data


class TestSplitterFunctionality:
    """Test the splitter feature in create_arc_prompt"""
    
    def test_splitter_disabled_returns_all_examples(self):
        """When splitter is disabled, all training examples should be included"""
        task_data = create_sample_task_data(num_train=5)
        prompt_loader = MockPromptLoader()
        
        system_msg, user_msg = create_arc_prompt(
            task_data, prompt_loader, prompt_version="soar", splitter=False
        )
        
        # Count training examples by looking for "## Input X" where X is a number, not "## Test Input"
        import re
        train_input_pattern = r"## Input \d+ \(grid shape:"
        test_input_pattern = r"## Test Input \d+ \(grid shape:"
        train_output_pattern = r"## Output \d+ \(grid shape:"
        
        train_input_count = len(re.findall(train_input_pattern, user_msg))
        test_input_count = len(re.findall(test_input_pattern, user_msg))
        output_count = len(re.findall(train_output_pattern, user_msg))
        
        # Should have 5 training inputs and 1 test input
        assert train_input_count == 5, f"Expected 5 training inputs, got {train_input_count}"
        assert test_input_count == 1, f"Expected 1 test input, got {test_input_count}"
        assert output_count == 5, f"Expected 5 outputs, got {output_count}"
    
    def test_splitter_enabled_with_single_example(self):
        """With only one training example, splitter should not affect output"""
        task_data = create_sample_task_data(num_train=1)
        prompt_loader = MockPromptLoader()
        
        system_msg, user_msg = create_arc_prompt(
            task_data, prompt_loader, prompt_version="soar", splitter=True
        )
        
        # Should still have 1 training example
        import re
        train_input_count = len(re.findall(r"## Input \d+ \(grid shape:", user_msg))
        test_input_count = len(re.findall(r"## Test Input \d+ \(grid shape:", user_msg))
        output_count = len(re.findall(r"## Output \d+ \(grid shape:", user_msg))
        
        assert train_input_count == 1, f"Expected 1 training input, got {train_input_count}"
        assert test_input_count == 1, f"Expected 1 test input, got {test_input_count}"
        assert output_count == 1, f"Expected 1 output, got {output_count}"
    
    def test_splitter_randomizes_selection(self):
        """Splitter should randomly select different numbers of examples"""
        task_data = create_sample_task_data(num_train=5)
        prompt_loader = MockPromptLoader()
        
        # Run multiple times to check for randomness
        example_counts = []
        for _ in range(20):
            system_msg, user_msg = create_arc_prompt(
                task_data, prompt_loader, prompt_version="soar", splitter=True
            )
            # Count training examples using regex
            import re
            train_count = len(re.findall(r"## Input \d+ \(grid shape:", user_msg))
            example_counts.append(train_count)
        
        # Should have varying counts between 1 and 5
        unique_counts = set(example_counts)
        assert len(unique_counts) > 1, "Splitter should produce varying numbers of examples"
        assert min(example_counts) >= 1, "Should have at least 1 training example"
        assert max(example_counts) <= 5, "Should have at most 5 training examples"
    
    def test_splitter_shuffles_order(self):
        """Splitter should shuffle the order of selected examples"""
        task_data = create_sample_task_data(num_train=3)
        prompt_loader = MockPromptLoader()
        
        # Mock random to always select all examples but shuffle them
        with patch('random.randint', return_value=3):
            # Collect multiple runs to check for different orderings
            orderings = []
            for _ in range(10):
                system_msg, user_msg = create_arc_prompt(
                    task_data, prompt_loader, prompt_version="soar", splitter=True
                )
                
                # Extract the grid values to identify ordering
                # Look for the first value in each input grid (0, 1, 2 for our test data)
                import re
                # Updated pattern to match our actual grid format: [[0 1] [2 3]]
                input_pattern = r"## Input \d+ \(grid shape: \d+ by \d+\):\n\[\[(\d+)"
                matches = re.findall(input_pattern, user_msg)
                # Only look at training inputs (exclude test inputs which start with 100+)
                train_matches = [m for m in matches if int(m) < 100]
                orderings.append(tuple(train_matches))
            
            # Check that we get different orderings
            unique_orderings = set(orderings)
            assert len(unique_orderings) > 1, "Splitter should produce different orderings"
    
    def test_splitter_preserves_test_examples(self):
        """Splitter should not affect test examples"""
        task_data = create_sample_task_data(num_train=3, num_test=2)
        prompt_loader = MockPromptLoader()
        
        # Run with and without splitter
        _, user_msg_no_split = create_arc_prompt(
            task_data, prompt_loader, prompt_version="soar", splitter=False
        )
        _, user_msg_with_split = create_arc_prompt(
            task_data, prompt_loader, prompt_version="soar", splitter=True
        )
        
        # Extract test sections
        test_section_no_split = user_msg_no_split.split("## Test Input")[1:]
        test_section_with_split = user_msg_with_split.split("## Test Input")[1:]
        
        # Both should have same number of test examples
        assert len(test_section_no_split) == 2
        assert len(test_section_with_split) == 2
        
        # Test examples should contain expected values (100-103 for first test)
        assert "[[100" in test_section_with_split[0]
        assert "[[101" in test_section_with_split[1] or "[[101" in test_section_with_split[0]
    
    def test_splitter_with_include_test_outputs(self):
        """Splitter should work correctly when test outputs are included"""
        task_data = create_sample_task_data(num_train=3, num_test=1)
        prompt_loader = MockPromptLoader()
        
        _, user_msg = create_arc_prompt(
            task_data, prompt_loader, 
            prompt_version="soar", 
            splitter=True,
            include_test_outputs=True
        )
        
        # Should have test output when include_test_outputs=True
        assert "## Expected Test Output" in user_msg
        assert "[[200" in user_msg  # Check for test output value
    
    def test_grid_formatting_preserved(self):
        """Ensure grid formatting is preserved with splitter"""
        task_data = create_sample_task_data(num_train=2)
        prompt_loader = MockPromptLoader()
        
        _, user_msg = create_arc_prompt(
            task_data, prompt_loader, prompt_version="soar", splitter=True
        )
        
        # Check that grids are properly formatted (no commas)
        assert "," not in user_msg.split("[[")[1].split("]]")[0]
        
        # Check that shape strings are present
        assert "grid shape:" in user_msg
        assert "2 by 2" in user_msg  # Our test grids are 2x2
    
    def test_deterministic_with_seed(self):
        """Test that results are reproducible with a fixed seed"""
        task_data = create_sample_task_data(num_train=5)
        prompt_loader = MockPromptLoader()
        
        # Set seed and generate prompt
        random.seed(42)
        _, user_msg1 = create_arc_prompt(
            task_data, prompt_loader, prompt_version="soar", splitter=True
        )
        
        # Reset seed and generate again
        random.seed(42)
        _, user_msg2 = create_arc_prompt(
            task_data, prompt_loader, prompt_version="soar", splitter=True
        )
        
        # Should be identical with same seed
        assert user_msg1 == user_msg2
        
        # Different seed should (likely) give different result
        random.seed(123)
        _, user_msg3 = create_arc_prompt(
            task_data, prompt_loader, prompt_version="soar", splitter=True
        )
        
        # Very likely to be different (not guaranteed but highly probable with 5 examples)
        # We'll check the length as a proxy for different selection
        if len(user_msg1) == len(user_msg3):
            # If same length, check content is different (could be different order)
            assert user_msg1 != user_msg3 or len(task_data['train']) == 1
    
    def test_multiple_attempts_produce_different_results(self):
        """Test that multiple calls to create_arc_prompt with splitter produce different results"""
        task_data = create_sample_task_data(num_train=5)
        prompt_loader = MockPromptLoader()
        
        # Generate multiple prompts
        results = []
        for _ in range(10):
            _, user_msg = create_arc_prompt(
                task_data, prompt_loader, prompt_version="soar", splitter=True
            )
            
            # Extract which training examples were selected
            import re
            pattern = r"## Input \d+ \(grid shape: \d+ by \d+\):\n\[\[(\d+)"
            matches = re.findall(pattern, user_msg)
            # Filter to just training examples (test examples have values >= 100)
            train_values = [int(m) for m in matches if int(m) < 100]
            results.append(tuple(sorted(train_values)))
        
        # Should have at least some variety in selections
        unique_results = set(results)
        assert len(unique_results) > 1, f"Expected variety in selections, got only: {unique_results}"
        
        # Should have variety in number of examples selected
        example_counts = [len(result) for result in results]
        unique_counts = set(example_counts)
        assert len(unique_counts) > 1, f"Expected variety in example counts, got only: {unique_counts}"
    
    def test_empty_task_data(self):
        """Test handling of empty or minimal task data"""
        # Empty training data
        task_data = {'train': [], 'test': [{'input': [[1]], 'output': [[2]]}]}
        prompt_loader = MockPromptLoader()
        
        _, user_msg = create_arc_prompt(
            task_data, prompt_loader, prompt_version="soar", splitter=True
        )
        
        # Should handle empty training data gracefully
        assert "## Test Input" in user_msg
        assert "## Input 1" not in user_msg  # No training inputs
    
    def test_large_number_of_examples(self):
        """Test with a large number of training examples"""
        task_data = create_sample_task_data(num_train=20)
        prompt_loader = MockPromptLoader()
        
        # Run multiple times
        example_counts = []
        for _ in range(10):
            _, user_msg = create_arc_prompt(
                task_data, prompt_loader, prompt_version="soar", splitter=True
            )
            import re
            train_count = len(re.findall(r"## Input \d+ \(grid shape:", user_msg))
            example_counts.append(train_count)
        
        # Should vary between 1 and 20
        assert min(example_counts) >= 1
        assert max(example_counts) <= 20
        # With 20 examples, should see good variation
        assert len(set(example_counts)) >= 3, "Should see variation with many examples"


class TestArgparseIntegration:
    """Test the argparse integration for splitter argument"""
    
    def test_splitter_argument_parsing(self):
        """Test that --splitter argument is properly parsed"""
        from run_arc_tasks_soar import main
        import argparse
        
        # Create parser (we'll test just the parser, not the full main)
        parser = argparse.ArgumentParser()
        
        # Add all the arguments (simplified version for testing)
        parser.add_argument("--dataset", default="arc-prize-2025")
        parser.add_argument("--subset", default="training")
        parser.add_argument("--model", default="gpt-4.1-mini")
        parser.add_argument("--limit", type=int)
        parser.add_argument("--base-url", type=str)
        parser.add_argument("--max_workers", type=int, default=1)
        parser.add_argument("--rate_limit_delay", type=float, default=0.0)
        parser.add_argument("--max_attempts", type=int, default=8)
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--max-tokens", type=int)
        parser.add_argument("--temperature", type=float)
        parser.add_argument("--reasoning_effort", type=str, default="low")
        parser.add_argument("--qwen-no-think", action="store_true")
        parser.add_argument("--prompt_version", type=str, default="soar")
        parser.add_argument("--unsafe-executor", action="store_true")
        parser.add_argument("--lora-adapter", type=str)
        parser.add_argument("--no-log-to-db", dest="log_to_db", action="store_false")
        parser.add_argument("--no-transductive-penalty", action="store_true")
        parser.add_argument("--parquet-output-dir", type=str)
        parser.add_argument("--splitter", action="store_true")
        
        # Test without --splitter
        args = parser.parse_args([])
        assert args.splitter == False
        
        # Test with --splitter
        args = parser.parse_args(["--splitter"])
        assert args.splitter == True
        
        # Test with other arguments
        args = parser.parse_args(["--model", "gpt-4", "--splitter", "--debug"])
        assert args.splitter == True
        assert args.model == "gpt-4"
        assert args.debug == True


class TestRunnerIntegration:
    """Test integration with ARCTaskRunnerSimple"""
    
    @patch('run_arc_tasks_soar.ARCAPIClient')
    @patch('run_arc_tasks_soar.ArcTester')
    @patch('run_arc_tasks_soar.PromptLoader')
    @patch('run_arc_tasks_soar.SoarDatasetCollector')
    @patch('run_arc_tasks_soar.CodeTransductionClassifier')
    def test_runner_initialization_with_splitter(self, mock_classifier, mock_collector, 
                                                 mock_prompt_loader, mock_tester, mock_api_client):
        """Test that ARCTaskRunnerSimple properly initializes with splitter"""
        from run_arc_tasks_soar import ARCTaskRunnerSimple
        
        # Create runner with splitter enabled
        runner = ARCTaskRunnerSimple(
            model="test-model",
            splitter=True
        )
        
        # Check that splitter is stored
        assert runner.splitter == True
        
        # Create runner with splitter disabled
        runner2 = ARCTaskRunnerSimple(
            model="test-model",
            splitter=False
        )
        assert runner2.splitter == False
    
    @patch('run_arc_tasks_soar.create_arc_prompt')
    @patch('run_arc_tasks_soar.ARCAPIClient')
    @patch('run_arc_tasks_soar.ArcTester')
    @patch('run_arc_tasks_soar.PromptLoader')
    @patch('run_arc_tasks_soar.SoarDatasetCollector')
    @patch('run_arc_tasks_soar.CodeTransductionClassifier')
    def test_create_prompt_passes_splitter(self, mock_classifier, mock_collector,
                                          mock_prompt_loader, mock_tester, mock_api_client,
                                          mock_create_arc_prompt):
        """Test that create_prompt method passes splitter flag correctly"""
        from run_arc_tasks_soar import ARCTaskRunnerSimple
        
        # Setup mock
        mock_create_arc_prompt.return_value = ("system", "user")
        
        # Create runner with splitter enabled
        runner = ARCTaskRunnerSimple(model="test-model", splitter=True)
        
        # Call create_prompt
        task_data = create_sample_task_data()
        runner.create_prompt(task_data)
        
        # Verify create_arc_prompt was called with splitter=True
        mock_create_arc_prompt.assert_called_once()
        call_args = mock_create_arc_prompt.call_args
        assert call_args[1]['splitter'] == True
        
        # Reset mock
        mock_create_arc_prompt.reset_mock()
        
        # Test with splitter disabled
        runner2 = ARCTaskRunnerSimple(model="test-model", splitter=False)
        runner2.create_prompt(task_data)
        
        call_args = mock_create_arc_prompt.call_args
        assert call_args[1]['splitter'] == False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])