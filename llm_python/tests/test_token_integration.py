#!/usr/bin/env python3

import pytest
import unittest.mock as mock
from llm_python.run_arc_tasks_soar import ARCTaskRunnerSimple


class TestTokenIntegration:
    """Integration tests for token calculation in ARCTaskRunnerSimple"""

    @mock.patch('llm_python.run_arc_tasks_soar.ArcTester')
    @mock.patch('llm_python.run_arc_tasks_soar.ARCAPIClient')
    def test_qwen_model_initialization_without_max_tokens(self, mock_api_client, mock_arc_tester):
        """Test that models can be initialized without max_tokens (calculated dynamically per request)"""
        # Create runner without specifying max_tokens
        runner = ARCTaskRunnerSimple(
            model="qwen-14b",
            unsafe_executor=True  # Avoid Docker dependency
        )

        # Verify ARCAPIClient was called with None max_tokens (default from ARCAPIClient will be used)
        mock_api_client.assert_called_once()
        call_args = mock_api_client.call_args[1]  # Get keyword arguments
        assert call_args['max_tokens'] is None  # Should be None for dynamic calculation

    @mock.patch('llm_python.run_arc_tasks_soar.ArcTester')
    @mock.patch('llm_python.run_arc_tasks_soar.ARCAPIClient')
    def test_oss_model_initialization_without_max_tokens(self, mock_api_client, mock_arc_tester):
        """Test that OSS models can be initialized without max_tokens (calculated dynamically per request)"""
        # Create runner without specifying max_tokens
        runner = ARCTaskRunnerSimple(
            model="gpt-oss-120b",
            unsafe_executor=True  # Avoid Docker dependency
        )

        # Verify ARCAPIClient was called with None max_tokens
        mock_api_client.assert_called_once()
        call_args = mock_api_client.call_args[1]  # Get keyword arguments
        assert call_args['max_tokens'] is None  # Should be None for dynamic calculation

    @mock.patch('llm_python.run_arc_tasks_soar.ArcTester')
    @mock.patch('llm_python.run_arc_tasks_soar.ARCAPIClient')
    def test_explicit_max_tokens_override(self, mock_api_client, mock_arc_tester):
        """Test that explicit max_tokens parameter takes precedence"""
        # Create runner with explicit max_tokens
        runner = ARCTaskRunnerSimple(
            model="qwen-14b",
            max_tokens=5000,
            unsafe_executor=True  # Avoid Docker dependency
        )

        # Verify ARCAPIClient was called with the explicit max_tokens
        mock_api_client.assert_called_once()
        call_args = mock_api_client.call_args[1]  # Get keyword arguments
        assert call_args['max_tokens'] == 5000  # Should use explicit value

    @mock.patch('llm_python.run_arc_tasks_soar.ArcTester')
    @mock.patch('llm_python.run_arc_tasks_soar.ARCAPIClient')
    def test_other_model_initialization_without_max_tokens(self, mock_api_client, mock_arc_tester):
        """Test that other models can be initialized without max_tokens (calculated dynamically per request)"""
        # Create runner without specifying max_tokens
        runner = ARCTaskRunnerSimple(
            model="gpt-4o",
            unsafe_executor=True  # Avoid Docker dependency
        )

        # Verify ARCAPIClient was called with None max_tokens
        mock_api_client.assert_called_once()
        call_args = mock_api_client.call_args[1]  # Get keyword arguments
        assert call_args['max_tokens'] is None  # Should be None for dynamic calculation

    @mock.patch('llm_python.run_arc_tasks_soar.ArcTester')
    @mock.patch('llm_python.run_arc_tasks_soar.ARCAPIClient')
    def test_no_debug_mode_token_calculation(self, mock_api_client, mock_arc_tester):
        """Test that debug=False is passed during initialization"""
        # Create runner without debug mode
        runner = ARCTaskRunnerSimple(
            model="qwen-14b",
            debug=False,
            unsafe_executor=True  # Avoid Docker dependency
        )

        # Just verify the runner was created successfully
        assert runner.debug is False

    @mock.patch('llm_python.run_arc_tasks_soar.calculate_prompt_tokens')
    @mock.patch('llm_python.run_arc_tasks_soar.calculate_max_tokens_for_model')
    @mock.patch('llm_python.run_arc_tasks_soar.ArcTester')
    def test_dynamic_token_calculation_in_run_single_attempt(self, mock_arc_tester, mock_calc_max_tokens, mock_calc_prompt_tokens):
        """Test that token calculation happens dynamically during run_single_attempt"""
        mock_calc_prompt_tokens.return_value = 5000  # Mock actual prompt tokens
        mock_calc_max_tokens.return_value = 8000    # Mock calculated max tokens

        # Mock the API client with all required methods
        mock_api_client = mock.MagicMock()
        mock_api_client.get_model_pricing.return_value = (1.0, 2.0)  # Mock pricing
        mock_api_client.model = "qwen-14b"  # Set the model name

        # Create runner
        runner = ARCTaskRunnerSimple(
            model="qwen-14b",
            debug=True,
            unsafe_executor=True
        )
        runner.api_client = mock_api_client
        runner.call_chat_completions_api = mock.MagicMock(return_value=(None, {}))
        runner.extract_code_from_response = mock.MagicMock(return_value="print('test')")
        runner.executor = mock.MagicMock()
        runner.executor.run_program.return_value = ([], "success", False, False)  # outputs, summary, timed_out, error
        runner.transduction_classifier = mock.MagicMock()
        runner.transduction_classifier.is_transductive.return_value = False

        # Mock task data with proper structure
        task_data = {
            "train": [],
            "test": [{"input": [[1, 2], [3, 4]], "output": [[1, 1], [1, 1]]}]
        }
        full_prompt = {
            "system": "You are a test assistant.",
            "user": "Solve this puzzle."
        }

        # Run single attempt
        result = runner.run_single_attempt(
            task_id="test_task",
            task_data=task_data,
            attempt_num=0,
            full_prompt=full_prompt
        )

        # Verify dynamic token calculation was called
        mock_calc_prompt_tokens.assert_called_once_with(
            "You are a test assistant.", "Solve this puzzle.", "qwen-14b", debug=True, cached_tokenizer=None
        )
        mock_calc_max_tokens.assert_called_once_with(
            "qwen-14b", estimated_prompt_tokens=5000, debug=True
        )
        # Verify API client max_tokens was updated
        mock_api_client.set_max_tokens_for_request.assert_called_once_with(8000)

    @mock.patch('llm_python.run_arc_tasks_soar.calculate_prompt_tokens')
    @mock.patch('llm_python.run_arc_tasks_soar.calculate_max_tokens_for_model')
    @mock.patch('llm_python.run_arc_tasks_soar.ArcTester')
    def test_explicit_max_tokens_vs_dynamic_calculation(self, mock_arc_tester, mock_calc_max_tokens, mock_calc_prompt_tokens):
        """Test that explicit max_tokens is respected but capped by dynamic calculation"""
        mock_calc_prompt_tokens.return_value = 5000
        mock_calc_max_tokens.return_value = 3000  # Dynamic calculation gives 3000

        # Mock the API client
        mock_api_client = mock.MagicMock()
        mock_api_client.get_model_pricing.return_value = (1.0, 2.0)
        mock_api_client.model = "qwen-14b"

        # Create runner with explicit max_tokens=5000
        runner = ARCTaskRunnerSimple(
            model="qwen-14b",
            max_tokens=5000,  # User wants 5000
            debug=True,
            unsafe_executor=True
        )
        runner.api_client = mock_api_client
        runner.call_chat_completions_api = mock.MagicMock(return_value=(None, {}))
        runner.extract_code_from_response = mock.MagicMock(return_value="print('test')")
        runner.executor = mock.MagicMock()
        runner.executor.run_program.return_value = ([], "success", False, False)
        runner.transduction_classifier = mock.MagicMock()
        runner.transduction_classifier.is_transductive.return_value = False

        # Mock task data
        task_data = {
            "train": [],
            "test": [{"input": [[1, 2]], "output": [[1, 1]]}]
        }
        full_prompt = {
            "system": "Test system prompt",
            "user": "Test user prompt"
        }

        # Run single attempt
        result = runner.run_single_attempt(
            task_id="test_task",
            task_data=task_data,
            attempt_num=0,
            full_prompt=full_prompt
        )

        # Verify dynamic calculation was called
        mock_calc_prompt_tokens.assert_called_once()
        mock_calc_max_tokens.assert_called_once()
        # Should use min(5000, 3000) = 3000
        mock_api_client.set_max_tokens_for_request.assert_called_once_with(3000)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])