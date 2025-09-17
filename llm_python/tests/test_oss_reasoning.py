"""
Test for OSS model reasoning instruction enhancement.
"""

import pytest
from unittest.mock import patch, MagicMock
from llm_python.run_arc_tasks_soar import ARCTaskRunnerSimple


class TestOSSReasoningInstruction:
    """Test OSS model reasoning instruction enhancement"""

    def test_oss_model_adds_reasoning_instruction(self):
        """Test that OSS models get 'Reasoning: high' appended to system prompt"""
        with patch('llm_python.utils.api_client.OpenAI') as mock_openai:
            mock_openai.return_value = MagicMock()

            # Create runner with OSS model
            runner = ARCTaskRunnerSimple(
                model="Qwen/Qwen2.5-Coder-32B-Instruct-OSS",
                max_workers=1,
                max_attempts=1,
                debug=False,
                unsafe_executor=True,
            )

            # Mock the API call to capture the conversation history
            with patch.object(runner, 'call_chat_completions_api') as mock_api:
                mock_api.return_value = (MagicMock(choices=[MagicMock(message=MagicMock(content="def solve(): pass"))]), {})

                # Create a simple full_prompt
                full_prompt = {
                    "system": "You are an AI assistant that solves ARC tasks.",
                    "user": "Please solve this task."
                }

                # Mock task data
                task_data = {
                    "train": [],
                    "test": [{"input": [[1, 2]], "output": None}]
                }

                # Run single attempt
                runner.run_single_attempt(
                    task_id="test_task",
                    task_data=task_data,
                    attempt_num=1,
                    full_prompt=full_prompt
                )

                # Verify the API was called with reasoning instruction
                mock_api.assert_called_once()
                call_args = mock_api.call_args[0]  # Get positional arguments
                conversation_history = call_args[0]  # First argument is conversation_history

                # Check that system message contains the reasoning instruction
                system_message = conversation_history[0]["content"]
                assert system_message.endswith(" Reasoning: high")
                assert "You are an AI assistant that solves ARC tasks. Reasoning: high" == system_message

    def test_non_oss_model_no_reasoning_instruction(self):
        """Test that non-OSS models do NOT get reasoning instruction"""
        with patch('llm_python.utils.api_client.OpenAI') as mock_openai:
            mock_openai.return_value = MagicMock()

            # Create runner with non-OSS model
            runner = ARCTaskRunnerSimple(
                model="gpt-4o-mini",
                max_workers=1,
                max_attempts=1,
                debug=False,
                unsafe_executor=True,
            )

            # Mock the API call to capture the conversation history
            with patch.object(runner, 'call_chat_completions_api') as mock_api:
                mock_api.return_value = (MagicMock(choices=[MagicMock(message=MagicMock(content="def solve(): pass"))]), {})

                # Create a simple full_prompt
                full_prompt = {
                    "system": "You are an AI assistant that solves ARC tasks.",
                    "user": "Please solve this task."
                }

                # Mock task data
                task_data = {
                    "train": [],
                    "test": [{"input": [[1, 2]], "output": None}]
                }

                # Run single attempt
                runner.run_single_attempt(
                    task_id="test_task",
                    task_data=task_data,
                    attempt_num=1,
                    full_prompt=full_prompt
                )

                # Verify the API was called without reasoning instruction
                mock_api.assert_called_once()
                call_args = mock_api.call_args[0]  # Get positional arguments
                conversation_history = call_args[0]  # First argument is conversation_history

                # Check that system message does NOT contain reasoning instruction
                system_message = conversation_history[0]["content"]
                assert not system_message.endswith(" Reasoning: high")
                assert "You are an AI assistant that solves ARC tasks." == system_message

    def test_oss_case_insensitive(self):
        """Test that OSS detection is case insensitive"""
        with patch('llm_python.utils.api_client.OpenAI') as mock_openai:
            mock_openai.return_value = MagicMock()

            # Test different cases
            test_models = [
                "Qwen/Qwen2.5-Coder-32B-Instruct-OSS",
                "some-model-OSS-version",
                "model-oss-test",
                "OSS-Model-Name",
            ]

            for model_name in test_models:
                runner = ARCTaskRunnerSimple(
                    model=model_name,
                    max_workers=1,
                    max_attempts=1,
                    debug=False,
                    unsafe_executor=True,
                )

                with patch.object(runner, 'call_chat_completions_api') as mock_api:
                    mock_api.return_value = (MagicMock(choices=[MagicMock(message=MagicMock(content="def solve(): pass"))]), {})

                    full_prompt = {
                        "system": "Test system prompt.",
                        "user": "Test user prompt."
                    }

                    task_data = {
                        "train": [],
                        "test": [{"input": [[1, 2]], "output": None}]
                    }

                    runner.run_single_attempt(
                        task_id="test_task",
                        task_data=task_data,
                        attempt_num=1,
                        full_prompt=full_prompt
                    )

                    # Verify reasoning instruction was added
                    call_args = mock_api.call_args[0]
                    conversation_history = call_args[0]
                    system_message = conversation_history[0]["content"]
                    assert system_message.endswith(" Reasoning: high"), f"Failed for model: {model_name}"