#!/usr/bin/env python3

import pytest
import unittest.mock as mock
from llm_python.utils.token_utils import (
    get_model_context_length,
    calculate_prompt_tokens,
    calculate_max_tokens_for_model
)


class TestTokenUtils:
    """Test cases for token utility functions"""

    def test_get_model_context_length_qwen(self):
        """Test context length for Qwen models"""
        assert get_model_context_length("qwen-14b") == 40960
        assert get_model_context_length("Qwen-2.5-14B") == 40960
        assert get_model_context_length("QWEN-CHAT") == 40960

    def test_get_model_context_length_oss(self):
        """Test context length for OSS models"""
        assert get_model_context_length("gpt-oss-120b") == 65536
        assert get_model_context_length("openai/gpt-oss-20b") == 65536
        assert get_model_context_length("GPT-OSS-INSTRUCT") == 65536

    def test_get_model_context_length_other(self):
        """Test context length for other models (default)"""
        assert get_model_context_length("gpt-4o") == 40960
        assert get_model_context_length("claude-3") == 40960
        assert get_model_context_length("random-model") == 40960

    def test_calculate_max_tokens_for_model_qwen(self):
        """Test max_tokens calculation for Qwen models"""
        max_tokens = calculate_max_tokens_for_model("qwen-14b")
        # Should be 40960 - 24000 = 16960 (no cap anymore)
        assert max_tokens == 16960

    def test_calculate_max_tokens_for_model_oss(self):
        """Test max_tokens calculation for OSS models"""
        max_tokens = calculate_max_tokens_for_model("gpt-oss-120b")
        # Should be 65536 - 24000 = 41536 (no cap anymore)
        assert max_tokens == 41536

    def test_calculate_max_tokens_for_model_custom_prompt_length(self):
        """Test max_tokens calculation with custom prompt length"""
        max_tokens = calculate_max_tokens_for_model("qwen-14b", estimated_prompt_tokens=30000)
        # Should be 40960 - 30000 = 10960
        assert max_tokens == 10960

    def test_calculate_max_tokens_for_model_minimum_enforced(self):
        """Test that minimum max_tokens warning is shown"""
        max_tokens = calculate_max_tokens_for_model("qwen-14b", estimated_prompt_tokens=39000)
        # Should be 40960 - 39000 = 1960, and warning is printed but value returned as-is
        assert max_tokens == 1960

    @mock.patch('llm_python.utils.token_utils.AutoTokenizer')
    def test_calculate_prompt_tokens_success(self, mock_tokenizer_class):
        """Test successful prompt token calculation"""
        # Mock tokenizer
        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.apply_chat_template.return_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 10 tokens
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        system_content = "You are a helpful assistant."
        user_content = "Hello, how are you?"

        result = calculate_prompt_tokens(system_content, user_content, "test-model")

        assert result == 10
        mock_tokenizer_class.from_pretrained.assert_called_once_with(
            "test-model", trust_remote_code=True, use_fast=True
        )
        mock_tokenizer.apply_chat_template.assert_called_once_with(
            [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None
        )

    @mock.patch('llm_python.utils.token_utils.AutoTokenizer')
    def test_calculate_prompt_tokens_model_path(self, mock_tokenizer_class):
        """Test prompt token calculation handles model paths correctly"""
        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.apply_chat_template.return_value = [1, 2, 3]
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        calculate_prompt_tokens("sys", "user", "org/model-name")

        # Should first try the model name as-is (HF hub)
        mock_tokenizer_class.from_pretrained.assert_called_with(
            "org/model-name", trust_remote_code=True, use_fast=True
        )

    @mock.patch('llm_python.utils.token_utils.AutoTokenizer')
    def test_calculate_prompt_tokens_fallback(self, mock_tokenizer_class):
        """Test fallback when tokenizer fails"""
        # Mock both HF hub and local path attempts to fail
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Model not found")

        system_content = "You are a helpful assistant."  # 28 chars
        user_content = "Hello, how are you?"  # 19 chars
        # Total: 47 chars, estimated tokens: 47 // 4 = 11

        result = calculate_prompt_tokens(system_content, user_content, "nonexistent-model", debug=True)

        assert result == 11
        # Should only be called once for HF hub (no local path attempt for simple model names)
        assert mock_tokenizer_class.from_pretrained.call_count == 1

    @mock.patch('llm_python.utils.token_utils.AutoTokenizer')
    def test_calculate_prompt_tokens_local_path_strategy(self, mock_tokenizer_class):
        """Test that local path loading is attempted when HF hub fails"""
        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.apply_chat_template.return_value = [1, 2, 3, 4, 5]

        # First call (HF hub) fails, second call (local path) succeeds
        mock_tokenizer_class.from_pretrained.side_effect = [Exception("HF hub failed"), mock_tokenizer]

        result = calculate_prompt_tokens("sys", "user", "/local/path/model")

        assert result == 5
        # Should be called twice: HF hub then local path
        assert mock_tokenizer_class.from_pretrained.call_count == 2
        # Check both calls were made with the same path
        calls = mock_tokenizer_class.from_pretrained.call_args_list
        assert calls[0][0][0] == "/local/path/model"  # First call (HF hub attempt)
        assert calls[1][0][0] == "/local/path/model"  # Second call (local path attempt)

    @mock.patch('llm_python.utils.token_utils.AutoTokenizer')
    def test_calculate_prompt_tokens_empty_content(self, mock_tokenizer_class):
        """Test prompt token calculation with empty content"""
        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.apply_chat_template.return_value = [1, 2]  # Just system tokens
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        result = calculate_prompt_tokens("", "", "test-model")
        assert result == 2

    def test_calculate_max_tokens_debug_output(self, capsys):
        """Test that debug output is produced when enabled"""
        calculate_max_tokens_for_model("qwen-14b", debug=True)

        captured = capsys.readouterr()
        assert "🧮 Calculated max_tokens for qwen-14b" in captured.out
        assert "context: 40960" in captured.out
        assert "estimated prompt: 24000" in captured.out

    def test_calculate_max_tokens_no_debug_output(self, capsys):
        """Test that no debug output is produced when debug=False"""
        calculate_max_tokens_for_model("qwen-14b", debug=False)

        captured = capsys.readouterr()
        assert captured.out == ""

    @mock.patch('llm_python.utils.token_utils.AutoTokenizer')
    def test_calculate_prompt_tokens_with_tokenizer_path_success(self, mock_tokenizer_class):
        """Test that tokenizer_path takes precedence over model name"""
        # Mock tokenizer
        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.apply_chat_template.return_value = [1, 2, 3, 4, 5]  # 5 tokens
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        system_content = "You are a helpful assistant."
        user_content = "Hello, how are you?"
        model_name = "gpt-4o"
        tokenizer_path = "custom/tokenizer-path"

        result = calculate_prompt_tokens(
            system_content, user_content, model_name,
            debug=True, tokenizer_path=tokenizer_path
        )

        assert result == 5
        # Should use tokenizer_path, not model name
        mock_tokenizer_class.from_pretrained.assert_called_once_with(
            tokenizer_path, trust_remote_code=True, use_fast=True
        )

    @mock.patch('llm_python.utils.token_utils.AutoTokenizer')
    def test_calculate_prompt_tokens_tokenizer_path_fallback_to_model(self, mock_tokenizer_class):
        """Test fallback to model name when tokenizer_path fails"""
        mock_tokenizer = mock.MagicMock()
        mock_tokenizer.apply_chat_template.return_value = [1, 2, 3]  # 3 tokens

        # First call (tokenizer_path HF hub) fails, second call (tokenizer_path local) fails,
        # third call (model name HF hub) succeeds
        mock_tokenizer_class.from_pretrained.side_effect = [
            Exception("HF hub failed"),
            Exception("Local path failed"),
            mock_tokenizer
        ]

        result = calculate_prompt_tokens(
            "sys", "user", "gpt-4o", debug=True, tokenizer_path="nonexistent/tokenizer"
        )

        assert result == 3
        # Should have been called 3 times: tokenizer_path (HF), tokenizer_path (local), model name (HF)
        assert mock_tokenizer_class.from_pretrained.call_count == 3
        calls = mock_tokenizer_class.from_pretrained.call_args_list
        assert calls[0][0][0] == "nonexistent/tokenizer"  # First attempt with tokenizer_path
        assert calls[1][0][0] == "nonexistent/tokenizer"  # Second attempt with tokenizer_path
        assert calls[2][0][0] == "gpt-4o"  # Third attempt with model name

    @mock.patch('llm_python.utils.token_utils.AutoTokenizer')
    def test_calculate_prompt_tokens_debug_fallback(self, mock_tokenizer_class, capsys):
        """Test debug output during fallback"""
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Test error")

        calculate_prompt_tokens("test", "test", "bad-model", debug=True)

        captured = capsys.readouterr()
        assert "Strategy 2 (HuggingFace Hub with model name) failed: Test error" in captured.out
        assert "Using character-based fallback estimate" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])