#!/usr/bin/env python3

import os
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from .serialization import ResponseSerializer


class ARCAPIClient:
    """Handles API calls and model-specific configuration for ARC task solving"""

    def __init__(
        self,
        model: str = "gpt-4.1-nano",
        base_url: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: Optional[float] = None,
        reasoning_effort: str = "low",
        qwen_no_think: bool = False,
        lora_adapter: Optional[str] = None,
        api_timeout: int = 120,
    ):
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.qwen_no_think = qwen_no_think
        self.lora_adapter = lora_adapter
        self.api_timeout = api_timeout

        # Set API key based on endpoint
        self.api_key = self._get_api_key()

        # Initialize OpenAI client
        client_timeout = self.api_timeout
        if base_url:
            self.client = OpenAI(
                api_key=self.api_key, base_url=base_url, timeout=client_timeout
            )
        else:
            self.client = OpenAI(api_key=self.api_key, timeout=client_timeout)

        # Check models endpoint if using custom base URL
        if base_url:
            self._check_models_endpoint()

    def _get_api_key(self) -> Optional[str]:
        """Get appropriate API key based on endpoint"""
        if self.base_url == "https://router.huggingface.co/v1":
            return self._get_hf_token()
        elif self.base_url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1":
            return os.getenv("DASHSCOPE_API_KEY")
        else:
            return os.getenv("OPENAI_API_KEY") or "EMPTY"

    def _get_hf_token(self) -> Optional[str]:
        """Get Hugging Face token from environment"""
        return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    def _check_models_endpoint(self):
        """Check what models are available at the endpoint"""
        try:
            models = self.client.models.list()
            available_models = [model.id for model in models]
            if self.model not in available_models:
                print(f"⚠️ Model '{self.model}' not found in available models")
                print(
                    f"📋 Available models: {', '.join(available_models[:10])}{'...' if len(available_models) > 10 else ''}"
                )
        except Exception as e:
            if self.base_url:
                print(f"⚠️ Could not check models endpoint at {self.base_url}: {e}")

    def get_model_pricing(self) -> Tuple[float, float]:
        """Get input and output pricing rates for a model in $/1M tokens"""
        model_lower = self.model.lower()

        # Reasoning models
        if model_lower.startswith("o3-pro"):
            return (20.00, 80.00)
        elif model_lower.startswith("o3-mini"):
            return (1.10, 4.40)
        elif model_lower.startswith("o3"):
            return (2.00, 8.00)
        elif model_lower.startswith("o4-mini"):
            return (1.10, 4.40)

        # GPT-5 models
        elif model_lower.startswith("gpt-5-mini"):
            return (0.25, 2.00)
        elif model_lower.startswith("gpt-5-nano"):
            return (0.05, 0.40)
        
        # GPT-4 models
        elif model_lower.startswith("gpt-4.1-nano"):
            return (0.10, 0.40)
        elif model_lower.startswith("gpt-4.1-mini"):
            return (0.40, 1.60)
        elif model_lower.startswith("gpt-4.1"):
            return (2.00, 8.00)
        elif model_lower.startswith("gpt-4o-mini"):
            return (0.15, 0.60)
        elif model_lower.startswith("gpt-4o"):
            return (2.50, 10.00)
        elif model_lower.startswith("gpt-oss-120b"):
            return (0.073, 0.29)
        elif model_lower.startswith("openai/gpt-oss-20b") or model_lower.startswith("gpt-oss-20b"):
            return (0.04, 0.15)

        # Google models
        elif model_lower.startswith("google/gemini-2.5-flash"):
            return (0.30, 2.50)

        # Default fallback
        else:
            return (0.15, 0.60)

    def get_sampling_parameters(self) -> Dict:
        """Get the sampling parameters that will be used for API calls"""
        kwargs = {}

        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        # Set temperature (use instance value or default to 1.0)
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        else:
            kwargs["temperature"] = 1.0  # Default temperature

        # Add reasoning parameters for OpenRouter
        if self.base_url and "openrouter" in self.base_url.lower():
            reasoning_tokens = {"low": 2000, "medium": 8000, "high": 16000}
            if self.reasoning_effort in reasoning_tokens:
                if "gemini" in self.model.lower():
                    kwargs["extra_body"] = {
                        "reasoning": {
                            "max_tokens": reasoning_tokens[self.reasoning_effort]
                        }
                    }
                else:
                    if self.max_tokens is None:
                        kwargs["max_tokens"] = reasoning_tokens[self.reasoning_effort]

        # Add thinking_budget for DashScope (Qwen thinking models)
        if self.base_url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1":
            thinking_budget_tokens = {"low": 2000, "medium": 8000, "high": 32000}
            if self.reasoning_effort in thinking_budget_tokens:
                if "extra_body" not in kwargs:
                    kwargs["extra_body"] = {}
                kwargs["extra_body"]["thinking_budget"] = thinking_budget_tokens[
                    self.reasoning_effort
                ]

        # Add sampling parameters based on endpoint type
        if "top_p" not in kwargs and "min_p" not in kwargs:
            # For TCP endpoints, use min_p instead of top_p/top_k
            if (
                self.base_url
                and ":" in self.base_url
                and not self.base_url.startswith("https://")
            ):
                if "extra_body" not in kwargs:
                    kwargs["extra_body"] = {}
                kwargs["extra_body"]["min_p"] = 0.05
                # kwargs["top_p"] = 0.9
                # if "extra_body" not in kwargs:
                #     kwargs["extra_body"] = {}
                # if "top_k" not in kwargs["extra_body"]:
                #     kwargs["extra_body"]["top_k"] = 20
            else:
                # For most endpoints, use top_p and top_k defaults
                kwargs["top_p"] = 0.95
                if "extra_body" not in kwargs:
                    kwargs["extra_body"] = {}
                if "top_k" not in kwargs["extra_body"]:
                    kwargs["extra_body"]["top_k"] = 50

        # Add Qwen-specific parameters (only for no-think flag)
        if "qwen" in self.model.lower() and self.base_url and self.qwen_no_think:
            if "extra_body" not in kwargs:
                kwargs["extra_body"] = {}
            # Note: DashScope commercial models don't support enable_thinking=False
            if (
                self.base_url
                != "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            ):
                kwargs["extra_body"]["chat_template_kwargs"] = {
                    "enable_thinking": False
                }

        # Add LORA adapter specification if provided
        if self.lora_adapter:
            if "extra_body" not in kwargs:
                kwargs["extra_body"] = {}
            kwargs["extra_body"]["lora"] = self.lora_adapter

        return kwargs

    def call_chat_completions_api(self, messages: List[Dict]) -> Dict:
        """Call the OpenAI Chat Completions API"""
        try:
            kwargs = {"model": self.model, "messages": messages}
            kwargs.update(self.get_sampling_parameters())

            response = self.client.chat.completions.create(
                **kwargs
            )

            if not response:
                return {
                    "success": False,
                    "error": "Empty response from API",
                    "api_timeout": False,
                    "empty_response": True,
                    "hit_max_tokens": False,
                    "raw_response": None,
                    "sampling_params": kwargs,
                }

            # Check if we hit max tokens
            hit_max_tokens = False
            if (
                hasattr(response, "choices")
                and len(response.choices) > 0
                and hasattr(response.choices[0], "finish_reason")
                and response.choices[0].finish_reason == "length"
            ):
                hit_max_tokens = True

            return {
                "success": True,
                "error": None,
                "api_timeout": False,
                "empty_response": False,
                "hit_max_tokens": hit_max_tokens,
                "raw_response": ResponseSerializer.serialize_response(response),
                "sampling_params": kwargs,
                "response": response,
            }

        except Exception as e:
            error_str = str(e).lower()
            is_timeout = any(
                timeout_indicator in error_str
                for timeout_indicator in [
                    "timeout",
                    "timed out",
                    "time out",
                    "deadline exceeded",
                ]
            )

            return {
                "success": False,
                "error": str(e),
                "api_timeout": is_timeout,
                "empty_response": False,
                "hit_max_tokens": False,
                "raw_response": None,
                "sampling_params": kwargs,
            }
