#!/usr/bin/env python3
from transformers import AutoTokenizer

def get_model_context_length(model: str) -> int:
    """Get the maximum context length for a model"""
    model_lower = model.lower()

    # Qwen models get 40960 tokens (40k)
    if 'qwen' in model_lower:
        return 40960

    # OSS models get 65536 tokens (64k)
    if 'oss' in model_lower:
        return 65536

    # Default fallback for other models
    return 40960


def calculate_prompt_tokens(system_content: str, user_content: str, model: str, debug: bool = False, cached_tokenizer=None) -> int:
    """Calculate the number of tokens in the prompt (excluding assistant response)

    Args:
        system_content: System message content
        user_content: User message content
        model: Model name/path
        debug: Enable debug output
        cached_tokenizer: Optional pre-loaded tokenizer to avoid reloading
    """

    def try_load_tokenizer(model_path_or_slug):
        """Try to load tokenizer from either HF hub or local path"""
        return AutoTokenizer.from_pretrained(
            model_path_or_slug,
            trust_remote_code=True,
            use_fast=True
        )

    # Use cached tokenizer if provided
    if cached_tokenizer is not None:
        tokenizer = cached_tokenizer
        if debug:
            print(f"✅ Using cached tokenizer")
    else:
        tokenizer = None

        # Strategy 1: Try model name as HuggingFace slug (works if online)
        try:
            if debug:
                print(f"🔍 Trying to load tokenizer for '{model}' from HuggingFace Hub...")
            tokenizer = try_load_tokenizer(model)
            if debug:
                print(f"✅ Successfully loaded tokenizer from HuggingFace Hub")
        except Exception as e:
            if debug:
                print(f"❌ HuggingFace Hub failed: {e}")

        # Strategy 2: If that fails, try model as local path (model might be a path to local model)
        if tokenizer is None:
            try:
                # Check if model looks like a path (contains '/' and possibly exists)
                import os
                if '/' in model or os.path.exists(model):
                    if debug:
                        print(f"🔍 Trying to load tokenizer from local path '{model}'...")
                    tokenizer = try_load_tokenizer(model)
                    if debug:
                        print(f"✅ Successfully loaded tokenizer from local path")
            except Exception as e:
                if debug:
                    print(f"❌ Local path failed: {e}")

    # Strategy 3: If we have a tokenizer, use it
    if tokenizer is not None:
        try:
            # Create messages for prompt only (no assistant response)
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]

            # Apply chat template with tokenization
            prompt_tokens = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,  # Include the prompt for assistant response
                return_tensors=None  # Return list of token IDs
            )

            token_count = len(prompt_tokens)
            if debug:
                print(f"🧮 Actual token count: {token_count}")
            return token_count

        except Exception as e:
            if debug:
                print(f"❌ Tokenization failed: {e}")

    # Strategy 4: Fallback to character-based estimation
    total_chars = len(system_content) + len(user_content)
    estimated_tokens = total_chars // 4
    if debug:
        print(f"⚠️ Using character-based fallback estimate: {estimated_tokens} tokens")
    return estimated_tokens

def calculate_max_tokens_for_model(model: str, estimated_prompt_tokens: int = 24000, debug: bool = False) -> int:
    """Calculate max_tokens based on model type and estimated prompt length"""
    # Get model's maximum context length
    max_context = get_model_context_length(model)

    # Reserve tokens for the assistant response
    max_tokens = max_context - estimated_prompt_tokens

    # Ensure we don't go below a reasonable minimum
    if max_tokens < 2000:
        print(f"Warning! max_tokens has gone below 2k to {max_tokens} tokens.")

    if debug:
        print(f"🧮 Calculated max_tokens for {model}: {max_tokens} (context: {max_context}, estimated prompt: {estimated_prompt_tokens})")

    return max_tokens


