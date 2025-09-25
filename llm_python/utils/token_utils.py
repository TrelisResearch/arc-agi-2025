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


def calculate_prompt_tokens(system_content: str, user_content: str, model: str, debug: bool = False, cached_tokenizer=None, skip_loading: bool = False, tokenizer_path: str = None) -> int:
    """Calculate the number of tokens in the prompt (excluding assistant response)

    Args:
        system_content: System message content
        user_content: User message content
        model: Model name (used for HuggingFace Hub lookup)
        debug: Enable debug output
        cached_tokenizer: Optional pre-loaded tokenizer to avoid reloading
        skip_loading: If True, skip tokenizer loading and go straight to character estimation
        tokenizer_path: Optional explicit path to tokenizer (tried first as HF slug, then as local path)
    """

    def try_load_tokenizer(model_path_or_slug):
        """Try to load tokenizer from either HF hub or local path"""
        return AutoTokenizer.from_pretrained(
            model_path_or_slug,
            trust_remote_code=True,
            use_fast=True
        )

    # Strategy selection based on inputs
    if cached_tokenizer is not None:
        # Use the cached tokenizer
        tokenizer = cached_tokenizer
        if debug:
            print(f"✅ Using cached tokenizer")
    elif skip_loading:
        # Skip loading, go straight to fallback
        tokenizer = None
        if debug:
            print(f"⚠️ Skipping tokenizer loading (using character estimation)")
    else:
        # Try to load tokenizer
        tokenizer = None
        if debug:
            print(f"🔍 Attempting to load tokenizer for '{model}'...")

        # Strategy 1: Try explicit tokenizer_path if provided (as HF slug first, then local path)
        if tokenizer_path:
            # Try as HuggingFace slug first
            try:
                if debug:
                    print(f"🔍 Strategy 1a: Attempting to load tokenizer from HuggingFace Hub using explicit tokenizer_path '{tokenizer_path}'")
                tokenizer = try_load_tokenizer(tokenizer_path)
                if debug:
                    print(f"✅ Successfully loaded tokenizer from HuggingFace Hub using tokenizer_path")
            except Exception as e:
                if debug:
                    print(f"❌ Strategy 1a (HuggingFace Hub with tokenizer_path) failed: {e}")

                # Try as local path
                try:
                    if debug:
                        print(f"🔍 Strategy 1b: Attempting to load tokenizer from local path using tokenizer_path '{tokenizer_path}'")
                    tokenizer = try_load_tokenizer(tokenizer_path)
                    if debug:
                        print(f"✅ Successfully loaded tokenizer from local path using tokenizer_path")
                except Exception as e2:
                    if debug:
                        print(f"❌ Strategy 1b (Local path with tokenizer_path) failed: {e2}")

        # Strategy 2: Try model name as HuggingFace slug (works if online)
        if tokenizer is None:
            try:
                if debug:
                    print(f"🔍 Strategy 2: Attempting to load tokenizer from HuggingFace Hub using model name '{model}'")
                tokenizer = try_load_tokenizer(model)
                if debug:
                    print(f"✅ Successfully loaded tokenizer from HuggingFace Hub using model name")
            except Exception as e:
                if debug:
                    print(f"❌ Strategy 2 (HuggingFace Hub with model name) failed: {e}")

        # Strategy 3: If that fails, try model as local path (model might be a path to local model)
        if tokenizer is None:
            try:
                # Check if model looks like a path (contains '/' and possibly exists)
                import os
                if '/' in model or os.path.exists(model):
                    if debug:
                        print(f"🔍 Strategy 3: Model contains '/' or exists as path, attempting to load tokenizer from local path '{model}'")
                    tokenizer = try_load_tokenizer(model)
                    if debug:
                        print(f"✅ Successfully loaded tokenizer from local path using model name")
                else:
                    if debug:
                        print(f"🔍 Strategy 3: Model '{model}' doesn't look like a local path (no '/' and doesn't exist), skipping local path attempt")
            except Exception as e:
                if debug:
                    print(f"❌ Strategy 3 (Local path with model name) failed: {e}")

    # Strategy 4: If we have a tokenizer, use it
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

    # Strategy 5: Fallback to character-based estimation
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


