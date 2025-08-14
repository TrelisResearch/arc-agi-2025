# Classification Tests

This directory contains test scripts for validating the ARC program classification analysis pipeline components.

## Test Scripts

### `test_nomic_embeddings.py`
**Purpose**: Test Nomic embeddings API connectivity and functionality  
**Model**: `nomic-embed-text-v1.5` with official Nomic Python client  
**Requirements**: `NOMIC_API_KEY` in root `.env` file  
**Output**: 768-dimensional embeddings validation

### `test_nomic_code_embeddings.py`  
**Purpose**: Test code-specific embedding models with fallback mechanisms  
**Models**: `nomic-embed-code`, `nomic-embed-text-v1.5` with different task types  
**Features**: Automatic fallback between models and task types  
**Requirements**: `NOMIC_API_KEY` in root `.env` file

### `test_gemini_classification.py`
**Purpose**: Test Gemini 2.5 Flash classification via OpenRouter  
**Model**: `google/gemini-2.5-flash` with detailed reasoning  
**Features**: Single program classification with full response parsing  
**Requirements**: `OPENAI_API_KEY` in `llm_python/.env` file

## Usage

```bash
# Test individual components
uv run python classification/tests/test_nomic_embeddings.py
uv run python classification/tests/test_nomic_code_embeddings.py  
uv run python classification/tests/test_gemini_classification.py

# Or run all tests from tests directory
cd classification/tests
uv run python test_nomic_embeddings.py
uv run python test_nomic_code_embeddings.py
uv run python test_gemini_classification.py
```

## Test Results Summary

- ✅ **Nomic Embeddings**: 768D vectors, stable API connectivity
- ✅ **Code Embeddings**: Multi-model fallback mechanism working  
- ✅ **Gemini Classification**: Detailed reasoning with proper response parsing
- ✅ **API Authentication**: All services accessible with proper keys

## What Each Test Does

### `test_nomic_embeddings.py`
- Tests embedding a Fibonacci function using `nomic-embed-text-v1.5` model
- Tests embedding multiple different code samples
- Validates embeddings are properly generated (768 dimensions, normalized vectors)
- Shows cosine similarity between different code samples

### `test_nomic_code_embeddings.py`
- Attempts to use `nomic-embed-code` model for code-specific embeddings
- Falls back to `nomic-embed-text-v1.5` with different task types if needed
- Tests robustness of the embedding pipeline with multiple model options

### `test_gemini_classification.py`
- Tests single program classification using Gemini 2.5 Flash
- Validates response parsing and classification extraction
- Shows detailed reasoning process for overfitting vs general determination

These tests validate the individual components before running the full analysis pipeline.