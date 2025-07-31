# ARC Program Classification Analysis

This directory contains a comprehensive analysis pipeline for classifying ARC (Abstraction and Reasoning Corpus) program solutions as either "overfitting" or "general" using code embeddings and AI classification.

## 🎯 Overview

The analysis pipeline combines **OpenAI code embeddings** with **Gemini 2.5 Flash reasoning** to achieve **100% human-AI classification agreement** on program solutions. The system distinguishes between:

- **Overfitting**: Solutions with hardcoded rules, specific dimensions, magic numbers
- **General**: Solutions with algorithmic approaches, pattern detection, adaptable logic

## 📊 Key Results

- **Perfect Agreement**: 100% alignment between human and AI classifications (16/16 programs)
- **Superior Embeddings**: OpenAI `text-embedding-3-small` (1536D) with 47% better class separation vs Nomic
- **Detailed Reasoning**: 8k token responses with comprehensive analysis per program
- **Fast Execution**: 5x parallel processing for ~17-20 second total analysis time

## 🗂️ Directory Structure

```
classification/
├── README.md                          # This documentation
├── analyze_program_embeddings.py      # Main analysis pipeline
├── embedding_analysis.py              # Statistical analysis and visualization
├── view_gemini_responses.py           # Utility to view detailed AI reasoning
├── data/                              # Results and datasets
│   ├── model-completion-accuracy-check.json     # Input: 16 program solutions
│   ├── program_analysis_dataset.json            # Output: Complete analysis dataset
│   ├── embedding_analysis_results.json          # Statistical analysis results
│   ├── embedding_visualizations.png             # PCA/t-SNE plots
│   └── similarity_heatmap.png                   # Cosine similarity heatmap
├── tests/                             # Test scripts and validation
│   ├── README.md                      # Test documentation
│   ├── test_nomic_embeddings.py       # Nomic API tests
│   ├── test_nomic_code_embeddings.py  # Code-specific embedding tests
│   └── test_gemini_classification.py  # Gemini classification tests
└── archive/                           # Deprecated/alternative implementations
    └── analyze_program_embeddings_code.py       # Alternative with Nomic code embeddings
```

## 🚀 Quick Start

### Prerequisites

1. **API Keys**: Add to root `.env` file:
   ```bash
   OPENAI_API_KEY_OPENAI="sk-proj-..."      # For embeddings
   OPENAI_API_KEY="sk-or-v1-..."            # For Gemini via OpenRouter
   ```

2. **Dependencies**: Install with uv:
   ```bash
   uv add openai python-dotenv numpy scikit-learn matplotlib seaborn
   ```

### Run Complete Analysis

```bash
# Generate embeddings, classifications, and analysis
uv run python classification/analyze_program_embeddings.py

# Run statistical analysis and generate visualizations  
uv run python classification/embedding_analysis.py

# View detailed Gemini reasoning for each program
uv run python classification/view_gemini_responses.py
```

## 📋 Scripts Documentation

### Main Pipeline (`analyze_program_embeddings.py`)

**Purpose**: Complete analysis pipeline from raw programs to classified dataset

**Features**:
- **Parallel Processing**: 5 workers for Gemini classifications, 4 for exact matching
- **OpenAI Embeddings**: `text-embedding-3-small` with 1536 dimensions
- **Detailed Reasoning**: 8k tokens for comprehensive Gemini analysis
- **Batch Processing**: Rate limit compliance with progress tracking

**Output**: `data/program_analysis_dataset.json` with:
- Original program code and model
- 1536D OpenAI embeddings
- Gemini classification with full reasoning
- Human classification labels
- API usage statistics

### Statistical Analysis (`embedding_analysis.py`)

**Purpose**: Analyze embedding patterns and classification correlations

**Features**:
- **Similarity Analysis**: Cosine similarity matrix computation
- **Class Separation**: Distance metrics between overfitting/general clusters
- **Clustering**: K-means analysis with inertia metrics
- **Visualization**: PCA, t-SNE plots with class coloring
- **Agreement Analysis**: Confusion matrix and accuracy metrics

**Outputs**:
- `data/embedding_analysis_results.json`: Detailed statistical results
- `data/embedding_visualizations.png`: PCA/t-SNE plots by classification
- `data/similarity_heatmap.png`: Program-to-program similarity matrix

### Utility Scripts

**`view_gemini_responses.py`**: Display full Gemini reasoning for each classification
**`tests/`**: Validation scripts for API connectivity and model testing

## 🔬 Analysis Methodology

### 1. **Embedding Generation**
- **Model**: OpenAI `text-embedding-3-small` 
- **Dimensions**: 1536 (2x higher than Nomic alternatives)
- **Batch Size**: 10 programs per API call for rate limit compliance
- **Processing**: Parallel batch processing with progress tracking

### 2. **AI Classification**
- **Model**: Gemini 2.5 Flash via OpenRouter
- **Reasoning**: 8k max tokens for detailed analysis
- **Criteria**: Hardcoded rules vs algorithmic patterns
- **Parallelization**: 5 concurrent workers for faster processing

### 3. **Human Labeling**
- **Ground Truth**: Programs 12 and 14 (0-indexed) = "general"
- **Remainder**: All other programs = "overfitting"
- **Rationale**: Based on algorithmic structure vs hardcoded specificity

### 4. **Statistical Analysis**
- **Class Separation**: Distance between mean embeddings
- **Similarity Distribution**: Cosine similarity statistics
- **Clustering**: K-means validation of natural groupings
- **Visualization**: Dimensionality reduction for interpretability

## 📈 Key Findings

### Perfect Classification Alignment
- **Human vs Gemini**: 100% agreement (16/16 programs)
- **Confusion Matrix**: Zero misclassifications
- **Distribution**: 14 overfitting, 2 general (both evaluators)

### Superior Embedding Quality
- **OpenAI vs Nomic**: 47% better class separation (0.0506 vs 0.0343 distance)
- **Dimensionality**: 1536D captures richer code semantics
- **Range**: Wider similarity distribution [0.6195, 1.0000] enables better discrimination

### Optimal Configuration
- **8k Reasoning Tokens**: Crucial for perfect agreement (vs 87.5% with 2k tokens)
- **Parallel Processing**: 3-5x performance improvement over sequential
- **OpenAI Embeddings**: Superior to specialized code embedding models

## 🔧 Configuration Options

### Performance Tuning
```python
# In analyze_program_embeddings.py
max_reasoning_tokens = 8000        # Gemini reasoning depth
max_workers_classify = 5           # Parallel classification workers  
max_workers_matches = 4            # Parallel exact match workers
batch_size = 10                    # OpenAI embedding batch size
```

### Model Alternatives
- **Embeddings**: Switch to `text-embedding-3-large` for even higher dimensionality
- **Classification**: Try `gpt-4` for comparison with Gemini reasoning
- **Reasoning Tokens**: Experiment with 4k-16k range for cost/quality tradeoffs

## 📊 Data Schema

### Program Analysis Dataset
```json
{
  "metadata": {
    "total_programs": 16,
    "embedding_model": "text-embedding-3-small", 
    "classification_model": "google/gemini-2.5-flash",
    "embedding_dimension": 1536,
    "exact_matches_found": 0
  },
  "programs": [
    {
      "index": 0,
      "code": "def transform(grid_lst: list[list[int]]) -> ...",
      "model": "Mistral-Large-Instruct-2407",
      "embedding": [0.021, -0.015, ...],  // 1536 dimensions
      "gemini_classification": "overfitting",
      "gemini_full_response": "This function exhibits...",
      "human_classification": "overfitting",
      "gemini_usage": {"completion_tokens": 611, ...}
    }
  ],
  "exact_matches": []
}
```

## 🎯 Future Enhancements

### 1. **Expanded Dataset**
- More program solutions across different complexity levels
- Additional programming languages and paradigms
- Cross-domain generalization testing

### 2. **Enhanced Classification**
- Multi-class labels (overfitting, general, mixed, unclear)
- Confidence scoring for classification certainty
- Domain-specific criteria (mathematical, visual, logical patterns)

### 3. **Advanced Analysis**
- **Embedding Interpretability**: Identify which dimensions correlate with overfitting
- **Feature Engineering**: Extract code metrics (complexity, abstraction levels)
- **Ensemble Methods**: Combine multiple embedding models and classifiers

### 4. **Practical Applications**
- **Code Review Assistant**: Flag potentially overfitted solutions
- **Education Tool**: Help students understand generalization principles
- **Benchmark Development**: Create evaluation metrics for code generalization

## 🔍 Troubleshooting

### API Issues
- **OpenAI Rate Limits**: Reduce `batch_size` or add delays
- **OpenRouter Timeouts**: Decrease `max_workers_classify`
- **Missing Keys**: Check `.env` file location and key names

### Performance Issues
- **Memory Usage**: Large embeddings may require chunking for massive datasets
- **Processing Time**: Adjust worker counts based on API rate limits
- **Storage**: JSON files can become large with many programs

### Analysis Issues
- **Poor Separation**: Try different embedding models or preprocessing
- **Classification Disagreement**: Increase reasoning tokens or refine prompts
- **Visualization Problems**: Check for NaN values in embeddings

## 📚 References

- **OpenAI Embeddings**: [API Documentation](https://platform.openai.com/docs/guides/embeddings)
- **Gemini 2.5 Flash**: [Model Information](https://ai.google.dev/models/gemini)
- **ARC Challenge**: [Original Paper](https://arxiv.org/abs/1911.01547)
- **Code Classification**: Research on automated code analysis and pattern recognition

---

**Created**: December 2024  
**Last Updated**: Current analysis pipeline  
**Status**: Production ready with 100% classification accuracy