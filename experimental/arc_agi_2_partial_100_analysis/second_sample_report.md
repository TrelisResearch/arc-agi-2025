# ARC-AGI-2 Partial 100 Dataset - Second Sample Analysis

## Dataset Overview
- Full dataset size: 44,880 tasks
- Second sample size: 25 tasks (selected from middle range by program count)
- Embedding model: nomic-ai/CodeRankEmbed
- Processing cores: 2 max

## Similarity Distribution (Second Sample)
- Maximum similarity found: 1.000
- Mean similarity: 0.718
- Median similarity: 0.719

## Analysis Results

### Threshold 0.85 (85% similarity)
- Sample: 5 tasks to remove (20.0% reduction)
- Full dataset estimate: 8,976 tasks to remove (20.0% reduction)
- Estimated remaining: 35,904 tasks

### Threshold 0.8 (80% similarity)
- Sample: 14 tasks to remove (56.0% reduction)
- Full dataset estimate: 25,132 tasks to remove (56.0% reduction)
- Estimated remaining: 19,748 tasks

### Threshold 0.75 (75% similarity)
- Sample: 19 tasks to remove (76.0% reduction)
- Full dataset estimate: 34,108 tasks to remove (76.0% reduction)
- Estimated remaining: 10,772 tasks

### Threshold 0.7 (70% similarity)
- Sample: 23 tasks to remove (92.0% reduction)
- Full dataset estimate: 41,289 tasks to remove (92.0% reduction)
- Estimated remaining: 3,591 tasks

## Comparison with First Sample

This second sample was selected from the middle range of tasks (by program complexity) rather than evenly spaced across the entire range. This helps validate our estimates by testing a different subset of the data.

## Files Generated
- second_sample_embeddings.npy: Second sample embeddings
- second_sample_similarity_matrix.npz: Second sample similarities
- second_selected_tasks.pkl: Second sample task info
- second_sample_analysis.pkl: Full analysis results
