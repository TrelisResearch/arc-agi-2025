# ARC-AGI-2 Partial 100 Dataset Deduplication Analysis (Sample)

## Dataset Overview
- Full dataset size: 44,880 tasks
- Sample size: 25 tasks (selected by program count, evenly spaced)
- Embedding model: nomic-ai/CodeRankEmbed
- Processing cores: 2 max

## Sample Analysis Results

### Threshold 0.9 (90% similarity) - Sample
- Duplicate pairs found: 0
- Tasks to remove: 0
- Remaining tasks: 25
- Reduction: 0.0%

### Threshold 0.95 (95% similarity) - Sample
- Duplicate pairs found: 0
- Tasks to remove: 0
- Remaining tasks: 25
- Reduction: 0.0%

## Extrapolated Full Dataset Results

### Threshold 0.9 (90% similarity) - Full Dataset Estimate
- Estimated tasks to remove: 0
- Estimated remaining: 44,880
- Estimated reduction: 0.0%

### Threshold 0.95 (95% similarity) - Full Dataset Estimate
- Estimated tasks to remove: 0
- Estimated remaining: 44,880
- Estimated reduction: 0.0%

## Files Generated
- sample_embeddings.npy: Sample task embeddings
- sample_similarity_matrix.npz: Sample pairwise similarities
- selected_tasks.pkl: Selected sample task info
- sample_deduplication_analysis.pkl: Full analysis results

## Note
Results are extrapolated from a representative sample. Actual full dataset results may vary.
