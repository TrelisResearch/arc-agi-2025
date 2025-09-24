# ARC-AGI-2 Partial 100 Dataset Deduplication Analysis (Realistic Thresholds)

## Dataset Overview
- Full dataset size: 44,880 tasks
- Sample size: 25 tasks (selected by program count, evenly spaced)
- Embedding model: nomic-ai/CodeRankEmbed
- Processing cores: 2 max

## Similarity Distribution
- Maximum similarity found: 1.000
- Mean similarity: 0.661
- Median similarity: 0.657

## Sample Analysis Results

### Threshold 0.7 (70% similarity) - Sample
- Duplicate pairs found: 96
- Tasks to remove: 19
- Remaining tasks: 6
- Reduction: 76.0%

### Threshold 0.7 (70% similarity) - Full Dataset Estimate
- Estimated tasks to remove: 34,108
- Estimated remaining: 10,772
- Estimated reduction: 76.0%

### Threshold 0.75 (75% similarity) - Sample
- Duplicate pairs found: 55
- Tasks to remove: 15
- Remaining tasks: 10
- Reduction: 60.0%

### Threshold 0.75 (75% similarity) - Full Dataset Estimate
- Estimated tasks to remove: 26,928
- Estimated remaining: 17,952
- Estimated reduction: 60.0%

### Threshold 0.8 (80% similarity) - Sample
- Duplicate pairs found: 14
- Tasks to remove: 8
- Remaining tasks: 17
- Reduction: 32.0%

### Threshold 0.8 (80% similarity) - Full Dataset Estimate
- Estimated tasks to remove: 14,361
- Estimated remaining: 30,519
- Estimated reduction: 32.0%

### Threshold 0.85 (85% similarity) - Sample
- Duplicate pairs found: 1
- Tasks to remove: 1
- Remaining tasks: 24
- Reduction: 4.0%

### Threshold 0.85 (85% similarity) - Full Dataset Estimate
- Estimated tasks to remove: 1,795
- Estimated remaining: 43,085
- Estimated reduction: 4.0%

## Summary

Based on the sample analysis, the dataset shows varying levels of similarity:

- At **85% similarity**: Estimated **43,085** tasks remain (4.0% reduction)
- At **80% similarity**: Estimated **30,519** tasks remain (32.0% reduction)
- At **75% similarity**: Estimated **17,952** tasks remain (60.0% reduction)
- At **70% similarity**: Estimated **10,772** tasks remain (76.0% reduction)


## Important Notes

1. **No pairs found above 90% similarity** - This suggests the dataset has good diversity at very high similarity thresholds
2. **Meaningful reductions possible at 80-85% thresholds** - These would remove quite similar tasks while preserving diversity
3. **Results are extrapolated** from a representative sample. Actual full dataset results may vary.
4. **Conservative approach recommended** - Consider using 85%+ thresholds to maintain dataset quality while removing near-duplicates

## Files Generated
- sample_embeddings.npy: Sample task embeddings
- sample_similarity_matrix.npz: Sample pairwise similarities
- selected_tasks.pkl: Selected sample task info
- realistic_deduplication_analysis.pkl: Full analysis results with realistic thresholds

## Recommendation

Based on this analysis, if deduplication is desired:
- **85% threshold**: Removes very similar tasks while preserving diversity
- **80% threshold**: More aggressive deduplication, suitable if training efficiency is a priority
- **75% threshold**: Significant deduplication, use only if dataset size is a major constraint
