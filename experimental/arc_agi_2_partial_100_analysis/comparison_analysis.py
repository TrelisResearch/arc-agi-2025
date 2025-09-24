#!/usr/bin/env python3

import pickle
import numpy as np

def load_both_samples():
    """Load results from both sample analyses"""
    print("Loading both sample analysis results...")

    # Load first sample
    with open('realistic_deduplication_analysis.pkl', 'rb') as f:
        first_results = pickle.load(f)

    # Load second sample
    with open('second_sample_analysis.pkl', 'rb') as f:
        second_results = pickle.load(f)

    return first_results, second_results

def create_comparison_report(first_results, second_results):
    """Create comprehensive comparison report"""

    report = f"""# ARC-AGI-2 Partial 100 Dataset Deduplication Analysis - Final Report

## Executive Summary

We analyzed two independent samples of 25 tasks each from the ARC-AGI-2 Partial 100 dataset (44,880 total tasks) using CodeRankEmbed embeddings to estimate deduplication impact at various similarity thresholds.

## Sample Selection Strategy

- **First Sample**: Evenly spaced selection across all tasks, sorted by program complexity
- **Second Sample**: Middle-range selection (tasks ranked roughly 1/6 to 5/6 by complexity)

## Similarity Distribution Comparison

| Metric | First Sample | Second Sample |
|--------|--------------|---------------|
| Max Similarity | {first_results['similarity_stats']['max_similarity']:.3f} | {second_results['similarity_stats']['max_similarity']:.3f} |
| Mean Similarity | {first_results['similarity_stats']['mean_similarity']:.3f} | {second_results['similarity_stats']['mean_similarity']:.3f} |
| Median Similarity | {first_results['similarity_stats']['median_similarity']:.3f} | {second_results['similarity_stats']['median_similarity']:.3f} |

## Deduplication Impact Estimates

### 85% Similarity Threshold (Conservative)
| Sample | Tasks Removed | Remaining | Reduction % |
|---------|---------------|-----------|-------------|
| First | {first_results['full_dataset_extrapolation'][0.85]['estimated_reduction_count']:,} | {first_results['full_dataset_extrapolation'][0.85]['estimated_remaining']:,} | {first_results['full_dataset_extrapolation'][0.85]['estimated_reduction_percent']:.1f}% |
| Second | {second_results['full_dataset_extrapolation'][0.85]['estimated_reduction_count']:,} | {second_results['full_dataset_extrapolation'][0.85]['estimated_remaining']:,} | {second_results['full_dataset_extrapolation'][0.85]['estimated_reduction_percent']:.1f}% |

### 80% Similarity Threshold (Moderate)
| Sample | Tasks Removed | Remaining | Reduction % |
|---------|---------------|-----------|-------------|
| First | {first_results['full_dataset_extrapolation'][0.8]['estimated_reduction_count']:,} | {first_results['full_dataset_extrapolation'][0.8]['estimated_remaining']:,} | {first_results['full_dataset_extrapolation'][0.8]['estimated_reduction_percent']:.1f}% |
| Second | {second_results['full_dataset_extrapolation'][0.8]['estimated_reduction_count']:,} | {second_results['full_dataset_extrapolation'][0.8]['estimated_remaining']:,} | {second_results['full_dataset_extrapolation'][0.8]['estimated_reduction_percent']:.1f}% |

### 75% Similarity Threshold (Aggressive)
| Sample | Tasks Removed | Remaining | Reduction % |
|---------|---------------|-----------|-------------|
| First | {first_results['full_dataset_extrapolation'][0.75]['estimated_reduction_count']:,} | {first_results['full_dataset_extrapolation'][0.75]['estimated_remaining']:,} | {first_results['full_dataset_extrapolation'][0.75]['estimated_reduction_percent']:.1f}% |
| Second | {second_results['full_dataset_extrapolation'][0.75]['estimated_reduction_count']:,} | {second_results['full_dataset_extrapolation'][0.75]['estimated_remaining']:,} | {second_results['full_dataset_extrapolation'][0.75]['estimated_reduction_percent']:.1f}% |

## Key Findings

1. **No High Similarity Pairs**: Neither sample found any pairs above 90% similarity, confirming good dataset diversity at very high thresholds.

2. **Consistency Between Samples**: While exact numbers vary, both samples show similar patterns:
   - Low reduction at 85% threshold ({first_results['full_dataset_extrapolation'][0.85]['estimated_reduction_percent']:.1f}% vs {second_results['full_dataset_extrapolation'][0.85]['estimated_reduction_percent']:.1f}%)
   - Moderate reduction at 80% threshold ({first_results['full_dataset_extrapolation'][0.8]['estimated_reduction_percent']:.1f}% vs {second_results['full_dataset_extrapolation'][0.8]['estimated_reduction_percent']:.1f}%)
   - Significant reduction at 75% threshold ({first_results['full_dataset_extrapolation'][0.75]['estimated_reduction_percent']:.1f}% vs {second_results['full_dataset_extrapolation'][0.75]['estimated_reduction_percent']:.1f}%)

3. **Sample Variation**: The second sample (middle-complexity tasks) showed higher similarity overall, suggesting that tasks of similar complexity tend to be more similar to each other.

## Estimated Dataset Sizes After Deduplication

Based on the average of both samples:

| Threshold | Avg. Remaining Tasks | Avg. Reduction |
|-----------|---------------------|----------------|
| **85%** | **{(first_results['full_dataset_extrapolation'][0.85]['estimated_remaining'] + second_results['full_dataset_extrapolation'][0.85]['estimated_remaining'])//2:,}** | **{(first_results['full_dataset_extrapolation'][0.85]['estimated_reduction_percent'] + second_results['full_dataset_extrapolation'][0.85]['estimated_reduction_percent'])/2:.1f}%** |
| **80%** | **{(first_results['full_dataset_extrapolation'][0.8]['estimated_remaining'] + second_results['full_dataset_extrapolation'][0.8]['estimated_remaining'])//2:,}** | **{(first_results['full_dataset_extrapolation'][0.8]['estimated_reduction_percent'] + second_results['full_dataset_extrapolation'][0.8]['estimated_reduction_percent'])/2:.1f}%** |
| **75%** | **{(first_results['full_dataset_extrapolation'][0.75]['estimated_remaining'] + second_results['full_dataset_extrapolation'][0.75]['estimated_remaining'])//2:,}** | **{(first_results['full_dataset_extrapolation'][0.75]['estimated_reduction_percent'] + second_results['full_dataset_extrapolation'][0.75]['estimated_reduction_percent'])/2:.1f}%** |

## Recommendations

1. **For Training Efficiency**: Use **80% similarity threshold** → ~**{(first_results['full_dataset_extrapolation'][0.8]['estimated_remaining'] + second_results['full_dataset_extrapolation'][0.8]['estimated_remaining'])//2:,} tasks** (significant reduction while preserving diversity)

2. **For Maximum Quality**: Use **85% similarity threshold** → ~**{(first_results['full_dataset_extrapolation'][0.85]['estimated_remaining'] + second_results['full_dataset_extrapolation'][0.85]['estimated_remaining'])//2:,} tasks** (minimal reduction, removes only very similar tasks)

3. **For Storage Constraints**: Use **75% similarity threshold** → ~**{(first_results['full_dataset_extrapolation'][0.75]['estimated_remaining'] + second_results['full_dataset_extrapolation'][0.75]['estimated_remaining'])//2:,} tasks** (aggressive reduction, use with caution)

## Methodology Notes

- **Embedding Model**: nomic-ai/CodeRankEmbed (768-dimensional embeddings)
- **Similarity Metric**: Cosine similarity
- **Deduplication Strategy**: Greedy removal (keeps first occurrence in similarity-sorted order)
- **Sample Size**: 50 tasks total (2 samples of 25 each)
- **Confidence**: Medium to High (consistent patterns across different samples)

## Important Limitations

1. **Sample Size**: Results are extrapolated from 50 tasks out of 44,880
2. **Selection Bias**: Samples may not perfectly represent full dataset diversity
3. **Threshold Sensitivity**: Small changes in threshold can have large impacts
4. **Greedy Strategy**: More sophisticated deduplication strategies might yield different results

## Files Generated

- `realistic_analysis_report.md`: First sample detailed analysis
- `second_sample_report.md`: Second sample detailed analysis
- `realistic_deduplication_analysis.pkl`: First sample results
- `second_sample_analysis.pkl`: Second sample results
- Various embedding and similarity matrix files for both samples
"""

    return report

def main():
    """Generate final comparison analysis"""
    print("Creating comprehensive comparison analysis...")

    first_results, second_results = load_both_samples()

    report = create_comparison_report(first_results, second_results)

    with open('final_deduplication_analysis.md', 'w') as f:
        f.write(report)

    print("Final analysis report saved to: final_deduplication_analysis.md")

    # Print summary to console
    print("\n" + "="*70)
    print("FINAL SUMMARY - ARC-AGI-2 PARTIAL 100 DEDUPLICATION ANALYSIS")
    print("="*70)
    print(f"Original dataset: 44,880 tasks")
    print("\nEstimated dataset sizes after deduplication (average of both samples):")

    avg_85 = (first_results['full_dataset_extrapolation'][0.85]['estimated_remaining'] +
              second_results['full_dataset_extrapolation'][0.85]['estimated_remaining']) // 2
    avg_80 = (first_results['full_dataset_extrapolation'][0.8]['estimated_remaining'] +
              second_results['full_dataset_extrapolation'][0.8]['estimated_remaining']) // 2
    avg_75 = (first_results['full_dataset_extrapolation'][0.75]['estimated_remaining'] +
              second_results['full_dataset_extrapolation'][0.75]['estimated_remaining']) // 2

    avg_red_85 = (first_results['full_dataset_extrapolation'][0.85]['estimated_reduction_percent'] +
                  second_results['full_dataset_extrapolation'][0.85]['estimated_reduction_percent']) / 2
    avg_red_80 = (first_results['full_dataset_extrapolation'][0.8]['estimated_reduction_percent'] +
                  second_results['full_dataset_extrapolation'][0.8]['estimated_reduction_percent']) / 2
    avg_red_75 = (first_results['full_dataset_extrapolation'][0.75]['estimated_reduction_percent'] +
                  second_results['full_dataset_extrapolation'][0.75]['estimated_reduction_percent']) / 2

    print(f"• 85% threshold: {avg_85:,} tasks ({avg_red_85:.1f}% reduction)")
    print(f"• 80% threshold: {avg_80:,} tasks ({avg_red_80:.1f}% reduction)  ← RECOMMENDED")
    print(f"• 75% threshold: {avg_75:,} tasks ({avg_red_75:.1f}% reduction)")
    print("="*70)

if __name__ == "__main__":
    main()