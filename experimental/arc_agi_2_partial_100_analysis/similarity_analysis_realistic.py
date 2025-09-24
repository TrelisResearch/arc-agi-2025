#!/usr/bin/env python3

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def load_existing_data():
    """Load the existing similarity analysis results"""
    print("Loading existing similarity analysis results...")

    # Load similarity matrix
    sim_data = np.load('sample_similarity_matrix.npz')
    similarity_matrix = sim_data['similarity_matrix']

    # Load task info
    with open('selected_tasks.pkl', 'rb') as f:
        selected_tasks = pickle.load(f)

    task_ids = [task['task_id'] for task in selected_tasks]

    return similarity_matrix, task_ids, selected_tasks

def analyze_duplicates(similarity_matrix, task_ids, threshold=0.8):
    """Analyze duplicates based on similarity threshold"""
    print(f"Analyzing duplicates at similarity threshold {threshold}")

    n_tasks = len(task_ids)
    to_remove = set()
    duplicate_pairs = []

    # Find duplicate pairs above threshold (excluding diagonal)
    for i in range(n_tasks):
        for j in range(i + 1, n_tasks):
            if similarity_matrix[i, j] >= threshold:
                duplicate_pairs.append((i, j, similarity_matrix[i, j], task_ids[i], task_ids[j]))
                to_remove.add(j)  # Remove the second occurrence

    unique_tasks = n_tasks - len(to_remove)

    print(f"Sample size: {n_tasks}")
    print(f"Duplicate pairs found: {len(duplicate_pairs)}")
    print(f"Tasks to remove: {len(to_remove)}")
    print(f"Remaining after deduplication: {unique_tasks}")
    print(f"Reduction: {len(to_remove)} tasks ({len(to_remove)/n_tasks*100:.1f}%)")

    # Show some example pairs
    if duplicate_pairs:
        print("\nExample duplicate pairs:")
        for i, (idx1, idx2, sim, task1, task2) in enumerate(duplicate_pairs[:5]):
            print(f"  Pair {i+1}: {task1} <-> {task2} (similarity: {sim:.3f})")

    return {
        'original_size': n_tasks,
        'duplicate_pairs': len(duplicate_pairs),
        'tasks_to_remove': len(to_remove),
        'deduplicated_size': unique_tasks,
        'reduction_count': len(to_remove),
        'reduction_percent': len(to_remove)/n_tasks*100 if n_tasks > 0 else 0,
        'pairs': duplicate_pairs
    }

def extrapolate_to_full_dataset(sample_analysis, sample_size, full_dataset_size):
    """Extrapolate sample results to estimate full dataset impact"""
    print(f"\nExtrapolating results from sample of {sample_size} to full dataset of {full_dataset_size}...")

    sample_reduction_rate = sample_analysis['reduction_percent'] / 100
    estimated_full_reduction = int(full_dataset_size * sample_reduction_rate)
    estimated_remaining = full_dataset_size - estimated_full_reduction

    return {
        'estimated_reduction_count': estimated_full_reduction,
        'estimated_remaining': estimated_remaining,
        'estimated_reduction_percent': sample_reduction_rate * 100
    }

def save_realistic_results(similarity_matrix, task_ids, selected_tasks, analyses, full_extrapolations, full_dataset_size):
    """Save realistic threshold analysis results"""
    print("Saving realistic threshold results...")

    # Save analysis results
    results = {
        'sample_analysis': analyses,
        'full_dataset_extrapolation': full_extrapolations,
        'sample_size': len(selected_tasks),
        'original_full_size': full_dataset_size,
        'similarity_stats': {
            'max_similarity': float(similarity_matrix.max()),
            'mean_similarity': float(similarity_matrix.mean()),
            'median_similarity': float(np.median(similarity_matrix))
        }
    }

    with open('realistic_deduplication_analysis.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Create comprehensive report
    report = f"""# ARC-AGI-2 Partial 100 Dataset Deduplication Analysis (Realistic Thresholds)

## Dataset Overview
- Full dataset size: {full_dataset_size:,} tasks
- Sample size: {len(selected_tasks)} tasks (selected by program count, evenly spaced)
- Embedding model: nomic-ai/CodeRankEmbed
- Processing cores: 2 max

## Similarity Distribution
- Maximum similarity found: {similarity_matrix.max():.3f}
- Mean similarity: {similarity_matrix.mean():.3f}
- Median similarity: {np.median(similarity_matrix):.3f}

## Sample Analysis Results

"""

    for threshold in sorted(analyses.keys()):
        analysis = analyses[threshold]
        extrapolation = full_extrapolations[threshold]

        report += f"""### Threshold {threshold} ({threshold*100:.0f}% similarity) - Sample
- Duplicate pairs found: {analysis['duplicate_pairs']}
- Tasks to remove: {analysis['tasks_to_remove']}
- Remaining tasks: {analysis['deduplicated_size']}
- Reduction: {analysis['reduction_percent']:.1f}%

### Threshold {threshold} ({threshold*100:.0f}% similarity) - Full Dataset Estimate
- Estimated tasks to remove: {extrapolation['estimated_reduction_count']:,}
- Estimated remaining: {extrapolation['estimated_remaining']:,}
- Estimated reduction: {extrapolation['estimated_reduction_percent']:.1f}%

"""

    report += f"""## Summary

Based on the sample analysis, the dataset shows varying levels of similarity:

"""
    for threshold in sorted(analyses.keys(), reverse=True):
        extrapolation = full_extrapolations[threshold]
        report += f"- At **{threshold*100:.0f}% similarity**: Estimated **{extrapolation['estimated_remaining']:,}** tasks remain ({extrapolation['estimated_reduction_percent']:.1f}% reduction)\n"

    report += f"""

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
"""

    with open('realistic_analysis_report.md', 'w') as f:
        f.write(report)

    print("Realistic threshold analysis saved!")
    return results

def main():
    """Main analysis with realistic thresholds"""
    print("Starting realistic threshold analysis for ARC-AGI-2 Partial 100...")

    # Load existing data
    similarity_matrix, task_ids, selected_tasks = load_existing_data()
    full_dataset_size = 44880  # From previous analysis

    # Test realistic thresholds based on observed similarity distribution
    thresholds = [0.85, 0.8, 0.75, 0.7]

    analyses = {}
    full_extrapolations = {}

    for threshold in thresholds:
        print("\n" + "="*50)
        analysis = analyze_duplicates(similarity_matrix, task_ids, threshold=threshold)
        analyses[threshold] = analysis

        full_extrapolation = extrapolate_to_full_dataset(analysis, len(selected_tasks), full_dataset_size)
        full_extrapolations[threshold] = full_extrapolation

    # Save results
    print("\n" + "="*50)
    results = save_realistic_results(similarity_matrix, task_ids, selected_tasks,
                                   analyses, full_extrapolations, full_dataset_size)

    print("\n" + "="*50)
    print("REALISTIC THRESHOLD SUMMARY:")
    print(f"Sample size: {len(task_ids)} tasks")
    for threshold in sorted(thresholds, reverse=True):
        analysis = analyses[threshold]
        extrapolation = full_extrapolations[threshold]
        print(f"At {threshold*100:.0f}% similarity threshold:")
        print(f"  Sample: {analysis['deduplicated_size']} tasks remain ({analysis['reduction_percent']:.1f}% reduction)")
        print(f"  Full dataset estimate: {extrapolation['estimated_remaining']:,} tasks remain ({extrapolation['estimated_reduction_percent']:.1f}% reduction)")

if __name__ == "__main__":
    main()