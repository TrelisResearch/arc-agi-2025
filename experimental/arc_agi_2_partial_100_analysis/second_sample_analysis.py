#!/usr/bin/env python3

import os
import pickle
import numpy as np
import pandas as pd
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import multiprocessing as mp

def load_dataset():
    """Load the arc-agi-2-partial-100 dataset"""
    print("Loading dataset...")
    dataset = load_from_disk("./arc_agi_2_partial_100_data")
    return dataset['train']

def select_second_sample_tasks(dataset, sample_size=25):
    """Select a different sample of 25 tasks - use middle range this time"""
    print(f"Selecting second sample of {sample_size} tasks from {len(dataset)} total tasks...")

    # Extract task info with program counts
    task_info = []
    for idx, example in enumerate(tqdm(dataset, desc="Analyzing tasks")):
        task_id = example.get('task_id', f'task_{idx}')
        code = example.get('code', '')
        reasoning = example.get('reasoning', '')

        # Count programs/functions in code (rough estimate)
        program_count = 0
        if code and isinstance(code, str):
            # Count 'def ' occurrences as rough proxy for program count
            program_count = code.count('def ')
            # Also count lines as another measure
            line_count = len([line for line in code.split('\n') if line.strip()])
        else:
            line_count = 0

        task_info.append({
            'idx': idx,
            'task_id': task_id,
            'code': code,
            'reasoning': reasoning,
            'program_count': program_count,
            'line_count': line_count,
            'total_size': len(code) if code else 0
        })

    # Sort by program count, then by line count as tiebreaker
    task_info.sort(key=lambda x: (x['program_count'], x['line_count']), reverse=True)

    # This time, select from the MIDDLE range instead of evenly spaced
    # Take tasks from positions 5000-40000 (middle portion of dataset)
    start_idx = len(task_info) // 6  # Start at about 1/6 through
    end_idx = 5 * len(task_info) // 6  # End at about 5/6 through
    middle_tasks = task_info[start_idx:end_idx]

    # Select evenly spaced samples from this middle range
    indices = np.linspace(0, len(middle_tasks) - 1, sample_size, dtype=int)
    selected_tasks = [middle_tasks[i] for i in indices]

    print(f"Selected {len(selected_tasks)} tasks from middle range:")
    print("Top 5 by program count in sample:")
    for i, task in enumerate(selected_tasks[:5]):
        print(f"  {i+1}. Task {task['task_id']}: {task['program_count']} programs, {task['line_count']} lines")

    return selected_tasks

def compute_embeddings(codes, model_name="nomic-ai/CodeRankEmbed", max_workers=2):
    """Compute embeddings for code examples using CodeRankEmbed with limited cores"""
    print(f"Loading {model_name} model...")

    # Set number of threads for the model to use max_workers cores
    import torch
    torch.set_num_threads(max_workers)

    model = SentenceTransformer(model_name, trust_remote_code=True)

    print(f"Computing embeddings for {len(codes)} code examples using max {max_workers} cores...")

    # Process codes in smaller batches for sample
    batch_size = 8
    all_embeddings = []

    for i in tqdm(range(0, len(codes), batch_size), desc="Computing embeddings"):
        batch = codes[i:i + batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.append(batch_embeddings)

    embeddings = np.vstack(all_embeddings)

    print(f"Generated embeddings shape: {embeddings.shape}")
    return embeddings

def compute_similarity_matrix(embeddings):
    """Compute cosine similarity matrix"""
    print("Computing pairwise cosine similarities...")
    similarity_matrix = cosine_similarity(embeddings)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    return similarity_matrix

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
        for i, (idx1, idx2, sim, task1, task2) in enumerate(duplicate_pairs[:3]):
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

def save_second_sample_results(selected_tasks, embeddings, similarity_matrix, analyses, full_extrapolations, full_dataset_size):
    """Save second sample results to files"""
    print("Saving second sample results...")

    # Save sample data
    np.save('second_sample_embeddings.npy', embeddings)
    np.savez_compressed('second_sample_similarity_matrix.npz', similarity_matrix=similarity_matrix)

    # Save task info
    with open('second_selected_tasks.pkl', 'wb') as f:
        pickle.dump(selected_tasks, f)

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

    with open('second_sample_analysis.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Create summary report
    report = f"""# ARC-AGI-2 Partial 100 Dataset - Second Sample Analysis

## Dataset Overview
- Full dataset size: {full_dataset_size:,} tasks
- Second sample size: {len(selected_tasks)} tasks (selected from middle range by program count)
- Embedding model: nomic-ai/CodeRankEmbed
- Processing cores: 2 max

## Similarity Distribution (Second Sample)
- Maximum similarity found: {similarity_matrix.max():.3f}
- Mean similarity: {similarity_matrix.mean():.3f}
- Median similarity: {np.median(similarity_matrix):.3f}

## Analysis Results

"""

    for threshold in sorted(analyses.keys(), reverse=True):
        analysis = analyses[threshold]
        extrapolation = full_extrapolations[threshold]

        report += f"""### Threshold {threshold} ({threshold*100:.0f}% similarity)
- Sample: {analysis['tasks_to_remove']} tasks to remove ({analysis['reduction_percent']:.1f}% reduction)
- Full dataset estimate: {extrapolation['estimated_reduction_count']:,} tasks to remove ({extrapolation['estimated_reduction_percent']:.1f}% reduction)
- Estimated remaining: {extrapolation['estimated_remaining']:,} tasks

"""

    report += f"""## Comparison with First Sample

This second sample was selected from the middle range of tasks (by program complexity) rather than evenly spaced across the entire range. This helps validate our estimates by testing a different subset of the data.

## Files Generated
- second_sample_embeddings.npy: Second sample embeddings
- second_sample_similarity_matrix.npz: Second sample similarities
- second_selected_tasks.pkl: Second sample task info
- second_sample_analysis.pkl: Full analysis results
"""

    with open('second_sample_report.md', 'w') as f:
        f.write(report)

    print("Second sample results saved!")
    return results

def main():
    """Main analysis pipeline for second sample"""
    print("Starting ARC-AGI-2 Partial 100 SECOND SAMPLE analysis...")

    # Load dataset
    dataset = load_dataset()
    full_dataset_size = len(dataset)

    # Select second sample tasks (from middle range)
    selected_tasks = select_second_sample_tasks(dataset, sample_size=25)

    # Extract codes and task IDs from selected tasks
    codes = [task['code'] for task in selected_tasks if task['code']]
    task_ids = [task['task_id'] for task in selected_tasks if task['code']]

    if len(codes) == 0:
        print("No valid code examples found in second sample!")
        return

    print(f"Processing {len(codes)} valid code examples from second sample...")

    # Compute embeddings (using 2 cores max)
    embeddings = compute_embeddings(codes, max_workers=2)

    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(embeddings)

    # Analyze at different thresholds
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
    results = save_second_sample_results(selected_tasks, embeddings, similarity_matrix,
                                       analyses, full_extrapolations, full_dataset_size)

    print("\n" + "="*50)
    print("SECOND SAMPLE SUMMARY:")
    print(f"Sample size: {len(codes)} tasks")
    for threshold in sorted(thresholds, reverse=True):
        analysis = analyses[threshold]
        extrapolation = full_extrapolations[threshold]
        print(f"At {threshold*100:.0f}% similarity threshold:")
        print(f"  Sample: {analysis['deduplicated_size']} tasks remain ({analysis['reduction_percent']:.1f}% reduction)")
        print(f"  Full dataset estimate: {extrapolation['estimated_remaining']:,} tasks remain ({extrapolation['estimated_reduction_percent']:.1f}% reduction)")

if __name__ == "__main__":
    main()