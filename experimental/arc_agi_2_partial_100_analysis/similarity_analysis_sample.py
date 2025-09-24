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

def select_sample_tasks(dataset, sample_size=25):
    """Select sample tasks by sorting by program count and taking evenly spaced samples"""
    print(f"Selecting {sample_size} sample tasks from {len(dataset)} total tasks...")

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

    # Select evenly spaced samples
    total_tasks = len(task_info)
    indices = np.linspace(0, total_tasks - 1, sample_size, dtype=int)
    selected_tasks = [task_info[i] for i in indices]

    print(f"Selected {len(selected_tasks)} tasks:")
    print("Top 5 by program count:")
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

def analyze_duplicates(similarity_matrix, task_ids, threshold=0.9):
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

def save_results(selected_tasks, embeddings, similarity_matrix, analysis_09, analysis_095, full_extrapolation_09, full_extrapolation_095, full_dataset_size):
    """Save all results to files"""
    print("Saving results...")

    # Save sample data
    np.save('sample_embeddings.npy', embeddings)
    np.savez_compressed('sample_similarity_matrix.npz', similarity_matrix=similarity_matrix)

    # Save task info
    with open('selected_tasks.pkl', 'wb') as f:
        pickle.dump(selected_tasks, f)

    # Save analysis results
    results = {
        'sample_analysis': {
            'threshold_0.9': analysis_09,
            'threshold_0.95': analysis_095,
            'sample_size': len(selected_tasks)
        },
        'full_dataset_extrapolation': {
            'threshold_0.9': full_extrapolation_09,
            'threshold_0.95': full_extrapolation_095,
            'original_full_size': full_dataset_size
        }
    }

    with open('sample_deduplication_analysis.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Create summary report
    report = f"""# ARC-AGI-2 Partial 100 Dataset Deduplication Analysis (Sample)

## Dataset Overview
- Full dataset size: {full_dataset_size:,} tasks
- Sample size: {len(selected_tasks)} tasks (selected by program count, evenly spaced)
- Embedding model: nomic-ai/CodeRankEmbed
- Processing cores: 2 max

## Sample Analysis Results

### Threshold 0.9 (90% similarity) - Sample
- Duplicate pairs found: {analysis_09['duplicate_pairs']}
- Tasks to remove: {analysis_09['tasks_to_remove']}
- Remaining tasks: {analysis_09['deduplicated_size']}
- Reduction: {analysis_09['reduction_percent']:.1f}%

### Threshold 0.95 (95% similarity) - Sample
- Duplicate pairs found: {analysis_095['duplicate_pairs']}
- Tasks to remove: {analysis_095['tasks_to_remove']}
- Remaining tasks: {analysis_095['deduplicated_size']}
- Reduction: {analysis_095['reduction_percent']:.1f}%

## Extrapolated Full Dataset Results

### Threshold 0.9 (90% similarity) - Full Dataset Estimate
- Estimated tasks to remove: {full_extrapolation_09['estimated_reduction_count']:,}
- Estimated remaining: {full_extrapolation_09['estimated_remaining']:,}
- Estimated reduction: {full_extrapolation_09['estimated_reduction_percent']:.1f}%

### Threshold 0.95 (95% similarity) - Full Dataset Estimate
- Estimated tasks to remove: {full_extrapolation_095['estimated_reduction_count']:,}
- Estimated remaining: {full_extrapolation_095['estimated_remaining']:,}
- Estimated reduction: {full_extrapolation_095['estimated_reduction_percent']:.1f}%

## Files Generated
- sample_embeddings.npy: Sample task embeddings
- sample_similarity_matrix.npz: Sample pairwise similarities
- selected_tasks.pkl: Selected sample task info
- sample_deduplication_analysis.pkl: Full analysis results

## Note
Results are extrapolated from a representative sample. Actual full dataset results may vary.
"""

    with open('sample_analysis_report.md', 'w') as f:
        f.write(report)

    print("Results saved!")
    return results

def main():
    """Main analysis pipeline for sample"""
    print("Starting ARC-AGI-2 Partial 100 similarity analysis (SAMPLE)...")

    # Load dataset
    dataset = load_dataset()
    full_dataset_size = len(dataset)

    # Select sample tasks
    selected_tasks = select_sample_tasks(dataset, sample_size=25)

    # Extract codes and task IDs from selected tasks
    codes = [task['code'] for task in selected_tasks if task['code']]
    task_ids = [task['task_id'] for task in selected_tasks if task['code']]

    if len(codes) == 0:
        print("No valid code examples found in sample!")
        return

    print(f"Processing {len(codes)} valid code examples from sample...")

    # Compute embeddings (using 2 cores max)
    embeddings = compute_embeddings(codes, max_workers=2)

    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(embeddings)

    # Analyze at different thresholds
    print("\n" + "="*50)
    analysis_09 = analyze_duplicates(similarity_matrix, task_ids, threshold=0.9)

    print("\n" + "="*50)
    analysis_095 = analyze_duplicates(similarity_matrix, task_ids, threshold=0.95)

    # Extrapolate to full dataset
    print("\n" + "="*50)
    full_extrapolation_09 = extrapolate_to_full_dataset(analysis_09, len(selected_tasks), full_dataset_size)
    full_extrapolation_095 = extrapolate_to_full_dataset(analysis_095, len(selected_tasks), full_dataset_size)

    # Save results
    print("\n" + "="*50)
    results = save_results(selected_tasks, embeddings, similarity_matrix, analysis_09, analysis_095,
                          full_extrapolation_09, full_extrapolation_095, full_dataset_size)

    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"Sample analysis: {len(codes)} tasks")
    print(f"At 90% similarity threshold:")
    print(f"  Sample: {analysis_09['deduplicated_size']} tasks remain ({analysis_09['reduction_percent']:.1f}% reduction)")
    print(f"  Full dataset estimate: {full_extrapolation_09['estimated_remaining']:,} tasks remain ({full_extrapolation_09['estimated_reduction_percent']:.1f}% reduction)")
    print(f"At 95% similarity threshold:")
    print(f"  Sample: {analysis_095['deduplicated_size']} tasks remain ({analysis_095['reduction_percent']:.1f}% reduction)")
    print(f"  Full dataset estimate: {full_extrapolation_095['estimated_remaining']:,} tasks remain ({full_extrapolation_095['estimated_reduction_percent']:.1f}% reduction)")

if __name__ == "__main__":
    main()