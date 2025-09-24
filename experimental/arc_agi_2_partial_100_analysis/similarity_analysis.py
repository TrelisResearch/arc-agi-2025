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

def extract_code_from_dataset(dataset):
    """Extract code strings from the dataset for embedding"""
    print(f"Extracting code from {len(dataset)} examples...")
    codes = []
    task_ids = []

    for idx, example in enumerate(tqdm(dataset)):
        task_id = example.get('task_id', f'task_{idx}')
        code = example.get('code', '')

        if code and isinstance(code, str) and len(code.strip()) > 0:
            codes.append(code)
            task_ids.append(task_id)

    print(f"Found {len(codes)} valid code examples")
    return codes, task_ids

def compute_embeddings(codes, model_name="nomic-ai/CodeRankEmbed", max_workers=6):
    """Compute embeddings for all code examples using CodeRankEmbed"""
    print(f"Loading {model_name} model...")

    # Set number of threads for the model to use max_workers cores
    import torch
    torch.set_num_threads(max_workers)

    model = SentenceTransformer(model_name, trust_remote_code=True)

    print(f"Computing embeddings for {len(codes)} code examples using max {max_workers} cores...")

    # Process codes in batches to manage memory
    batch_size = 32
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
                duplicate_pairs.append((i, j, similarity_matrix[i, j]))
                to_remove.add(j)  # Remove the second occurrence

    unique_tasks = n_tasks - len(to_remove)

    print(f"Original dataset size: {n_tasks}")
    print(f"Duplicate pairs found: {len(duplicate_pairs)}")
    print(f"Tasks to remove: {len(to_remove)}")
    print(f"Remaining after deduplication: {unique_tasks}")
    print(f"Reduction: {len(to_remove)} tasks ({len(to_remove)/n_tasks*100:.1f}%)")

    return {
        'original_size': n_tasks,
        'duplicate_pairs': len(duplicate_pairs),
        'tasks_to_remove': len(to_remove),
        'deduplicated_size': unique_tasks,
        'reduction_count': len(to_remove),
        'reduction_percent': len(to_remove)/n_tasks*100,
        'pairs': duplicate_pairs[:10]  # Store first 10 pairs as examples
    }

def save_results(embeddings, similarity_matrix, task_ids, analysis_09, analysis_095):
    """Save all results to files"""
    print("Saving results...")

    # Save embeddings
    np.save('embeddings.npy', embeddings)

    # Save similarity matrix (compressed)
    np.savez_compressed('similarity_matrix.npz', similarity_matrix=similarity_matrix)

    # Save task IDs
    with open('task_ids.pkl', 'wb') as f:
        pickle.dump(task_ids, f)

    # Save analysis results
    results = {
        'threshold_0.9': analysis_09,
        'threshold_0.95': analysis_095,
        'total_tasks': len(task_ids)
    }

    with open('deduplication_analysis.pkl', 'wb') as f:
        pickle.dump(results, f)

    # Create summary report
    report = f"""# ARC-AGI-2 Partial 100 Dataset Deduplication Analysis

## Dataset Overview
- Total tasks: {len(task_ids):,}
- Embedding model: nomic-ai/CodeRankEmbed
- Processing cores: 6 max

## Similarity Analysis Results

### Threshold 0.9 (90% similarity)
- Duplicate pairs found: {analysis_09['duplicate_pairs']:,}
- Tasks to remove: {analysis_09['tasks_to_remove']:,}
- Remaining tasks: {analysis_09['deduplicated_size']:,}
- Reduction: {analysis_09['reduction_percent']:.1f}%

### Threshold 0.95 (95% similarity)
- Duplicate pairs found: {analysis_095['duplicate_pairs']:,}
- Tasks to remove: {analysis_095['tasks_to_remove']:,}
- Remaining tasks: {analysis_095['deduplicated_size']:,}
- Reduction: {analysis_095['reduction_percent']:.1f}%

## Files Generated
- embeddings.npy: Task embeddings
- similarity_matrix.npz: Pairwise similarities (compressed)
- task_ids.pkl: Task identifiers
- deduplication_analysis.pkl: Full analysis results
"""

    with open('analysis_report.md', 'w') as f:
        f.write(report)

    print("Results saved!")
    return results

def main():
    """Main analysis pipeline"""
    print("Starting ARC-AGI-2 Partial 100 similarity analysis...")

    # Load dataset
    dataset = load_dataset()

    # Extract code
    codes, task_ids = extract_code_from_dataset(dataset)

    if len(codes) == 0:
        print("No valid code examples found!")
        return

    # Compute embeddings
    embeddings = compute_embeddings(codes, max_workers=6)

    # Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(embeddings)

    # Analyze at different thresholds
    print("\n" + "="*50)
    analysis_09 = analyze_duplicates(similarity_matrix, task_ids, threshold=0.9)

    print("\n" + "="*50)
    analysis_095 = analyze_duplicates(similarity_matrix, task_ids, threshold=0.95)

    # Save results
    print("\n" + "="*50)
    results = save_results(embeddings, similarity_matrix, task_ids, analysis_09, analysis_095)

    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"Original dataset: {len(task_ids):,} tasks")
    print(f"At 90% similarity threshold: {analysis_09['deduplicated_size']:,} tasks remain ({analysis_09['reduction_percent']:.1f}% reduction)")
    print(f"At 95% similarity threshold: {analysis_095['deduplicated_size']:,} tasks remain ({analysis_095['reduction_percent']:.1f}% reduction)")

if __name__ == "__main__":
    main()