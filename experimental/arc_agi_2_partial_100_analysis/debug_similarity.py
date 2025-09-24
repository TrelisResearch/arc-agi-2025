#!/usr/bin/env python3

import numpy as np
import pickle
from tqdm import tqdm

def analyze_similarity_matrix():
    """Debug the similarity analysis results"""
    print("Loading similarity matrix and task data...")

    # Load the similarity matrix
    sim_data = np.load('sample_similarity_matrix.npz')
    similarity_matrix = sim_data['similarity_matrix']

    # Load task info
    with open('selected_tasks.pkl', 'rb') as f:
        selected_tasks = pickle.load(f)

    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(f"Number of tasks: {len(selected_tasks)}")

    # Analyze similarity distribution
    print("\nSimilarity matrix statistics:")
    print(f"Min similarity: {similarity_matrix.min():.4f}")
    print(f"Max similarity: {similarity_matrix.max():.4f}")
    print(f"Mean similarity: {similarity_matrix.mean():.4f}")

    # Check diagonal (should be 1.0)
    diagonal_values = np.diag(similarity_matrix)
    print(f"Diagonal min: {diagonal_values.min():.4f}")
    print(f"Diagonal max: {diagonal_values.max():.4f}")

    # Exclude diagonal and find highest similarities
    n = similarity_matrix.shape[0]
    upper_triangle_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    upper_triangle_sims = similarity_matrix[upper_triangle_mask]

    print(f"\nOff-diagonal similarities:")
    print(f"Min: {upper_triangle_sims.min():.4f}")
    print(f"Max: {upper_triangle_sims.max():.4f}")
    print(f"Mean: {upper_triangle_sims.mean():.4f}")
    print(f"Median: {np.median(upper_triangle_sims):.4f}")

    # Find top 10 most similar pairs
    print(f"\nTop 10 most similar task pairs:")
    indices = np.where(upper_triangle_mask)
    similarities = similarity_matrix[indices]
    sorted_indices = np.argsort(similarities)[::-1]

    for rank, idx in enumerate(sorted_indices[:10]):
        i, j = indices[0][idx], indices[1][idx]
        sim_score = similarities[idx]
        task_i = selected_tasks[i]['task_id']
        task_j = selected_tasks[j]['task_id']
        print(f"  {rank+1}. {task_i} <-> {task_j}: {sim_score:.4f}")

    # Count pairs above different thresholds
    thresholds = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
    print(f"\nPairs above similarity thresholds:")
    for threshold in thresholds:
        count = np.sum(upper_triangle_sims >= threshold)
        print(f"  >= {threshold}: {count} pairs ({count/len(upper_triangle_sims)*100:.1f}%)")

    # Show examples of code for the most similar pairs
    print(f"\nCode examples from most similar pairs:")
    for rank, idx in enumerate(sorted_indices[:3]):
        i, j = indices[0][idx], indices[1][idx]
        sim_score = similarities[idx]
        task_i = selected_tasks[i]
        task_j = selected_tasks[j]

        print(f"\nPair {rank+1}: {task_i['task_id']} <-> {task_j['task_id']} (similarity: {sim_score:.4f})")
        print(f"Task {task_i['task_id']} code length: {len(task_i['code'])} chars, {task_i['program_count']} programs")
        print(f"Task {task_j['task_id']} code length: {len(task_j['code'])} chars, {task_j['program_count']} programs")

        # Show first few lines of each code
        code_i_lines = task_i['code'].split('\n')[:5]
        code_j_lines = task_j['code'].split('\n')[:5]

        print(f"\nTask {task_i['task_id']} (first 5 lines):")
        for line in code_i_lines:
            print(f"  {line}")

        print(f"\nTask {task_j['task_id']} (first 5 lines):")
        for line in code_j_lines:
            print(f"  {line}")

if __name__ == "__main__":
    analyze_similarity_matrix()