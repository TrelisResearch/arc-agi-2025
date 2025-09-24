#!/usr/bin/env python3

import os
import pickle
import numpy as np
import pandas as pd
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from collections import defaultdict, Counter

def load_dataset():
    """Load the arc-agi-2-partial-100 dataset"""
    print("Loading dataset...")
    dataset = load_from_disk("./arc_agi_2_partial_100_data")
    return dataset['train']

def group_programs_by_task(dataset):
    """Group all programs by their task_id"""
    print("Grouping programs by task_id...")

    task_programs = defaultdict(list)

    for idx, example in enumerate(tqdm(dataset, desc="Processing examples")):
        task_id = example['task_id']
        code = example.get('code', '')

        if code and isinstance(code, str) and len(code.strip()) > 0:
            task_programs[task_id].append({
                'idx': idx,
                'code': code,
                'task_id': task_id
            })

    print(f"Found {len(task_programs)} unique tasks")

    # Show task statistics
    program_counts = [len(programs) for programs in task_programs.values()]
    print(f"Programs per task - Min: {min(program_counts)}, Max: {max(program_counts)}, Mean: {np.mean(program_counts):.1f}")

    return task_programs

def select_sample_tasks(task_programs, sample_size=25):
    """Select sample tasks with varying numbers of programs"""
    print(f"Selecting {sample_size} sample tasks...")

    # Sort tasks by number of programs
    tasks_by_program_count = [(task_id, len(programs)) for task_id, programs in task_programs.items()]
    tasks_by_program_count.sort(key=lambda x: x[1], reverse=True)

    # Select a mix: some high-program tasks, some medium, some low
    selected_tasks = []
    total_tasks = len(tasks_by_program_count)

    # Take evenly spaced samples across the range
    indices = np.linspace(0, total_tasks - 1, sample_size, dtype=int)
    selected_task_ids = [tasks_by_program_count[i][0] for i in indices]

    for task_id in selected_task_ids:
        selected_tasks.append((task_id, task_programs[task_id]))

    print("Selected tasks:")
    for i, (task_id, programs) in enumerate(selected_tasks):
        print(f"  {i+1}. Task {task_id}: {len(programs)} programs")

    return selected_tasks

def compute_embeddings_for_task(programs, model, task_id):
    """Compute embeddings for all programs in a single task"""
    codes = [prog['code'] for prog in programs]

    # Process in batches
    batch_size = 8
    all_embeddings = []

    for i in range(0, len(codes), batch_size):
        batch = codes[i:i + batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.append(batch_embeddings)

    if all_embeddings:
        embeddings = np.vstack(all_embeddings)
    else:
        embeddings = np.array([])

    return embeddings

def analyze_within_task_duplicates(programs, embeddings, task_id, threshold=0.8):
    """Analyze duplicates within a single task"""
    if len(programs) <= 1:
        return {
            'task_id': task_id,
            'original_count': len(programs),
            'duplicate_pairs': 0,
            'programs_to_remove': 0,
            'deduplicated_count': len(programs),
            'reduction_percent': 0.0,
            'pairs': []
        }

    n_programs = len(programs)
    similarity_matrix = cosine_similarity(embeddings)

    to_remove = set()
    duplicate_pairs = []

    # Find duplicate pairs above threshold (excluding diagonal)
    for i in range(n_programs):
        for j in range(i + 1, n_programs):
            if similarity_matrix[i, j] >= threshold:
                duplicate_pairs.append((i, j, similarity_matrix[i, j]))
                to_remove.add(j)  # Remove the second occurrence

    deduplicated_count = n_programs - len(to_remove)
    reduction_percent = (len(to_remove) / n_programs * 100) if n_programs > 0 else 0

    return {
        'task_id': task_id,
        'original_count': n_programs,
        'duplicate_pairs': len(duplicate_pairs),
        'programs_to_remove': len(to_remove),
        'deduplicated_count': deduplicated_count,
        'reduction_percent': reduction_percent,
        'pairs': duplicate_pairs[:5],  # Store first 5 pairs as examples
        'max_similarity': float(similarity_matrix.max()) if len(similarity_matrix) > 0 else 0.0
    }

def run_within_task_analysis(selected_tasks, thresholds=[0.99, 0.95, 0.9, 0.85, 0.8], max_workers=2):
    """Run deduplication analysis within each task"""
    print("Loading CodeRankEmbed model...")

    import torch
    torch.set_num_threads(max_workers)

    model = SentenceTransformer("nomic-ai/CodeRankEmbed", trust_remote_code=True)

    results_by_threshold = {threshold: [] for threshold in thresholds}
    task_embeddings = {}

    print(f"Analyzing {len(selected_tasks)} tasks...")

    for task_id, programs in tqdm(selected_tasks, desc="Processing tasks"):
        if len(programs) <= 1:
            # Skip tasks with only 1 program
            for threshold in thresholds:
                results_by_threshold[threshold].append({
                    'task_id': task_id,
                    'original_count': len(programs),
                    'duplicate_pairs': 0,
                    'programs_to_remove': 0,
                    'deduplicated_count': len(programs),
                    'reduction_percent': 0.0
                })
            continue

        # Compute embeddings for this task's programs
        embeddings = compute_embeddings_for_task(programs, model, task_id)
        task_embeddings[task_id] = embeddings

        # Analyze at each threshold
        for threshold in thresholds:
            analysis = analyze_within_task_duplicates(programs, embeddings, task_id, threshold)
            results_by_threshold[threshold].append(analysis)

    return results_by_threshold, task_embeddings

def calculate_summary_stats(results_by_threshold, total_programs_in_dataset):
    """Calculate summary statistics across all thresholds"""
    summary = {}

    for threshold, results in results_by_threshold.items():
        total_original = sum(r['original_count'] for r in results)
        total_removed = sum(r['programs_to_remove'] for r in results)
        total_remaining = total_original - total_removed
        overall_reduction = (total_removed / total_original * 100) if total_original > 0 else 0

        # Extrapolate to full dataset
        if total_original > 0:
            full_dataset_reduction_rate = total_removed / total_original
            estimated_full_removed = int(total_programs_in_dataset * full_dataset_reduction_rate)
            estimated_full_remaining = total_programs_in_dataset - estimated_full_removed
            estimated_full_reduction_percent = estimated_full_removed / total_programs_in_dataset * 100
        else:
            estimated_full_removed = 0
            estimated_full_remaining = total_programs_in_dataset
            estimated_full_reduction_percent = 0

        summary[threshold] = {
            'sample_original': total_original,
            'sample_removed': total_removed,
            'sample_remaining': total_remaining,
            'sample_reduction_percent': overall_reduction,
            'estimated_full_removed': estimated_full_removed,
            'estimated_full_remaining': estimated_full_remaining,
            'estimated_full_reduction_percent': estimated_full_reduction_percent
        }

    return summary

def save_results(results_by_threshold, summary, task_embeddings, selected_tasks, total_programs):
    """Save all analysis results"""
    print("Saving results...")

    # Save detailed results
    with open('within_task_dedup_results.pkl', 'wb') as f:
        pickle.dump({
            'results_by_threshold': results_by_threshold,
            'summary': summary,
            'selected_tasks_info': [(task_id, len(programs)) for task_id, programs in selected_tasks],
            'total_programs_in_dataset': total_programs
        }, f)

    # Save embeddings
    with open('task_embeddings.pkl', 'wb') as f:
        pickle.dump(task_embeddings, f)

    # Create report
    report = f"""# ARC-AGI-2 Partial 100: Within-Task Program Deduplication Analysis

## Overview
- **Analysis Type**: Deduplication within each task (comparing programs for the same task_id)
- **Total Dataset**: {total_programs:,} programs across 863 tasks
- **Sample Analyzed**: {len(selected_tasks)} tasks with {summary[0.8]['sample_original']} programs total

## Sample Task Details
"""

    for i, (task_id, programs) in enumerate(selected_tasks):
        report += f"- **{task_id}**: {len(programs)} programs\n"

    report += "\n## Deduplication Results\n\n"

    for threshold in sorted(summary.keys(), reverse=True):
        stats = summary[threshold]
        report += f"""### {threshold*100:.0f}% Similarity Threshold

**Sample Results:**
- Original programs: {stats['sample_original']:,}
- Programs to remove: {stats['sample_removed']:,}
- Programs remaining: {stats['sample_remaining']:,}
- Reduction: {stats['sample_reduction_percent']:.1f}%

**Estimated Full Dataset:**
- Programs to remove: {stats['estimated_full_removed']:,}
- Programs remaining: {stats['estimated_full_remaining']:,}
- Reduction: {stats['estimated_full_reduction_percent']:.1f}%

"""

    report += f"""## Key Findings

1. **Within-task similarity**: Programs solving the same task show varying levels of similarity
2. **Deduplication potential**: Significant reduction possible while maintaining solution diversity
3. **Task-specific patterns**: Some tasks have many similar solutions, others have diverse approaches

## Methodology
- **Embedding Model**: nomic-ai/CodeRankEmbed
- **Similarity Metric**: Cosine similarity
- **Deduplication Strategy**: Greedy removal (keep first occurrence)
- **Processing**: 2 CPU cores maximum

## Limitations
- Results extrapolated from sample of {len(selected_tasks)} tasks
- Actual full dataset results may vary
- Some tasks might have different similarity patterns than sampled tasks
"""

    with open('within_task_dedup_report.md', 'w') as f:
        f.write(report)

    print("Results saved!")
    return summary

def main():
    """Main analysis pipeline for within-task deduplication"""
    print("Starting WITHIN-TASK deduplication analysis for ARC-AGI-2 Partial 100...")

    # Load and group data
    dataset = load_dataset()
    task_programs = group_programs_by_task(dataset)

    total_programs = len(dataset)

    # Select sample tasks
    selected_tasks = select_sample_tasks(task_programs, sample_size=25)

    # Run analysis
    results_by_threshold, task_embeddings = run_within_task_analysis(selected_tasks)

    # Calculate summary statistics
    summary = calculate_summary_stats(results_by_threshold, total_programs)

    # Save results
    final_summary = save_results(results_by_threshold, summary, task_embeddings, selected_tasks, total_programs)

    # Print summary
    print("\n" + "="*70)
    print("WITHIN-TASK DEDUPLICATION SUMMARY")
    print("="*70)
    print(f"Original dataset: {total_programs:,} programs across 863 tasks")
    print(f"Sample analyzed: {len(selected_tasks)} tasks")
    print()

    for threshold in sorted(final_summary.keys(), reverse=True):
        stats = final_summary[threshold]
        print(f"At {threshold*100:.0f}% similarity threshold:")
        print(f"  Estimated remaining: {stats['estimated_full_remaining']:,} programs")
        print(f"  Estimated reduction: {stats['estimated_full_reduction_percent']:.1f}%")

    print("="*70)

if __name__ == "__main__":
    main()