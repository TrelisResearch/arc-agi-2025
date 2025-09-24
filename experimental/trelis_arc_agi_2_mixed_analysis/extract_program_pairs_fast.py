#!/usr/bin/env python3
"""
Fast extraction of program pairs within same task for each accuracy category.
Finds pairs with similarity close to 0.9.
"""

import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import gc
import random

def load_and_classify():
    """Load dataset and classify programs"""
    print("Loading and classifying dataset...")
    table = pq.read_table('superking_aa2.parquet')
    df = table.to_pandas(strings_to_categorical=False, types_mapper=pd.ArrowDtype)

    def get_accuracy_category(row):
        train_correct = row['correct_train_input']
        test_correct = row['correct_test_input']

        train_correct_count = sum(train_correct) if train_correct is not None and len(train_correct) > 0 else 0
        test_correct_count = sum(test_correct) if test_correct is not None and len(test_correct) > 0 else 0

        train_total = len(train_correct) if train_correct is not None else 0
        test_total = len(test_correct) if test_correct is not None else 0

        if train_correct_count == train_total and test_correct_count == test_total and train_total > 0 and test_total > 0:
            return 'all_correct'
        elif train_correct_count == 1 and train_total > 1:
            return 'partially_correct_train'
        elif train_correct_count == 0 and test_correct_count == 0 and train_total > 0 and test_total > 0:
            return 'all_incorrect'
        return 'other'

    df['accuracy_category'] = df.apply(get_accuracy_category, axis=1)

    category_counts = df['accuracy_category'].value_counts()
    print("Category distribution:")
    for category, count in category_counts.items():
        print(f"  {category}: {count}")

    return df

def find_pair_for_category(df, category, target_sim=0.9, max_tasks_to_check=50):
    """Find best pair for a category, checking only a limited number of tasks"""
    print(f"\nSearching for pair in category: {category}")

    category_df = df[df['accuracy_category'] == category]
    task_groups = category_df.groupby('task_id')
    valid_tasks = [(task_id, group) for task_id, group in task_groups if len(group) >= 2]

    if not valid_tasks:
        print(f"No valid tasks found for {category}")
        return None

    print(f"Found {len(valid_tasks)} tasks with multiple programs")

    # Sample tasks to check (for speed)
    tasks_to_check = min(max_tasks_to_check, len(valid_tasks))
    sampled_tasks = random.sample(valid_tasks, tasks_to_check)

    print(f"Checking {tasks_to_check} tasks")

    # Load model once
    model = SentenceTransformer("nomic-ai/CodeRankEmbed", trust_remote_code=True)

    best_pair = None
    best_similarity = 0.0
    best_task_id = None
    best_score = float('inf')

    for i, (task_id, task_df) in enumerate(sampled_tasks):
        print(f"  Task {i+1}/{tasks_to_check}: {task_id} ({len(task_df)} programs)")

        codes = task_df['code'].fillna('').tolist()
        embeddings = model.encode(codes, batch_size=8, show_progress_bar=False)
        sim_matrix = cosine_similarity(embeddings)

        # Find best pair in this task
        for idx1 in range(len(embeddings)):
            for idx2 in range(idx1 + 1, len(embeddings)):
                similarity = sim_matrix[idx1][idx2]

                # Score based on closeness to target
                score = abs(similarity - target_sim)

                if score < best_score:
                    best_score = score
                    best_similarity = similarity
                    best_task_id = task_id
                    best_pair = task_df.iloc[[idx1, idx2]].copy()

        print(f"    Best similarity in task: {sim_matrix.max():.3f}")

        # Early stopping if we find a very good pair
        if best_score < 0.02:  # Within 0.02 of target
            print(f"    Found excellent pair (similarity {best_similarity:.3f}), stopping early")
            break

    if best_pair is not None:
        best_pair['pair_similarity'] = best_similarity
        print(f"Selected pair from task {best_task_id}, similarity: {best_similarity:.3f}")

    # Clean up
    del model
    gc.collect()

    return best_pair

def main():
    random.seed(42)

    # Load data
    df = load_and_classify()

    categories = ['all_correct', 'partially_correct_train', 'all_incorrect']
    all_pairs = []

    print(f"\n{'='*60}")
    print("SEARCHING FOR PROGRAM PAIRS")
    print(f"{'='*60}")

    for category in categories:
        print(f"\n{'-'*40}")
        print(f"Category: {category}")
        print(f"{'-'*40}")

        pair = find_pair_for_category(df, category, target_sim=0.99, max_tasks_to_check=30)

        if pair is not None:
            pair['selected_category'] = category
            all_pairs.append(pair)

    if all_pairs:
        final_df = pd.concat(all_pairs, ignore_index=True)

        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Total programs: {len(final_df)} (3 pairs)")

        print("\nPair details:")
        for category in categories:
            category_data = final_df[final_df['selected_category'] == category]
            if len(category_data) > 0:
                task_id = category_data['task_id'].iloc[0]
                similarity = category_data['pair_similarity'].iloc[0]
                row_ids = category_data['row_id'].tolist()
                print(f"  {category}:")
                print(f"    Task: {task_id}")
                print(f"    Similarity: {similarity:.3f}")
                print(f"    Row IDs: {row_ids}")

        # Save results
        output_file = 'selected_program_pairs_fast_sim0.99.parquet'
        final_df.to_parquet(output_file, index=False)
        print(f"\nSaved to: {output_file}")

        # Save summary
        summary = []
        for category in categories:
            category_data = final_df[final_df['selected_category'] == category]
            if len(category_data) > 0:
                summary.append({
                    'category': category,
                    'task_id': category_data['task_id'].iloc[0],
                    'pair_similarity': float(category_data['pair_similarity'].iloc[0]),
                    'row_ids': category_data['row_id'].tolist()
                })

        with open('selected_program_pairs_fast_sim0.99_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print("Summary saved to: selected_program_pairs_fast_sim0.99_summary.json")

        return final_df
    else:
        print("No pairs found!")
        return None

if __name__ == "__main__":
    main()