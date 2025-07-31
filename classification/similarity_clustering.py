#!/usr/bin/env python3
"""
Similarity-based clustering of programs within tasks using OpenAI embeddings and elbow method.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dotenv import load_dotenv
from datasets import load_dataset
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import List, Dict, Any, Tuple
import argparse

def load_api_key():
    """Load OpenAI API key from .env file."""
    # Load .env from root for OpenAI API key
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
    
    api_key = os.getenv("OPENAI_API_KEY_OPENAI") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY_OPENAI or OPENAI_API_KEY not found in .env file")
    
    return api_key

def get_embedding_batch(codes: List[str], client: OpenAI, model: str = "text-embedding-3-small") -> List[List[float]]:
    """Get embeddings for a batch of code snippets."""
    try:
        response = client.embeddings.create(
            input=codes,
            model=model
        )
        return [embedding.embedding for embedding in response.data]
    except Exception as e:
        print(f"   ❌ Error in batch: {str(e)}")
        return []

def calculate_embeddings_parallel(programs: List[Dict[str, Any]], max_workers: int = 20) -> List[Dict[str, Any]]:
    """Calculate embeddings for programs using parallel processing."""
    print(f"🔄 Calculating embeddings for {len(programs)} programs...")
    
    api_key = load_api_key()
    client = OpenAI(api_key=api_key)
    
    # Process in batches for efficiency
    batch_size = 50  # OpenAI allows up to 2048 inputs per request
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for i in range(0, len(programs), batch_size):
            batch_programs = programs[i:i + batch_size]
            batch_codes = [p['code'] for p in batch_programs]
            
            future = executor.submit(get_embedding_batch, batch_codes, client)
            futures.append((future, batch_programs, i))
        
        completed = 0
        for future, batch_programs, start_idx in futures:
            try:
                embeddings = future.result()
                if embeddings:
                    for j, (program, embedding) in enumerate(zip(batch_programs, embeddings)):
                        results.append({
                            **program,
                            'embedding': embedding,
                            'embedding_model': 'text-embedding-3-small'
                        })
                        completed += 1
                        if completed % 10 == 0:
                            print(f"   ✅ Completed {completed}/{len(programs)} embeddings")
                else:
                    print(f"   ⚠️  Failed batch starting at {start_idx}")
            except Exception as e:
                print(f"   ❌ Error processing batch: {str(e)}")
    
    print(f"✅ Generated {len(results)} embeddings")
    return results

def find_optimal_clusters(embeddings: np.ndarray, max_k: int = None) -> Tuple[int, Dict[str, Any]]:
    """Find optimal number of clusters using elbow method and silhouette analysis."""
    if max_k is None:
        max_k = min(10, len(embeddings) - 1)
    
    if max_k < 2:
        return 1, {'method': 'single_cluster', 'reason': 'insufficient_data'}
    
    k_range = range(2, max_k + 1)
    inertias = []
    silhouette_scores = []
    
    print(f"   🔍 Testing k from 2 to {max_k}...")
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        inertias.append(kmeans.inertia_)
        sil_score = silhouette_score(embeddings, cluster_labels)
        silhouette_scores.append(sil_score)
        
        print(f"     k={k}: inertia={kmeans.inertia_:.3f}, silhouette={sil_score:.3f}")
    
    # Find elbow using rate of change
    if len(inertias) >= 2:
        # Calculate second derivative to find elbow
        deltas = np.diff(inertias)
        delta2 = np.diff(deltas)
        
        # Find the point where curvature is highest (most negative second derivative)
        if len(delta2) > 0:
            elbow_idx = np.argmin(delta2)
            elbow_k = k_range[elbow_idx + 1]  # +1 because of double diff
        else:
            elbow_k = k_range[0]
    else:
        elbow_k = k_range[0]
    
    # Best silhouette score
    best_sil_idx = np.argmax(silhouette_scores)
    best_sil_k = k_range[best_sil_idx]
    
    # Choose between elbow and silhouette (prefer silhouette if close)
    if abs(elbow_k - best_sil_k) <= 1:
        optimal_k = best_sil_k
        method = 'silhouette'
    else:
        optimal_k = elbow_k
        method = 'elbow'
    
    analysis = {
        'method': method,
        'elbow_k': elbow_k,
        'silhouette_k': best_sil_k,
        'chosen_k': optimal_k,
        'k_range': list(k_range),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'silhouette_optimal': silhouette_scores[best_sil_idx]
    }
    
    print(f"   📊 Optimal k: {optimal_k} (method: {method})")
    return optimal_k, analysis

def cluster_programs(programs_with_embeddings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cluster programs and analyze results."""
    print(f"🎯 Clustering {len(programs_with_embeddings)} programs...")
    
    # Extract embeddings
    embeddings = np.array([p['embedding'] for p in programs_with_embeddings])
    
    # Find optimal number of clusters
    optimal_k, cluster_analysis = find_optimal_clusters(embeddings)
    
    # Perform final clustering
    if optimal_k == 1:
        cluster_labels = np.zeros(len(embeddings))
        cluster_centers = np.mean(embeddings, axis=0, keepdims=True)
    else:
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        cluster_centers = kmeans.cluster_centers_
    
    # Assign cluster labels to programs
    for i, program in enumerate(programs_with_embeddings):
        program['cluster'] = int(cluster_labels[i])
    
    # Group programs by cluster
    clusters = {}
    for program in programs_with_embeddings:
        cluster_id = program['cluster']
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(program)
    
    # Calculate cluster statistics
    cluster_stats = {}
    for cluster_id, cluster_programs in clusters.items():
        cluster_embeddings = np.array([p['embedding'] for p in cluster_programs])
        center = cluster_centers[cluster_id] if optimal_k > 1 else cluster_centers[0]
        
        # Calculate distances to center
        distances = [np.linalg.norm(emb - center) for emb in cluster_embeddings]
        
        cluster_stats[cluster_id] = {
            'size': len(cluster_programs),
            'avg_distance_to_center': float(np.mean(distances)),
            'max_distance_to_center': float(np.max(distances)),
            'models': list(set(p['model'] for p in cluster_programs)),
            'avg_code_length': float(np.mean([len(p['code']) for p in cluster_programs]))
        }
    
    return {
        'programs': programs_with_embeddings,
        'clusters': clusters,
        'cluster_stats': cluster_stats,
        'cluster_analysis': cluster_analysis,
        'optimal_k': optimal_k,
        'total_programs': len(programs_with_embeddings)
    }

def create_visualization(clustering_result: Dict[str, Any], task_id: str, output_dir: str):
    """Create visualizations for the clustering results."""
    programs = clustering_result['programs']
    embeddings = np.array([p['embedding'] for p in programs])
    cluster_labels = [p['cluster'] for p in programs]
    
    # Reduce dimensionality for visualization
    n_samples, n_features = embeddings.shape
    
    if len(embeddings) > 3:
        # Use PCA first to reduce to reasonable dimensions, then t-SNE
        # Ensure n_components doesn't exceed min(n_samples, n_features)
        max_pca_components = min(50, n_samples - 1, n_features)
        
        if n_features > max_pca_components and max_pca_components > 2:
            pca = PCA(n_components=max_pca_components)
            embeddings_pca = pca.fit_transform(embeddings)
        else:
            embeddings_pca = embeddings
            
        # For t-SNE, perplexity must be less than n_samples
        perplexity = min(30, (len(embeddings) - 1) // 3, 5)  # Conservative perplexity
        if perplexity < 1:
            perplexity = 1
            
        if len(embeddings_pca) > perplexity:
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            embeddings_2d = tsne.fit_transform(embeddings_pca)
        else:
            # Fall back to PCA for very small datasets
            pca_2d = PCA(n_components=2)
            embeddings_2d = pca_2d.fit_transform(embeddings_pca)
    else:
        # For very small datasets, just use PCA
        n_components = min(2, n_samples - 1, n_features)
        if n_components < 2:
            # Can't visualize properly, create dummy 2D coordinates
            embeddings_2d = np.column_stack([
                np.arange(len(embeddings)),
                np.zeros(len(embeddings))
            ])
        else:
            pca = PCA(n_components=n_components)
            embeddings_reduced = pca.fit_transform(embeddings)
            if n_components == 1:
                embeddings_2d = np.column_stack([embeddings_reduced.flatten(), np.zeros(len(embeddings))])
            else:
                embeddings_2d = embeddings_reduced
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot clusters
    unique_clusters = set(cluster_labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
    
    for cluster_id, color in zip(unique_clusters, colors):
        cluster_mask = [label == cluster_id for label in cluster_labels]
        cluster_points = embeddings_2d[cluster_mask]
        
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   c=[color], label=f'Cluster {cluster_id}', 
                   alpha=0.7, s=60)
        
        # Add program indices as text
        for i, (x, y) in enumerate(cluster_points):
            original_idx = [j for j, label in enumerate(cluster_labels) if label == cluster_id][i]
            plt.annotate(str(original_idx), (x, y), xytext=(2, 2), 
                        textcoords='offset points', fontsize=8, alpha=0.8)
    
    plt.title(f'Program Clustering for Task {task_id}\\n'
              f'{len(programs)} programs, {clustering_result["optimal_k"]} clusters')
    plt.xlabel('t-SNE/PCA Dimension 1')
    plt.ylabel('t-SNE/PCA Dimension 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_file = f"{output_dir}/clustering_{task_id}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 Saved visualization: {plot_file}")

def save_readable_results(clustering_result: Dict[str, Any], task_id: str, output_dir: str):
    """Save clustering results in an easily readable format."""
    
    # Create a readable summary
    readable_data = {
        'task_id': task_id,
        'summary': {
            'total_programs': clustering_result['total_programs'],
            'optimal_clusters': clustering_result['optimal_k'],
            'clustering_method': clustering_result['cluster_analysis']['method']
        },
        'cluster_analysis': clustering_result['cluster_analysis'],
        'clusters': {}
    }
    
    # Process each cluster
    for cluster_id, cluster_programs in clustering_result['clusters'].items():
        readable_data['clusters'][f'cluster_{cluster_id}'] = {
            'stats': clustering_result['cluster_stats'][cluster_id],
            'programs': []
        }
        
        for i, program in enumerate(cluster_programs):
            program_data = {
                'index': i,
                'model': program['model'],
                'code_length': len(program['code']),
                'code_preview': program['code'][:200] + "..." if len(program['code']) > 200 else program['code'],
                'full_code': program['code']
            }
            readable_data['clusters'][f'cluster_{cluster_id}']['programs'].append(program_data)
    
    # Save readable JSON
    readable_file = f"{output_dir}/clustering_readable_{task_id}.json"
    with open(readable_file, 'w') as f:
        json.dump(readable_data, f, indent=2)
    
    # Save full data (with embeddings)
    full_file = f"{output_dir}/clustering_full_{task_id}.json"
    with open(full_file, 'w') as f:
        json.dump(clustering_result, f, indent=2)
    
    print(f"   💾 Saved readable results: {readable_file}")
    print(f"   💾 Saved full results: {full_file}")
    
    # Print summary
    print(f"   📋 Cluster Summary for {task_id}:")
    for cluster_id, stats in clustering_result['cluster_stats'].items():
        print(f"     Cluster {cluster_id}: {stats['size']} programs, "
              f"models: {', '.join(stats['models'])}")

def analyze_task_similarity(task_id: str, max_workers: int = 20, output_dir: str = "classification/data"):
    """Analyze similarity for a specific task."""
    print(f"\\n🎯 Analyzing task: {task_id}")
    print("=" * 50)
    
    # Load dataset and filter for specific task
    print("📁 Loading SOAR dataset...")
    dataset = load_dataset("Trelis/soar-program-samples")
    df = dataset['train'].to_pandas()
    
    task_programs = df[df['task_id'] == task_id].to_dict('records')
    print(f"✅ Found {len(task_programs)} programs for task {task_id}")
    
    if len(task_programs) < 2:
        print("⚠️  Not enough programs for clustering analysis")
        return None
    
    # Calculate embeddings
    programs_with_embeddings = calculate_embeddings_parallel(task_programs, max_workers)
    
    if len(programs_with_embeddings) < 2:
        print("❌ Failed to generate enough embeddings")
        return None
    
    # Cluster programs
    clustering_result = cluster_programs(programs_with_embeddings)
    
    # Create visualizations
    os.makedirs(output_dir, exist_ok=True)
    create_visualization(clustering_result, task_id, output_dir)
    
    # Save results
    save_readable_results(clustering_result, task_id, output_dir)
    
    return clustering_result

def main():
    """Main execution pipeline."""
    parser = argparse.ArgumentParser(description="Analyze program similarity within tasks")
    parser.add_argument("--tasks", nargs="+", 
                       default=["3e980e27", "4522001f", "a48eeaf7"],
                       help="Task IDs to analyze")
    parser.add_argument("--workers", type=int, default=20,
                       help="Number of parallel workers for embeddings")
    parser.add_argument("--output", default="classification/data",
                       help="Output directory")
    
    args = parser.parse_args()
    
    print("🚀 Program Similarity Analysis")
    print("=" * 60)
    print(f"📋 Analyzing tasks: {', '.join(args.tasks)}")
    print(f"⚡ Using {args.workers} parallel workers")
    
    results = {}
    start_time = time.time()
    
    for task_id in args.tasks:
        try:
            result = analyze_task_similarity(task_id, args.workers, args.output)
            if result:
                results[task_id] = result
        except Exception as e:
            print(f"❌ Error analyzing task {task_id}: {str(e)}")
    
    total_time = time.time() - start_time
    print(f"\\n⏱️  Total analysis time: {total_time:.2f} seconds")
    print(f"✅ Successfully analyzed {len(results)} tasks")
    
    return results

if __name__ == "__main__":
    main()