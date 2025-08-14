#!/usr/bin/env python3
"""
Analysis script to examine correlations between code embeddings and classifications.
"""

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple
import pandas as pd

def load_analysis_dataset(file_path: str) -> Dict[str, Any]:
    """Load the analysis dataset."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_embeddings_and_labels(dataset: Dict[str, Any]) -> Tuple[np.ndarray, List[str], List[str], List[str]]:
    """Extract embeddings and classification labels."""
    programs = dataset['programs']
    
    embeddings = np.array([program['embedding'] for program in programs])
    human_labels = [program['human_classification'] for program in programs]
    gemini_labels = [program['gemini_classification'] for program in programs]
    models = [program['model'] for program in programs]
    
    return embeddings, human_labels, gemini_labels, models

def calculate_embedding_similarities(embeddings: np.ndarray) -> np.ndarray:
    """Calculate cosine similarities between all embeddings."""
    return cosine_similarity(embeddings)

def analyze_classification_separation(embeddings: np.ndarray, human_labels: List[str], gemini_labels: List[str]) -> Dict[str, Any]:
    """Analyze how well embeddings separate different classifications."""
    
    # Convert labels to numeric for analysis
    human_numeric = [1 if label == 'general' else 0 for label in human_labels]
    gemini_numeric = [1 if label == 'general' else 0 for label in gemini_labels]
    
    # Calculate mean embeddings for each class
    human_overfitting_mask = np.array([label == 'overfitting' for label in human_labels])
    human_general_mask = np.array([label == 'general' for label in human_labels])
    
    gemini_overfitting_mask = np.array([label == 'overfitting' for label in gemini_labels])
    gemini_general_mask = np.array([label == 'general' for label in gemini_labels])
    
    results = {}
    
    if np.any(human_general_mask) and np.any(human_overfitting_mask):
        human_overfitting_mean = embeddings[human_overfitting_mask].mean(axis=0)
        human_general_mean = embeddings[human_general_mask].mean(axis=0)
        human_class_similarity = cosine_similarity(
            human_overfitting_mean.reshape(1, -1), 
            human_general_mean.reshape(1, -1)
        )[0][0]
        results['human_class_separation'] = {
            'overfitting_mean': human_overfitting_mean,
            'general_mean': human_general_mean,
            'class_similarity': human_class_similarity,
            'class_distance': 1 - human_class_similarity
        }
    
    if np.any(gemini_general_mask) and np.any(gemini_overfitting_mask):
        gemini_overfitting_mean = embeddings[gemini_overfitting_mask].mean(axis=0)
        gemini_general_mean = embeddings[gemini_general_mask].mean(axis=0)
        gemini_class_similarity = cosine_similarity(
            gemini_overfitting_mean.reshape(1, -1), 
            gemini_general_mean.reshape(1, -1)
        )[0][0]
        results['gemini_class_separation'] = {
            'overfitting_mean': gemini_overfitting_mean,
            'general_mean': gemini_general_mean,
            'class_similarity': gemini_class_similarity,
            'class_distance': 1 - gemini_class_similarity
        }
    
    return results

def perform_clustering_analysis(embeddings: np.ndarray, n_clusters: int = 2) -> Dict[str, Any]:
    """Perform K-means clustering on embeddings."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    return {
        'cluster_labels': cluster_labels,
        'cluster_centers': kmeans.cluster_centers_,
        'inertia': kmeans.inertia_
    }

def calculate_classification_agreement(human_labels: List[str], gemini_labels: List[str]) -> Dict[str, Any]:
    """Calculate agreement between human and Gemini classifications."""
    
    # Filter out error cases
    valid_indices = [i for i, label in enumerate(gemini_labels) if label != 'error']
    
    if not valid_indices:
        return {'error': 'No valid Gemini classifications'}
    
    valid_human = [human_labels[i] for i in valid_indices]
    valid_gemini = [gemini_labels[i] for i in valid_indices]
    
    agreements = sum(1 for h, g in zip(valid_human, valid_gemini) if h == g)
    total = len(valid_human)
    
    agreement_rate = agreements / total if total > 0 else 0
    
    # Create confusion matrix
    confusion = {}
    for h_label in ['overfitting', 'general']:
        confusion[h_label] = {}
        for g_label in ['overfitting', 'general']:
            confusion[h_label][g_label] = sum(1 for h, g in zip(valid_human, valid_gemini) 
                                            if h == h_label and g == g_label)
    
    return {
        'agreement_rate': agreement_rate,
        'agreements': agreements,
        'total_valid': total,
        'confusion_matrix': confusion,
        'valid_indices': valid_indices
    }

def visualize_embeddings(embeddings: np.ndarray, human_labels: List[str], gemini_labels: List[str], 
                        models: List[str], output_dir: str = "classification/data") -> None:
    """Create visualizations of the embeddings."""
    
    # Reduce dimensionality for visualization
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d_pca = pca.fit_transform(embeddings)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    embeddings_2d_tsne = tsne.fit_transform(embeddings)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Code Embedding Analysis', fontsize=16)
    
    # Colors for classifications
    color_map = {'overfitting': 'red', 'general': 'blue', 'error': 'gray'}
    
    # PCA plot - Human labels
    ax1 = axes[0, 0]
    for label in set(human_labels):
        mask = np.array([l == label for l in human_labels])
        ax1.scatter(embeddings_2d_pca[mask, 0], embeddings_2d_pca[mask, 1], 
                   c=color_map.get(label, 'black'), label=label, alpha=0.7)
    ax1.set_title('PCA - Human Classifications')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # PCA plot - Gemini labels
    ax2 = axes[0, 1]
    for label in set(gemini_labels):
        mask = np.array([l == label for l in gemini_labels])
        ax2.scatter(embeddings_2d_pca[mask, 0], embeddings_2d_pca[mask, 1], 
                   c=color_map.get(label, 'black'), label=label, alpha=0.7)
    ax2.set_title('PCA - Gemini Classifications')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # t-SNE plot - Human labels
    ax3 = axes[1, 0]
    for label in set(human_labels):
        mask = np.array([l == label for l in human_labels])
        ax3.scatter(embeddings_2d_tsne[mask, 0], embeddings_2d_tsne[mask, 1], 
                   c=color_map.get(label, 'black'), label=label, alpha=0.7)
    ax3.set_title('t-SNE - Human Classifications')
    ax3.set_xlabel('t-SNE 1')
    ax3.set_ylabel('t-SNE 2')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # t-SNE plot - Gemini labels
    ax4 = axes[1, 1]
    for label in set(gemini_labels):
        mask = np.array([l == label for l in gemini_labels])
        ax4.scatter(embeddings_2d_tsne[mask, 0], embeddings_2d_tsne[mask, 1], 
                   c=color_map.get(label, 'black'), label=label, alpha=0.7)
    ax4.set_title('t-SNE - Gemini Classifications')
    ax4.set_xlabel('t-SNE 1')
    ax4.set_ylabel('t-SNE 2')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/embedding_visualizations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved embedding visualization to {output_dir}/embedding_visualizations.png")

def create_similarity_heatmap(similarities: np.ndarray, human_labels: List[str], 
                             models: List[str], output_dir: str = "classification/data") -> None:
    """Create a heatmap of embedding similarities."""
    
    # Create labels for the heatmap
    labels = [f"{i}: {label[:4]}" for i, label in enumerate(human_labels)]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarities, 
                xticklabels=labels, 
                yticklabels=labels,
                cmap='coolwarm', 
                center=0.5,
                square=True,
                linewidths=0.1,
                cbar_kws={"shrink": .8})
    
    plt.title('Cosine Similarity Between Code Embeddings')
    plt.xlabel('Programs')
    plt.ylabel('Programs')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/similarity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved similarity heatmap to {output_dir}/similarity_heatmap.png")

def perform_comprehensive_analysis(dataset_file: str) -> Dict[str, Any]:
    """Perform comprehensive analysis of embeddings and classifications."""
    
    print("🔬 Performing comprehensive embedding analysis...")
    print("=" * 60)
    
    # Load dataset
    dataset = load_analysis_dataset(dataset_file)
    embeddings, human_labels, gemini_labels, models = extract_embeddings_and_labels(dataset)
    
    print(f"📊 Dataset: {len(embeddings)} programs, {embeddings.shape[1]}-dimensional embeddings")
    
    # Calculate similarities
    similarities = calculate_embedding_similarities(embeddings)
    
    # Analyze classification separation
    separation_analysis = analyze_classification_separation(embeddings, human_labels, gemini_labels)
    
    # Perform clustering
    clustering_analysis = perform_clustering_analysis(embeddings)
    
    # Calculate classification agreement
    agreement_analysis = calculate_classification_agreement(human_labels, gemini_labels)
    
    # Create visualizations
    visualize_embeddings(embeddings, human_labels, gemini_labels, models)
    create_similarity_heatmap(similarities, human_labels, models)
    
    # Compile results
    analysis_results = {
        'dataset_info': {
            'n_programs': len(embeddings),
            'embedding_dim': embeddings.shape[1],
            'human_labels_dist': {label: human_labels.count(label) for label in set(human_labels)},
            'gemini_labels_dist': {label: gemini_labels.count(label) for label in set(gemini_labels)}
        },
        'embedding_stats': {
            'mean_similarity': float(similarities.mean()),
            'std_similarity': float(similarities.std()),
            'min_similarity': float(similarities.min()),
            'max_similarity': float(similarities.max())
        },
        'separation_analysis': separation_analysis,
        'clustering_analysis': {
            'cluster_labels': clustering_analysis['cluster_labels'].tolist(),
            'inertia': float(clustering_analysis['inertia'])
        },
        'classification_agreement': agreement_analysis
    }
    
    return analysis_results

def print_analysis_summary(results: Dict[str, Any]) -> None:
    """Print a comprehensive summary of the analysis."""
    
    print("\n" + "=" * 60)
    print("🎯 EMBEDDING ANALYSIS SUMMARY")
    print("=" * 60)
    
    # Dataset info
    info = results['dataset_info']
    print(f"📊 Dataset: {info['n_programs']} programs, {info['embedding_dim']}-D embeddings")
    print(f"   Human labels: {info['human_labels_dist']}")
    print(f"   Gemini labels: {info['gemini_labels_dist']}")
    
    # Embedding statistics
    stats = results['embedding_stats']
    print(f"\n📈 Embedding Similarities:")
    print(f"   Mean: {stats['mean_similarity']:.4f}")
    print(f"   Std:  {stats['std_similarity']:.4f}")
    print(f"   Range: [{stats['min_similarity']:.4f}, {stats['max_similarity']:.4f}]")
    
    # Classification agreement
    agreement = results['classification_agreement']
    if 'error' not in agreement:
        print(f"\n🤝 Human-Gemini Agreement:")
        print(f"   Agreement rate: {agreement['agreement_rate']:.2%}")
        print(f"   Agreements: {agreement['agreements']}/{agreement['total_valid']}")
        
        confusion = agreement['confusion_matrix']
        print(f"\n📋 Confusion Matrix:")
        print(f"   Human\\Gemini    Overfitting  General")
        print(f"   Overfitting     {confusion['overfitting']['overfitting']:>10}  {confusion['overfitting']['general']:>7}")
        print(f"   General         {confusion['general']['overfitting']:>10}  {confusion['general']['general']:>7}")
    
    # Class separation analysis
    separation = results['separation_analysis']
    if 'human_class_separation' in separation:
        human_sep = separation['human_class_separation']
        print(f"\n🎯 Human Classification Separation:")
        print(f"   Class distance: {human_sep['class_distance']:.4f}")
        print(f"   Class similarity: {human_sep['class_similarity']:.4f}")
    
    if 'gemini_class_separation' in separation:
        gemini_sep = separation['gemini_class_separation']
        print(f"\n🤖 Gemini Classification Separation:")
        print(f"   Class distance: {gemini_sep['class_distance']:.4f}")
        print(f"   Class similarity: {gemini_sep['class_similarity']:.4f}")
    
    # Clustering analysis
    clustering = results['clustering_analysis']
    print(f"\n🔍 K-means Clustering (k=2):")
    print(f"   Inertia: {clustering['inertia']:.2f}")
    
    # Analysis conclusion
    print(f"\n💡 ANALYSIS INSIGHTS:")
    
    # Embedding signal strength
    if 'human_class_separation' in separation:
        human_distance = separation['human_class_separation']['class_distance']
        if human_distance > 0.1:
            print(f"   ✅ Strong embedding signal for human classifications (distance: {human_distance:.4f})")
        elif human_distance > 0.05:
            print(f"   ⚠️  Moderate embedding signal for human classifications (distance: {human_distance:.4f})")
        else:
            print(f"   ❌ Weak embedding signal for human classifications (distance: {human_distance:.4f})")
    
    if 'gemini_class_separation' in separation:
        gemini_distance = separation['gemini_class_separation']['class_distance']
        if gemini_distance > 0.1:
            print(f"   ✅ Strong embedding signal for Gemini classifications (distance: {gemini_distance:.4f})")
        elif gemini_distance > 0.05:
            print(f"   ⚠️  Moderate embedding signal for Gemini classifications (distance: {gemini_distance:.4f})")
        else:
            print(f"   ❌ Weak embedding signal for Gemini classifications (distance: {gemini_distance:.4f})")
    
    # Agreement assessment
    if 'error' not in agreement:
        agreement_rate = agreement['agreement_rate']
        if agreement_rate > 0.8:
            print(f"   ✅ High human-Gemini agreement ({agreement_rate:.1%})")
        elif agreement_rate > 0.6:
            print(f"   ⚠️  Moderate human-Gemini agreement ({agreement_rate:.1%})")
        else:
            print(f"   ❌ Low human-Gemini agreement ({agreement_rate:.1%})")

if __name__ == "__main__":
    dataset_file = "classification/data/program_analysis_dataset.json"
    
    # Perform analysis
    results = perform_comprehensive_analysis(dataset_file)
    
    # Print summary
    print_analysis_summary(results)
    
    # Save detailed results
    output_file = "classification/data/embedding_analysis_results.json"
    
    # Convert numpy arrays to lists for JSON serialization
    results_serializable = results.copy()
    if 'separation_analysis' in results_serializable:
        for key, value in results_serializable['separation_analysis'].items():
            if 'overfitting_mean' in value:
                value['overfitting_mean'] = value['overfitting_mean'].tolist()
            if 'general_mean' in value:
                value['general_mean'] = value['general_mean'].tolist()
    
    with open(output_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n💾 Detailed analysis saved to {output_file}")
    print("🖼️  Visualizations saved to classification/data/")