import json
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from collections import defaultdict

def analyze_dataset_lengths():
    print("Loading Trelis/arc-agi-2-partial-100 dataset...")
    dataset = load_dataset("Trelis/arc-agi-2-partial-100", split="train")
    
    lengths = defaultdict(list)
    sample_indices = []
    
    lengths['total_samples'] = len(dataset)
    
    for idx, sample in enumerate(dataset):
        if idx % 2000 == 0:
            print(f"Processing sample {idx}/{len(dataset)}...")
        
        sample_indices.append(idx)
        
        lengths['reasoning_length'].append(len(sample.get('reasoning', '')))
        lengths['code_length'].append(len(sample.get('code', '')))
        
        if 'correct_train_input' in sample and sample['correct_train_input']:
            try:
                total_train_cells = sum(len(grid) * len(grid[0]) if grid and isinstance(grid, list) and len(grid) > 0 else 0 
                                      for grid in sample['correct_train_input'])
                lengths['train_input_cells'].append(total_train_cells)
                lengths['num_train_examples'].append(len(sample['correct_train_input']))
            except (TypeError, IndexError):
                pass
        
        if 'correct_test_input' in sample and sample['correct_test_input']:
            try:
                total_test_cells = sum(len(grid) * len(grid[0]) if grid and isinstance(grid, list) and len(grid) > 0 else 0 
                                     for grid in sample['correct_test_input'])
                lengths['test_input_cells'].append(total_test_cells)
                lengths['num_test_examples'].append(len(sample['correct_test_input']))
            except (TypeError, IndexError):
                pass
        
        if 'predicted_train_output' in sample and sample['predicted_train_output']:
            try:
                total_train_out_cells = sum(len(grid) * len(grid[0]) if grid and isinstance(grid, list) and len(grid) > 0 else 0 
                                          for grid in sample['predicted_train_output'])
                lengths['train_output_cells'].append(total_train_out_cells)
            except (TypeError, IndexError):
                pass
        
        if 'predicted_test_output' in sample and sample['predicted_test_output']:
            try:
                total_test_out_cells = sum(len(grid) * len(grid[0]) if grid and isinstance(grid, list) and len(grid) > 0 else 0 
                                         for grid in sample['predicted_test_output'])
                lengths['test_output_cells'].append(total_test_out_cells)
            except (TypeError, IndexError):
                pass
        
        total_length = (len(sample.get('reasoning', '')) + 
                       len(sample.get('code', '')) + 
                       len(str(sample.get('correct_train_input', []))) + 
                       len(str(sample.get('correct_test_input', []))) +
                       len(str(sample.get('predicted_train_output', []))) +
                       len(str(sample.get('predicted_test_output', []))))
        lengths['total_sample_length'].append(total_length)
    
    print("\n" + "="*60)
    print("DATASET LENGTH ANALYSIS")
    print("="*60)
    print(f"Total samples: {lengths['total_samples']}")
    
    outlier_info = {}
    
    for key in ['total_sample_length', 'reasoning_length', 'code_length', 
                'train_input_cells', 'test_input_cells', 
                'train_output_cells', 'test_output_cells',
                'num_train_examples', 'num_test_examples']:
        if key in lengths and lengths[key]:
            data = np.array(lengths[key])
            
            if len(data) == 0:
                continue
                
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            print(f"\n{key.replace('_', ' ').title()}:")
            print(f"  Count: {len(data)}")
            print(f"  Mean: {np.mean(data):.2f}")
            print(f"  Median: {np.median(data):.2f}")
            print(f"  Min: {np.min(data):.0f}")
            print(f"  Max: {np.max(data):.0f}")
            print(f"  Std: {np.std(data):.2f}")
            print(f"  Q1: {q1:.2f}, Q3: {q3:.2f}")
            print(f"  IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
            print(f"  Number of outliers: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)")
            
            if len(outliers) > 0:
                outlier_indices = [i for i, val in enumerate(data) if (val < lower_bound) or (val > upper_bound)]
                outlier_info[key] = {
                    'indices': outlier_indices,
                    'values': outliers
                }
                print(f"  Top 10 outlier values: {sorted(outliers, reverse=True)[:10]}")
                print(f"  Sample indices with top outliers: {[outlier_indices[i] for i in np.argsort(outliers)[-10:][::-1]]}")
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Dataset Length Distribution Analysis (Box Plots)', fontsize=16)
    
    plot_keys = ['total_sample_length', 'reasoning_length', 'code_length', 
                 'train_input_cells', 'test_input_cells', 
                 'train_output_cells', 'test_output_cells',
                 'num_train_examples', 'num_test_examples']
    
    for idx, key in enumerate(plot_keys):
        if idx >= 9:
            break
        ax = axes[idx // 3, idx % 3]
        if key in lengths and lengths[key]:
            ax.boxplot(lengths[key])
            ax.set_title(key.replace('_', ' ').title(), fontsize=10)
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experimental/trelis_dataset_inspection/outliers_boxplot.png', dpi=150)
    print(f"\nBoxplot saved to experimental/trelis_dataset_inspection/outliers_boxplot.png")
    
    fig2, axes2 = plt.subplots(3, 3, figsize=(15, 12))
    fig2.suptitle('Dataset Length Distribution (Histograms)', fontsize=16)
    
    for idx, key in enumerate(plot_keys):
        if idx >= 9:
            break
        ax = axes2[idx // 3, idx % 3]
        if key in lengths and lengths[key] and len(lengths[key]) > 0:
            ax.hist(lengths[key], bins=50, edgecolor='black', alpha=0.7)
            ax.set_title(key.replace('_', ' ').title(), fontsize=10)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experimental/trelis_dataset_inspection/length_histograms.png', dpi=150)
    print(f"Histograms saved to experimental/trelis_dataset_inspection/length_histograms.png")
    
    print(f"\n{'='*60}")
    print("TOP OUTLIER SAMPLES BY TOTAL LENGTH")
    print(f"{'='*60}")
    
    if 'total_sample_length' in outlier_info:
        top_outlier_indices = [outlier_info['total_sample_length']['indices'][i] 
                              for i in np.argsort(outlier_info['total_sample_length']['values'])[-5:][::-1]]
        
        for idx in top_outlier_indices:
            sample = dataset[idx]
            print(f"\nSample {idx}:")
            print(f"  Task ID: {sample.get('task_id', 'N/A')}")
            print(f"  Total length: {lengths['total_sample_length'][idx]:,} characters")
            print(f"  Code length: {len(sample.get('code', '')):,} characters")
            print(f"  Reasoning length: {len(sample.get('reasoning', '')):,} characters")
            print(f"  Model: {sample.get('model', 'N/A')}")
            
            if sample.get('code'):
                print(f"  Code preview (first 200 chars):")
                print(f"    {sample['code'][:200]}...")

if __name__ == "__main__":
    analyze_dataset_lengths()