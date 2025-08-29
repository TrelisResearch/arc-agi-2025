import json
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from collections import defaultdict

def analyze_dataset_lengths():
    print("Loading Trelis/arc-agi-2-partial-100 dataset...")
    dataset = load_dataset("Trelis/arc-agi-2-partial-100", split="train")
    
    lengths = defaultdict(list)
    
    lengths['total_samples'] = len(dataset)
    
    sample_indices = []
    text_lengths = []
    
    for idx, sample in enumerate(dataset):
        if idx % 1000 == 0:
            print(f"Processing sample {idx}/{len(dataset)}...")
        
        sample_text_len = 0
        
        if 'input' in sample:
            sample_text_len += len(sample['input']) if isinstance(sample['input'], str) else 0
            
        if 'output' in sample:
            sample_text_len += len(sample['output']) if isinstance(sample['output'], str) else 0
            
        if 'train' in sample:
            train_str = sample['train'] if isinstance(sample['train'], str) else json.dumps(sample['train'])
            sample_text_len += len(train_str)
            
            train_data = json.loads(sample['train']) if isinstance(sample['train'], str) else sample['train']
            lengths['num_train_examples'].append(len(train_data))
            
            for train_ex in train_data:
                if 'input' in train_ex and train_ex['input']:
                    lengths['train_input_size'].append(len(train_ex['input']) * len(train_ex['input'][0]))
                if 'output' in train_ex and train_ex['output']:
                    lengths['train_output_size'].append(len(train_ex['output']) * len(train_ex['output'][0]))
        
        if 'test' in sample:
            test_str = sample['test'] if isinstance(sample['test'], str) else json.dumps(sample['test'])
            sample_text_len += len(test_str)
            
            test_data = json.loads(sample['test']) if isinstance(sample['test'], str) else sample['test']
            lengths['num_test_examples'].append(len(test_data))
            
            for test_ex in test_data:
                if 'input' in test_ex and test_ex['input']:
                    lengths['test_input_size'].append(len(test_ex['input']) * len(test_ex['input'][0]))
                if 'output' in test_ex and test_ex['output'] and test_ex['output']:
                    lengths['test_output_size'].append(len(test_ex['output']) * len(test_ex['output'][0]))
        
        text_lengths.append(sample_text_len)
        sample_indices.append(idx)
    
    lengths['total_text_length'] = text_lengths
    
    print("\n" + "="*60)
    print("DATASET LENGTH ANALYSIS")
    print("="*60)
    print(f"Total samples: {lengths['total_samples']}")
    
    outlier_samples = {}
    
    for key in ['total_text_length', 'num_train_examples', 'num_test_examples', 'train_input_size', 
                'train_output_size', 'test_input_size', 'test_output_size']:
        if key in lengths and lengths[key]:
            data = np.array(lengths[key])
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            print(f"\n{key}:")
            print(f"  Mean: {np.mean(data):.2f}")
            print(f"  Median: {np.median(data):.2f}")
            print(f"  Min: {np.min(data)}")
            print(f"  Max: {np.max(data)}")
            print(f"  Std: {np.std(data):.2f}")
            print(f"  Q1: {q1:.2f}, Q3: {q3:.2f}")
            print(f"  IQR bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
            print(f"  Number of outliers: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)")
            if len(outliers) > 0:
                print(f"  Outlier values: {sorted(outliers)[:10]}{'...' if len(outliers) > 10 else ''}")
                
                if key == 'total_text_length':
                    outlier_indices = [i for i, val in enumerate(data) if (val < lower_bound) or (val > upper_bound)]
                    outlier_samples[key] = outlier_indices[:5]
                    print(f"  Sample indices with outliers (first 5): {outlier_indices[:5]}")
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Dataset Length Distribution Analysis', fontsize=16)
    
    plot_keys = ['total_text_length', 'num_train_examples', 'num_test_examples', 
                 'train_input_size', 'train_output_size', 'test_input_size', 'test_output_size']
    
    for idx, key in enumerate(plot_keys):
        ax = axes[idx // 3, idx % 3]
        if key in lengths and lengths[key]:
            ax.boxplot(lengths[key])
            ax.set_title(key.replace('_', ' ').title())
            ax.set_ylabel('Count/Size')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experimental/trelis_dataset_inspection/outliers_boxplot.png', dpi=150)
    print(f"\nBoxplot saved to experimental/trelis_dataset_inspection/outliers_boxplot.png")
    
    for idx in range(len(plot_keys), 9):
        fig.delaxes(axes[idx // 3, idx % 3])
    
    fig2, axes2 = plt.subplots(3, 3, figsize=(15, 12))
    fig2.suptitle('Dataset Length Histograms', fontsize=16)
    
    for idx, key in enumerate(plot_keys):
        ax = axes2[idx // 3, idx % 3]
        if key in lengths and lengths[key]:
            ax.hist(lengths[key], bins=30, edgecolor='black', alpha=0.7)
            ax.set_title(key.replace('_', ' ').title())
            ax.set_xlabel('Count/Size')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
    
    for idx in range(len(plot_keys), 9):
        fig2.delaxes(axes2[idx // 3, idx % 3])
    
    plt.tight_layout()
    plt.savefig('experimental/trelis_dataset_inspection/length_histograms.png', dpi=150)
    print(f"Histograms saved to experimental/trelis_dataset_inspection/length_histograms.png")
    
    if 'total_text_length' in outlier_samples and outlier_samples['total_text_length']:
        print(f"\n{'='*60}")
        print("EXAMINING SPECIFIC OUTLIER SAMPLES")
        print(f"{'='*60}")
        for idx in outlier_samples['total_text_length']:
            sample = dataset[idx]
            print(f"\nSample {idx}:")
            print(f"  Total text length: {text_lengths[idx]:,} characters")
            if 'train' in sample:
                train_data = json.loads(sample['train']) if isinstance(sample['train'], str) else sample['train']
                print(f"  Number of train examples: {len(train_data)}")
            if 'test' in sample:
                test_data = json.loads(sample['test']) if isinstance(sample['test'], str) else sample['test']
                print(f"  Number of test examples: {len(test_data)}")

if __name__ == "__main__":
    analyze_dataset_lengths()