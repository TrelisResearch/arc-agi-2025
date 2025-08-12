import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Data extracted from experiment notes (Weighted Voting Pass2 metrics)
models = {
    'Best Length Filtered (Constant LR)': {
        'checkpoints': ['c904', 'c1808', 'c2712', 'c3614'],
        'means': [12.4, None, 13.3, 11.3],  # c1808 missing
        'std_devs': [1.5, None, 0.9, 1.0],
        'color': '#2E86AB',
        'marker': 'o'
    },
    'Best Length Filtered (Annealing)': {
        'checkpoints': ['c904', 'c1808', 'c2712', 'c3616'],
        'means': [15.6, 15.4, 14.1, 10.2],
        'std_devs': [2.4, 0.4, 1.3, 0.4],
        'color': '#A23B72',
        'marker': 's'
    },
    '50 Correct 200 Partial (Constant LR)': {
        'checkpoints': ['c1057', 'c2114', 'c3171', 'c4228'],
        'means': [12.2, 13.8, 16.7, 13.6],
        'std_devs': [2.0, 1.6, 1.0, 0.4],
        'color': '#F18F01',
        'marker': '^'
    }
}

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

# Plot each model
for model_name, data in models.items():
    x_positions = np.arange(len(data['checkpoints']))
    
    # Filter out None values for CST LR model
    valid_indices = [i for i, m in enumerate(data['means']) if m is not None]
    valid_x = [x_positions[i] for i in valid_indices]
    valid_means = [data['means'][i] for i in valid_indices]
    valid_stds = [data['std_devs'][i] for i in valid_indices]
    
    # Plot with error bars (using 2*std for ~95% confidence interval)
    error_bars = [2 * std for std in valid_stds]
    ax.errorbar(valid_x, valid_means, yerr=error_bars,
                label=model_name, color=data['color'], 
                marker=data['marker'], markersize=10,
                linewidth=2, capsize=5, capthick=2,
                alpha=0.8)
    
    # Add value labels
    for x, y, std in zip(valid_x, valid_means, valid_stds):
        ax.annotate(f'{y:.1f}%', 
                   xy=(x, y), 
                   xytext=(0, 10),
                   textcoords='offset points',
                   ha='center',
                   fontsize=9,
                   color=data['color'],
                   fontweight='bold')

# Customize plot
ax.set_xlabel('Checkpoint', fontsize=14, fontweight='bold')
ax.set_ylabel('Weighted Voting Pass@2 Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('Model Performance Across Training Checkpoints\n(3 runs, 8 attempts, arc-agi-1 evaluation set)\nError bars show 2σ (~95% confidence interval)', 
             fontsize=16, fontweight='bold', pad=20)

# Set x-axis labels
all_checkpoints = ['Checkpoint 1', 'Checkpoint 2', 'Checkpoint 3', 'Checkpoint 4']
ax.set_xticks(np.arange(len(all_checkpoints)))
ax.set_xticklabels(all_checkpoints)

# Add grid
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Customize legend
ax.legend(loc='upper left', fontsize=11, framealpha=0.95, 
          edgecolor='black', fancybox=True, shadow=True)

# Set y-axis limits with some padding
ax.set_ylim(8, 20)

# Add background color
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

# Tight layout
plt.tight_layout()

# Save figure
plt.savefig('/Users/ronanmcgovern/TR/arc-agi-2025/experimental/model_checkpoint_comparison/checkpoint_performance_comparison.png', 
            dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.savefig('/Users/ronanmcgovern/TR/arc-agi-2025/experimental/model_checkpoint_comparison/checkpoint_performance_comparison.pdf', 
            bbox_inches='tight', facecolor='white', edgecolor='none')

print("Plot saved as checkpoint_performance_comparison.png and .pdf")