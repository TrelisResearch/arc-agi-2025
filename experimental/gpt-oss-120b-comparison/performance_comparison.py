#!/usr/bin/env python3
"""
GPT-OSS-120B Performance Comparison Across ARC-AGI Subsets
Generate a bar chart comparing Pass@2 performance across different test subsets.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data extracted from experiment notes
subsets = [
    'ARC-AGI-2\nUnique Training',
    'ARC-AGI-2\nAll Evaluation', 
    'ARC-AGI-1\nAll Evaluation'
]

pass_at_2_scores = [12.4, 1.7, 29.8]  # Pass@2 (Weighted Voting) percentages
task_counts = [233, 120, 400]  # Number of tasks in each subset

# Create the bar chart
fig, ax = plt.subplots(figsize=(14, 10))

# Create bars with different colors
colors = ['#2E86AB', '#A23B72', '#F18F01']
bars = ax.bar(subsets, pass_at_2_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2, width=0.6)

# Customize the chart
ax.set_ylabel('Pass@2 Score (%)', fontsize=16, fontweight='bold')
ax.set_title('GPT-OSS-120B Performance Comparison\nAcross ARC-AGI Subsets (1 Attempt)', 
             fontsize=18, fontweight='bold', pad=30)

# Add value labels on bars with better positioning
for bar, score, count in zip(bars, pass_at_2_scores, task_counts):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
            f'{score}%', 
            ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
            f'({count} tasks)', 
            ha='center', va='bottom', fontsize=11, style='italic', color='gray')

# Customize grid and styling
ax.grid(True, alpha=0.3, axis='y')
ax.set_axisbelow(True)
ax.set_ylim(0, max(pass_at_2_scores) * 1.4)

# Add horizontal line at 10% for reference
ax.axhline(y=10, color='red', linestyle='--', alpha=0.7, linewidth=1)
ax.text(0.02, 10.5, '10% Reference Line', transform=ax.get_yaxis_transform(), 
        fontsize=10, color='red', alpha=0.8)

# Styling
plt.xticks(rotation=0, fontsize=13)
plt.yticks(fontsize=13)
plt.tight_layout()

# Add footer with model details - positioned higher to avoid overlap
fig.text(0.5, 0.04, 'Model: openai/gpt-oss-120b via OpenRouter | Max Tokens: 64000 | Single Attempt per Task', 
         ha='center', fontsize=11, style='italic', color='gray')

# Save the chart
plt.savefig('/Users/ronanmcgovern/TR/arc-agi-2025/experimental/gpt-oss-120b-comparison/performance_comparison.png', 
            dpi=300, bbox_inches='tight')
plt.savefig('/Users/ronanmcgovern/TR/arc-agi-2025/experimental/gpt-oss-120b-comparison/performance_comparison.pdf', 
            bbox_inches='tight')

print("Bar chart saved successfully!")
print("\nKey Findings:")
print(f"- Best performance: ARC-AGI-1 All Evaluation ({pass_at_2_scores[2]}%)")
print(f"- Worst performance: ARC-AGI-2 All Evaluation ({pass_at_2_scores[1]}%)")
print(f"- Training vs Evaluation gap (ARC-AGI-2): {pass_at_2_scores[0] - pass_at_2_scores[1]:.1f} percentage points")
print(f"- ARC-AGI-1 vs ARC-AGI-2 evaluation gap: {pass_at_2_scores[2] - pass_at_2_scores[1]:.1f} percentage points")