#!/usr/bin/env python3
"""
Plot showing progress of refinement iterations for manual dataset with gpt-5-mini.
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from the refinement experiments
iterations = ['Initial', 'Refinement 1', 'Refinement 2', 'Refinement 3']

# Metrics data (percentages)
weighted_voting = [20.0, 20.0, 20.0, 20.0]
weighted_voting_excl = [10.0, 10.0, 10.0, 20.0]  # Excluding transductive

oracle = [20.0, 20.0, 20.0, 20.0]
oracle_excl = [10.0, 10.0, 10.0, 20.0]

all_train_correct = [20.0, 40.0, 40.0, 50.0]
all_train_correct_excl = [10.0, 30.0, 20.0, 50.0]

min_1_train_correct = [80.0, 90.0, 80.0, 80.0]
min_1_train_correct_excl = [40.0, 50.0, 60.0, 80.0]

min_1_code_success = [90.0, 100.0, 100.0, 100.0]

# Calculate transductive portions (total - excluding transductive)
weighted_voting_trans = [w - e for w, e in zip(weighted_voting, weighted_voting_excl)]
oracle_trans = [o - e for o, e in zip(oracle, oracle_excl)]
all_train_trans = [a - e for a, e in zip(all_train_correct, all_train_correct_excl)]
min_1_train_trans = [m - e for m, e in zip(min_1_train_correct, min_1_train_correct_excl)]

# Set up the plot
fig, ax = plt.subplots(figsize=(14, 8))

# Bar width and positions
bar_width = 0.15
x = np.arange(len(iterations))

# Create bars for each metric
bars1_excl = ax.bar(x - 2*bar_width, weighted_voting_excl, bar_width,
                   label='Weighted Voting (excl. trans)', color='#1f77b4', alpha=0.8)
bars1_trans = ax.bar(x - 2*bar_width, weighted_voting_trans, bar_width,
                    bottom=weighted_voting_excl, color='#1f77b4', alpha=0.4)

bars2_excl = ax.bar(x - bar_width, oracle_excl, bar_width,
                   label='Oracle (excl. trans)', color='#ff7f0e', alpha=0.8)
bars2_trans = ax.bar(x - bar_width, oracle_trans, bar_width,
                    bottom=oracle_excl, color='#ff7f0e', alpha=0.4)

bars3_excl = ax.bar(x, all_train_correct_excl, bar_width,
                   label='All Train Correct (excl. trans)', color='#2ca02c', alpha=0.8)
bars3_trans = ax.bar(x, all_train_trans, bar_width,
                    bottom=all_train_correct_excl, color='#2ca02c', alpha=0.4)

bars4_excl = ax.bar(x + bar_width, min_1_train_correct_excl, bar_width,
                   label='Min 1 Train Correct (excl. trans)', color='#d62728', alpha=0.8)
bars4_trans = ax.bar(x + bar_width, min_1_train_trans, bar_width,
                    bottom=min_1_train_correct_excl, color='#d62728', alpha=0.4)

bars5 = ax.bar(x + 2*bar_width, min_1_code_success, bar_width,
               label='Min 1 Code Success', color='#9467bd', alpha=0.8)

# Add value labels on bars
def add_labels(bars, values, bottom_values=None):
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        if height > 0:
            y_pos = height / 2
            if bottom_values:
                y_pos += bottom_values[i]
            ax.annotate(f'{val:.0f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                       xytext=(0, 0), textcoords='offset points',
                       ha='center', va='center', fontsize=8, fontweight='bold')

# Add labels for non-transductive parts
add_labels(bars1_excl, weighted_voting_excl)
add_labels(bars2_excl, oracle_excl)
add_labels(bars3_excl, all_train_correct_excl)
add_labels(bars4_excl, min_1_train_correct_excl)
add_labels(bars5, min_1_code_success)

# Add labels for transductive parts (only if > 0)
add_labels(bars1_trans, [t if t > 0 else 0 for t in weighted_voting_trans], weighted_voting_excl)
add_labels(bars2_trans, [t if t > 0 else 0 for t in oracle_trans], oracle_excl)
add_labels(bars3_trans, [t if t > 0 else 0 for t in all_train_trans], all_train_correct_excl)
add_labels(bars4_trans, [t if t > 0 else 0 for t in min_1_train_trans], min_1_train_correct_excl)

# Customize the plot
ax.set_xlabel('Refinement Iteration', fontsize=12, fontweight='bold')
ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('ARC Task Performance Progress Across Refinement Iterations\n(Manual Dataset, GPT-5-Mini)',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(iterations)
ax.set_ylim(0, 105)

# Add grid for better readability
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Create custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1f77b4', alpha=0.8, label='Weighted Voting (excl. trans)'),
    Patch(facecolor='#ff7f0e', alpha=0.8, label='Oracle (excl. trans)'),
    Patch(facecolor='#2ca02c', alpha=0.8, label='All Train Correct (excl. trans)'),
    Patch(facecolor='#d62728', alpha=0.8, label='Min 1 Train Correct (excl. trans)'),
    Patch(facecolor='#9467bd', alpha=0.8, label='Min 1 Code Success'),
    Patch(facecolor='gray', alpha=0.4, label='Transductive portion')
]

ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98))

# Add summary statistics text box
summary_text = f"""Summary Statistics:
• Total Tasks: 10
• Model: GPT-5-Mini
• Max Attempts: 2
• Final All Train Correct: {all_train_correct[-1]:.0f}% ({all_train_correct_excl[-1]:.0f}% excl. trans)
• Final Min 1 Code Success: {min_1_code_success[-1]:.0f}%"""

ax.text(0.98, 0.02, summary_text, transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        verticalalignment='bottom', horizontalalignment='right')

plt.tight_layout()
plt.savefig('/Users/ronanmcgovern/TR/arc-agi-2025/experimental/refinement_progress.png',
            dpi=300, bbox_inches='tight')
plt.show()

print("📊 Plot saved as refinement_progress.png")
print("\n🔍 Key Observations:")
print(f"• Weighted Voting remained stable at {weighted_voting[0]:.0f}% across all iterations")
print(f"• All Train Correct improved from {all_train_correct[0]:.0f}% to {all_train_correct[-1]:.0f}%")
print(f"• Min 1 Train Correct fluctuated between {min(min_1_train_correct):.0f}%-{max(min_1_train_correct):.0f}%")
print(f"• Min 1 Code Success improved from {min_1_code_success[0]:.0f}% to {min_1_code_success[-1]:.0f}%")
print(f"• Transductive learning was eliminated by iteration 3 (all scores became 'excl. trans')")