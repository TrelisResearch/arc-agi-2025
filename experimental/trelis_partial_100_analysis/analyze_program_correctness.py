from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import pandas as pd

print("Loading Trelis/arc-agi-2-partial-100 dataset...")
dataset = load_dataset("Trelis/arc-agi-2-partial-100", split="train")

# Dictionary to store results per task
task_results = defaultdict(lambda: {"all_train_correct": 0, "at_least_one_train_correct": 0, "total_programs": 0})

print("Analyzing program correctness across tasks...")
for idx, sample in enumerate(dataset):
    if idx % 1000 == 0:
        print(f"  Processed {idx} samples...")
    
    task_id = sample['task_id']
    correct_train = sample['correct_train_input']
    
    # Count total programs for this task
    task_results[task_id]["total_programs"] += 1
    
    # Check if all train examples are correct
    if all(correct_train):
        task_results[task_id]["all_train_correct"] += 1
    
    # Check if at least one train example is correct
    if any(correct_train):
        task_results[task_id]["at_least_one_train_correct"] += 1

print(f"\nTotal unique tasks: {len(task_results)}")
print(f"Total programs analyzed: {idx + 1}")

# Convert to DataFrame for easier manipulation
df_data = []
for task_id, counts in task_results.items():
    df_data.append({
        'task_id': task_id,
        'all_correct': counts['all_train_correct'],
        'at_least_one_correct': counts['at_least_one_train_correct'],
        'total': counts['total_programs']
    })

df = pd.DataFrame(df_data)

# Calculate programs with at least one correct but not all correct
df['some_but_not_all'] = df['at_least_one_correct'] - df['all_correct']

# Sort by number of all_correct programs (descending)
df = df.sort_values('all_correct', ascending=False)

# Print statistics
print("\n=== Overall Statistics ===")
print(f"Total tasks: {len(df)}")
print(f"Total programs: {df['total'].sum()}")
print(f"Programs with all train correct: {df['all_correct'].sum()}")
print(f"Programs with at least one train correct: {df['at_least_one_correct'].sum()}")
print(f"Average programs per task: {df['total'].mean():.2f}")

# Create stacked column chart
fig, ax = plt.subplots(figsize=(20, 10))

# Prepare data for stacking
task_ids = df['task_id'].values
all_correct = df['all_correct'].values
some_but_not_all = df['some_but_not_all'].values

# Create x-axis positions
x_pos = np.arange(len(task_ids))

# Create stacked bars
p1 = ax.bar(x_pos, all_correct, color='#2ecc71', label='All train examples correct')
p2 = ax.bar(x_pos, some_but_not_all, bottom=all_correct, color='#f39c12', label='At least one (but not all) correct')

# Customize chart
ax.set_xlabel('Tasks (sorted by "all correct" count)', fontsize=12)
ax.set_ylabel('Number of Programs', fontsize=12)
ax.set_title('Program Correctness by Task (Trelis/arc-agi-2-partial-100)\nSorted by Number of Programs with All Train Examples Correct', fontsize=14, pad=20)

# Add rough ticks on x-axis for task count
tick_interval = 100  # Show ticks every 100 tasks
tick_positions = list(range(0, len(task_ids), tick_interval))
if len(task_ids) - 1 not in tick_positions:  # Add the last position if not already included
    tick_positions.append(len(task_ids) - 1)
ax.set_xticks(tick_positions)
ax.set_xticklabels([str(pos) for pos in tick_positions])

# Add legend
ax.legend(loc='upper right', fontsize=10)

# Add grid for better readability
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add summary statistics as text
stats_text = f"Total Tasks: {len(df)} | Total Programs: {df['total'].sum():,}\n"
stats_text += f"All Correct: {df['all_correct'].sum():,} | At Least One Correct: {df['at_least_one_correct'].sum():,}"
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()

# Save the figure
output_file = 'program_correctness_stacked_chart.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nChart saved as: {output_file}")

# Show top 10 tasks with most programs
print("\n=== Top 10 Tasks by Program Count ===")
top_10 = df.nlargest(10, 'total')
for _, row in top_10.iterrows():
    print(f"Task {row['task_id']}: {row['total']} programs "
          f"({row['all_correct']} all correct, {row['some_but_not_all']} partially correct)")

# Show tasks with highest success rate (min 10 programs)
df_filtered = df[df['total'] >= 10].copy()
df_filtered['success_rate'] = df_filtered['all_correct'] / df_filtered['total']
print("\n=== Top 10 Tasks by Success Rate (min 10 programs) ===")
top_success = df_filtered.nlargest(10, 'success_rate')
for _, row in top_success.iterrows():
    print(f"Task {row['task_id']}: {row['all_correct']}/{row['total']} = {row['success_rate']*100:.1f}% all correct")

plt.show()