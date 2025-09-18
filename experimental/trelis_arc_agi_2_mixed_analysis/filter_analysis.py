import pandas as pd
import json
import pyarrow.parquet as pq
from collections import defaultdict

# Load the Trelis dataset
print("Loading Trelis dataset...")
table = pq.read_table('superking_aa2.parquet')
df = table.to_pandas(strings_to_categorical=False, types_mapper=pd.ArrowDtype)
print(f"Dataset shape: {df.shape}")

# Load ARC-Prize-2025 training tasks
print("Loading ARC-Prize-2025 training tasks...")
with open('../../data/arc-prize-2025/arc-agi_training_challenges.json', 'r') as f:
    arc_training_tasks = json.load(f)

arc_training_task_ids = set(arc_training_tasks.keys())
print(f"Found {len(arc_training_task_ids)} training task IDs")

# Filter for tasks that are in the arc-prize-2025 training subset
print("\nFiltering for ARC-Prize-2025 training tasks...")
arc_training_df = df[df['task_id'].isin(arc_training_task_ids)]
print(f"Rows in ARC training subset: {len(arc_training_df)}")

# Filter for non-null refined_from_id
print("\nFiltering for non-null refined_from_id...")
refined_df = arc_training_df[arc_training_df['refined_from_id'].notna()]
print(f"Rows with non-null refined_from_id: {len(refined_df)}")

# Filter for correct_train_input all True
print("\nFiltering for correct_train_input all True...")

def all_true(bool_list):
    """Check if all elements in the boolean list are True"""
    try:
        if bool_list is None:
            return False
        # Convert to list and check if all are True
        bool_values = bool_list.tolist() if hasattr(bool_list, 'tolist') else bool_list
        return all(bool_values) if bool_values else False
    except:
        return False

# Apply the filter
correct_train_df = refined_df[refined_df['correct_train_input'].apply(all_true)]
print(f"Rows with all correct_train_input True: {len(correct_train_df)}")

# Generate the report
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)

if len(correct_train_df) == 0:
    print("No tasks found matching all criteria.")
else:
    # Group by task_id and count programs
    task_program_counts = correct_train_df.groupby('task_id').size().sort_values(ascending=False)

    print(f"\nFound {len(task_program_counts)} unique tasks with {len(correct_train_df)} total programs")
    print("\nTask IDs and program counts:")
    print("-" * 40)

    for task_id, count in task_program_counts.items():
        print(f"{task_id}: {count} programs")

    print(f"\nSummary:")
    print(f"- Total unique tasks: {len(task_program_counts)}")
    print(f"- Total programs: {len(correct_train_df)}")
    print(f"- Average programs per task: {len(correct_train_df) / len(task_program_counts):.2f}")
    print(f"- Max programs per task: {task_program_counts.max()}")
    print(f"- Min programs per task: {task_program_counts.min()}")

# Also show some sample data for verification
print("\nSample of filtered data:")
print(correct_train_df[['task_id', 'refined_from_id', 'correct_train_input']].head(10))

# Save the results to a file
with open('filtered_task_results.txt', 'w') as f:
    if len(correct_train_df) == 0:
        f.write("No tasks found matching all criteria.\n")
    else:
        f.write(f"Tasks from Trelis/arc-agi-2-mixed-finetuning-20 that are:\n")
        f.write(f"- In arc-prize-2025 training subset\n")
        f.write(f"- Have non-null refined_from_id\n")
        f.write(f"- Have all correct_train_input values as True\n\n")
        f.write(f"Total unique tasks: {len(task_program_counts)}\n")
        f.write(f"Total programs: {len(correct_train_df)}\n\n")
        f.write("Task IDs and program counts:\n")
        f.write("-" * 40 + "\n")
        for task_id, count in task_program_counts.items():
            f.write(f"{task_id}: {count} programs\n")

print(f"\nResults saved to: filtered_task_results.txt")