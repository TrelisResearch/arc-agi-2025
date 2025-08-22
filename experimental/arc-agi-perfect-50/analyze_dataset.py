import pandas as pd
import json
from collections import defaultdict

print("Loading dataset...")
df = pd.read_parquet('dataset.parquet')

print(f"Total rows in dataset: {len(df)}")
print(f"Unique tasks: {df['task_id'].nunique()}")

# Group by task_id
task_groups = df.groupby('task_id')

# Analyze each task
task_stats = []
for task_id, group in task_groups:
    num_programs = len(group)
    
    # Count programs with all train predictions correct
    all_train_correct = 0
    # Count programs with all test predictions correct
    all_test_correct = 0
    
    for _, row in group.iterrows():
        if all(row['correct_train_input']):
            all_train_correct += 1
        if all(row['correct_test_input']):
            all_test_correct += 1
    
    task_stats.append({
        'task_id': task_id,
        'num_programs': num_programs,
        'all_train_correct': all_train_correct,
        'all_test_correct': all_test_correct,
        'both_correct': all_train_correct if all_train_correct == all_test_correct else 0
    })

stats_df = pd.DataFrame(task_stats)

print("\n" + "="*50)
print("TASK 1: Find task with 10+ programs where ALL predictions are correct")
print("="*50)
perfect_tasks = stats_df[(stats_df['num_programs'] >= 10) & 
                         (stats_df['all_train_correct'] == stats_df['num_programs']) &
                         (stats_df['all_test_correct'] == stats_df['num_programs'])]

if len(perfect_tasks) > 0:
    best_perfect = perfect_tasks.iloc[0]
    print(f"Found {len(perfect_tasks)} tasks with 10+ perfect programs")
    print(f"Selected task: {best_perfect['task_id']}")
    print(f"  - Programs: {best_perfect['num_programs']}")
    print(f"  - All train correct: {best_perfect['all_train_correct']}")
    print(f"  - All test correct: {best_perfect['all_test_correct']}")
else:
    print("No tasks found with 10+ programs where all are perfect")

print("\n" + "="*50)
print("TASK 2: Find task with NO correct test predictions")
print("="*50)
no_test_correct = stats_df[stats_df['all_test_correct'] == 0]
if len(no_test_correct) > 0:
    # Pick one with most programs for variety
    no_test_correct = no_test_correct.sort_values('num_programs', ascending=False)
    selected_no_test = no_test_correct.iloc[0]
    print(f"Found {len(no_test_correct)} tasks with no correct test predictions")
    print(f"Selected task: {selected_no_test['task_id']}")
    print(f"  - Programs: {selected_no_test['num_programs']}")
    print(f"  - All train correct: {selected_no_test['all_train_correct']}")
    print(f"  - All test correct: {selected_no_test['all_test_correct']}")
else:
    print("No tasks found with zero correct test predictions")

print("\n" + "="*50)
print("TASK 3: Find task with FEWEST rows/programs")
print("="*50)
fewest_rows = stats_df.sort_values('num_programs').iloc[0]
print(f"Task with fewest programs: {fewest_rows['task_id']}")
print(f"  - Programs: {fewest_rows['num_programs']}")
print(f"  - All train correct: {fewest_rows['all_train_correct']}")
print(f"  - All test correct: {fewest_rows['all_test_correct']}")

# Save results for later use
results = {
    'perfect_task': best_perfect['task_id'] if len(perfect_tasks) > 0 else None,
    'no_test_task': selected_no_test['task_id'] if len(no_test_correct) > 0 else None,
    'fewest_task': fewest_rows['task_id']
}

with open('selected_tasks.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSelected tasks saved to selected_tasks.json")

# Print some overall statistics
print("\n" + "="*50)
print("OVERALL STATISTICS")
print("="*50)
print(f"Tasks with at least 1 perfect program: {len(stats_df[stats_df['all_test_correct'] > 0])}")
print(f"Tasks with all programs perfect: {len(stats_df[(stats_df['all_test_correct'] == stats_df['num_programs']) & (stats_df['all_train_correct'] == stats_df['num_programs'])])}")
print(f"Average programs per task: {stats_df['num_programs'].mean():.2f}")
print(f"Min programs per task: {stats_df['num_programs'].min()}")
print(f"Max programs per task: {stats_df['num_programs'].max()}")