import pandas as pd

df = pd.read_parquet('dataset.parquet')

# Group by task and count programs
task_counts = df.groupby('task_id').size().reset_index(name='program_count')

# Filter for tasks with around 10 programs (let's say 8-15)
mid_size_tasks = task_counts[(task_counts['program_count'] >= 8) & 
                              (task_counts['program_count'] <= 15)].sort_values('program_count')

print("Tasks with 8-15 programs (all are perfect in this dataset):")
print("-" * 50)
for _, row in mid_size_tasks.iterrows():
    task_id = row['task_id']
    count = row['program_count']
    
    # Get the task data to verify all are correct
    task_data = df[df['task_id'] == task_id]
    
    # Check if all predictions are correct
    all_correct = True
    for _, prog in task_data.iterrows():
        if not (all(prog['correct_train_input']) and all(prog['correct_test_input'])):
            all_correct = False
            break
    
    if all_correct:
        print(f"  {task_id}: {count} programs (✓ all perfect)")

# Pick one with exactly 10 if exists, otherwise closest to 10
target = 10
if len(mid_size_tasks) > 0:
    mid_size_tasks['distance'] = abs(mid_size_tasks['program_count'] - target)
    best = mid_size_tasks.sort_values('distance').iloc[0]
    
    print(f"\n📌 Recommended task: {best['task_id']} with {best['program_count']} programs")
    print(f"\nTo browse it: uv run python paginate_programs.py --task {best['task_id']}")