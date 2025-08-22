import pandas as pd

df = pd.read_parquet('dataset.parquet')

# Calculate code length for each program
df['code_length'] = df['code'].str.len()

# Find the longest program
longest = df.loc[df['code_length'].idxmax()]

print(f"LONGEST PROGRAM FOUND:")
print(f"="*50)
print(f"Task ID: {longest['task_id']}")
print(f"Code length: {longest['code_length']:,} characters")
print(f"Model: {longest['model']}")
print(f"Train correct: {sum(longest['correct_train_input'])}/{len(longest['correct_train_input'])}")
print(f"Test correct: {sum(longest['correct_test_input'])}/{len(longest['correct_test_input'])}")

# Show top 10 longest programs
print(f"\n" + "="*50)
print("TOP 10 LONGEST PROGRAMS:")
print(f"="*50)
top_10 = df.nlargest(10, 'code_length')[['task_id', 'code_length', 'model']]
for i, (idx, row) in enumerate(top_10.iterrows(), 1):
    print(f"{i:2d}. Task {row['task_id']}: {row['code_length']:,} chars ({row['model']})")

# Show stats by task for the longest program's task
print(f"\n" + "="*50)
print(f"ALL PROGRAMS FOR TASK {longest['task_id']}:")
print(f"="*50)
task_programs = df[df['task_id'] == longest['task_id']].sort_values('code_length', ascending=False)
print(f"Total programs in this task: {len(task_programs)}")
print(f"Code lengths range: {task_programs['code_length'].min():,} - {task_programs['code_length'].max():,} characters")
print(f"Average code length: {task_programs['code_length'].mean():,.0f} characters")

print(f"\nTo view the longest program:")
print(f"uv run python paginate_programs.py --task {longest['task_id']}")

# Save the longest code to a file for inspection
with open('longest_program.py', 'w') as f:
    f.write(f"# Task: {longest['task_id']}\n")
    f.write(f"# Model: {longest['model']}\n")
    f.write(f"# Length: {longest['code_length']:,} characters\n")
    f.write(f"# Train correct: {sum(longest['correct_train_input'])}/{len(longest['correct_train_input'])}\n")
    f.write(f"# Test correct: {sum(longest['correct_test_input'])}/{len(longest['correct_test_input'])}\n")
    f.write("\n" + longest['code'])

print(f"\nLongest program code saved to: longest_program.py")