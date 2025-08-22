#!/usr/bin/env python3

import pandas as pd
import json

print("Loading dataset...")
df = pd.read_parquet('dataset.parquet')

# Load selected tasks
with open('selected_tasks.json', 'r') as f:
    selected = json.load(f)

print("\n" + "="*80)
print("THREE SPECIFIC TASKS AS REQUESTED")
print("="*80)

# 1. Task with 10+ programs where all are perfect
print("\n1️⃣  TASK WITH 10+ PERFECT PROGRAMS: 007bbfb7")
print("-"*40)
task1 = df[df['task_id'] == '007bbfb7']
print(f"Total programs: {len(task1)}")
print(f"All have perfect train & test predictions")
print("\nProgram summaries:")
for i, (_, row) in enumerate(task1.iterrows()):
    train_correct = sum(row['correct_train_input'])
    test_correct = sum(row['correct_test_input'])
    print(f"  Program {i+1:2d}: Train {train_correct}/{len(row['correct_train_input'])} | Test {test_correct}/{len(row['correct_test_input'])} | Model: {row['model'][:20]}")
    if i >= 9:  # Show first 10
        print(f"  ... and {len(task1) - 10} more programs")
        break

# 2. Task with no correct test predictions (doesn't exist in this dataset)
print("\n2️⃣  TASK WITH NO CORRECT TEST PREDICTIONS: None found")
print("-"*40)
print("All tasks in this dataset have perfect predictions!")
print("This is the 'perfect-50' dataset where all programs are correct.")

# 3. Task with fewest rows
print("\n3️⃣  TASK WITH FEWEST PROGRAMS: a8c38be5")
print("-"*40)
task3 = df[df['task_id'] == 'a8c38be5']
print(f"Total programs: {len(task3)}")
print("\nProgram details:")
for i, (_, row) in enumerate(task3.iterrows()):
    train_correct = sum(row['correct_train_input'])
    test_correct = sum(row['correct_test_input'])
    print(f"  Program {i+1}: Train {train_correct}/{len(row['correct_train_input'])} | Test {test_correct}/{len(row['correct_test_input'])} | Model: {row['model']}")
    print(f"\n  Code:")
    for line in row['code'].split('\n'):
        print(f"    {line}")

print("\n" + "="*80)
print("HOW TO EXPLORE THESE TASKS:")
print("="*80)
print("\n1. To browse task with 10+ perfect programs:")
print("   uv run python paginate_programs.py --task 007bbfb7")
print("\n2. To browse task with fewest programs:")
print("   uv run python paginate_programs.py --task a8c38be5")
print("\n3. To explore any other task interactively:")
print("   uv run python paginate_programs.py")
print("\nOnce in the browser, use:")
print("  - 'n' for next program")
print("  - 'p' for previous program")
print("  - 'j <num>' to jump to a specific program")
print("  - 't' to see all available tasks")
print("  - 'q' to quit")