from datasets import load_dataset
import json
from collections import defaultdict

print("Loading dataset to debug correctness logic...")
dataset = load_dataset("Trelis/arc-agi-2-partial-100", split="train")

# Let's look at a few specific tasks that showed surprising patterns
target_tasks = ['a740d043', 'fc754716', '87ab05b8']  # Tasks that showed 100% all correct, 0% partial

task_samples = defaultdict(list)
sample_count = 0

print("Collecting samples for target tasks...")
for sample in dataset:
    sample_count += 1
    if sample['task_id'] in target_tasks:
        task_samples[sample['task_id']].append(sample)
    
    # Stop after we have enough samples for our target tasks
    if all(len(samples) >= 5 for samples in task_samples.values()) or sample_count > 10000:
        break

print(f"\nAnalyzed {sample_count} samples")

for task_id in target_tasks:
    if task_id in task_samples:
        print(f"\n=== Task {task_id} ===")
        samples = task_samples[task_id][:5]  # Look at first 5 samples
        
        for i, sample in enumerate(samples):
            correct_train = sample['correct_train_input']
            print(f"  Sample {i+1}: correct_train_input = {correct_train}")
            print(f"    - All correct: {all(correct_train)}")
            print(f"    - At least one correct: {any(correct_train)}")
            print(f"    - Length of train examples: {len(correct_train)}")
            
            # Let's also check if there are any edge cases
            if len(correct_train) == 0:
                print(f"    - WARNING: Empty correct_train_input!")
            
            # Check the actual train data structure if available
            print(f"    - Train data available: {'train' in sample}")

print("\n=== Let's also check some tasks with mixed results ===")
# Let's find a task that should have mixed results
mixed_task_found = False
sample_count = 0

for sample in dataset:
    sample_count += 1
    correct_train = sample['correct_train_input']
    
    # Look for a task where we have both all-correct and partial-correct programs
    if len(correct_train) > 1 and any(correct_train) and not all(correct_train):
        print(f"\nFound mixed result task: {sample['task_id']}")
        print(f"  correct_train_input = {correct_train}")
        print(f"  All correct: {all(correct_train)} | At least one correct: {any(correct_train)}")
        mixed_task_found = True
        break
    
    if sample_count > 5000:  # Don't search forever
        break

if not mixed_task_found:
    print("\nNo mixed results found in first 5000 samples - this might explain the pattern!")

print("\n=== Summary of Logic ===")
print("Our current logic:")
print("- 'All correct': all(correct_train_input) == True")
print("- 'At least one correct': any(correct_train_input) == True") 
print("- 'Some but not all': any(correct_train_input) == True AND all(correct_train_input) == False")
print("\nIf most programs either get ALL training examples right or ALL wrong,")
print("then we'd see very few 'partial' cases, which might explain the pattern.")