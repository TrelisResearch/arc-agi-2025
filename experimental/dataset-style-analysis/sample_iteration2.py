#!/usr/bin/env python3
import random
from datasets import load_dataset

# Filter for correct programs
dataset = load_dataset("Trelis/arc-agi-2-mixed-finetuning-20")
data = dataset['train']

correct_indices = []
for idx in range(len(data)):
    row = data[idx]
    if all(row['correct_train_input']) and all(row['correct_test_input']):
        correct_indices.append(idx)

# Sample iteration 2 with different seed
random.seed(789)
sampled_indices = random.sample(correct_indices, 10)

print("="*50)
print("ITERATION 2: Sampling 10 fully correct programs")
print("="*50)

for i, idx in enumerate(sampled_indices):
    row = data[idx]
    print(f"\nProgram {i+1} (index {idx}):")
    print("-" * 40)
    print(row['code'])
    print("-" * 40)