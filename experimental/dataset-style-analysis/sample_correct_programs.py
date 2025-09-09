#!/usr/bin/env python3
import random
from datasets import load_dataset

def filter_fully_correct_programs():
    """Filter dataset to only include programs where all train AND test are correct."""
    print("Loading dataset and filtering for fully correct programs...")
    dataset = load_dataset("Trelis/arc-agi-2-mixed-finetuning-20")
    data = dataset['train']
    
    correct_indices = []
    for idx in range(len(data)):
        row = data[idx]
        train_correct = row['correct_train_input']
        test_correct = row['correct_test_input']
        
        # Check if all train inputs are correct AND all test inputs are correct
        if all(train_correct) and all(test_correct):
            correct_indices.append(idx)
    
    print(f"Found {len(correct_indices)} fully correct programs out of {len(data)} total")
    return data, correct_indices

def sample_correct_programs(data, correct_indices, n=10, iteration=1, seed=None):
    """Sample n programs from the filtered correct programs."""
    if seed:
        random.seed(seed)
    
    print(f"\n{'='*50}")
    print(f"ITERATION {iteration}: Sampling {n} fully correct programs")
    print(f"Available correct programs: {len(correct_indices)}")
    print(f"{'='*50}")
    
    # Sample from the correct indices
    sampled_indices = random.sample(correct_indices, min(n, len(correct_indices)))
    
    programs = []
    for i, idx in enumerate(sampled_indices):
        row = data[idx]
        program_text = row['code']
        
        print(f"\nProgram {i+1} (index {idx}):")
        print(f"Train correct: {row['correct_train_input']}")
        print(f"Test correct: {row['correct_test_input']}")
        print("-" * 40)
        print(program_text)
        print("-" * 40)
        
        programs.append(program_text)
    
    return programs

if __name__ == "__main__":
    data, correct_indices = filter_fully_correct_programs()
    programs_iter1 = sample_correct_programs(data, correct_indices, 10, 1, seed=42)