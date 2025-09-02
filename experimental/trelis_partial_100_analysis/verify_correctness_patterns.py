from datasets import load_dataset
from collections import defaultdict, Counter

print("Loading dataset to verify correctness patterns...")
dataset = load_dataset("Trelis/arc-agi-2-partial-100", split="train")

# Track detailed patterns
pattern_counts = Counter()
task_patterns = defaultdict(lambda: {"all_correct": 0, "partial": 0, "all_wrong": 0, "samples": []})

print("Analyzing all samples...")
for idx, sample in enumerate(dataset):
    if idx % 5000 == 0:
        print(f"  Processed {idx} samples...")
    
    task_id = sample['task_id']
    correct_train = sample['correct_train_input']
    
    # Categorize this program
    if all(correct_train):
        category = "all_correct"
        task_patterns[task_id]["all_correct"] += 1
    elif any(correct_train):
        category = "partial"
        task_patterns[task_id]["partial"] += 1
    else:
        category = "all_wrong"
        task_patterns[task_id]["all_wrong"] += 1
    
    # Track the specific pattern
    pattern = tuple(correct_train)
    pattern_counts[pattern] += 1
    
    # Keep some examples for analysis
    if len(task_patterns[task_id]["samples"]) < 3:
        task_patterns[task_id]["samples"].append({
            "correct_train": correct_train,
            "category": category
        })

total_samples = idx + 1
print(f"\nTotal samples: {total_samples}")

# Analyze the patterns
print(f"\n=== Most Common Correctness Patterns ===")
for pattern, count in pattern_counts.most_common(10):
    percentage = (count / total_samples) * 100
    print(f"{pattern}: {count:,} ({percentage:.1f}%)")

# Look at tasks with surprising patterns (high all_correct, low partial)
print(f"\n=== Tasks with Surprising Patterns (High All-Correct, Low Partial) ===")
surprising_tasks = []
for task_id, counts in task_patterns.items():
    total_task = counts["all_correct"] + counts["partial"] + counts["all_wrong"]
    if total_task >= 50:  # Only look at tasks with reasonable sample size
        all_correct_pct = (counts["all_correct"] / total_task) * 100
        partial_pct = (counts["partial"] / total_task) * 100
        
        # Flag tasks with >80% all correct and <5% partial
        if all_correct_pct > 80 and partial_pct < 5:
            surprising_tasks.append((task_id, all_correct_pct, partial_pct, total_task))

surprising_tasks.sort(key=lambda x: x[1], reverse=True)  # Sort by all_correct percentage

print(f"Found {len(surprising_tasks)} tasks with >80% all correct and <5% partial:")
for task_id, all_pct, partial_pct, total in surprising_tasks[:10]:
    print(f"  {task_id}: {all_pct:.1f}% all correct, {partial_pct:.1f}% partial ({total} programs)")
    
    # Show some examples
    samples = task_patterns[task_id]["samples"]
    for i, sample in enumerate(samples):
        print(f"    Example {i+1}: {sample['correct_train']} -> {sample['category']}")

# Overall statistics
all_correct_total = sum(counts["all_correct"] for counts in task_patterns.values())
partial_total = sum(counts["partial"] for counts in task_patterns.values())
all_wrong_total = sum(counts["all_wrong"] for counts in task_patterns.values())

print(f"\n=== Overall Distribution ===")
print(f"All correct: {all_correct_total:,} ({(all_correct_total/total_samples)*100:.1f}%)")
print(f"Partial: {partial_total:,} ({(partial_total/total_samples)*100:.1f}%)")
print(f"All wrong: {all_wrong_total:,} ({(all_wrong_total/total_samples)*100:.1f}%)")

# Check if this makes sense
print(f"\n=== Validation ===")
calculated_total = all_correct_total + partial_total + all_wrong_total
print(f"Sum of categories: {calculated_total}")
print(f"Actual total: {total_samples}")
print(f"Match: {calculated_total == total_samples}")