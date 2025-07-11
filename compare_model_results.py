#!/usr/bin/env python3

from pathlib import Path

def read_task_ids(filename):
    """Read task IDs from a text file"""
    filepath = Path("data/subsets/arc-agi-1") / filename
    
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return set()
    
    with open(filepath, 'r') as f:
        task_ids = set(line.strip() for line in f if line.strip())
    
    return task_ids

def compare_models():
    """Compare the results between o4-mini and gpt_4_1"""
    print("Comparing model results...")
    print("=" * 60)
    
    # Read both files
    o4_mini_tasks = read_task_ids("o4-mini.txt")
    gpt_4_1_tasks = read_task_ids("gpt_4_1.txt")
    
    if not o4_mini_tasks:
        print("❌ Could not read o4-mini.txt")
        return
    
    if not gpt_4_1_tasks:
        print("❌ Could not read gpt_4_1.txt")
        return
    
    print(f"o4-mini solved: {len(o4_mini_tasks)} tasks")
    print(f"gpt-4.1 solved: {len(gpt_4_1_tasks)} tasks")
    print()
    
    # Calculate overlaps and differences
    overlap = o4_mini_tasks & gpt_4_1_tasks
    only_in_gpt_4_1 = gpt_4_1_tasks - o4_mini_tasks
    only_in_o4_mini = o4_mini_tasks - gpt_4_1_tasks
    
    # Display results
    print(f"OVERLAP (both models solved): {len(overlap)} tasks")
    print(f"ONLY gpt-4.1 solved: {len(only_in_gpt_4_1)} tasks")
    print(f"ONLY o4-mini solved: {len(only_in_o4_mini)} tasks")
    print()
    
    # Show overlap percentage
    total_unique = len(o4_mini_tasks | gpt_4_1_tasks)
    overlap_pct_of_total = (len(overlap) / total_unique) * 100 if total_unique > 0 else 0
    overlap_pct_of_gpt = (len(overlap) / len(gpt_4_1_tasks)) * 100 if gpt_4_1_tasks else 0
    overlap_pct_of_o4 = (len(overlap) / len(o4_mini_tasks)) * 100 if o4_mini_tasks else 0
    
    print(f"Overlap as % of total unique tasks: {overlap_pct_of_total:.1f}%")
    print(f"Overlap as % of gpt-4.1 tasks: {overlap_pct_of_gpt:.1f}%")
    print(f"Overlap as % of o4-mini tasks: {overlap_pct_of_o4:.1f}%")
    print()
    
    # Show tasks that gpt-4.1 solved but o4-mini didn't
    if only_in_gpt_4_1:
        print(f"TASKS ONLY SOLVED BY GPT-4.1 ({len(only_in_gpt_4_1)} tasks):")
        print("-" * 40)
        for task_id in sorted(only_in_gpt_4_1):
            print(f"  {task_id}")
        print()
    
    # Show first few tasks that o4-mini solved but gpt-4.1 didn't
    if only_in_o4_mini:
        print(f"TASKS ONLY SOLVED BY O4-MINI ({len(only_in_o4_mini)} tasks):")
        print("-" * 40)
        # Show first 20 to avoid overwhelming output
        shown_tasks = sorted(only_in_o4_mini)[:20]
        for task_id in shown_tasks:
            print(f"  {task_id}")
        
        if len(only_in_o4_mini) > 20:
            print(f"  ... and {len(only_in_o4_mini) - 20} more")
        print()
    
    # Summary statistics
    print("SUMMARY:")
    print("-" * 40)
    print(f"Total unique tasks solved by either model: {total_unique}")
    print(f"gpt-4.1 performance: {len(gpt_4_1_tasks)} solved")
    print(f"o4-mini performance: {len(o4_mini_tasks)} solved")
    
    if len(gpt_4_1_tasks) > len(o4_mini_tasks):
        diff = len(gpt_4_1_tasks) - len(o4_mini_tasks)
        print(f"gpt-4.1 solved {diff} more tasks than o4-mini")
    elif len(o4_mini_tasks) > len(gpt_4_1_tasks):
        diff = len(o4_mini_tasks) - len(gpt_4_1_tasks)
        print(f"o4-mini solved {diff} more tasks than gpt-4.1")
    else:
        print("Both models solved the same number of tasks")

def main():
    compare_models()

if __name__ == "__main__":
    main() 