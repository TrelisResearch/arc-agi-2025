#!/usr/bin/env python3

import json
import sys
from pathlib import Path

def load_task_results(summary_file_path):
    """Load task results from a summary file"""
    try:
        with open(summary_file_path, 'r') as f:
            data = json.load(f)
        
        # Extract results
        results = data.get('results', [])
        
        # Create a mapping of task_id -> best result (True if any attempt succeeded)
        task_results = {}
        for result in results:
            task_id = result['task_id']
            attempt_details = result.get('attempt_details', [])
            
            # Check if any attempt was successful
            any_success = False
            for attempt in attempt_details:
                if attempt.get('test_correct', False):
                    any_success = True
                    break
            
            task_results[task_id] = any_success
        
        return task_results
    except Exception as e:
        print(f"Error loading {summary_file_path}: {e}")
        return {}

def create_visual_table():
    """Create a visual table with green checks and red X's"""
    
    # File paths for the 3 runs
    forward_files = [
        "/Users/ronanmcgovern/TR/arc-agi-2025/llm-python/logs/20250728_140522/20250728_140551_summary_arc-agi-1_shortest_evaluation_30_simple_run1.json",
        "/Users/ronanmcgovern/TR/arc-agi-2025/llm-python/logs/20250728_140553/20250728_140625_summary_arc-agi-1_shortest_evaluation_30_simple_run2.json",
        "/Users/ronanmcgovern/TR/arc-agi-2025/llm-python/logs/20250728_140626/20250728_140653_summary_arc-agi-1_shortest_evaluation_30_simple_run3.json"
    ]
    
    reverse_files = [
        "/Users/ronanmcgovern/TR/arc-agi-2025/llm-python/logs/20250728_133752/20250728_133821_summary_arc-agi-1r_shortest_evaluation_30r_simple_run1.json",
        "/Users/ronanmcgovern/TR/arc-agi-2025/llm-python/logs/20250728_133823/20250728_133849_summary_arc-agi-1r_shortest_evaluation_30r_simple_run2.json", 
        "/Users/ronanmcgovern/TR/arc-agi-2025/llm-python/logs/20250728_133851/20250728_133919_summary_arc-agi-1r_shortest_evaluation_30r_simple_run3.json"
    ]
    
    # Load all results
    forward_results = []
    reverse_results = []
    
    print("Loading data...")
    for i, (forward_file, reverse_file) in enumerate(zip(forward_files, reverse_files)):
        forward_data = load_task_results(forward_file)
        reverse_data = load_task_results(reverse_file)
        forward_results.append(forward_data)
        reverse_results.append(reverse_data)
        print(f"  Run {i+1}: {len(forward_data)} forward, {len(reverse_data)} reverse tasks")
    
    # Get all task IDs (remove 'r' suffix from reverse to match)
    all_task_ids = set()
    
    # From forward results
    for forward_data in forward_results:
        all_task_ids.update(forward_data.keys())
    
    # From reverse results (remove 'r' suffix)
    for reverse_data in reverse_results:
        for task_id in reverse_data.keys():
            base_task_id = task_id[:-1] if task_id.endswith('r') else task_id
            all_task_ids.add(base_task_id)
    
    task_ids_sorted = sorted(all_task_ids)
    
    print(f"\n" + "="*80)
    print("🎯 ARC-AGI FORWARD vs REVERSE PERFORMANCE")
    print("="*80)
    print(f"📊 {len(task_ids_sorted)} tasks × 3 runs × 2 directions")
    print()
    
    # Create header
    header = "Task ID".ljust(12)
    header += "Fwd R1".ljust(8) + "Fwd R2".ljust(8) + "Fwd R3".ljust(8) + "FwdWgt".ljust(8)
    header += "Rev R1".ljust(8) + "Rev R2".ljust(8) + "Rev R3".ljust(8) + "RevWgt".ljust(8)
    
    print(header)
    print("-" * len(header))
    
    # Calculate weighted scores for each task and sort
    task_scores = []
    
    for task_id in task_ids_sorted:
        forward_task_id = task_id
        reverse_task_id = task_id + 'r'
        
        # Calculate forward weighted score (percentage of runs successful)
        forward_successes_task = 0
        for run_idx in range(3):
            if forward_results[run_idx].get(forward_task_id, False):
                forward_successes_task += 1
        forward_weighted = forward_successes_task / 3.0
        
        # Calculate reverse weighted score
        reverse_successes_task = 0
        for run_idx in range(3):
            if reverse_results[run_idx].get(reverse_task_id, False):
                reverse_successes_task += 1
        reverse_weighted = reverse_successes_task / 3.0
        
        task_scores.append((task_id, forward_weighted, reverse_weighted))
    
    # Sort by forward weighted score (descending), then by reverse weighted score (descending)
    task_scores.sort(key=lambda x: (-x[1], -x[2]))
    
    # Counters for statistics
    forward_successes = [0, 0, 0]  # Run 1, 2, 3
    reverse_successes = [0, 0, 0]
    total_tasks = len(task_scores)
    
    # Process each task in sorted order
    for task_id, forward_weighted, reverse_weighted in task_scores:
        forward_task_id = task_id
        reverse_task_id = task_id + 'r'
        
        row = task_id.ljust(12)
        
        # Add all forward results first
        for run_idx in range(3):
            forward_success = forward_results[run_idx].get(forward_task_id, False)
            if forward_success:
                forward_successes[run_idx] += 1
            forward_symbol = "✅" if forward_success else "❌"
            row += f"  {forward_symbol}  ".ljust(8)
        
        # Add forward weighted score
        row += f"{forward_weighted:.2f}".ljust(8)
        
        # Then add all reverse results
        for run_idx in range(3):
            reverse_success = reverse_results[run_idx].get(reverse_task_id, False)
            if reverse_success:
                reverse_successes[run_idx] += 1
            reverse_symbol = "✅" if reverse_success else "❌"
            row += f"  {reverse_symbol}  ".ljust(8)
        
        # Add reverse weighted score
        row += f"{reverse_weighted:.2f}".ljust(8)
        
        print(row)
    
    # Summary statistics
    print("\n" + "="*80)
    print("📈 SUMMARY STATISTICS")
    print("="*80)
    
    summary_header = "".ljust(12) + "Fwd R1".ljust(8) + "Fwd R2".ljust(8) + "Fwd R3".ljust(8) + "FwdWgt".ljust(8) + "Rev R1".ljust(8) + "Rev R2".ljust(8) + "Rev R3".ljust(8) + "RevWgt".ljust(8)
    print(summary_header)
    print("-" * len(summary_header))
    
    # Success counts
    success_row = "Success:".ljust(12)
    # Forward counts first
    for run_idx in range(3):
        success_row += f"{forward_successes[run_idx]:2d}/30".ljust(8)
    # Average forward weighted score
    avg_forward_weighted = sum(score[1] for score in task_scores) / len(task_scores)
    success_row += f"{avg_forward_weighted:.2f}".ljust(8)
    # Then reverse counts
    for run_idx in range(3):
        success_row += f"{reverse_successes[run_idx]:2d}/30".ljust(8)
    # Average reverse weighted score
    avg_reverse_weighted = sum(score[2] for score in task_scores) / len(task_scores)
    success_row += f"{avg_reverse_weighted:.2f}".ljust(8)
    print(success_row)
    
    # Percentages
    percent_row = "Percent:".ljust(12)
    # Forward percentages first
    for run_idx in range(3):
        forward_pct = forward_successes[run_idx] / total_tasks * 100
        percent_row += f"{forward_pct:4.1f}%".ljust(8)
    # Average forward weighted percentage
    percent_row += f"{avg_forward_weighted*100:4.1f}%".ljust(8)
    # Then reverse percentages
    for run_idx in range(3):
        reverse_pct = reverse_successes[run_idx] / total_tasks * 100
        percent_row += f"{reverse_pct:4.1f}%".ljust(8)
    # Average reverse weighted percentage
    percent_row += f"{avg_reverse_weighted*100:4.1f}%".ljust(8)
    print(percent_row)
    
    # Ratio (reverse/forward)
    ratio_row = "Ratio:".ljust(12)
    # Skip forward ratios (they're always 1.0)
    for run_idx in range(3):
        ratio_row += "".ljust(8)
    # Weighted ratio
    if avg_forward_weighted > 0:
        weighted_ratio = avg_reverse_weighted / avg_forward_weighted
        ratio_row += f"{weighted_ratio:.2f}x".ljust(8)
    else:
        ratio_row += "N/A".ljust(8)
    # Add reverse/forward ratios
    for run_idx in range(3):
        if forward_successes[run_idx] > 0:
            ratio = reverse_successes[run_idx] / forward_successes[run_idx]
            ratio_row += f"{ratio:.2f}x".ljust(8)
        else:
            ratio_row += "N/A".ljust(8)
    ratio_row += "".ljust(8)  # Skip reverse weighted ratio (redundant)
    print(ratio_row)
    
    print("\n" + "="*80)
    print("🔍 KEY INSIGHTS")
    print("="*80)
    
    # Overall statistics
    total_forward = sum(forward_successes)
    total_reverse = sum(reverse_successes)
    total_attempts = total_tasks * 3
    
    print(f"Overall Forward Performance: {total_forward}/{total_attempts} = {total_forward/total_attempts*100:.1f}%")
    print(f"Overall Reverse Performance: {total_reverse}/{total_attempts} = {total_reverse/total_attempts*100:.1f}%")
    print(f"Reverse Difficulty Factor: {total_reverse/total_forward:.2f}x harder than forward")
    
    # Consistency analysis
    forward_consistent = sum(1 for i in range(total_tasks) 
                           if all(task_ids_sorted[i] in forward_results[j] and forward_results[j][task_ids_sorted[i]] 
                                 for j in range(3)))
    reverse_consistent = sum(1 for i in range(total_tasks) 
                           if all(task_ids_sorted[i] + 'r' in reverse_results[j] and reverse_results[j][task_ids_sorted[i] + 'r'] 
                                 for j in range(3)))
    
    print(f"Tasks solved consistently (all 3 runs):")
    print(f"  Forward: {forward_consistent}/{total_tasks} tasks")
    print(f"  Reverse: {reverse_consistent}/{total_tasks} tasks")

def main():
    create_visual_table()

if __name__ == "__main__":
    main() 