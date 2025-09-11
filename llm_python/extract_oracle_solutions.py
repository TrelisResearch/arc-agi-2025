#!/usr/bin/env python3
"""
Extract oracle solutions from recent experiment runs and update solution counts.

Oracle criteria: test_correct=True AND train_accuracy=1.0
"""

import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import defaultdict
import argparse
from datetime import datetime

def load_solution_counts(file_path: str) -> Dict[str, Optional[int]]:
    """Load current solution counts from JSON file."""
    if not os.path.exists(file_path):
        print(f"⚠️  Solution counts file not found: {file_path}")
        return {}
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to dict for easier access
    counts = {}
    for item in data:
        counts[item['task_id']] = item['solution_programs']
    
    print(f"📊 Loaded solution counts for {len(counts)} tasks")
    return counts

def extract_oracle_programs(log_dir: str, verbose: bool = False) -> Dict[str, List[str]]:
    """Extract oracle programs from a log directory."""
    oracle_programs = defaultdict(list)
    
    task_files = glob.glob(os.path.join(log_dir, "*_simple.json"))
    task_files = [f for f in task_files if not f.endswith("_summary_*_simple.json")]
    
    print(f"🔍 Analyzing {len(task_files)} task files in {os.path.basename(log_dir)}")
    
    for task_file in task_files:
        try:
            with open(task_file, 'r') as f:
                task_data = json.load(f)
            
            task_id = task_data.get('task_id')
            if not task_id:
                continue
                
            # Check each attempt for oracle solutions
            attempt_details = task_data.get('attempt_details', [])
            
            for attempt in attempt_details:
                # Oracle criteria: all_test_correct=True AND all train_results correct
                test_correct = attempt.get('all_test_correct', False)
                train_results = attempt.get('train_results', [])
                
                # Check if all training examples are correct
                all_train_correct = (len(train_results) > 0 and 
                                   all(tr.get('correct', False) for tr in train_results))
                
                if test_correct and all_train_correct:
                    program_code = attempt.get('code', '')
                    if program_code and program_code.strip():
                        oracle_programs[task_id].append(program_code)
                        if verbose:
                            print(f"  ✅ Oracle found for {task_id} (test: {test_correct}, train: {len(train_results)} all correct)")
                        
        except Exception as e:
            if verbose:
                print(f"  ❌ Error processing {task_file}: {e}")
            continue
    
    # Deduplicate programs per task
    for task_id in oracle_programs:
        unique_programs = list(set(oracle_programs[task_id]))
        oracle_programs[task_id] = unique_programs
    
    oracle_tasks = len(oracle_programs)
    total_programs = sum(len(progs) for progs in oracle_programs.values())
    
    print(f"📈 Found {total_programs} oracle programs across {oracle_tasks} tasks")
    
    return dict(oracle_programs)

def update_solution_counts(current_counts: Dict[str, Optional[int]], 
                         new_oracle_programs: Dict[str, List[str]]) -> Dict[str, Optional[int]]:
    """Update solution counts with new oracle programs."""
    updated_counts = current_counts.copy()
    
    for task_id, programs in new_oracle_programs.items():
        current_count = current_counts.get(task_id, 0) or 0
        # Handle case where count might be stored as string
        current_count = int(current_count) if isinstance(current_count, str) else current_count
        new_count = current_count + len(programs)
        updated_counts[task_id] = new_count
        
        print(f"📊 {task_id}: {current_count} → {new_count} (+{len(programs)})")
    
    return updated_counts

def save_enhanced_solution_counts(counts: Dict[str, Optional[int]], output_path: str):
    """Save enhanced solution counts to JSON file."""
    # Convert back to list format
    data = [{"task_id": task_id, "solution_programs": count} 
            for task_id, count in counts.items()]
    
    # Sort by task_id for consistency
    data.sort(key=lambda x: x['task_id'])
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"💾 Saved enhanced solution counts to: {output_path}")

def create_new_subset(counts: Dict[str, Optional[int]], 
                     max_solutions: int = 7,
                     output_path: str = None) -> List[str]:
    """Create a new subset of tasks with ≤max_solutions correct programs."""
    
    tricky_tasks = []
    
    for task_id, count in counts.items():
        # Convert count to int if it's a string, handle None
        if count is None:
            count_int = 0
        else:
            count_int = int(count) if isinstance(count, str) else count
        
        if count is None or count_int <= max_solutions:
            tricky_tasks.append(task_id)
    
    # Sort for consistency
    tricky_tasks.sort()
    
    # Count breakdown
    breakdown = defaultdict(int)
    for task_id in tricky_tasks:
        count = counts[task_id]
        if count is None:
            breakdown['null'] += 1
        else:
            count_int = int(count) if isinstance(count, str) else count
            breakdown[count_int] += 1
    
    print(f"📋 New subset: {len(tricky_tasks)} tasks with ≤{max_solutions} solutions")
    print(f"   Breakdown: {dict(breakdown)}")
    
    if output_path:
        with open(output_path, 'w') as f:
            for task_id in tricky_tasks:
                f.write(f"{task_id}\n")
        print(f"💾 Saved new subset to: {output_path}")
    
    return tricky_tasks

def main():
    parser = argparse.ArgumentParser(description="Extract oracle solutions and create new subset")
    parser.add_argument("--log-dirs", nargs="+", required=True, 
                       help="Log directories to analyze")
    parser.add_argument("--solution-counts", required=True,
                       help="Path to current solution counts JSON file") 
    parser.add_argument("--max-solutions", type=int, default=7,
                       help="Maximum solutions for new subset (default: 7)")
    parser.add_argument("--output-dir", default=".",
                       help="Output directory for enhanced files")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Load current solution counts
    current_counts = load_solution_counts(args.solution_counts)
    
    # Extract oracle programs from all log directories
    all_oracle_programs = defaultdict(list)
    
    for log_dir in args.log_dirs:
        if not os.path.exists(log_dir):
            print(f"⚠️  Log directory not found: {log_dir}")
            continue
            
        print(f"\n🔍 Processing: {log_dir}")
        oracle_programs = extract_oracle_programs(log_dir, args.verbose)
        
        # Merge with all oracle programs
        for task_id, programs in oracle_programs.items():
            all_oracle_programs[task_id].extend(programs)
    
    # Deduplicate across all runs
    for task_id in all_oracle_programs:
        unique_programs = list(set(all_oracle_programs[task_id]))
        all_oracle_programs[task_id] = unique_programs
    
    total_oracle_tasks = len(all_oracle_programs)
    total_oracle_programs = sum(len(progs) for progs in all_oracle_programs.values())
    
    print(f"\n📈 TOTAL: {total_oracle_programs} oracle programs across {total_oracle_tasks} tasks")
    
    if not all_oracle_programs:
        print("❌ No oracle programs found!")
        return
    
    # Update solution counts
    print(f"\n📊 UPDATING SOLUTION COUNTS:")
    updated_counts = update_solution_counts(current_counts, all_oracle_programs)
    
    # Save enhanced solution counts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    enhanced_path = os.path.join(args.output_dir, 
                               f"soar_arc_training_solution_counts_enhanced_{timestamp}.json")
    save_enhanced_solution_counts(updated_counts, enhanced_path)
    
    # Create new subset
    print(f"\n📋 CREATING NEW SUBSET (≤{args.max_solutions} solutions):")
    subset_path = os.path.join(args.output_dir, 
                              f"training_new_tricky_{timestamp}.txt")
    new_subset = create_new_subset(updated_counts, args.max_solutions, subset_path)
    
    print(f"\n✅ SUMMARY:")
    print(f"   📊 Enhanced solution counts: {enhanced_path}")
    print(f"   📋 New subset ({len(new_subset)} tasks): {subset_path}")

if __name__ == "__main__":
    main()