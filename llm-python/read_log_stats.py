#!/usr/bin/env python3
"""
Script to retrospectively read and display statistics from ARC task runner log directories.

Usage:
    # Single directory
    python read_log_stats.py logs/20250728_114716
    
    # Multiple directories (for repeated runs)  
    python read_log_stats.py logs/20250728_113731 logs/20250728_114716 logs/20250728_115648
    
    # Auto-discover all runs from a date
    python read_log_stats.py --pattern 20250728
"""

import json
import argparse
from pathlib import Path
import statistics
from typing import List, Dict, Any

def load_summary_files(log_dir: Path) -> List[Dict[str, Any]]:
    """Load all summary JSON files from a log directory"""
    summary_files = []
    
    # Look for summary files (both single run and multi-run formats)
    patterns = ["*summary*.json"]
    
    for pattern in patterns:
        for file_path in log_dir.glob(pattern):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    data['_filename'] = file_path.name
                    data['_directory'] = str(log_dir)
                    summary_files.append(data)
                print(f"✅ Loaded: {file_path.name}")
            except Exception as e:
                print(f"❌ Failed to load {file_path.name}: {e}")
    
    return summary_files

def display_single_run_stats(data: Dict[str, Any]):
    """Display statistics for a single run"""
    print("\n" + "="*60)
    print(f"SINGLE RUN STATISTICS")
    print("="*60)
    
    # Basic info
    print(f"Dataset: {data.get('dataset', 'Unknown')}")
    print(f"Subset: {data.get('subset', 'Unknown')}")
    print(f"Model: {data.get('model', 'Unknown')}")
    print(f"Timestamp: {data.get('timestamp', 'Unknown')}")
    
    # Task stats
    total_tasks = data.get('total_tasks', 0)
    successful_api = data.get('successful_api_calls', 0)
    print(f"Total tasks: {total_tasks}")
    print(f"Successful API calls: {successful_api}/{total_tasks} ({successful_api/total_tasks:.1%})")
    
    # Cost and tokens
    total_tokens = data.get('total_tokens', 0)
    total_cost = data.get('total_cost', 0.0)
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total cost: ${total_cost:.6f}")
    
    # Core metrics
    metrics = data.get('metrics', {})
    if metrics:
        print("\n📊 CORE METRICS:")
        print(f"  Pass@2 (Weighted Voting): {metrics.get('weighted_voting_pass2', 0):.1%}")
        print(f"  Pass@2 (Train Majority):  {metrics.get('train_majority_pass2', 0):.1%}")
        print(f"  Oracle (Best Attempt):    {metrics.get('oracle_correct', 0):.1%}")
        print(f"  All Train Correct:        {metrics.get('all_train_correct', 0):.1%}")
        print(f"  Min 1 Train Correct:      {metrics.get('min1_train_correct', 0):.1%}")
        print(f"  Max Length Responses:     {metrics.get('max_length_responses', 0):.1%}")
        print(f"  Timeout Responses:        {metrics.get('timeout_responses', 0):.1%}")
        print(f"  API Failure Responses:    {metrics.get('api_failure_responses', 0):.1%}")

def display_multi_run_stats(summaries: List[Dict[str, Any]]):
    """Display aggregate statistics for multiple runs"""
    print("\n" + "="*60)
    print(f"MULTI-RUN AGGREGATE STATISTICS ({len(summaries)} runs)")
    print("="*60)
    
    # Basic info from first run
    first_run = summaries[0]
    print(f"Dataset: {first_run.get('dataset', 'Unknown')}")
    print(f"Subset: {first_run.get('subset', 'Unknown')}")
    print(f"Model: {first_run.get('model', 'Unknown')}")
    print(f"Number of runs: {len(summaries)}")
    
    # Individual run results table
    print("\nINDIVIDUAL RUN RESULTS:")
    print("-" * 80)
    print("Run  Tasks  Weighted   Train-Maj  Oracle   All-Train  Min1-Train  Max-Len")
    print("-" * 80)
    
    for i, summary in enumerate(summaries, 1):
        metrics = summary.get('metrics', {})
        tasks = summary.get('total_tasks', 0)
        print(f"{i:<4} {tasks:<6} {metrics.get('weighted_voting_pass2', 0):<10.1%} "
              f"{metrics.get('train_majority_pass2', 0):<10.1%} {metrics.get('oracle_correct', 0):<8.1%} "
              f"{metrics.get('all_train_correct', 0):<10.1%} {metrics.get('min1_train_correct', 0):<11.1%} "
              f"{metrics.get('max_length_responses', 0):<7.1%}")
    
    # Aggregate statistics
    print("\nAGGREGATE STATISTICS:")
    print("-" * 80)
    
    metric_names = [
        ('weighted_voting_pass2', 'Weighted Voting Pass2'),
        ('train_majority_pass2', 'Train Majority Pass2'),
        ('oracle_correct', 'Oracle Correct'),
        ('all_train_correct', 'All Train Correct'),
        ('min1_train_correct', 'Min1 Train Correct'),
        ('max_length_responses', 'Max Length Responses'),
        ('timeout_responses', 'Timeout Responses'),
        ('api_failure_responses', 'API Failure Responses')
    ]
    
    for metric_key, metric_name in metric_names:
        values = [summary.get('metrics', {}).get(metric_key, 0) for summary in summaries]
        if values and any(v > 0 for v in values):
            mean_val = statistics.mean(values)
            std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
            
            # 95% confidence interval
            margin = 1.96 * std_dev if len(values) > 1 else 0.0
            ci_low = max(0, mean_val - margin)
            ci_high = min(1, mean_val + margin)
            
            print(f"{metric_name}:")
            print(f"  Mean: {mean_val:.1%}")
            print(f"  Std Dev: {std_dev:.1%}")
            print(f"  95% CI: [{ci_low:.1%}, {ci_high:.1%}]")
            print()

def main():
    parser = argparse.ArgumentParser(description="Read and display ARC task runner log statistics")
    parser.add_argument("log_dirs", nargs="*", help="Path(s) to log directory(ies) (e.g., logs/20250728_114716 logs/20250728_115648)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    parser.add_argument("--pattern", help="Auto-find directories matching date pattern (e.g., 20250728)")
    
    args = parser.parse_args()
    
    # Validate that either log_dirs or pattern is provided
    if not args.log_dirs and not args.pattern:
        parser.error("Must provide either log directories or --pattern")
    
    if args.log_dirs and args.pattern:
        parser.error("Cannot use both log_dirs and --pattern at the same time")
    
    # Handle pattern matching for auto-discovery
    if args.pattern:
        log_dirs = []
        logs_parent = Path("logs")
        if logs_parent.exists():
            for dir_path in logs_parent.glob(f"{args.pattern}_*"):
                if dir_path.is_dir():
                    log_dirs.append(dir_path)
            log_dirs.sort()  # Sort chronologically
        else:
            print(f"❌ logs/ directory not found")
            return
        
        if not log_dirs:
            print(f"❌ No directories found matching pattern: {args.pattern}")
            return
            
        print(f"🔍 Auto-discovered {len(log_dirs)} directories matching '{args.pattern}':")
        for d in log_dirs:
            print(f"   {d}")
    else:
        # Use provided directories
        log_dirs = [Path(d) for d in args.log_dirs]
        
        # Validate directories
        for log_dir in log_dirs:
            if not log_dir.exists():
                print(f"❌ Log directory not found: {log_dir}")
                return
            if not log_dir.is_dir():
                print(f"❌ Path is not a directory: {log_dir}")
                return
    
    print(f"\n🔍 Reading log statistics from {len(log_dirs)} directories...")
    
    # Load all summary files from all directories
    all_summaries = []
    for log_dir in log_dirs:
        print(f"\n📁 Processing: {log_dir}")
        summaries = load_summary_files(log_dir)
        all_summaries.extend(summaries)
    
    if not all_summaries:
        print("❌ No summary files found in any directories")
        return
    
    print(f"\n📊 Found {len(all_summaries)} summary file(s) total")
    
    # Filter out aggregate summary files (they're duplicates)
    run_summaries = [s for s in all_summaries if not 'aggregate_summary' in s.get('_filename', '')]
    
    if not run_summaries:
        print("❌ No individual run summary files found")
        return
    
    print(f"📊 Found {len(run_summaries)} individual run summary file(s)")
    
    # Sort by run number if available
    def get_run_number(summary):
        return summary.get('run_number', 0)
    
    run_summaries.sort(key=get_run_number)
    
    if len(run_summaries) == 1:
        display_single_run_stats(run_summaries[0])
    else:
        display_multi_run_stats(run_summaries)
    
    # Show file details if verbose
    if args.verbose:
        print("\n" + "="*60)
        print("FILE DETAILS:")
        print("="*60)
        for summary in run_summaries:
            print(f"📄 {summary['_filename']}")
            print(f"   Directory: {summary.get('_directory', 'Unknown')}")
            print(f"   Tasks: {summary.get('total_tasks', 0)}")
            print(f"   Tokens: {summary.get('total_tokens', 0):,}")
            print(f"   Cost: ${summary.get('total_cost', 0.0):.6f}")
            print(f"   Run: {summary.get('run_number', 'N/A')}")
            print()

if __name__ == "__main__":
    main() 