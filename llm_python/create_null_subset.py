#!/usr/bin/env python3
"""
Create a subset of tasks with null solution programs (no oracle solutions found yet).
"""

import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Create subset of tasks with null solutions")
    parser.add_argument("--solution-counts", required=True,
                       help="Path to solution counts JSON file")
    parser.add_argument("--output", required=True,
                       help="Output subset file path")
    parser.add_argument("--limit", type=int, default=30,
                       help="Maximum number of tasks to include (default: 30)")
    
    args = parser.parse_args()
    
    # Load solution counts
    with open(args.solution_counts, 'r') as f:
        data = json.load(f)
    
    # Find tasks with null solutions
    null_tasks = []
    for item in data:
        if item['solution_programs'] is None:
            null_tasks.append(item['task_id'])
    
    print(f"📊 Found {len(null_tasks)} tasks with null solutions")
    
    # Limit to requested number
    if args.limit and len(null_tasks) > args.limit:
        null_tasks = null_tasks[:args.limit]
        print(f"🎯 Limited to first {args.limit} tasks")
    
    # Sort for consistency
    null_tasks.sort()
    
    # Write subset file
    with open(args.output, 'w') as f:
        for task_id in null_tasks:
            f.write(f"{task_id}\n")
    
    print(f"💾 Created subset with {len(null_tasks)} tasks: {args.output}")
    print(f"📋 Tasks: {', '.join(null_tasks[:10])}{'...' if len(null_tasks) > 10 else ''}")

if __name__ == "__main__":
    main()