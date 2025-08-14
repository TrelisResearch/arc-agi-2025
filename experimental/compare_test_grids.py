#!/usr/bin/env python3
"""
Compare the first test example (input and output grids) for the 6 overlapping tasks
between ARC-AGI-1 and ARC-AGI-2 to determine if they're truly the same tasks.
"""

import json
import os

def load_task(file_path):
    """Load a task JSON file and return the parsed data."""
    with open(file_path, 'r') as f:
        return json.load(f)

def compare_grids(grid1, grid2):
    """Compare two grids and return True if identical."""
    if len(grid1) != len(grid2):
        return False
    for i, row in enumerate(grid1):
        if len(row) != len(grid2[i]):
            return False
        for j, cell in enumerate(row):
            if cell != grid2[i][j]:
                return False
    return True

def main():
    base_path = "/Users/ronanmcgovern/TR/arc-agi-2025/data"
    contaminated_tasks = ["0934a4d8", "136b0064", "16b78196", "981571dc", "aa4ec2a5", "da515329"]
    
    print("=== COMPARING FIRST TEST EXAMPLES FOR OVERLAPPING TASKS ===\n")
    
    for task_id in contaminated_tasks:
        print(f"Task: {task_id}")
        print("-" * 40)
        
        # Load both versions
        arc1_path = f"{base_path}/arc-agi-1/evaluation/{task_id}.json"
        arc2_path = f"{base_path}/arc-agi-2/evaluation/{task_id}.json"
        
        try:
            arc1_task = load_task(arc1_path)
            arc2_task = load_task(arc2_path)
            
            # Get first test example from each
            arc1_test = arc1_task['test'][0]
            arc2_test = arc2_task['test'][0]
            
            # Compare input grids
            input_match = compare_grids(arc1_test['input'], arc2_test['input'])
            
            # Compare output grids
            output_match = compare_grids(arc1_test['output'], arc2_test['output'])
            
            print(f"  Input grids match: {'✓' if input_match else '✗'}")
            print(f"  Output grids match: {'✓' if output_match else '✗'}")
            
            if input_match and output_match:
                print("  ✓ IDENTICAL TASK - Same input and output grids")
            else:
                print("  ✗ DIFFERENT TASKS - Grid content differs")
                
            # Show grid dimensions for context
            arc1_input_dims = f"{len(arc1_test['input'])}x{len(arc1_test['input'][0])}"
            arc1_output_dims = f"{len(arc1_test['output'])}x{len(arc1_test['output'][0])}"
            arc2_input_dims = f"{len(arc2_test['input'])}x{len(arc2_test['input'][0])}"
            arc2_output_dims = f"{len(arc2_test['output'])}x{len(arc2_test['output'][0])}"
            
            print(f"  ARC-AGI-1 dimensions: {arc1_input_dims} -> {arc1_output_dims}")
            print(f"  ARC-AGI-2 dimensions: {arc2_input_dims} -> {arc2_output_dims}")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            
        print()

if __name__ == "__main__":
    main()