#!/usr/bin/env python3

import pandas as pd
from pathlib import Path
import sys

# Add llm_python to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from llm_python.datasets.io import read_soar_parquet

def check_all_correct(x):
    """Helper to check if all examples are correct"""
    if x is None or (hasattr(x, '__len__') and len(x) == 0):
        return False
    if hasattr(x, 'to_pylist'):
        x = x.to_pylist()
    return all(x)

def analyze_parquet_for_success(parquet_path):
    """Analyze a parquet file and return tasks with all-correct programs"""
    df = read_soar_parquet(parquet_path)
    
    # Filter non-transductive only
    non_trans_df = df[df['is_transductive'] == False].copy()
    
    # Add all-correct analysis
    non_trans_df['all_train_correct'] = non_trans_df['correct_train_input'].apply(check_all_correct)
    non_trans_df['all_test_correct'] = non_trans_df['correct_test_input'].apply(check_all_correct)
    non_trans_df['all_correct'] = non_trans_df['all_train_correct'] & non_trans_df['all_test_correct']
    
    # Get task-level statistics
    task_stats = non_trans_df.groupby('task_id').agg({
        'all_correct': 'sum',
        'row_id': 'count'
    }).rename(columns={'row_id': 'total_programs', 'all_correct': 'all_correct_count'})
    
    # Return dict of task_id -> count of all-correct programs
    return task_stats[task_stats['all_correct_count'] > 0]['all_correct_count'].to_dict()

def combine_results(baseline_results, other_results):
    """Combine baseline results with another experiment's results"""
    combined = {}
    all_tasks = set(baseline_results.keys()) | set(other_results.keys())
    
    for task in all_tasks:
        # For combined analysis, we take the MAX of either run
        # (assuming we'd pick the best solution from either approach)
        baseline_count = baseline_results.get(task, 0)
        other_count = other_results.get(task, 0)
        if baseline_count > 0 or other_count > 0:
            combined[task] = max(baseline_count, other_count)
    
    return combined

def main():
    # Define the experiments
    experiments = [
        {
            'file': '20250902_142348_openai_gpt-oss-120b_arc-prize-2025_training-hard.parquet',
            'label': 'Baseline 1',
            'description': 'No refinement (Sept 2)'
        },
        {
            'file': '20250903_093510_openai_gpt-oss-120b_arc-prize-2025_training-hard.parquet',
            'label': 'Baseline 2', 
            'description': 'No refinement (Sept 3)'
        },
        {
            'file': '20250903_093511_openai_gpt-oss-120b_arc-prize-2025_training-hard.parquet',
            'label': 'Refine + Diff',
            'description': 'Refinement with output diffs'
        },
        {
            'file': '20250903_094032_openai_gpt-oss-120b_arc-prize-2025_training-hard.parquet',
            'label': 'Refine + Full',
            'description': 'Refinement with full outputs'
        },
        {
            'file': '20250903_094037_openai_gpt-oss-120b_arc-prize-2025_training-hard.parquet',
            'label': 'Refine Only',
            'description': 'Refinement with program only'
        }
    ]
    
    parquet_dir = Path(__file__).parent.parent.parent / "llm_python/datasets/inference"
    
    # Load all individual results
    print("Loading individual experiment results...")
    for exp in experiments:
        parquet_path = parquet_dir / exp['file']
        if parquet_path.exists():
            exp['results'] = analyze_parquet_for_success(parquet_path)
            print(f"  - {exp['label']}: {len(exp['results'])} tasks with success")
        else:
            print(f"  ⚠️ {exp['label']}: File not found")
            exp['results'] = {}
    
    # Get baseline 1 results
    baseline1_results = experiments[0]['results']
    
    # Create combined results
    combined_experiments = []
    for exp in experiments[1:]:  # Skip baseline 1 itself
        combined_results = combine_results(baseline1_results, exp['results'])
        combined_experiments.append({
            'label': f"Baseline 1 + {exp['label']}",
            'description': f"Combined: Baseline 1 + {exp['description']}",
            'results': combined_results,
            'individual_exp': exp
        })
    
    # Collect all tasks that appear in any experiment
    all_tasks = set()
    for exp in experiments:
        all_tasks.update(exp['results'].keys())
    for comb in combined_experiments:
        all_tasks.update(comb['results'].keys())
    all_tasks = sorted(all_tasks)
    
    print(f"\nTotal tasks with at least one success across all experiments: {len(all_tasks)}")
    
    # Create the combined table
    with open(Path(__file__).parent / "combined_experiment_table.md", 'w') as f:
        f.write("# Combined Analysis: Baseline 1 + Each Refinement Approach\n\n")
        f.write("This table shows the combined results when Baseline 1 is merged with each other approach.\n")
        f.write("For combined columns, we take the MAX success count from either Baseline 1 or the other approach.\n\n")
        
        # Experiment descriptions
        f.write("## Experiment Descriptions\n\n")
        f.write("**Individual Runs:**\n")
        for exp in experiments:
            f.write(f"- **{exp['label']}**: {exp['description']}\n")
        
        f.write("\n**Combined Runs (Baseline 1 + X):**\n")
        for comb in combined_experiments:
            f.write(f"- **{comb['label']}**: Best of either approach per task\n")
        
        # Results table
        f.write("\n## Results Table\n\n")
        f.write("Numbers show count of all-correct programs. Combined columns show MAX(Baseline 1, Other).\n\n")
        
        # Create header
        headers = ['Task ID']
        # Individual columns
        for exp in experiments:
            headers.append(exp['label'])
        # Combined columns
        for comb in combined_experiments:
            headers.append(comb['label'].replace('Baseline 1 + ', 'B1+'))
        
        f.write('| ' + ' | '.join(headers) + ' |\n')
        f.write('|' + '---|' * len(headers) + '\n')
        
        # Data rows
        for task in all_tasks:
            row_values = [task]
            # Individual results
            for exp in experiments:
                count = exp['results'].get(task, 0)
                row_values.append(str(count) if count > 0 else '')
            # Combined results
            for comb in combined_experiments:
                count = comb['results'].get(task, 0)
                row_values.append(str(count) if count > 0 else '')
            
            f.write('| ' + ' | '.join(row_values) + ' |\n')
        
        # Add totals row
        totals = ['**TOTAL TASKS**']
        # Individual totals
        for exp in experiments:
            totals.append(str(len(exp['results'])))
        # Combined totals
        for comb in combined_experiments:
            totals.append(f"**{len(comb['results'])}**")
        f.write('| ' + ' | '.join(totals) + ' |\n')
        
        # Summary statistics
        f.write("\n## Summary Statistics\n\n")
        
        f.write("### Individual Experiments\n\n")
        for exp in experiments:
            total_programs = sum(exp['results'].values())
            f.write(f"**{exp['label']}**:\n")
            f.write(f"- Tasks with ≥1 all-correct: {len(exp['results'])}\n")
            f.write(f"- Total all-correct programs: {total_programs}\n\n")
        
        f.write("### Combined Results (Baseline 1 + Each Approach)\n\n")
        for comb in combined_experiments:
            total_programs = sum(comb['results'].values())
            baseline_only = len([t for t in baseline1_results if t not in comb['individual_exp']['results']])
            other_only = len([t for t in comb['individual_exp']['results'] if t not in baseline1_results])
            both = len([t for t in comb['results'] if t in baseline1_results and t in comb['individual_exp']['results']])
            
            f.write(f"**{comb['label']}**:\n")
            f.write(f"- Tasks with ≥1 all-correct: **{len(comb['results'])}** ")
            f.write(f"(+{len(comb['results']) - len(baseline1_results)} vs Baseline 1 alone)\n")
            f.write(f"- Breakdown: {baseline_only} from B1 only, {other_only} from other only, {both} from both\n\n")
        
        # Improvement analysis
        f.write("## Improvement Analysis\n\n")
        f.write("How much does combining Baseline 1 with each approach improve over Baseline 1 alone?\n\n")
        
        baseline_task_count = len(baseline1_results)
        for comb in combined_experiments:
            combined_count = len(comb['results'])
            improvement = combined_count - baseline_task_count
            percent_improvement = (improvement / baseline_task_count * 100) if baseline_task_count > 0 else 0
            
            f.write(f"**{comb['label'].replace('Baseline 1 + ', '')}**: ")
            f.write(f"{baseline_task_count} → {combined_count} tasks ")
            f.write(f"(+{improvement} tasks, {percent_improvement:.1f}% improvement)\n")
    
    print("\n✅ Combined analysis saved to combined_experiment_table.md")

if __name__ == "__main__":
    main()