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

def main():
    # Define the 5 parquet files and their descriptions (from logs)
    experiments = [
        {
            'file': '20250902_142348_openai_gpt-oss-120b_arc-prize-2025_training-hard.parquet',
            'label': 'Baseline 1',
            'description': 'No refinement (Sept 2 - used as source for refinement)'
        },
        {
            'file': '20250903_093510_openai_gpt-oss-120b_arc-prize-2025_training-hard.parquet',
            'label': 'Baseline 2', 
            'description': 'No refinement (Sept 3 control run)'
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
            'description': 'Refinement with program only (no outputs)'
        }
    ]
    
    parquet_dir = Path(__file__).parent.parent.parent / "llm_python/datasets/inference"
    
    # Collect results for each experiment
    all_results = {}
    all_tasks = set()
    
    print("Analyzing parquet files...")
    for exp in experiments:
        parquet_path = parquet_dir / exp['file']
        if not parquet_path.exists():
            print(f"⚠️ Warning: {exp['file']} not found")
            exp['results'] = {}
        else:
            print(f"  - {exp['label']}: {exp['file']}")
            exp['results'] = analyze_parquet_for_success(parquet_path)
            all_tasks.update(exp['results'].keys())
    
    # Sort tasks
    all_tasks = sorted(all_tasks)
    
    print(f"\nFound {len(all_tasks)} tasks with at least one all-correct program across all experiments")
    
    # Create DataFrame for the table
    table_data = []
    for task in all_tasks:
        row = {'Task ID': task}
        for exp in experiments:
            count = exp['results'].get(task, 0)
            row[exp['label']] = count if count > 0 else ''
        table_data.append(row)
    
    df_table = pd.DataFrame(table_data)
    
    # Add totals row
    totals = {'Task ID': 'TOTAL TASKS'}
    for exp in experiments:
        count = len([t for t in all_tasks if exp['results'].get(t, 0) > 0])
        totals[exp['label']] = count
    df_table = pd.concat([df_table, pd.DataFrame([totals])], ignore_index=True)
    
    # Save as Markdown
    md_path = Path(__file__).parent / "experiment_comparison_table.md"
    with open(md_path, 'w') as f:
        f.write("# All-Correct Programs by Task Across Experiments\n\n")
        f.write("Comparison of 5 experimental runs showing count of all-correct (non-transductive) programs per task.\n\n")
        
        # Add experiment descriptions
        f.write("## Experiment Descriptions\n\n")
        for exp in experiments:
            f.write(f"- **{exp['label']}**: {exp['description']}\n")
        
        f.write("\n## Results Table\n\n")
        f.write("Numbers show count of all-correct programs for each task. Empty cells indicate zero all-correct programs.\n\n")
        
        # Create markdown table manually
        # Header
        headers = ['Task ID'] + [exp['label'] for exp in experiments]
        f.write('| ' + ' | '.join(headers) + ' |\n')
        f.write('|' + '---|' * len(headers) + '\n')
        
        # Data rows
        for _, row in df_table.iterrows():
            values = [str(row[col]) if row[col] != '' else '' for col in headers]
            f.write('| ' + ' | '.join(values) + ' |\n')
        
        f.write("\n\n## Summary Statistics\n\n")
        
        # Calculate summary stats
        for exp in experiments:
            if exp['results']:
                tasks_with_success = len(exp['results'])
                total_all_correct = sum(exp['results'].values())
                f.write(f"**{exp['label']}**:\n")
                f.write(f"- Tasks with ≥1 all-correct: {tasks_with_success}\n")
                f.write(f"- Total all-correct programs: {total_all_correct}\n\n")
    
    # Also save as CSV for easier viewing
    csv_path = Path(__file__).parent / "experiment_comparison_table.csv"
    df_table.to_csv(csv_path, index=False)
    
    # Create a more detailed comparison for the best tasks
    print("\n📊 Top Tasks Across Experiments:")
    print("="*80)
    
    # Find tasks that appear in multiple experiments
    task_appearances = {}
    for task in all_tasks:
        appearances = sum(1 for exp in experiments if exp['results'].get(task, 0) > 0)
        if appearances >= 2:  # Task successful in at least 2 experiments
            task_appearances[task] = appearances
    
    # Sort by number of appearances and print
    for task, count in sorted(task_appearances.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"\n{task} (appears in {count} experiments):")
        for exp in experiments:
            if exp['results'].get(task, 0) > 0:
                print(f"  - {exp['label']}: {exp['results'][task]} all-correct")
    
    print(f"\n✅ Tables saved to:")
    print(f"  - {md_path.name} (Markdown)")
    print(f"  - {csv_path.name} (CSV)")

if __name__ == "__main__":
    main()