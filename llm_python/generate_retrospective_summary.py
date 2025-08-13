#!/usr/bin/env python3

import os
import json
import argparse
import datetime
import re
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

try:
    # Try relative imports first (when run as module)
    from .utils.metrics_utils import calculate_task_metrics, format_metrics_display, metrics_to_percentages
except ImportError:
    # Fall back to absolute imports (when run directly)
    from utils.metrics_utils import calculate_task_metrics, format_metrics_display, metrics_to_percentages

def parse_task_filename(filename: str) -> Optional[Dict[str, str]]:
    """Parse a task result filename to extract metadata"""
    # Pattern: {timestamp}_{thread_id}_{task_id}_simple[_run{run_number}].json
    # Also handle partial attempts: {timestamp}_{thread_id}_{task_id}_simple_run{run_number}.json
    
    patterns = [
        r'^(\d{8}_\d{6}_\d+)_(\d+)_([^_]+)_simple(?:_run(\d+))?\.json$',  # Standard pattern
        r'^(\d{8}_\d{6}_\d+)_(\d+)_([^_]+)_simple\.json$',  # No run number
    ]
    
    for pattern in patterns:
        match = re.match(pattern, filename)
        if match:
            timestamp, thread_id, task_id = match.groups()[:3]
            run_number = match.group(4) if len(match.groups()) > 3 and match.group(4) else "0"
            
            return {
                'timestamp': timestamp,
                'thread_id': thread_id,
                'task_id': task_id,
                'run_number': run_number,
                'filename': filename
            }
    
    return None

def load_task_results_from_directory(results_dir: Path) -> List[Dict]:
    """Load all task results from a directory"""
    print(f"📁 Scanning directory: {results_dir}")
    
    task_files = []
    summary_files = []
    other_files = []
    
    # Categorize files
    for file_path in results_dir.glob("*.json"):
        filename = file_path.name
        
        if "summary" in filename:
            summary_files.append(filename)
        elif parse_task_filename(filename):
            task_files.append(file_path)
        else:
            other_files.append(filename)
    
    print(f"📊 Found {len(task_files)} task files, {len(summary_files)} summary files, {len(other_files)} other files")
    
    if summary_files:
        print(f"📋 Existing summary files: {summary_files}")
    
    # Load task results
    task_results = []
    failed_loads = 0
    
    for file_path in task_files:
        try:
            with open(file_path, 'r') as f:
                result = json.load(f)
            task_results.append(result)
        except Exception as e:
            print(f"❌ Failed to load {file_path.name}: {e}")
            failed_loads += 1
    
    if failed_loads > 0:
        print(f"⚠️ {failed_loads} files failed to load")
    
    print(f"✅ Successfully loaded {len(task_results)} task results")
    return task_results

def extract_metadata_from_results(task_results: List[Dict]) -> Dict:
    """Extract common metadata from task results"""
    if not task_results:
        return {}
    
    # Get metadata from first valid result
    sample_result = task_results[0]
    
    metadata = {
        'dataset': sample_result.get('dataset', 'unknown'),
        'subset': sample_result.get('subset', 'unknown'),
        'model': sample_result.get('model', 'unknown'),
        'api_type': sample_result.get('api_type', 'unknown'),
    }
    
    # Check if all results have consistent metadata
    inconsistent_fields = []
    for field in ['dataset', 'subset', 'model']:
        values = set(result.get(field) for result in task_results if result.get(field))
        if len(values) > 1:
            inconsistent_fields.append(f"{field}: {values}")
    
    if inconsistent_fields:
        print(f"⚠️ Inconsistent metadata found: {', '.join(inconsistent_fields)}")
    
    return metadata

def analyze_run_completeness(task_results: List[Dict]) -> Dict:
    """Analyze how complete the runs are"""
    task_attempts = defaultdict(list)
    
    # Group attempts by task
    for result in task_results:
        task_id = result.get('task_id')
        if task_id and 'attempt_details' in result:
            task_attempts[task_id].extend(result['attempt_details'])
    
    # Analyze completeness
    total_tasks = len(task_attempts)
    if total_tasks == 0:
        return {'total_tasks': 0, 'analysis': 'No valid tasks found'}
    
    attempt_counts = [len(attempts) for attempts in task_attempts.values()]
    max_attempts = max(attempt_counts) if attempt_counts else 0
    min_attempts = min(attempt_counts) if attempt_counts else 0
    avg_attempts = sum(attempt_counts) / len(attempt_counts) if attempt_counts else 0
    
    # Count tasks by attempt completeness
    attempt_distribution = defaultdict(int)
    for count in attempt_counts:
        attempt_distribution[count] += 1
    
    analysis = {
        'total_tasks': total_tasks,
        'max_attempts': max_attempts,
        'min_attempts': min_attempts,
        'avg_attempts': avg_attempts,
        'attempt_distribution': dict(attempt_distribution),
        'complete_tasks': sum(1 for count in attempt_counts if count >= max_attempts),
        'partial_tasks': sum(1 for count in attempt_counts if 0 < count < max_attempts),
        'empty_tasks': sum(1 for count in attempt_counts if count == 0)
    }
    
    return analysis

def convert_to_results_format(task_results: List[Dict]) -> List[Dict]:
    """Convert task results to the format expected by metrics calculation"""
    # The metrics functions expect a list of results where each result represents a task
    # with attempt_details containing all attempts for that task
    
    # Group by task_id
    task_groups = defaultdict(lambda: {'attempt_details': [], 'task_data': None})
    
    for result in task_results:
        task_id = result.get('task_id')
        if not task_id:
            continue
            
        # Add attempt details
        if 'attempt_details' in result:
            task_groups[task_id]['attempt_details'].extend(result['attempt_details'])
        
        # Store task data and other metadata
        if 'task_data' in result:
            task_groups[task_id]['task_data'] = result['task_data']
        
        # Copy over other fields from the first result for this task
        for field in ['model', 'api_type', 'dataset', 'subset']:
            if field in result and field not in task_groups[task_id]:
                task_groups[task_id][field] = result[field]
    
    # Convert to list format
    results = []
    for task_id, task_group in task_groups.items():
        if task_group['attempt_details']:  # Only include tasks with attempts
            result = {
                'task_id': task_id,
                'attempt_details': sorted(task_group['attempt_details'], key=lambda x: x.get('attempt_number', 0)),
                **{k: v for k, v in task_group.items() if k != 'attempt_details'}
            }
            results.append(result)
    
    return results

def generate_retrospective_summary(results_dir: Path, output_dir: Path = None, max_tokens: int = None) -> Dict:
    """Generate a retrospective summary for a results directory"""
    
    if output_dir is None:
        output_dir = results_dir
    
    print(f"🔄 Generating retrospective summary for: {results_dir}")
    print("-" * 60)
    
    # Load task results
    task_results = load_task_results_from_directory(results_dir)
    
    if not task_results:
        print("❌ No task results found")
        return {}
    
    # Extract metadata
    metadata = extract_metadata_from_results(task_results)
    print(f"📋 Dataset: {metadata.get('dataset')}")
    print(f"📋 Subset: {metadata.get('subset')}")
    print(f"📋 Model: {metadata.get('model')}")
    
    # Analyze completeness
    completeness = analyze_run_completeness(task_results)
    print(f"\n📊 COMPLETENESS ANALYSIS:")
    print(f"   Total tasks: {completeness['total_tasks']}")
    print(f"   Max attempts per task: {completeness['max_attempts']}")
    print(f"   Min attempts per task: {completeness['min_attempts']}")
    print(f"   Average attempts per task: {completeness['avg_attempts']:.1f}")
    
    if completeness['attempt_distribution']:
        print(f"   Attempt distribution:")
        for attempts, count in sorted(completeness['attempt_distribution'].items()):
            print(f"     {attempts} attempts: {count} tasks")
    
    # Convert to format expected by metrics calculation
    results_for_metrics = convert_to_results_format(task_results)
    
    if not results_for_metrics:
        print("❌ No valid results for metrics calculation")
        return {}
    
    print(f"\n📊 CALCULATING METRICS for {len(results_for_metrics)} tasks...")
    
    # Calculate metrics
    try:
        metrics = calculate_task_metrics(results_for_metrics, max_tokens=max_tokens)
        percentage_metrics = metrics_to_percentages(metrics)
    except Exception as e:
        print(f"❌ Error calculating metrics: {e}")
        return {}
    
    # Calculate costs
    total_cost = sum(
        sum(attempt.get('attempt_cost', 0.0) for attempt in result.get('attempt_details', []))
        for result in results_for_metrics
    )
    
    total_tokens = sum(
        sum(attempt.get('input_tokens', 0) + attempt.get('output_tokens', 0) 
            for attempt in result.get('attempt_details', []))
        for result in results_for_metrics
    )
    
    # Create summary
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary = {
        'timestamp': timestamp,
        'retrospective_analysis': True,
        'source_directory': str(results_dir),
        'dataset': metadata.get('dataset'),
        'subset': metadata.get('subset'),
        'model': metadata.get('model'),
        'api_type': metadata.get('api_type'),
        'total_tasks': len(results_for_metrics),
        'completeness_analysis': completeness,
        'total_tokens': total_tokens,
        'total_cost': total_cost,
        'metrics': percentage_metrics,
        'results': results_for_metrics,  # Include full results for further analysis
        'task_ids': [result['task_id'] for result in results_for_metrics]
    }
    
    # Save summary
    summary_filename = f"{timestamp}_retrospective_summary_{metadata.get('dataset', 'unknown')}_{metadata.get('subset', 'unknown')}.json"
    summary_path = output_dir / summary_filename
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print metrics
    print(f"\n📊 CORE METRICS:")
    print(f"  Pass@2 (Weighted Voting): {percentage_metrics['weighted_voting_pass2']:.1%}")
    print(f"  Pass@2 (Train Majority):  {percentage_metrics['train_majority_pass2']:.1%}")
    print(f"  Oracle (Best Attempt):    {percentage_metrics['all_test_correct']:.1%}")
    print(f"  All Train Correct:        {percentage_metrics['all_train_correct']:.1%}")
    print(f"  Min 1 Train Correct:      {percentage_metrics['min1_train_correct']:.1%}")
    print(f"  Min 1 Code Success:       {percentage_metrics['min1_code_success']:.1%}")
    print(f"  Max Length Responses:     {percentage_metrics['max_length_responses']:.1%}")
    print(f"  Timeout Responses:        {percentage_metrics['timeout_responses']:.1%}")
    print(f"  API Failure Responses:    {percentage_metrics['api_failure_responses']:.1%}")
    
    print(f"\n💰 COST ANALYSIS:")
    print(f"  Total tokens used: {total_tokens:,}")
    print(f"  Total cost: ${total_cost:.6f}")
    
    print(f"\n💾 Summary saved to: {summary_path}")
    
    return summary

def deduplicate_task_results_by_run(directories: List[Path]) -> Dict[str, List[Dict]]:
    """
    Load and deduplicate task results across multiple directories, grouped by run number.
    
    Returns a dict mapping run_number -> list of deduplicated task results
    """
    # Group all task results by run number
    runs_data = defaultdict(lambda: defaultdict(dict))  # run_number -> task_id -> task_data
    
    for directory in directories:
        if not directory.exists() or not directory.is_dir():
            continue
            
        for file_path in directory.glob("*.json"):
            if "summary" in file_path.name:
                continue
                
            parsed = parse_task_filename(file_path.name)
            if not parsed:
                continue
                
            try:
                with open(file_path, 'r') as f:
                    task_data = json.load(f)
                    
                run_number = parsed['run_number']
                task_id = task_data.get('task_id')
                
                if not task_id:
                    continue
                    
                # Keep the version with more attempts (or newer if same attempts)
                existing = runs_data[run_number].get(task_id)
                if existing:
                    existing_attempts = len(existing.get('attempt_details', []))
                    new_attempts = len(task_data.get('attempt_details', []))
                    
                    if new_attempts > existing_attempts:
                        runs_data[run_number][task_id] = task_data
                        print(f"  📝 Replaced task {task_id} in run {run_number} ({existing_attempts} → {new_attempts} attempts)")
                    elif new_attempts == existing_attempts:
                        # Use timestamp as tiebreaker
                        if parsed['timestamp'] > existing.get('_timestamp', ''):
                            runs_data[run_number][task_id] = task_data
                            task_data['_timestamp'] = parsed['timestamp']
                else:
                    task_data['_timestamp'] = parsed['timestamp']
                    runs_data[run_number][task_id] = task_data
                    
            except Exception as e:
                print(f"  ⚠️ Failed to load {file_path.name}: {e}")
    
    # Convert to list format and verify consistency across runs
    runs_list = {}
    for run_number, tasks_dict in runs_data.items():
        runs_list[run_number] = list(tasks_dict.values())
        print(f"📊 Run {run_number}: {len(tasks_dict)} unique tasks")
    
    return runs_list

def process_multiple_directories(directories: List[Path], output_dir: Path = None, max_tokens: int = None) -> List[Dict]:
    """Process multiple results directories with deduplication and run grouping"""
    
    print(f"🔄 Processing {len(directories)} directories with deduplication...")
    
    # Load and deduplicate all task results grouped by run
    runs_data = deduplicate_task_results_by_run(directories)
    
    if not runs_data:
        print("❌ No valid task results found")
        return []
    
    # Check if all runs have the same task IDs (for aggregation)
    run_numbers = sorted(runs_data.keys())
    if len(run_numbers) > 1:
        # Get task IDs for each run
        task_ids_per_run = {}
        for run_num in run_numbers:
            task_ids = set(task['task_id'] for task in runs_data[run_num] if 'task_id' in task)
            task_ids_per_run[run_num] = task_ids
            
        # Check if all runs have identical task sets
        first_run_ids = task_ids_per_run[run_numbers[0]]
        all_identical = all(task_ids_per_run[run] == first_run_ids for run in run_numbers[1:])
        
        if all_identical:
            print(f"✅ All {len(run_numbers)} runs have identical task sets ({len(first_run_ids)} tasks)")
            print("   → Will calculate mean ± std across runs")
        else:
            # Report differences
            print(f"⚠️ Runs have different task sets:")
            for run in run_numbers:
                print(f"   Run {run}: {len(task_ids_per_run[run])} tasks")
            
            # Find union and intersection
            all_tasks = set.union(*task_ids_per_run.values())
            common_tasks = set.intersection(*task_ids_per_run.values()) if len(task_ids_per_run) > 1 else all_tasks
            
            print(f"   Union: {len(all_tasks)} unique tasks total")
            print(f"   Intersection: {len(common_tasks)} common tasks")
            
            if len(run_numbers) > 1 and len(common_tasks) < len(all_tasks):
                print("❌ Cannot calculate meaningful statistics across runs with different task sets")
                print("   Treating as single combined dataset instead")
                
                # Combine all runs into a single dataset
                all_results = []
                seen_tasks = set()
                for run_num in run_numbers:
                    for task in runs_data[run_num]:
                        task_id = task.get('task_id')
                        if task_id and task_id not in seen_tasks:
                            all_results.append(task)
                            seen_tasks.add(task_id)
                
                runs_data = {'combined': all_results}
                print(f"📊 Combined into single dataset: {len(all_results)} unique tasks")
    
    # Process each run
    summaries = []
    for run_number in sorted(runs_data.keys()):
        task_results = runs_data[run_number]
        
        print(f"\n{'='*80}")
        print(f"Processing Run {run_number}: {len(task_results)} tasks")
        print(f"{'='*80}")
        
        if not task_results:
            continue
            
        # Generate summary for this run
        summary = generate_summary_from_results(task_results, run_number, output_dir, max_tokens)
        if summary:
            summaries.append(summary)
    
    return summaries

def generate_summary_from_results(task_results: List[Dict], run_identifier: str, output_dir: Path, max_tokens: int = None) -> Dict:
    """Generate a summary from a list of deduplicated task results"""
    
    # Extract metadata
    metadata = extract_metadata_from_results(task_results)
    print(f"📋 Dataset: {metadata.get('dataset')}")
    print(f"📋 Subset: {metadata.get('subset')}")
    print(f"📋 Model: {metadata.get('model')}")
    
    # Analyze completeness
    completeness = analyze_run_completeness(task_results)
    print(f"\n📊 COMPLETENESS ANALYSIS:")
    print(f"   Total tasks: {completeness['total_tasks']}")
    print(f"   Max attempts per task: {completeness['max_attempts']}")
    print(f"   Min attempts per task: {completeness['min_attempts']}")
    print(f"   Average attempts per task: {completeness['avg_attempts']:.1f}")
    
    # Convert to format expected by metrics calculation
    results_for_metrics = convert_to_results_format(task_results)
    
    if not results_for_metrics:
        print("❌ No valid results for metrics calculation")
        return {}
    
    print(f"\n📊 CALCULATING METRICS for {len(results_for_metrics)} tasks...")
    
    # Calculate metrics
    try:
        metrics = calculate_task_metrics(results_for_metrics, max_tokens=max_tokens)
        percentage_metrics = metrics_to_percentages(metrics)
    except Exception as e:
        print(f"❌ Error calculating metrics: {e}")
        return {}
    
    # Calculate costs
    total_cost = sum(
        sum(attempt.get('attempt_cost', 0.0) for attempt in result.get('attempt_details', []))
        for result in results_for_metrics
    )
    
    total_tokens = sum(
        sum(attempt.get('input_tokens', 0) + attempt.get('output_tokens', 0) 
            for attempt in result.get('attempt_details', []))
        for result in results_for_metrics
    )
    
    # Create summary
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary = {
        'timestamp': timestamp,
        'run_identifier': run_identifier,
        'retrospective_analysis': True,
        'dataset': metadata.get('dataset'),
        'subset': metadata.get('subset'),
        'model': metadata.get('model'),
        'api_type': metadata.get('api_type'),
        'total_tasks': len(results_for_metrics),
        'completeness_analysis': completeness,
        'total_tokens': total_tokens,
        'total_cost': total_cost,
        'metrics': percentage_metrics,
        'results': results_for_metrics,
        'task_ids': sorted([result['task_id'] for result in results_for_metrics])
    }
    
    # Save summary
    summary_filename = f"{timestamp}_retrospective_summary_run_{run_identifier}_{metadata.get('dataset', 'unknown')}_{metadata.get('subset', 'unknown')}.json"
    summary_path = output_dir / summary_filename
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print metrics
    print(f"\n📊 CORE METRICS:")
    print(f"  Pass@2 (Weighted Voting): {percentage_metrics['weighted_voting_pass2']:.1%}")
    print(f"  Pass@2 (Train Majority):  {percentage_metrics['train_majority_pass2']:.1%}")
    print(f"  Oracle (Best Attempt):    {percentage_metrics['all_test_correct']:.1%}")
    print(f"  All Train Correct:        {percentage_metrics['all_train_correct']:.1%}")
    print(f"  Min 1 Train Correct:      {percentage_metrics['min1_train_correct']:.1%}")
    print(f"  Min 1 Code Success:       {percentage_metrics['min1_code_success']:.1%}")
    
    print(f"\n💰 COST ANALYSIS:")
    print(f"  Total tokens used: {total_tokens:,}")
    print(f"  Total cost: ${total_cost:.6f}")
    
    print(f"\n💾 Summary saved to: {summary_path}")
    
    return summary

def aggregate_multiple_summaries(summaries: List[Dict], output_dir: Path) -> Optional[Dict]:
    """Aggregate multiple summaries into a combined analysis"""
    
    if len(summaries) < 2:
        print("ℹ️ Need at least 2 summaries for aggregation")
        return None
    
    print(f"\n📊 AGGREGATING {len(summaries)} SUMMARIES...")
    
    # Check if all summaries have the same task_ids (for valid aggregation)
    task_id_sets = []
    for summary in summaries:
        if 'task_ids' in summary:
            task_id_sets.append(set(summary['task_ids']))
    
    if len(task_id_sets) > 1:
        first_set = task_id_sets[0]
        all_identical = all(task_set == first_set for task_set in task_id_sets[1:])
        
        if not all_identical:
            print("❌ ERROR: Summaries have different task sets - cannot aggregate")
            for i, summary in enumerate(summaries):
                run_id = summary.get('run_identifier', f'summary_{i}')
                task_count = len(summary.get('task_ids', []))
                print(f"   {run_id}: {task_count} tasks")
            print("   These appear to be different experiments, not repeated runs")
            return None
    
    # Extract metrics from each summary
    all_metrics = []
    for summary in summaries:
        if 'metrics' in summary:
            metrics = summary['metrics'].copy()
            metrics['run_identifier'] = summary.get('run_identifier', 'unknown')
            all_metrics.append(metrics)
    
    if not all_metrics:
        print("❌ No metrics found in summaries")
        return None
    
    # Calculate aggregate statistics
    import numpy as np
    
    metric_names = ['weighted_voting_pass2', 'train_majority_pass2', 'all_test_correct',
                   'all_train_correct', 'min1_train_correct', 'min1_code_success']
    
    aggregate_stats = {}
    for metric_name in metric_names:
        values = [m.get(metric_name, 0.0) for m in all_metrics if metric_name in m]
        if values:
            aggregate_stats[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values, ddof=1) if len(values) > 1 else 0.0,
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
    
    # Create aggregate summary
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    aggregate_summary = {
        'timestamp': timestamp,
        'aggregated_analysis': True,
        'num_summaries': len(summaries),
        'individual_metrics': all_metrics,
        'aggregate_statistics': aggregate_stats,
        'run_identifiers': [s.get('run_identifier') for s in summaries]
    }
    
    # Save aggregate summary
    filename = f"{timestamp}_aggregate_retrospective_summary_{len(summaries)}_runs.json"
    filepath = output_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(aggregate_summary, f, indent=2)
    
    # Print aggregate results
    print(f"\n📊 AGGREGATE RESULTS ACROSS {len(summaries)} RUNS:")
    print("-" * 80)
    
    for metric_name, stats in aggregate_stats.items():
        metric_display = metric_name.replace('_', ' ').title()
        print(f"{metric_display}:")
        print(f"  Mean: {stats['mean']:.1%}")
        print(f"  Std Dev: {stats['std']:.1%}")
        print(f"  Range: {stats['min']:.1%} - {stats['max']:.1%}")
        if len(summaries) > 1:
            ci_lower = max(0, stats['mean'] - 1.96 * stats['std'])
            ci_upper = min(1, stats['mean'] + 1.96 * stats['std'])
            print(f"  95% CI: [{ci_lower:.1%}, {ci_upper:.1%}]")
        print()
    
    print(f"💾 Aggregate summary saved to: {filepath}")
    
    return aggregate_summary

def main():
    parser = argparse.ArgumentParser(description="Generate retrospective summaries from ARC task results")
    parser.add_argument("directories", nargs='+', help="Results directories to process")
    parser.add_argument("--output-dir", type=str, help="Output directory for summaries (defaults to first input directory)")
    parser.add_argument("--max-tokens", type=int, help="Max tokens used in original runs (for metrics calculation)")
    parser.add_argument("--aggregate", action="store_true", help="Generate aggregate statistics across multiple directories")
    
    args = parser.parse_args()
    
    # Convert to Path objects
    directories = [Path(d) for d in args.directories]
    output_dir = Path(args.output_dir) if args.output_dir else directories[0]
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process directories
    summaries = process_multiple_directories(directories, output_dir, args.max_tokens)
    
    print(f"\n✅ Successfully processed {len(summaries)}/{len(directories)} directories")
    
    # Generate aggregate if requested and we have multiple summaries
    if args.aggregate and len(summaries) > 1:
        aggregate_summary = aggregate_multiple_summaries(summaries, output_dir)

if __name__ == "__main__":
    main()