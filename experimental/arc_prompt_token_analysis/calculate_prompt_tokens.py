#!/usr/bin/env python3
"""
Script to calculate token lengths for ARC task prompts in all datasets and splits.
Creates templated prompts and saves example to txt file for approval.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import pandas as pd

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import ARC utilities
try:
    from llm_python.utils.task_loader import TaskLoader
    from llm_python.utils.prompt_utils import create_arc_prompt
    from llm_python.utils.prompt_loader import PromptLoader
except ImportError as e:
    print(f"Error importing utilities: {e}")
    sys.exit(1)


def setup_tokenizer():
    """Setup tokenizer for token counting."""
    try:
        from transformers import AutoTokenizer
        # Use Qwen3-4B tokenizer as requested
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        return tokenizer
    except ImportError:
        print("Error: transformers library not installed. Install with: pip install transformers")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)


def create_chat_messages(system_content: str, user_content: str) -> List[Dict[str, str]]:
    """Create chat messages in the format expected by apply_chat_template."""
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]


def calculate_prompt_tokens(task_id: str, task_data: Dict, prompt_loader: PromptLoader, tokenizer) -> Dict[str, Any]:
    """Calculate token lengths for a single task's prompt."""
    try:
        # Create the prompt using the same method as the runner
        system_content, user_content = create_arc_prompt(task_data, prompt_loader, "soar")
        
        # Create chat messages
        messages = create_chat_messages(system_content, user_content)
        
        # Apply chat template with tokenize=False to get the templated text
        templated_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True,
            enable_thinking=True
        )
        
        # Count tokens in the templated text
        tokens = tokenizer.encode(templated_text)
        token_count = len(tokens)
        
        # Calculate individual component lengths
        system_tokens = len(tokenizer.encode(system_content))
        user_tokens = len(tokenizer.encode(user_content))
        
        # Calculate task data stats
        num_train_examples = len(task_data.get('train', []))
        num_test_examples = len(task_data.get('test', []))
        
        # Calculate total grid cells in task
        total_cells = 0
        for example in task_data.get('train', []):
            if 'input' in example:
                input_grid = example['input']
                total_cells += len(input_grid) * len(input_grid[0]) if input_grid else 0
            if 'output' in example:
                output_grid = example['output']
                total_cells += len(output_grid) * len(output_grid[0]) if output_grid else 0
                
        for example in task_data.get('test', []):
            if 'input' in example:
                input_grid = example['input']
                total_cells += len(input_grid) * len(input_grid[0]) if input_grid else 0
                
        return {
            'task_id': task_id,
            'total_tokens': token_count,
            'system_tokens': system_tokens,
            'user_tokens': user_tokens,
            'templated_text_length': len(templated_text),
            'system_content_length': len(system_content),
            'user_content_length': len(user_content),
            'num_train_examples': num_train_examples,
            'num_test_examples': num_test_examples,
            'total_grid_cells': total_cells,
            'templated_text': templated_text,  # Store for example output
            'system_content': system_content,
            'user_content': user_content
        }
        
    except Exception as e:
        print(f"Error processing task {task_id}: {e}")
        return {
            'task_id': task_id,
            'total_tokens': 0,
            'system_tokens': 0,
            'user_tokens': 0,
            'templated_text_length': 0,
            'system_content_length': 0,
            'user_content_length': 0,
            'num_train_examples': 0,
            'num_test_examples': 0,
            'total_grid_cells': 0,
            'error': str(e)
        }


def process_dataset_split(dataset: str, split: str, task_loader: TaskLoader, prompt_loader: PromptLoader, tokenizer) -> List[Dict[str, Any]]:
    """Process all tasks in a dataset split."""
    print(f"Processing {dataset}/{split}...")
    
    try:
        if split == "all_training":
            tasks = task_loader.load_tasks_from_subset("all_training", dataset)
        elif split == "all_evaluation":
            tasks = task_loader.load_tasks_from_subset("all_evaluation", dataset)
        else:
            print(f"Unknown split: {split}")
            return []
            
        results = []
        for i, (task_id, task_data) in enumerate(tasks):
            if i % 50 == 0:
                print(f"  Processed {i}/{len(tasks)} tasks...")
            
            result = calculate_prompt_tokens(task_id, task_data, prompt_loader, tokenizer)
            result['dataset'] = dataset
            result['split'] = split
            results.append(result)
            
        print(f"  Completed {dataset}/{split}: {len(results)} tasks")
        return results
        
    except Exception as e:
        print(f"Error processing {dataset}/{split}: {e}")
        return []


def save_example_templated_prompt(results: List[Dict[str, Any]], output_dir: Path) -> str:
    """Save an example templated prompt to a text file for user approval."""
    # Find a task with reasonable length for example
    example_task = None
    for result in results:
        if ('error' not in result and 
            1000 < result.get('total_tokens', 0) < 3000 and 
            result.get('num_train_examples', 0) >= 2):
            example_task = result
            break
    
    if not example_task:
        # Fallback to first valid task
        for result in results:
            if 'error' not in result and 'templated_text' in result:
                example_task = result
                break
    
    if not example_task:
        print("No valid tasks found for example generation")
        return ""
    
    # Create example file
    example_file = output_dir / "example_templated_prompt.txt"
    with open(example_file, 'w', encoding='utf-8') as f:
        f.write("# Example Templated ARC Task Prompt\n")
        f.write("# This shows the full prompt text that would be sent to the LLM\n")
        f.write(f"# Task ID: {example_task['task_id']}\n")
        f.write(f"# Dataset: {example_task['dataset']}\n")
        f.write(f"# Split: {example_task['split']}\n")
        f.write(f"# Total tokens: {example_task['total_tokens']}\n")
        f.write(f"# System tokens: {example_task['system_tokens']}\n")
        f.write(f"# User tokens: {example_task['user_tokens']}\n")
        f.write(f"# Train examples: {example_task['num_train_examples']}\n")
        f.write(f"# Test examples: {example_task['num_test_examples']}\n")
        f.write("=" * 80 + "\n\n")
        f.write(example_task['templated_text'])
    
    print(f"Example templated prompt saved to: {example_file}")
    return str(example_file)


def save_results(results: List[Dict[str, Any]], output_dir: Path) -> str:
    """Save token calculation results to CSV."""
    # Remove templated_text, system_content, and user_content from results for CSV
    csv_results = []
    for result in results:
        csv_result = {k: v for k, v in result.items() 
                     if k not in ['templated_text', 'system_content', 'user_content']}
        csv_results.append(csv_result)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(csv_results)
    csv_file = output_dir / "arc_prompt_token_analysis.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"Token analysis results saved to: {csv_file}")
    return str(csv_file)


def print_summary_stats(results: List[Dict[str, Any]]):
    """Print summary statistics of token analysis."""
    if not results:
        print("No results to summarize")
        return
    
    # Filter out error results
    valid_results = [r for r in results if 'error' not in r]
    
    if not valid_results:
        print("No valid results to summarize")
        return
    
    df = pd.DataFrame(valid_results)
    
    print("\n" + "="*80)
    print("TOKEN ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nTotal tasks analyzed: {len(valid_results)}")
    print(f"Tasks with errors: {len(results) - len(valid_results)}")
    
    # Overall statistics
    print(f"\nTOKEN STATISTICS:")
    print(f"  Mean tokens per prompt: {df['total_tokens'].mean():.1f}")
    print(f"  Median tokens per prompt: {df['total_tokens'].median():.1f}")
    print(f"  Min tokens per prompt: {df['total_tokens'].min()}")
    print(f"  Max tokens per prompt: {df['total_tokens'].max()}")
    print(f"  Std dev: {df['total_tokens'].std():.1f}")
    
    # Breakdown by dataset and split
    print(f"\nBREAKDOWN BY DATASET/SPLIT:")
    for dataset in df['dataset'].unique():
        for split in df['split'].unique():
            subset = df[(df['dataset'] == dataset) & (df['split'] == split)]
            if len(subset) > 0:
                print(f"  {dataset}/{split}: {len(subset)} tasks, "
                      f"mean {subset['total_tokens'].mean():.1f} tokens, "
                      f"range {subset['total_tokens'].min()}-{subset['total_tokens'].max()}")
    
    # Token distribution percentiles
    print(f"\nTOKEN DISTRIBUTION PERCENTILES:")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = df['total_tokens'].quantile(p/100)
        print(f"  {p}th percentile: {val:.0f} tokens")
    
    # Task complexity statistics
    print(f"\nTASK COMPLEXITY:")
    print(f"  Mean train examples per task: {df['num_train_examples'].mean():.1f}")
    print(f"  Mean test examples per task: {df['num_test_examples'].mean():.1f}")
    print(f"  Mean grid cells per task: {df['total_grid_cells'].mean():.1f}")


def main():
    """Main function to calculate token lengths for all ARC task prompts."""
    print("ARC Prompt Token Length Calculator")
    print("=" * 50)
    
    # Setup
    output_dir = Path(__file__).parent / "token_analysis_results_qwen"
    output_dir.mkdir(exist_ok=True)
    
    # Initialize components
    print("Initializing task loader, prompt loader, and tokenizer...")
    task_loader = TaskLoader()
    prompt_loader = PromptLoader()
    tokenizer = setup_tokenizer()
    
    # Define datasets and splits to process
    datasets_splits = [
        ("arc-agi-1", "all_training"),
        ("arc-agi-1", "all_evaluation"),
        ("arc-agi-2", "all_training"),
        ("arc-agi-2", "all_evaluation")
    ]
    
    # Process all dataset/split combinations
    all_results = []
    for dataset, split in datasets_splits:
        try:
            results = process_dataset_split(dataset, split, task_loader, prompt_loader, tokenizer)
            all_results.extend(results)
        except Exception as e:
            print(f"Failed to process {dataset}/{split}: {e}")
    
    if not all_results:
        print("No results generated. Exiting.")
        return
    
    # Save results and generate example
    print(f"\nSaving results...")
    csv_file = save_results(all_results, output_dir)
    example_file = save_example_templated_prompt(all_results, output_dir)
    
    # Print summary statistics
    print_summary_stats(all_results)
    
    print(f"\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {csv_file}")
    print(f"Example prompt saved to: {example_file}")
    print(f"Total tasks processed: {len(all_results)}")
    print("\nPlease review the example templated prompt to verify it looks correct!")


if __name__ == "__main__":
    main()