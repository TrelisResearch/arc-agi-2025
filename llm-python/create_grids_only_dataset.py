#!/usr/bin/env python3

import argparse
import datetime
import json
from pathlib import Path
from typing import Dict, List, Optional
from datasets import Dataset
from huggingface_hub import HfApi

def load_task_data(task_id: str, dataset: str = "arc-agi-1") -> Optional[Dict]:
    """Load task data from the data folder"""
    # Try both training and evaluation folders
    for folder in ["training", "evaluation"]:
        task_path = Path(f"../data/{dataset}/{folder}/{task_id}.json")
        if task_path.exists():
            with open(task_path, 'r') as f:
                return json.load(f)
    
    # Also try arc-agi-2
    if dataset == "arc-agi-1":
        return load_task_data(task_id, "arc-agi-2")
    return None

def load_task_ids_from_subset(dataset: str, subset: str) -> List[str]:
    """Load task IDs from a subset file"""
    subset_path = Path(f"../data/subsets/{dataset}/{subset}.txt")
    if not subset_path.exists():
        raise FileNotFoundError(f"Subset file not found: {subset_path}")
    
    with open(subset_path, 'r') as f:
        task_ids = [line.strip() for line in f if line.strip()]
    
    return task_ids

def create_simple_training_example(task_data: Dict, task_id: str) -> Dict:
    """Create a simplified training example without code or reasoning"""
    
    # Extract training inputs and outputs
    train_inputs = [example['input'] for example in task_data['train']]
    train_outputs = [example['output'] for example in task_data['train']]
    
    # Extract test inputs and outputs
    test_inputs = [example['input'] for example in task_data['test']]
    test_outputs = [example['output'] for example in task_data['test']]
    
    # Create empty lists for predicted outputs and correct flags
    predicted_train_outputs = [[] for _ in train_inputs]
    predicted_test_outputs = [[] for _ in test_inputs]
    correct_train_inputs = [False for _ in train_inputs]
    correct_test_inputs = [False for _ in test_inputs]
    
    # Create the training example with same structure but empty code/reasoning
    training_example = {
        "reasoning": "",  # Always empty
        "code": "",  # Always empty
        "correct_train_input": correct_train_inputs,  # Always False
        "train_input": train_inputs,
        "train_output": train_outputs,
        "predicted_train_output": predicted_train_outputs,  # Always empty lists
        "correct_test_input": correct_test_inputs,  # Always False
        "test_input": test_inputs,
        "test_output": test_outputs,
        "predicted_test_output": predicted_test_outputs,  # Always empty lists
        "task_id": task_id,
        "model": "",  # Always empty
        "generation": 0  # Always 0
    }
    
    return training_example

def create_and_push_hf_dataset(training_examples: List[Dict], validation_examples: List[Dict], args) -> None:
    """Create and push Hugging Face dataset to the hub"""
    
    # Generate dataset name if not provided
    if args.hf_dataset_name:
        dataset_name = args.hf_dataset_name
    else:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_part = args.dataset or "unknown"
        subset_part = args.subset or "unknown"
        dataset_name = f"grids_only_{dataset_part}_{subset_part}_{timestamp}"
    
    print(f"Creating Hugging Face dataset: {args.hf_org}/{dataset_name}")
    
    # Create datasets
    if validation_examples:
        # Create train and validation splits
        train_dataset = Dataset.from_list(training_examples)
        val_dataset = Dataset.from_list(validation_examples)
        
        print(f"Created training dataset with {len(training_examples)} examples")
        print(f"Created validation dataset with {len(validation_examples)} examples")
        
        # Push both splits
        train_dataset.push_to_hub(
            f"{args.hf_org}/{dataset_name}",
            split="train",
            private=args.hf_private
        )
        val_dataset.push_to_hub(
            f"{args.hf_org}/{dataset_name}",
            split="validation",
            private=args.hf_private
        )
        
        print(f"Successfully pushed training and validation splits to {args.hf_org}/{dataset_name}")
    else:
        # Create single training split
        train_dataset = Dataset.from_list(training_examples)
        
        print(f"Created training dataset with {len(training_examples)} examples")
        
        # Push training split
        train_dataset.push_to_hub(
            f"{args.hf_org}/{dataset_name}",
            split="train",
            private=args.hf_private
        )
        
        print(f"Successfully pushed training split to {args.hf_org}/{dataset_name}")
    
    # Print dataset URL
    visibility = "private" if args.hf_private else "public"
    print(f"Dataset URL: https://huggingface.co/datasets/{args.hf_org}/{dataset_name} ({visibility})")

def main():
    parser = argparse.ArgumentParser(description="Create grids-only training datasets (contains grid data without code or reasoning)")
    parser.add_argument("dataset", type=str, 
                       help="Dataset name (e.g., 'arc-agi-1', 'arc-agi-2')")
    parser.add_argument("subset", type=str,
                       help="Subset name (e.g., 'all_training', 'random_split_1_training')")
    parser.add_argument("--validation", action="store_true",
                       help="Create a validation split (10 percent or 32 examples, whichever is smaller)")
    parser.add_argument("--hf-dataset-name", type=str, default=None,
                       help="Name for the Hugging Face dataset. If not provided, will use simple_{dataset}_{subset}_DATETIME format")
    parser.add_argument("--hf-org", type=str, default="Trelis",
                       help="Hugging Face organization to push the dataset to (default: Trelis)")
    parser.add_argument("--hf-private", action="store_true",
                       help="Make the Hugging Face dataset private")
    parser.add_argument("--output", type=str, 
                       default=f"simple_dataset_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
                       help="Output JSONL file name (optional, for local saving)")
    parser.add_argument("--save-local", action="store_true",
                       help="Save dataset locally as JSONL file")
    
    args = parser.parse_args()
    
    print(f"Creating grids-only dataset from {args.dataset}/{args.subset}")
    
    # Load task IDs from subset
    try:
        task_ids = load_task_ids_from_subset(args.dataset, args.subset)
        print(f"Loaded {len(task_ids)} task IDs from subset")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Create training examples
    training_examples = []
    failed_tasks = []
    
    for task_id in task_ids:
        try:
            task_data = load_task_data(task_id, args.dataset)
            if task_data is None:
                failed_tasks.append(task_id)
                continue
            
            training_example = create_simple_training_example(task_data, task_id)
            training_examples.append(training_example)
            
        except Exception as e:
            print(f"Error processing task {task_id}: {e}")
            failed_tasks.append(task_id)
    
    if failed_tasks:
        print(f"Failed to process {len(failed_tasks)} tasks: {failed_tasks}")
    
    print(f"Successfully created {len(training_examples)} training examples")
    
    # Create validation split if requested
    validation_examples = []
    if args.validation and len(training_examples) > 1:
        val_size = min(32, max(1, len(training_examples) // 10))
        validation_examples = training_examples[-val_size:]
        training_examples = training_examples[:-val_size]
        print(f"Split into {len(training_examples)} training and {len(validation_examples)} validation examples")
    
    # Save locally if requested
    if args.save_local:
        # Create training_data directory if it doesn't exist
        output_dir = Path("training_data")
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / args.output
        with open(output_path, 'w') as f:
            for example in training_examples:
                f.write(json.dumps(example) + '\n')
        print(f"Saved {len(training_examples)} examples to {output_path}")
        
        if validation_examples:
            val_output_path = output_dir / f"val_{args.output}"
            with open(val_output_path, 'w') as f:
                for example in validation_examples:
                    f.write(json.dumps(example) + '\n')
            print(f"Saved {len(validation_examples)} validation examples to {val_output_path}")
    
    # Push to Hugging Face
    try:
        create_and_push_hf_dataset(training_examples, validation_examples, args)
    except Exception as e:
        print(f"Error pushing to Hugging Face: {e}")
        print("Dataset was still created locally if --save-local was specified")

if __name__ == "__main__":
    main() 