#!/usr/bin/env python3
"""
Standalone submission file generator for ARC tasks.

This script reads program execution results from parquet files and generates
submission.json files using weighted majority voting. It's designed to work
independently from the task runner.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from llm_python.datasets.io import read_soar_parquet
from llm_python.utils.task_loader import get_task_loader
from llm_python.utils.validator import ARCTaskValidator
from llm_python.utils.voting_utils import compute_weighted_majority_voting


class SubmissionGenerator:
    """Generates submission files from parquet data using weighted voting"""
    
    def __init__(
        self,
        no_transductive_penalty: bool = False,
        output_dir: Optional[str] = None,
        debug: bool = False,
    ):
        self.no_transductive_penalty = no_transductive_penalty
        self.output_dir = output_dir or "/kaggle/working"
        self.debug = debug
        self.task_loader = get_task_loader()
        
    def load_parquet_data(self, parquet_paths: List[Path]) -> pd.DataFrame:
        """Load and combine data from multiple parquet files"""
        if not parquet_paths:
            raise ValueError("No parquet files provided")
        
        all_data = []
        for path in parquet_paths:
            if not path.exists():
                print(f"⚠️ Parquet file not found: {path}")
                continue
            
            df = read_soar_parquet(path)
            all_data.append(df)
            if self.debug:
                print(f"✅ Loaded {len(df)} rows from {path}")
        
        if not all_data:
            raise ValueError("No valid parquet files could be loaded")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"📊 Combined data: {len(combined_df)} total rows from {len(all_data)} files")
        
        return combined_df
    
    def convert_parquet_for_voting(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Convert parquet DataFrame to minimal format needed by voting functions
        
        Creates dictionaries with only the fields actually used by voting:
        - test_predicted: The predictions to vote on (already proper Python lists from read_soar_parquet)
        - train_accuracy: For weighting (calculated from correct_train_input)
        - is_transductive: For penalty application
        - transduction_confidence: Fresh confidence scores computed from code
        
        Note: read_soar_parquet uses dtype_backend='pyarrow' so nested lists are already
        proper Python types, no numpy conversion needed.
        """
        from llm_python.transduction.code_classifier import CodeTransductionClassifier
        
        # Initialize transduction classifier for fresh scoring
        classifier = CodeTransductionClassifier()
        results_by_task = {}
        
        # Convert to dict for iteration (preserves PyArrow types from read_soar_parquet)
        records = df.to_dict('records')
        
        for row in records:
            task_id = str(row['task_id'])  # Ensure task_id is string
            
            # Calculate train accuracy from correct_train_input
            correct_train = row['correct_train_input']
            if correct_train and len(correct_train) > 0:
                # correct_train is already a Python list thanks to read_soar_parquet
                train_acc = float(sum(correct_train) / len(correct_train))
            else:
                train_acc = 0.0
            
            # Recompute transduction classification from code
            code = row.get('code', '')
            is_transductive = False
            transduction_confidence = 0.0
            
            if code and str(code).strip():
                try:
                    # Fresh transduction scoring from the actual code
                    is_transductive, transduction_confidence = classifier.is_transductive(str(code))
                    if self.debug:
                        old_trans = bool(row.get('is_transductive', False))
                        if old_trans != is_transductive:
                            print(f"📊 Task {task_id}: Transduction changed {old_trans} → {is_transductive} (conf: {transduction_confidence:.3f})")
                except Exception as e:
                    if self.debug:
                        print(f"⚠️ Failed to classify transduction for {task_id}: {e}")
                    # Fall back to stored value if classification fails
                    is_transductive = bool(row.get('is_transductive', False))
                    transduction_confidence = 1.0 if is_transductive else 0.0
            
            # Get test predictions - already proper Python lists from read_soar_parquet
            test_predicted = row['predicted_test_output']
            
            # Create minimal voting record with ONLY required fields
            voting_record = {
                'test_predicted': test_predicted,
                'train_accuracy': float(train_acc),
                'is_transductive': bool(is_transductive),
                'transduction_confidence': float(transduction_confidence),  # Fresh confidence score!
            }
            
            if task_id not in results_by_task:
                results_by_task[task_id] = []
            
            results_by_task[task_id].append(voting_record)
        
        if self.debug:
            # Summary of transduction recomputation
            total_records = len(records)
            total_transductive = sum(1 for r in results_by_task.values() for v in r if v['is_transductive'])
            print(f"📊 Transduction summary: {total_transductive}/{total_records} marked as transductive after recomputation")
        
        return results_by_task
    
    def find_recent_parquets(self, path: Union[str, Path], n_files: int = 1) -> List[Path]:
        """Find the most recent n parquet files at a given path"""
        path = Path(path)
        
        if path.is_file() and path.suffix == '.parquet':
            # Single file provided
            return [path]
        
        if path.is_dir():
            # Directory provided - find recent parquet files
            parquet_files = list(path.glob("*.parquet"))
            if not parquet_files:
                raise ValueError(f"No parquet files found in directory: {path}")
            
            # Sort by modification time (most recent first)
            parquet_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            
            selected_files = parquet_files[:n_files]
            print(f"🔍 Selected {len(selected_files)} most recent parquet files from {path}:")
            for f in selected_files:
                mtime = datetime.datetime.fromtimestamp(f.stat().st_mtime)
                print(f"  • {f.name} (modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
            
            return selected_files
        
        raise ValueError(f"Invalid path: {path} (must be a parquet file or directory)")
    
    def generate_submission(
        self,
        parquet_paths: List[Path], 
        dataset: str,
        subset: str,
        model_name: Optional[str] = None,
    ) -> str:
        """Generate submission file from parquet data"""
        
        # Load parquet data
        df = self.load_parquet_data(parquet_paths)
        
        # Convert to voting format with fresh transduction scores
        results_by_task = self.convert_parquet_for_voting(df)
        
        # Load ALL tasks from the dataset to ensure complete submission
        try:
            all_tasks = self.task_loader.get_subset_tasks(f"{dataset}/{subset}")
            all_task_ids = [task_id for task_id, _ in all_tasks]
            task_data_by_id = {task_id: task_data for task_id, task_data in all_tasks}
        except Exception as e:
            print(f"⚠️ Could not load all tasks from {dataset}/{subset}: {e}")
            # Fallback to tasks from parquet data
            all_task_ids = list(results_by_task.keys())
            task_data_by_id = {}
        
        print(f"🎯 Generating submission for {len(all_task_ids)} tasks from {dataset}/{subset}")
        
        # Create submission
        submission = {}
        tasks_with_predictions = 0
        tasks_with_duplicated_attempts = 0
        tasks_with_empty_fallback = 0
        
        for task_id in all_task_ids:
            if task_id in results_by_task:
                # We have attempts for this task
                attempts = results_by_task[task_id]
                task_data = task_data_by_id.get(task_id)
                
                # Use weighted voting to get top 2 predictions
                try:
                    top_predictions = compute_weighted_majority_voting(
                        attempts, 
                        top_k=2, 
                        no_transductive_penalty=self.no_transductive_penalty
                    )
                except Exception as e:
                    print(f"⚠️ Weighted voting failed for task {task_id}: {e}")
                    # Fallback to first available predictions
                    top_predictions = []
                    for attempt in attempts[:2]:
                        pred = attempt.get("test_predicted")
                        if pred is not None:
                            top_predictions.append(pred)
                
                # Determine number of test outputs
                if task_data:
                    num_test_outputs = len(task_data.get("test", []))
                else:
                    # Infer from prediction structure
                    if top_predictions and isinstance(top_predictions[0], list):
                        num_test_outputs = len(top_predictions[0])
                    else:
                        num_test_outputs = 1
                
                if num_test_outputs == 0:
                    num_test_outputs = 1  # Default fallback
                
                submission[task_id] = []
                
                for test_idx in range(num_test_outputs):
                    attempt_1_grid = [[0, 0], [0, 0]]  # Default fallback
                    attempt_2_grid = [[0, 0], [0, 0]]  # Default fallback
                    
                    # Extract attempt 1
                    if len(top_predictions) > 0 and top_predictions[0] is not None:
                        pred_1 = top_predictions[0]
                        
                        if isinstance(pred_1, list) and len(pred_1) > test_idx:
                            attempt_1_grid = pred_1[test_idx]
                            # Validate grid
                            if not ARCTaskValidator.validate_prediction(attempt_1_grid):
                                print(f"⚠️ Invalid grid from voting for {task_id} attempt 1")
                                attempt_1_grid = [[0, 0], [0, 0]]  # Fallback
                    
                    # Extract attempt 2
                    if len(top_predictions) > 1 and top_predictions[1] is not None:
                        pred_2 = top_predictions[1]
                        
                        if isinstance(pred_2, list) and len(pred_2) > test_idx:
                            attempt_2_grid = pred_2[test_idx]
                            # Validate grid
                            if not ARCTaskValidator.validate_prediction(attempt_2_grid):
                                print(f"⚠️ Invalid grid from voting for {task_id} attempt 2")
                                attempt_2_grid = [[0, 0], [0, 0]]  # Fallback
                    else:
                        # Only one prediction available, duplicate it
                        attempt_2_grid = attempt_1_grid
                        if test_idx == 0:  # Only count once per task
                            tasks_with_duplicated_attempts += 1
                    
                    submission[task_id].append({
                        "attempt_1": attempt_1_grid,
                        "attempt_2": attempt_2_grid
                    })
                
                tasks_with_predictions += 1
            
            else:
                # No attempts for this task - use empty fallback
                print(f"⚠️ No attempts for task {task_id}, using empty fallback")
                
                # Try to determine number of test outputs from task data
                task_data = task_data_by_id.get(task_id)
                if task_data:
                    num_test_outputs = len(task_data.get("test", []))
                else:
                    num_test_outputs = 1  # Default fallback
                
                if num_test_outputs == 0:
                    num_test_outputs = 1
                
                submission[task_id] = []
                for _ in range(num_test_outputs):
                    submission[task_id].append({
                        "attempt_1": [[0, 0], [0, 0]],
                        "attempt_2": [[0, 0], [0, 0]]
                    })
                
                tasks_with_empty_fallback += 1
        
        # Generate submission filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create model name from parquet paths if not provided
        if not model_name and parquet_paths:
            first_path = parquet_paths[0]
            # Extract model info from filename if possible
            filename_parts = first_path.stem.split('_')
            if len(filename_parts) >= 4:
                model_name = f"{filename_parts[1]}_{filename_parts[2]}"
            else:
                model_name = "unknown_model"
        
        model_name_safe = model_name.replace("/", "_").replace(":", "_") if model_name else "model"
        
        # Official submission file (required name)
        submission_filename = "submission.json"
        submission_path = Path(self.output_dir) / submission_filename
        
        # Backup with timestamp
        backup_filename = f"submission_{dataset}_{subset}_{model_name_safe}_{timestamp}.json"
        backup_path = Path(self.output_dir) / backup_filename
        
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save files
        with open(submission_path, "w") as f:
            json.dump(submission, f, indent=2)
        
        with open(backup_path, "w") as f:
            json.dump(submission, f, indent=2)
        
        # Print summary
        total_tasks = len(all_task_ids)
        print(f"\n✅ Submission files created:")
        print(f"📊 Summary:")
        print(f"  Total tasks in dataset: {total_tasks}")
        print(f"  Tasks with predictions: {tasks_with_predictions}")
        print(f"  Tasks with duplicated attempts: {tasks_with_duplicated_attempts}")
        print(f"  Tasks with empty fallback: {tasks_with_empty_fallback}")
        print(f"  Official file: {submission_path}")
        print(f"  Backup file: {backup_path}")
        
        # Note: Submission validation requires a challenges file path
        # For now, we skip validation to keep the script simple
        print("ℹ️ Note: Run separate validation if needed")
        
        return str(submission_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate ARC submission files from parquet data"
    )
    parser.add_argument(
        "--parquet-path",
        type=str,
        default=None,
        help="Path to parquet file or directory with parquet files. Defaults to most recent in llm_python/datasets/inference/",
    )
    parser.add_argument(
        "--n-files",
        type=int,
        default=1,
        help="Number of most recent parquet files to use (when providing directory)",
    )
    parser.add_argument(
        "--dataset",
        default="arc-prize-2024",
        help="Dataset name (e.g., arc-prize-2024, arc-agi-1)",
    )
    parser.add_argument(
        "--subset",
        default="evaluation",
        help="Dataset subset (e.g., evaluation, training)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for submission files (defaults to /kaggle/working or current dir)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        help="Model name for backup filename (inferred from parquet if not provided)",
    )
    parser.add_argument(
        "--no-transductive-penalty",
        action="store_true",
        help="Disable transductive penalty in weighted voting",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )
    
    args = parser.parse_args()
    
    # Determine default parquet path if not provided
    if args.parquet_path is None:
        default_inference_dir = Path(__file__).parent / "datasets" / "inference"
        if default_inference_dir.exists():
            args.parquet_path = str(default_inference_dir)
            print(f"🔍 Using default inference directory: {default_inference_dir}")
        else:
            parser.error("No --parquet-path provided and default inference directory not found")
    
    # Determine default output directory
    if args.output_dir is None:
        if os.path.exists("/kaggle/working"):
            args.output_dir = "/kaggle/working"
        else:
            args.output_dir = "."
    
    # Create generator
    generator = SubmissionGenerator(
        no_transductive_penalty=args.no_transductive_penalty,
        output_dir=args.output_dir,
        debug=args.debug,
    )
    
    try:
        # Find parquet files
        parquet_paths = generator.find_recent_parquets(args.parquet_path, args.n_files)
        
        # Generate submission
        submission_path = generator.generate_submission(
            parquet_paths=parquet_paths,
            dataset=args.dataset,
            subset=args.subset,
            model_name=args.model_name,
        )
        
        print(f"🎯 Submission generation complete: {submission_path}")
        
    except Exception as e:
        print(f"❌ Error generating submission: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())