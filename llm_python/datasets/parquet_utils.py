"""
Parquet utilities for fine-tuning dataset preparation.

Provides functions to load parquet data and convert it to datasets format
for fine-tuning, replacing the DuckDB-based approach.
"""

import os
from pathlib import Path
from typing import Optional, Union, Dict, Any
import pandas as pd
from datasets import Dataset

from llm_python.datasets.io import read_soar_parquet


def find_latest_parquet_file(parquet_path: Union[str, Path]) -> Path:
    """
    Find the latest parquet file based on timestamp in filename.
    
    Args:
        parquet_path: Either a specific parquet file path or directory containing parquet files
        
    Returns:
        Path to the latest parquet file
        
    Raises:
        FileNotFoundError: If no parquet files found
    """
    path = Path(parquet_path)
    
    if path.is_file() and path.suffix == '.parquet':
        return path
    
    if path.is_dir():
        # Find all parquet files and sort by timestamp in filename
        parquet_files = list(path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in directory: {path}")
        
        # Sort by filename (timestamp is at the beginning: YYYYMMDD_HHMMSS_...)
        latest_file = max(parquet_files, key=lambda p: p.name)
        return latest_file
    
    raise FileNotFoundError(f"Invalid path - not a file or directory: {path}")


def load_programs_for_finetuning(parquet_path: Union[str, Path], 
                                max_rows: Optional[int] = None,
                                filter_transductive: bool = True,
                                max_incorrect_per_task: int = 4) -> Dataset:
    """
    Load program data from parquet file(s) for fine-tuning.
    
    Follows filtering logic with fallback for incorrect programs:
    - ONLY non-transductive programs (if filter_transductive=True)  
    - Programs with ≥1 correct train example are always included
    - If max_incorrect_per_task > 0, includes up to N shortest incorrect programs per task as fallback
    
    Args:
        parquet_path: Path to parquet file or directory containing parquet files
        max_rows: Maximum number of rows to load (None for all)
        filter_transductive: Whether to filter out transductive programs (default True)
        max_incorrect_per_task: Max incorrect programs to include per task as fallback (default 4)
        
    Returns:
        Dataset containing program data compatible with fine-tuning format
    """
    # Find the specific parquet file to load
    file_path = find_latest_parquet_file(parquet_path)
    print(f"📊 Loading programs from parquet: {file_path}")
    
    # Use proper IO method for schema validation and type handling
    df = read_soar_parquet(file_path)
    print(f"✅ Loaded {len(df)} rows from parquet")
    
    # Apply row limit if specified
    if max_rows and len(df) > max_rows:
        df = df.head(max_rows)
        print(f"📏 Limited to {max_rows} rows")
    
    # Filter transductive programs if requested (default for fine-tuning)
    if filter_transductive:
        initial_count = len(df)
        df = df[~df['is_transductive']].copy()
        filtered_count = len(df)
        print(f"🔍 Filtered to {filtered_count} non-transductive programs (removed {initial_count - filtered_count} transductive)")
        
        if len(df) == 0:
            raise ValueError("No non-transductive programs found for fine-tuning")
    
    # Separate correct and incorrect programs
    def has_train_success(x):
        if hasattr(x, 'any'):  # numpy array
            return x.any()
        elif isinstance(x, list):
            return any(x)
        else:
            return bool(x)
    
    # Always include programs with ≥1 correct train example
    correct_programs = df[df['correct_train_input'].apply(has_train_success)].copy()
    
    # Add incorrect programs as fallback (up to max_incorrect_per_task per task)
    incorrect_programs_added = 0
    if max_incorrect_per_task > 0:
        incorrect_programs = df[~df['correct_train_input'].apply(has_train_success)].copy()
        
        if len(incorrect_programs) > 0:
            # Group by task and select shortest incorrect programs per task
            incorrect_fallback = []
            
            for task_id in incorrect_programs['task_id'].unique():
                task_incorrect = incorrect_programs[incorrect_programs['task_id'] == task_id].copy()
                
                # Sort by code length (shortest first) and take up to max_incorrect_per_task
                task_incorrect['code_length'] = task_incorrect['code'].fillna('').str.len()
                task_incorrect = task_incorrect.sort_values('code_length')
                selected = task_incorrect.head(max_incorrect_per_task)
                
                incorrect_fallback.append(selected.drop('code_length', axis=1))
            
            if incorrect_fallback:
                incorrect_fallback_df = pd.concat(incorrect_fallback, ignore_index=True)
                incorrect_programs_added = len(incorrect_fallback_df)
                df = pd.concat([correct_programs, incorrect_fallback_df], ignore_index=True)
            else:
                df = correct_programs
        else:
            df = correct_programs
    else:
        df = correct_programs
    
    correct_count = len(correct_programs)
    total_count = len(df)
    print(f"🔍 Selected {correct_count} programs with ≥1 train correct + {incorrect_programs_added} shortest incorrect as fallback = {total_count} total")
    
    if len(df) == 0:
        raise ValueError("No programs found for fine-tuning (neither correct nor incorrect)")
    
    # Prepare data for fine-tuning format
    programs_data = []
    
    for _, row in df.iterrows():
        # Determine success based on correct outputs
        train_correct = row.get('correct_train_input', [])
        test_correct = row.get('correct_test_input', [])
        
        # Convert numpy arrays to regular Python lists/bools if needed
        if hasattr(train_correct, 'tolist'):
            train_correct = train_correct.tolist()
        if hasattr(test_correct, 'tolist'):
            test_correct = test_correct.tolist()
        
        # Calculate success rates
        train_success = any(train_correct) if isinstance(train_correct, list) else bool(train_correct)
        test_success = any(test_correct) if isinstance(test_correct, list) else bool(test_correct)
        
        program_data = {
            'task_id': row['task_id'],
            'program': row['code'] or "",  # Handle None/empty code
            'model': row.get('model', 'unknown'),
            'is_transductive': row['is_transductive'],
            'train_success': train_success,
            'test_success': test_success,
            # Keep raw data for reference
            'correct_train_input': train_correct,
            'correct_test_input': test_correct,
            # Include reasoning if available
            'reasoning': row.get('reasoning', ''),
        }
        programs_data.append(program_data)
    
    print(f"📦 Prepared {len(programs_data)} program records for fine-tuning")
    
    # Convert to Dataset
    dataset = Dataset.from_list(programs_data)
    
    # Show some statistics
    if len(programs_data) > 0:
        train_success_rate = sum(p['train_success'] for p in programs_data) / len(programs_data)
        test_success_rate = sum(p['test_success'] for p in programs_data) / len(programs_data)
        print(f"📈 Success rates - Train: {train_success_rate:.2%}, Test: {test_success_rate:.2%}")
        
        # Show unique models
        models = set(p['model'] for p in programs_data)
        print(f"🤖 Models in dataset: {', '.join(models)}")
    
    return dataset


def get_parquet_path_from_config(data_config: Dict[str, Any], default_path: str = "../datasets/inference/") -> str:
    """
    Get parquet path from configuration with environment variable override.
    
    Args:
        data_config: Data configuration dictionary
        default_path: Default path if not specified in config or environment
        
    Returns:
        Resolved parquet path
    """
    return os.environ.get('ARC_PROGRAMS_PARQUET',
                         data_config.get('parquet', {}).get('path', default_path))


def validate_parquet_for_finetuning(parquet_path: Union[str, Path]) -> bool:
    """
    Validate that a parquet file/directory is suitable for fine-tuning.
    
    Args:
        parquet_path: Path to validate
        
    Returns:
        True if valid, raises exception otherwise
    """
    try:
        file_path = find_latest_parquet_file(parquet_path)
        df = read_soar_parquet(file_path)
        
        # Check required columns
        required_columns = {'task_id', 'code', 'is_transductive'}
        available_columns = set(df.columns)
        missing_columns = required_columns - available_columns
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for non-transductive programs
        non_transductive = df[~df['is_transductive']]
        if len(non_transductive) == 0:
            raise ValueError("No non-transductive programs found for fine-tuning")
        
        print(f"✅ Parquet validation passed: {len(non_transductive)} non-transductive programs available")
        return True
        
    except Exception as e:
        print(f"❌ Parquet validation failed: {e}")
        raise


# Simple interface for the fine-tuning notebook
def parquet_to_dataset(parquet_path: Union[str, Path], max_rows: Optional[int] = None, 
                      max_incorrect_per_task: int = 4) -> Dataset:
    """
    Simple interface to load parquet data as Dataset for fine-tuning.
    
    This is the main function that the fine-tuning notebook should use
    as a drop-in replacement for the DuckDB function.
    """
    return load_programs_for_finetuning(parquet_path, max_rows, filter_transductive=True, 
                                       max_incorrect_per_task=max_incorrect_per_task)


if __name__ == "__main__":
    # Test the utilities
    import sys
    
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    else:
        test_path = "../datasets/inference/"
    
    try:
        print(f"🧪 Testing parquet utilities with path: {test_path}")
        
        # Test validation
        validate_parquet_for_finetuning(test_path)
        
        # Test loading
        dataset = parquet_to_dataset(test_path, max_rows=5)
        print(f"✅ Successfully loaded {len(dataset)} examples")
        
        if len(dataset) > 0:
            print("\n📋 Sample data structure:")
            example = dataset[0]
            for key, value in example.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")
                    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)