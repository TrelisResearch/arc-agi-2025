from typing import TypedDict, List
import pandas as pd
import pyarrow as pa

class SoarProgramExample(TypedDict):
    """Schema for training examples enriched for actual training with separate train/test data"""
    task_id: str # Task ID from ARC
    reasoning: str # Reasoning trace if provided
    code: str # Program code that should define a `generate` function
    correct_train_input: List[bool] # Training inputs where program produced correct output
    correct_test_input: List[bool] # Test inputs where program produced correct output
    predicted_train_output: List[List[List[int]]] # Program's predicted outputs for training inputs
    predicted_test_output: List[List[List[int]]] # Program's predicted outputs for test inputs
    train_input: List[List[List[int]]] # Training input grids (optional)
    test_input: List[List[List[int]]] # Test input grids (optional)
    model: str # What model generated this example
    generation: int # Generation number (default 0)


# Define explicit PyArrow schema for our parquet file
PARQUET_SCHEMA = pa.schema(
    [
        ("task_id", pa.string()),
        ("reasoning", pa.string()),
        ("code", pa.string()),
        ("correct_train_input", pa.list_(pa.bool_())),
        ("correct_test_input", pa.list_(pa.bool_())),
        ("predicted_train_output", pa.list_(pa.list_(pa.list_(pa.int64())))),
        ("predicted_test_output", pa.list_(pa.list_(pa.list_(pa.int64())))),
        ("model", pa.string()),
    ]
)


class ValidationError(Exception):
    """Raised when dataset validation fails"""
    pass


def validate_soar_dataset(df: pd.DataFrame, max_grid_size: int = 40, silent: bool = False) -> None:
    """
    Validate a SOAR dataset DataFrame for correctness and compatibility.
    
    Args:
        df: The pandas DataFrame to validate
        max_grid_size: Maximum allowed grid dimension
        silent: If True, only print on errors
        
    Raises:
        ValidationError: If critical validation fails
    """
    try:
        if not silent:
            print("✓ Starting dataset validation...")
        
        # Basic structure validation
        required_columns = ['task_id', 'code', 'model', 'predicted_train_output', 
                           'predicted_test_output', 'correct_train_input', 'correct_test_input']
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValidationError(f"Missing required columns: {missing_columns}")
            
        if df.empty:
            raise ValidationError("Dataset is empty")
            
        # Calculate and print stats
        stats = {
            'total_programs': len(df),
            'unique_tasks': df['task_id'].nunique(),
            'programs_per_task': {
                'mean': df.groupby('task_id').size().mean(),
                'min': df.groupby('task_id').size().min(),
                'max': df.groupby('task_id').size().max()
            },
            'models': df['model'].value_counts().to_dict()
        }
        
        if not silent:
            print(f"✓ Dataset shape: {df.shape}")
            print(f"✓ Unique tasks: {stats['unique_tasks']}")
        
        # Data type validation on sample
        sample = df.iloc[0]
        
        # Check grid outputs (3D arrays)
        for output_col in ['predicted_train_output', 'predicted_test_output']:
            output_data = sample[output_col]
            # Convert numpy array back to list if needed
            if hasattr(output_data, 'tolist'):
                output_data = output_data.tolist()
            
            if not isinstance(output_data, list):
                raise ValidationError(f"{output_col} is not a list (got {type(output_data)})")
                
            # Check grid dimensions
            for i, grid in enumerate(output_data):
                if hasattr(grid, 'tolist'):
                    grid = grid.tolist()
                if not isinstance(grid, list):
                    raise ValidationError(f"{output_col}[{i}] is not a list (got {type(grid)})")
                    
                if len(grid) > max_grid_size:
                    raise ValidationError(f"{output_col}[{i}] height {len(grid)} exceeds max {max_grid_size}")
                    
                for j, row in enumerate(grid):
                    if hasattr(row, 'tolist'):
                        row = row.tolist()
                    if not isinstance(row, list):
                        raise ValidationError(f"{output_col}[{i}][{j}] is not a list (got {type(row)})")
                        
                    if len(row) > max_grid_size:
                        raise ValidationError(f"{output_col}[{i}][{j}] width {len(row)} exceeds max {max_grid_size}")
                        
                    for k, cell in enumerate(row):
                        if not isinstance(cell, (int, float)):
                            raise ValidationError(f"{output_col}[{i}][{j}][{k}] is not numeric (got {type(cell)})")
        
        # Check correctness arrays (1D boolean arrays)
        for correct_col in ['correct_train_input', 'correct_test_input']:
            correct_data = sample[correct_col]
            # Convert numpy array back to list if needed
            if hasattr(correct_data, 'tolist'):
                correct_data = correct_data.tolist()
                
            if not isinstance(correct_data, list):
                raise ValidationError(f"{correct_col} is not a list (got {type(correct_data)})")
                
            for i, val in enumerate(correct_data):
                # Handle numpy boolean types
                if hasattr(val, 'item'):
                    val = val.item()
                if not isinstance(val, bool):
                    raise ValidationError(f"{correct_col}[{i}] is not boolean (got {type(val)})")
        
        # DuckDB compatibility check is skipped since we're working with DataFrames
        # (DuckDB can read parquet files directly, but not DataFrames in memory)
        
        if not silent:
            print("✅ Dataset validation passed!")
            print(f"📊 {stats['total_programs']:,} programs across {stats['unique_tasks']} tasks")
            print(f"🤖 Models: {len(stats['models'])}")
            print(f"📏 Programs per task: {stats['programs_per_task']['min']}-{stats['programs_per_task']['max']} (avg: {stats['programs_per_task']['mean']:.1f})")
        
    except ValidationError:
        raise  # Re-raise ValidationError as-is
    except Exception as e:
        raise ValidationError(f"Validation failed with exception: {str(e)}")


def validate_soar_parquet_file(file_path: str, max_grid_size: int = 40, silent: bool = False) -> None:
    """
    Validate a SOAR dataset parquet file for correctness and compatibility.
    
    Args:
        file_path: Path to the parquet file
        max_grid_size: Maximum allowed grid dimension
        silent: If True, only print on errors
        
    Raises:
        ValidationError: If critical validation fails
    """
    try:
        df = pd.read_parquet(file_path)
        if not silent:
            print(f"✓ Successfully loaded parquet file: {file_path}")
        
        validate_soar_dataset(df, max_grid_size, silent)
        
    except ValidationError:
        raise  # Re-raise ValidationError as-is
    except Exception as e:
        raise ValidationError(f"Failed to load parquet file {file_path}: {str(e)}")

