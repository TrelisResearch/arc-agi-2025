#!/usr/bin/env python3

import gzip
import json
from typing import List, Dict, Tuple, Optional
import subprocess
import tempfile
import os

class GridScorer:
    """Scores predicted grids against ground truth"""
    
    def score_grid(self, predicted: List[List[int]], actual: List[List[int]]) -> Dict:
        """Score a predicted grid against the actual grid"""
        # Ensure grids have same dimensions
        if len(predicted) != len(actual):
            return {
                'correct': False,
                'pixel_accuracy': 0.0,
                'total_pixels': len(actual) * (len(actual[0]) if actual else 0),
                'correct_pixels': 0,
                'error': 'Grid height mismatch'
            }
        
        if any(len(row) != len(actual[0]) for row in predicted):
            return {
                'correct': False,
                'pixel_accuracy': 0.0,
                'total_pixels': len(actual) * len(actual[0]),
                'correct_pixels': 0,
                'error': 'Grid width mismatch'
            }
        
        # Count correct pixels
        total_pixels = 0
        correct_pixels = 0
        
        for pred_row, actual_row in zip(predicted, actual):
            for pred_cell, actual_cell in zip(pred_row, actual_row):
                total_pixels += 1
                if pred_cell == actual_cell:
                    correct_pixels += 1
        
        return {
            'correct': correct_pixels == total_pixels,
            'pixel_accuracy': correct_pixels / total_pixels if total_pixels > 0 else 0.0,
            'total_pixels': total_pixels,
            'correct_pixels': correct_pixels,
            'error': None
        }
    
    def calculate_residual_grid(self, predicted: List[List[int]], actual: List[List[int]]) -> List[List[int]]:
        """
        Calculate the residual grid using difference-based approach.
        
        The residual represents the "patch" needed to reconstruct the actual output:
        actual_output = predicted_output + residual
        
        This assumes predicted and actual have the same dimensions.
        Dimension mismatches should be handled at a higher level.
        """
        residual = []
        
        for pred_row, actual_row in zip(predicted, actual):
            residual_row = []
            for pred_cell, actual_cell in zip(pred_row, actual_row):
                # Difference-based: actual - predicted
                # This allows perfect reconstruction: predicted + residual = actual
                residual_row.append(actual_cell - pred_cell)
            residual.append(residual_row)
        
        return residual
    
    def gzip_compress_grid(self, data) -> int:
        """Compress data (grid or flattened list) and return the gzipped size in bytes"""
        # Convert data to a string representation
        data_str = json.dumps(data)
        
        # Compress and return size
        compressed = gzip.compress(data_str.encode('utf-8'))
        return len(compressed)
    
    def strip_comments_and_compress(self, program: str) -> int:
        """Strip comments from Python code and return gzipped size in bytes"""
        import ast
        import re
        
        # Remove single-line comments (lines starting with #)
        lines = program.split('\n')
        lines_no_comments = []
        for line in lines:
            # Remove inline comments but preserve # inside strings
            # Simple approach: split on # and take the first part if not in quotes
            if '#' in line:
                # This is a simple approach - doesn't handle # inside strings perfectly
                # but should work for most generated code
                comment_pos = line.find('#')
                line = line[:comment_pos].rstrip()
            if line.strip():  # Only keep non-empty lines
                lines_no_comments.append(line)
        
        cleaned_program = '\n'.join(lines_no_comments)
        
        # Compress and return size
        compressed = gzip.compress(cleaned_program.encode('utf-8'))
        return len(compressed)
    
    def calculate_mdl_score(self, program: str, residual_grid: List[List[int]]) -> Dict:
        """
        Calculate MDL (Minimum Description Length) score using gzip compression
        
        Args:
            program: The Python program as a string
            residual_grid: The residual grid (differences between predicted and actual)
        
        Returns:
            Dictionary with MDL components and total score (all in bytes)
        """
        # Use gzip compression for program (after stripping comments)
        program_bytes = self.strip_comments_and_compress(program)
        
        # Compress residual grid
        residual_bytes = self.gzip_compress_grid(residual_grid)
        
        # MDL score: program + residual (training examples are given, not part of description length)
        mdl_score = program_bytes + residual_bytes
        
        return {
            'program_bytes': program_bytes,
            'residual_bytes': residual_bytes,
            'mdl_score': mdl_score
        }
    
    def calculate_residual_reduction(self, program: str, training_examples: List[Dict], executor) -> Dict:
        """
        Calculate residual reduction percentage - a cleaner measure of pattern learning.
        
        This measures what percentage of the transformation pattern the program learned:
        residual_reduction = (null_residual_bytes - program_residual_bytes) / null_residual_bytes
        
        Args:
            program: The Python program as a string
            training_examples: List of {"input": grid, "output": grid} training examples
            executor: ProgramExecutor instance to run the program
        
        Returns:
            Dictionary with residual reduction components and percentage
        """
        # Calculate residuals for all training examples
        all_training_residuals = []
        training_errors = []
        training_executions = 0
        training_correct = 0
        
        for example in training_examples:
            train_input = example['input']
            train_expected = example['output']
            
            # Execute program on training input
            train_predicted, error, timed_out = executor.execute_program(program, train_input)
            
            if error or timed_out or train_predicted is None:
                # If program fails on training, use null baseline residual
                training_errors.append({
                    'input': train_input,
                    'expected': train_expected, 
                    'error': error or 'timeout' if timed_out else 'no output'
                })
                # Use null baseline: grid of zeros with correct output dimensions
                null_prediction = [[0 for _ in range(len(train_expected[0]))] for _ in range(len(train_expected))]
                residual = self.calculate_residual_grid(null_prediction, train_expected)
            else:
                training_executions += 1
                
                # Check dimensions first
                if (len(train_predicted) != len(train_expected) or 
                    any(len(row) != len(train_expected[0]) for row in train_predicted)):
                    # Wrong dimensions - use null baseline residual
                    null_prediction = [[0 for _ in range(len(train_expected[0]))] for _ in range(len(train_expected))]
                    residual = self.calculate_residual_grid(null_prediction, train_expected)
                else:
                    # Calculate difference-based residual
                    residual = self.calculate_residual_grid(train_predicted, train_expected)
                    
                    # Check if this training example was solved correctly
                    if train_predicted == train_expected:
                        training_correct += 1
            
            # Flatten residual grid properly - convert 2D grid to 1D list
            flat_residual = []
            for row in residual:
                if isinstance(row, list):
                    flat_residual.extend(row)
                else:
                    flat_residual.append(row)
            all_training_residuals.extend(flat_residual)
        
        # Compress program's training residuals
        program_residual_bytes = self.gzip_compress_grid(all_training_residuals)
        
        # Calculate null program residuals for comparison
        null_residuals = self.calculate_null_program_training_residuals(training_examples)
        null_residual_bytes = self.gzip_compress_grid(null_residuals)
        
        # Calculate residual reduction percentage
        if null_residual_bytes > 0:
            residual_reduction = (null_residual_bytes - program_residual_bytes) / null_residual_bytes
            residual_reduction = max(0.0, min(1.0, residual_reduction))  # Clamp to [0,1]
        else:
            residual_reduction = 1.0 if program_residual_bytes == 0 else 0.0
        
        return {
            'program_residual_bytes': program_residual_bytes,
            'null_residual_bytes': null_residual_bytes,
            'residual_reduction': residual_reduction,
            'pattern_learning_score': residual_reduction * 100,  # 0-100% score
            'training_examples_count': len(training_examples),
            'training_executions': training_executions,
            'training_correct': training_correct,
            'training_errors': training_errors
        }
    
    def calculate_null_program_training_residuals(self, training_examples: List[Dict]) -> List[int]:
        """
        Calculate raw residuals for null program on training examples.
        
        Null program predicts a grid of zeros with the correct output dimensions.
        This represents "knowing nothing except the output size".
        
        Args:
            training_examples: List of {"input": grid, "output": grid} training examples
        
        Returns:
            Flattened list of all training residuals for null program
        """
        all_training_residuals = []
        
        for example in training_examples:
            train_output = example['output']
            
            # Null program predicts grid of zeros with correct output dimensions
            null_prediction = [[0 for _ in range(len(train_output[0]))] for _ in range(len(train_output))]
            
            # Calculate residual: actual - null_prediction
            residual = self.calculate_residual_grid(null_prediction, train_output)
            
            # Flatten residual grid properly - convert 2D grid to 1D list
            flat_residual = []
            for row in residual:
                if isinstance(row, list):
                    flat_residual.extend(row)
                else:
                    flat_residual.append(row)
            all_training_residuals.extend(flat_residual)
        
        return all_training_residuals

    def calculate_null_program_training_mdl(self, training_examples: List[Dict]) -> Dict:
        """
        Calculate MDL score for the null program using training examples.
        
        Args:
            training_examples: List of {"input": grid, "output": grid} training examples
        
        Returns:
            Dictionary with null program training-based MDL components and total score
        """
        # Null program: def transform(grid): return grid
        null_program = "def transform(grid):\n    return grid"
        null_program_bytes = self.strip_comments_and_compress(null_program)
        
        # For null program applied to training examples:
        # predicted = input, actual = output, so residual represents full transformation cost
        all_training_residuals = []
        
        for example in training_examples:
            train_input = example['input']
            train_output = example['output']
            
            # Null program predicts input unchanged, so residual = output - input
            residual = self.calculate_residual_grid(train_input, train_output)
            all_training_residuals.extend(residual)
        
        # Compress all training residuals together
        training_residual_bytes = self.gzip_compress_grid(all_training_residuals)
        
        # Simple sum since both components are in bytes
        null_mdl_score = null_program_bytes + training_residual_bytes
        
        return {
            'null_program': null_program,
            'null_program_bytes': null_program_bytes,
            'null_training_residual_bytes': training_residual_bytes,
            'null_mdl_score': null_mdl_score,
            'training_examples_count': len(training_examples)
        }

    def calculate_null_program_mdl(self, actual_grid: List[List[int]]) -> Dict:
        """
        Calculate MDL score for the null program (returns input unchanged)
        
        Args:
            actual_grid: The expected output grid
        
        Returns:
            Dictionary with null program MDL components and total score (all in bytes)
        """
        # Null program: def transform(grid): return grid
        null_program = "def transform(grid):\n    return grid"
        null_program_bytes = self.strip_comments_and_compress(null_program)
        
        # For null program, residual is the full actual grid (since predicted = input, actual = output)
        # This represents the full "surprise" of the transformation
        null_residual_bytes = self.gzip_compress_grid(actual_grid)
        
        # Simple sum since both components are in bytes
        null_mdl_score = null_program_bytes + null_residual_bytes
        
        return {
            'null_program': null_program,
            'null_program_bytes': null_program_bytes,
            'null_residual_bytes': null_residual_bytes,
            'null_mdl_score': null_mdl_score
        }


class ProgramExecutor:
    """Executes Python programs with timeout and captures output"""
    
    def __init__(self, timeout: float = 0.1):
        self.timeout = timeout
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        import numpy as np
        
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        else:
            return obj
    
    def execute_program(self, program: str, test_input: List[List[int]]) -> Tuple[Optional[List[List[int]]], str, bool]:
        """
        Execute a Python program with the test input
        
        Returns:
            (output_grid, error_message, timed_out)
        """
        # Create a wrapper script that will execute the program
        wrapper = f"""
import json
import sys

def convert_numpy_types(obj):
    \"\"\"Convert numpy types to Python native types for JSON serialization\"\"\"
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
    except ImportError:
        pass  # numpy not available
    
    if isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

# Define the test input
test_input = {json.dumps(test_input)}

# Execute the provided program
{program}

# The program should define a function called 'transform' or 'solve'
# Try common function names
output = None
if 'transform' in locals():
    output = transform(test_input)
elif 'solve' in locals():
    output = solve(test_input)
elif 'apply_transform' in locals():
    output = apply_transform(test_input)
else:
    # If no function found, try to find any function that takes a grid
    for name, obj in locals().items():
        if callable(obj) and not name.startswith('_'):
            try:
                result = obj(test_input)
                if isinstance(result, list) and all(isinstance(row, list) for row in result):
                    output = result
                    break
            except:
                pass

if output is not None:
    # Convert numpy types before JSON serialization
    output = convert_numpy_types(output)
    print(json.dumps(output))
else:
    print("ERROR: No valid transformation function found")
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(wrapper)
            temp_file = f.name
        
        try:
            # Run the program with timeout
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode == 0 and result.stdout.strip():
                try:
                    if result.stdout.strip().startswith("ERROR:"):
                        return None, result.stdout.strip(), False
                    
                    output = json.loads(result.stdout.strip())
                    
                    # Convert numpy types to Python types if needed
                    if output is not None:
                        output = self._convert_numpy_types(output)
                    
                    return output, "", False
                except json.JSONDecodeError:
                    return None, f"Invalid output format: {result.stdout}", False
            else:
                error = result.stderr if result.stderr else "Program produced no output"
                return None, error, False
                
        except subprocess.TimeoutExpired:
            return None, f"Program exceeded timeout of {self.timeout}s", True
        except Exception as e:
            return None, str(e), False
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass


# Example usage
if __name__ == "__main__":
    scorer = GridScorer()
    
    # Test grid scoring
    predicted = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    actual = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]  # One pixel different
    
    score = scorer.score_grid(predicted, actual)
    print("Grid score:", json.dumps(score, indent=2))
    
    # Test difference-based residual and reconstruction
    residual = scorer.calculate_residual_grid(predicted, actual)
    print(f"\nDifference-based residual: {residual}")
    
    # Demonstrate perfect reconstruction: actual = predicted + residual
    reconstructed = []
    for pred_row, residual_row in zip(predicted, residual):
        recon_row = []
        for pred_cell, residual_cell in zip(pred_row, residual_row):
            recon_row.append(pred_cell + residual_cell)
        reconstructed.append(recon_row)
    
    print(f"Original actual:    {actual}")
    print(f"Reconstructed:      {reconstructed}")
    print(f"Perfect reconstruction? {reconstructed == actual}")
    
    program = "def transform(grid):\n    return [[grid[row][col] for col in range(len(grid[0]))] for row in range(len(grid))]"
    mdl = scorer.calculate_mdl_score(program, residual)
    print("\nMDL score:", json.dumps(mdl, indent=2))
    
    # Test program execution
    executor = ProgramExecutor()
    test_input = [[1, 2], [3, 4]]
    output, error, timed_out = executor.execute_program(program, test_input)
    print("\nProgram execution:")
    print(f"  Output: {output}")
    print(f"  Error: {error}")
    print(f"  Timed out: {timed_out}")