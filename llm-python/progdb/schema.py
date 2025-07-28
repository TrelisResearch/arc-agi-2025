from typing import TypedDict, List, Any, Optional

class LogData(TypedDict):
    """Schema for data extracted from log files"""
    task_id: str
    program: str
    reasoning: str
    model: str

class TrainingExample(TypedDict):
    """Schema for training examples extracted from logs"""
    code: str # Program code that should define a `generate` function
    reasoning: str # Reasoning trace if provided
    model: str # What model generated this example
    task_id: str # Task ID of the source task that this program was generated for
    train_correct_fraction: float # Fraction of training examples for the task ID this program solves
    test_correct_fraction: float # Fraction of test examples for the task ID this program solves
    sample_inputs: List[List[List[int]]]  # List of 2D grids (combined train+test)
    sample_outputs: List[List[List[int]]]  # List of 2D grids (combined train+test)

class EnrichedTrainingExample(TypedDict):
    """Schema for training examples enriched for actual training with separate train/test data"""
    reasoning: str # Reasoning trace if provided
    code: str # Program code that should define a `generate` function
    correct_train_input: List[List[List[int]]] # Training inputs where program produced correct output
    train_input: List[List[List[int]]] # All training inputs (2D grids)
    train_output: List[List[List[int]]] # All training outputs (2D grids)
    predicted_train_output: List[List[List[int]]] # Program's predicted outputs for training inputs
    correct_test_input: List[List[List[int]]] # Test inputs where program produced correct output
    test_input: List[List[List[int]]] # All test inputs (2D grids)
    test_output: List[List[List[int]]] # All test outputs (2D grids)
    predicted_test_output: List[List[List[int]]] # Program's predicted outputs for test inputs
    task_id: str # Task ID of the source task that this program was generated for
    model: str # What model generated this example
    generation: int # Generation number if applicable

