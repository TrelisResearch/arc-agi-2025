from typing import TypedDict, List

class SoarProgramExample(TypedDict):
    """Schema for training examples enriched for actual training with separate train/test data"""
    task_id: str # Task ID from ARC
    reasoning: str # Reasoning trace if provided
    code: str # Program code that should define a `generate` function
    correct_train_input: List[bool] # Training inputs where program produced correct output
    correct_test_input: List[bool] # Test inputs where program produced correct output
    predicted_train_output: List[List[List[int]]] # Program's predicted outputs for training inputs
    predicted_test_output: List[List[List[int]]] # Program's predicted outputs for test inputs
    train_input: List[List[List[int]]] # Actual training inputs
    test_input: List[List[List[int]]] # Actual test inputs
    model: str # What model generated this example
    generation: int # Generation number if applicable

