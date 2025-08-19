import pytest
import yaml
import os
from llm_python.transduction.augmentation_classifier import (
    AugmentationBasedTransductionClassifier,
)
from llm_python.transduction.code_classifier import CodeTransductionClassifier
from llm_python.utils.arc_tester import ArcTester
from llm_python.utils.task_loader import TaskLoader


def load_test_data():
    """Load test data from YAML file."""
    test_file = os.path.join(os.path.dirname(__file__), "test_data.yaml")
    with open(test_file, "r") as f:
        return yaml.safe_load(f)


# Load test cases for parametrization
test_cases = load_test_data()
arc_tester = ArcTester()
task_loader = TaskLoader()

# Initialize classifiers
augmentation_classifier = AugmentationBasedTransductionClassifier(arc_tester)
code_classifier = CodeTransductionClassifier()


@pytest.mark.parametrize("case", test_cases)
def test_both_classifiers(case):
    """Test that both classifiers produce the expected result."""
    code = case["code"]
    task_id = case["task_id"]
    expected = case["transductive"]
    task_data = task_loader.load_task(task_id)

    # Test augmentation-based classifier
    aug_result, aug_confidence = augmentation_classifier.is_transductive(code, task_data)

    # Test code-based classifier  
    code_result, code_confidence = code_classifier.is_transductive(code, task_data)

    # Both should match expected result
    assert aug_result == expected, (
        f"Augmentation classifier failed for {task_id}: expected {expected}, got {aug_result} (confidence: {aug_confidence})"
    )
    assert code_result == expected, (
        f"Code classifier failed for {task_id}: expected {expected}, got {code_result} (confidence: {code_confidence})"
    )
