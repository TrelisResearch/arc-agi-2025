from typing import Tuple

from llm_python.augmentations import (
    ColorRotationAugmentation,
    HorizontalFlipAugmentation,
    VerticalFlipAugmentation,
)
from llm_python.utils.arc_tester import ArcTester
from llm_python.utils.task_loader import TaskData


def grids_equal(grid1, grid2):
    """Check if two grids are equal."""
    if len(grid1) != len(grid2):
        return False
    for row1, row2 in zip(grid1, grid2):
        if len(row1) != len(row2):
            return False
        for cell1, cell2 in zip(row1, row2):
            if cell1 != cell2:
                return False
    return True


def count_matching_train_outputs(original_outputs, augmented_outputs):
    """Count how many training outputs match between original and augmented results."""
    if len(original_outputs) != len(augmented_outputs):
        return 0

    matches = 0
    for orig, aug in zip(original_outputs, augmented_outputs):
        if orig is None or aug is None:
            # If either is None, only count as match if both are None
            if orig is None and aug is None:
                matches += 1
        else:
            if grids_equal(orig, aug):
                matches += 1

    return matches


def detect_transduction_augmentation(
    program: str, task_data: TaskData, arc_tester: "ArcTester", debug: bool = False
) -> Tuple[bool, str]:
    """
    Detect if a program is transductive by testing augmentation invariance.

    A program is considered transductive if it's NOT invariant to ANY of the
    three augmentations (vertical flip, horizontal flip, color rotation).

    If a program is invariant to at least one augmentation, it's likely
    learning the actual pattern rather than memorizing specific values.

    Args:
        program: The program code string to test
        task_data: The original task data
        arc_tester: ArcTester instance to use for program execution
        debug: Whether to print debug information

    Returns:
        Tuple of (is_transductive, reason)
    """
    # Get original program results
    try:
        original_results = arc_tester.test_program(program, task_data)
    except Exception as e:
        if debug:
            print(f"    ❌ Program failed on original task: {e}")
        return True, f"Program failed on original task: {str(e)}"

    # Extract original training outputs
    original_train_outputs = original_results.train_outputs

    if not original_train_outputs:
        if debug:
            print("    ℹ️  No training outputs to compare")
        return False, "No training outputs to compare"

    # Check if all outputs are None (program failed to produce any valid outputs)
    if all(output is None for output in original_train_outputs):
        if debug:
            print("    🚫 Program failed to produce any valid outputs")
        return True, "Program failed to produce any valid outputs"

    # Define augmentations to test
    augmentations = [
        ("vertical_flip", VerticalFlipAugmentation()),
        ("horizontal_flip", HorizontalFlipAugmentation()),
        ("color_rotation", ColorRotationAugmentation(offset=1)),
    ]

    invariant_augmentations = []

    for aug_name, augmentation in augmentations:
        try:
            # Apply augmentation to task
            augmented_task = augmentation.forward_task(task_data)

            # Run program on augmented task
            augmented_results = arc_tester.test_program(program, augmented_task)

            # Create a TaskData structure from the augmented results to un-augment
            augmented_results_task = {
                "train": [
                    {
                        "input": augmented_results.train_inputs[i],
                        "output": augmented_results.train_outputs[i],
                    }
                    for i in range(len(augmented_results.train_inputs))
                ],
                "test": [
                    {
                        "input": augmented_results.test_inputs[i],
                        "output": augmented_results.test_outputs[i],
                    }
                    for i in range(len(augmented_results.test_inputs))
                ],
            }

            # Un-augment the results
            unaugmented_results = augmentation.backward_task(augmented_results_task)  # type: ignore

            # Extract training outputs from un-augmented results
            unaugmented_train_outputs = [
                example["output"] for example in unaugmented_results["train"]
            ]

            # Count matches
            matches = count_matching_train_outputs(
                original_train_outputs, unaugmented_train_outputs
            )
            total_outputs = len(original_train_outputs)

            if debug:
                print(
                    f"    {aug_name}: {matches}/{total_outputs} training outputs match"
                )

            # Consider invariant if all outputs match
            if matches == total_outputs:
                invariant_augmentations.append(aug_name)

        except Exception as e:
            if debug:
                print(f"    ⚠️  {aug_name} augmentation failed: {e}")
            # If augmentation fails, we can't determine invariance for this augmentation
            continue

    # Program is transductive if it's NOT invariant to ANY augmentation
    is_transductive = len(invariant_augmentations) == 0

    if is_transductive:
        reason = "Program is not invariant to any augmentation (vertical flip, horizontal flip, color rotation)"
        if debug:
            print(f"    🚫 Transductive: {reason}")
    else:
        reason = f"Program is invariant to: {', '.join(invariant_augmentations)}"
        if debug:
            print(f"    ✅ Not transductive: {reason}")

    return is_transductive, reason


def detect_transduction(
    program: str, task_data: TaskData, debug: bool = False
) -> Tuple[bool, str, bool, str]:
    """
    Legacy detect transduction function that checks for hardcoded values.

    Detect if a program hardcodes training or test outputs (transduction).
    Returns (is_train_transductive, train_reason, is_test_transductive, test_reason).

    Train transduction: Program hardcodes training outputs - excluded from voting
    Test transduction: Program hardcodes test outputs - tagged for analysis
    """
    # Check 1: Very long lines (likely hardcoded values) - counts as train transduction
    lines = program.split("\n")
    for line_num, line in enumerate(lines, 1):
        if len(line) > 200:
            reason = f"Line {line_num} exceeds 200 characters (likely hardcoded)"
            if debug:
                print(f"    🚫 Long line detected: {reason}")
                print(f"       Line: {line[:100]}...")
            return True, reason, False, ""

    # Check 2: Hardcoded output values
    train_examples = task_data.get("train", [])
    test_examples = task_data.get("test", [])

    # Collect outputs
    train_outputs = [ex["output"] for ex in train_examples if ex.get("output")]
    test_outputs = [ex["output"] for ex in test_examples if ex.get("output")]

    if not train_outputs and not test_outputs:
        return False, "", False, ""

    # String cleaning function
    flag_one = any(
        (1, 1) == (len(output), len(output[0]) if output else 0)
        for output in train_outputs
        if output is not None
    )
    clean_string = (
        (lambda s: str(s).replace(" ", ""))
        if flag_one
        else (lambda s: str(s).replace(" ", "").replace("[", "").replace("]", ""))
    )

    cleaned_code = clean_string(program)

    # Helper function to check outputs
    def check_outputs(outputs, output_type):
        for i, output in enumerate(outputs):
            if output is None:
                continue
            output_str = clean_string(output)
            if len(output_str) > 2 and output_str in cleaned_code:
                reason = f"{output_type} output {i + 1} hardcoded in program: {output_str[:50]}..."
                if debug:
                    symbol = "🚫" if output_type == "Training" else "⚠️"
                    print(f"    {symbol} {output_type} transduction detected: {reason}")
                    code_idx = cleaned_code.find(output_str)
                    context_start = max(0, code_idx - 30)
                    context_end = min(
                        len(cleaned_code), code_idx + len(output_str) + 30
                    )
                    print(
                        f"       Code context: ...{cleaned_code[context_start:context_end]}..."
                    )
                return True, reason
        return False, ""

    # Check train and test outputs
    train_transductive, train_reason = check_outputs(train_outputs, "Training")
    test_transductive, test_reason = check_outputs(test_outputs, "Test")

    return train_transductive, train_reason, test_transductive, test_reason
