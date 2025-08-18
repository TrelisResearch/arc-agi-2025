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

    Tests if a program is invariant to augmentations by:
    1. Augmenting the train example grids (both input and output)
    2. Running the function on augmented data
    3. Un-augmenting the results
    4. Checking if results match the function applied to original grids

    A program is considered NOT transductive (captures pattern essence) if it's 
    invariant to at least one augmentation across ALL training examples.

    A program is considered transductive (likely memorizing) if it's NOT 
    invariant to ANY of the augmentations.

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
    # (i.e., if it fails all augmentation tests, it's likely memorizing)
    is_transductive = len(invariant_augmentations) == 0

    if is_transductive:
        reason = "Program is not invariant to any augmentation (likely memorizing specific values)"
        if debug:
            print(f"    🚫 Transductive: {reason}")
    else:
        reason = f"Program is invariant to: {', '.join(invariant_augmentations)} (likely captures pattern essence)"
        if debug:
            print(f"    ✅ Not transductive: {reason}")

    return is_transductive, reason

