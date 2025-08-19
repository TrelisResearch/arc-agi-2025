from typing import Tuple, Optional

from llm_python.augmentations import (
    ColorRotationAugmentation,
    HorizontalFlipAugmentation,
    VerticalFlipAugmentation,
)
from llm_python.utils.arc_tester import ArcTester
from llm_python.utils.task_loader import TaskData


class AugmentationBasedTransductionClassifier:
    """
    Classifier that determines if a program is transductive based on augmentation invariance testing.
    """
    
    def __init__(self, arc_tester: ArcTester):
        """
        Initialize the augmentation-based transduction classifier.
        
        Args:
            arc_tester: ArcTester instance to use for program execution
        """
        self.arc_tester = arc_tester
    
    def is_transductive(self, program: str, task_data: Optional[TaskData] = None) -> Tuple[bool, None]:
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
            task_data: The task data (required for augmentation-based classification)

        Returns:
            Tuple of (is_transductive, confidence)
        """
        if task_data is None:
            raise ValueError("task_data is required for augmentation-based transduction classification")
        
        # Get original program results
        original_results = self.arc_tester.test_program(program, task_data)
        print(original_results)

        # Extract original training outputs
        original_train_outputs = original_results.train_outputs

        if not original_train_outputs:
            return False, None  # No training outputs to compare, return low confidence

        # Check if all outputs are None (program failed to produce any valid outputs)
        if all(output is None for output in original_train_outputs):
            return True, None  # Program failed to produce any valid outputs, likely transductive

        # Define augmentations to test
        augmentations = [
            ("vertical_flip", VerticalFlipAugmentation()),
            ("horizontal_flip", HorizontalFlipAugmentation()),
            ("color_rotation", ColorRotationAugmentation(offset=1)),
        ]

        invariant_augmentations: list[str] = []

        for aug_name, augmentation in augmentations:
            try:
                # Apply augmentation to task
                augmented_task = augmentation.forward_task(task_data)

                # Run program on augmented task
                augmented_results = self.arc_tester.test_program(program, augmented_task)

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
                matches = self._count_matching_train_outputs(
                    original_train_outputs, unaugmented_train_outputs
                )
                total_outputs = len(original_train_outputs)

                # Consider invariant if all outputs match
                if matches == total_outputs:
                    invariant_augmentations.append(aug_name)

            except Exception:
                # If augmentation fails, we can't determine invariance for this augmentation
                continue

        # Program is transductive if it's NOT invariant to ANY augmentation
        # (i.e., if it fails all augmentation tests, it's likely memorizing)
        is_transductive = len(invariant_augmentations) == 0

        return is_transductive, None
    
    def _count_matching_train_outputs(self, original_outputs, augmented_outputs):
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
                if self._grids_equal(orig, aug):
                    matches += 1

        return matches

    def _grids_equal(self, grid1, grid2):
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
