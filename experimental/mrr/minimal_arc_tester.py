import asyncio
import traceback
from typing import Callable, Coroutine, List
from dataclasses import dataclass

import numpy as np

from llm_python.utils.grids import grids_equal
from llm_python.utils.task_loader import Grid, TaskExample, get_task_loader


@dataclass
class EvaluationResult:
    pixel_accuracy: float
    correct_accuracy: float
    solver_error_rate: float


@dataclass
class SolverInput:
    task_id: str
    examples: List[TaskExample]
    test_inputs: List[Grid]


@dataclass
class SingleSolverInput:
    task_id: str
    examples: List[TaskExample]
    test_input: Grid


def split_multi_test(
    single_test_solver: Callable[[SingleSolverInput], Coroutine[None, None, Grid]]
) -> Callable[[SolverInput], Coroutine[None, None, List[Grid]]]:
    """
    Takes a solver that processes a single test input and returns a solver
    that processes multiple test inputs in parallel.
    """

    async def multi_test_solver(solver_input: SolverInput) -> List[Grid]:
        single_solver_inputs = [
            SingleSolverInput(
                task_id=solver_input.task_id,
                examples=solver_input.examples,
                test_input=test_input,
            )
            for test_input in solver_input.test_inputs
        ]

        results = await asyncio.gather(
            *[single_test_solver(inp) for inp in single_solver_inputs]
        )
        return results

    return multi_test_solver


async def evaluate_solver(
    solver: Callable[[SolverInput], Coroutine[None, None, List[Grid]]],
    subset: str,
) -> EvaluationResult:
    task_loader = get_task_loader()
    tasks = task_loader.get_subset_tasks(subset)

    correct_pixels = 0
    incorrect_pixels = 0
    total_correct = 0
    solver_errors = 0

    async def solve_task(task_id, task_data):
        nonlocal correct_pixels, incorrect_pixels, total_correct, solver_errors
        test_inputs = [test_example["input"] for test_example in task_data["test"]]
        try:
            solver_input = SolverInput(
                task_id=task_id,
                examples=task_data["train"],
                test_inputs=test_inputs,
            )
            outputs = await solver(solver_input)
            groundtruth_outputs = [
                example["output"] for example in task_data["test"]
            ]

            # Pad arrays to at least 30x30 for pixel comparison
            def pad_to_30x30(arr):
                h, w = arr.shape
                pad_h = max(0, 30 - h)
                pad_w = max(0, 30 - w)
                return np.pad(arr, ((0, pad_h), (0, pad_w)), mode="constant")

            for output, groundtruth in zip(outputs, groundtruth_outputs):
                if grids_equal(output, groundtruth):
                    total_correct += 1
                    print(f"Task {task_id} solved correctly!")
                else:
                    print(f"Task {task_id} solved incorrectly.")
                    print(f"Output:\n{np.array(output)}")
                    print(f"Groundtruth:\n{np.array(groundtruth)}")

                padded_output = pad_to_30x30(np.array(output))
                padded_groundtruth = pad_to_30x30(np.array(groundtruth))

                correct_pixels += int(
                    (padded_output == padded_groundtruth).flatten().sum()
                )
                incorrect_pixels += int(
                    (padded_output != padded_groundtruth).flatten().sum()
                )
        except Exception as e:
            print(f"Solver failed on task {task_id} with error: {e}")
            traceback.print_exc()
            solver_errors += 1

    await asyncio.gather(*[solve_task(task_id, task_data) for task_id, task_data in tasks])

    total_tests = sum(len(task_data["test"]) for _, task_data in tasks)
    accuracy = total_correct / total_tests if total_tests > 0 else 0
    solver_error_rate = solver_errors / len(tasks) if tasks else 0
    pixel_accuracy = (
        correct_pixels / (correct_pixels + incorrect_pixels)
        if (correct_pixels + incorrect_pixels) > 0
        else 0
    )

    return EvaluationResult(
        pixel_accuracy=pixel_accuracy,
        correct_accuracy=accuracy,
        solver_error_rate=solver_error_rate,
    )
