import asyncio
import logging
import traceback
from typing import Callable, Coroutine, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

from experimental.flags import flag
from llm_python.utils.grids import grids_equal
from llm_python.utils.task_loader import Grid, TaskData, TaskExample, get_task_loader
from collections import defaultdict

max_attempts_flag = flag(
    name="max_attempts", type=int, help="Number of attempts to make per task.", default=1
)

logger = logging.getLogger(__name__)

@dataclass
class Attempt:
    task_id: str
    test_outputs: List[Grid]
    correct: List[bool]
    correct_size: List[bool]
    pixel_accuracy: float
    error: str | None = None
    extra_info: dict | None = None


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
    single_test_solver: Callable[[SingleSolverInput], Coroutine[None, None, Grid]],
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
    tasks: List[Tuple[str, TaskData]]
) -> List[Attempt]:
    task_ids_with_any_correct = set()

    async def solve_task(task_id, task_data, attempt_number) -> Attempt:
        test_inputs = [test_example["input"] for test_example in task_data["test"]]
        try:
            solver_input = SolverInput(
                task_id=task_id,
                examples=task_data["train"],
                test_inputs=test_inputs,
            )
            outputs = await solver(solver_input)
            groundtruth_outputs = [example["output"] for example in task_data["test"]]

            # Pad arrays to at least 30x30 for pixel comparison
            def pad_to_30x30(arr):
                h, w = arr.shape
                pad_h = max(0, 30 - h)
                pad_w = max(0, 30 - w)
                return np.pad(arr, ((0, pad_h), (0, pad_w)), mode="constant")

            test_correct = []
            test_correct_size = []
            correct_pixels = 0
            incorrect_pixels = 0
            for output, groundtruth in zip(outputs, groundtruth_outputs):
                test_correct_size.append(
                    len(output) == len(groundtruth)
                    and len(output[0]) == len(groundtruth[0])
                )
                are_equal = grids_equal(output, groundtruth)
                test_correct.append(are_equal)

                if are_equal:
                    print(f"✅ Attempt {attempt_number} for task {task_id} passed!")
                    task_ids_with_any_correct.add(task_id)
                else:
                    print(f"❌ Attempt {attempt_number} for task {task_id} failed.")

                padded_output = pad_to_30x30(np.array(output))
                padded_groundtruth = pad_to_30x30(np.array(groundtruth))

                correct_pixels = int(
                    (padded_output == padded_groundtruth).flatten().mean()
                )
                incorrect_pixels = int(
                    (padded_output != padded_groundtruth).flatten().sum()
                )
            pixel_accuracy = correct_pixels / (correct_pixels + incorrect_pixels)
            return Attempt(
                task_id=task_id,
                test_outputs=outputs,
                correct=test_correct,
                correct_size=test_correct_size,
                pixel_accuracy=pixel_accuracy,
                error=None,
            )

        except Exception as e:
            print(f"Solver failed on task {task_id} with error: {e}")
            traceback.print_exc()
            return Attempt(
                task_id=task_id,
                test_outputs=[[[0]] * len(task_data["test"])],
                correct=[False] * len(task_data["test"]),
                correct_size=[False] * len(task_data["test"]),
                pixel_accuracy=0.0,
                error=str(e),
            )
        finally:
            print(f"ℹ️ Tasks with successes so far: {len(task_ids_with_any_correct)} out of {len(tasks)}")
    print(
        f"Evaluating solver on {len(tasks)} tasks with {max_attempts_flag()} attempts..."
    )

    solve_calls = []
    for task_id, task_data in tasks:
        for attempt_number in range(max_attempts_flag()):
            solve_calls.append(solve_task(task_id, task_data, attempt_number))

    attempts = await asyncio.gather(*solve_calls)

    return attempts


def score_attempts(attempts: List[Attempt]) -> dict:
    grouped_attempts = defaultdict(list)
    for attempt in attempts:
        grouped_attempts[attempt.task_id].append(attempt)

    oracle_scores = {}
    total_tasks = len(grouped_attempts)

    for task_id, attempts_list in grouped_attempts.items():
        num_tests = len(attempts_list[0].correct)

        best_attempt = max(
            attempts_list, key=lambda a: sum([int(c) for c in a.correct])
        )
        oracle_scores[task_id] = sum([int(c) for c in best_attempt.correct]) / num_tests

    accuracy = 100 * sum(oracle_scores.values()) / total_tasks if total_tasks else 0

    return {
        "total_tasks": total_tasks,
        "accuracy_percent": accuracy,
    }
