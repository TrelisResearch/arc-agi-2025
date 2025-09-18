import asyncio
import copy
import random
import traceback
from typing import Any, Callable, Coroutine, Dict, List, Optional, ParamSpec, Tuple, TypeVar

import pandas as pd
from dotenv import load_dotenv

from experimental.flags import flag, parse_flags
from llm_python.utils.task_loader import Task, get_task_loader
from sandbox import create_executor
from sandbox.subprocess_executor import ExecutionTimeout

load_dotenv()

validators_file_flag = flag(
    name="validators_file",
    type=str,
    required=True,
    help="Path to the input parquet file with validator code.",
)
task_subset_flag = flag(
    name="task_subset",
    type=str,
    default="arc-prize-2024/evaluation",
    help="The task subset to use for testing.",
)
parallelism_flag = flag(
    name="parallelism",
    type=int,
    default=4,
    help="Number of parallel validation tests to run.",
)

P = ParamSpec("P")
R = TypeVar("R")


def limit_parallelism(
    func: Callable[P, Coroutine[Any, Any, R]], parallelism: int
) -> Callable[P, Coroutine[Any, Any, R]]:
    semaphore = asyncio.Semaphore(parallelism)

    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        async with semaphore:
            return await func(*args, **kwargs)

    return wrapper


def shuffle_colors(grid: List[List[int]]) -> List[List[int]]:
    """Creates a new grid with colors remapped."""
    if not grid or not grid[0]:
        return grid
    
    colors = list(range(10))
    random.shuffle(colors)
    color_map = {i: colors[i] for i in range(10)}
    
    return [[color_map[cell] for cell in row] for row in grid]

def transpose_grid(grid: List[List[int]]) -> List[List[int]]:
    """Transposes the grid."""
    if not grid:
        return []
    return [list(row) for row in zip(*grid)]

def change_random_cell(grid: List[List[int]]) -> List[List[int]]:
    """Changes one cell's value to a random different color."""
    if not grid or not grid[0]:
        return grid
    
    new_grid = copy.deepcopy(grid)
    height, width = len(new_grid), len(new_grid[0])
    
    row_idx = random.randint(0, height - 1)
    col_idx = random.randint(0, width - 1)
    
    original_color = new_grid[row_idx][col_idx]
    new_color = original_color
    while new_color == original_color:
        new_color = random.randint(0, 9)
        
    new_grid[row_idx][col_idx] = new_color
    return new_grid


class ValidatorTester:
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self._executor = create_executor("unrestricted")
        self._executor_context = self._executor.__enter__()

    def __del__(self):
        if self._executor:
            self._executor.__exit__(None, None, None)

    def execute_validator(
        self, validator_code: str, input_grid: List[List[int]], output_grid: List[List[int]]
    ) -> Tuple[Optional[bool], Optional[str]]:
        code = f"""
{validator_code}

result = validate({repr(input_grid)}, {repr(output_grid)})
"""
        try:
            result, error = self._executor_context.execute_code(code, timeout=self.timeout)
            if error:
                if isinstance(error, ExecutionTimeout):
                    return None, f"Timeout: {error}"
                return None, str(error)
            if isinstance(result, bool):
                return result, None
            return None, f"Validator returned non-boolean value: {type(result)}"
        except Exception as e:
            return None, f"Execution failed: {e}"

    async def test_validator(
        self, task: Task, validator_code: str
    ) -> Dict[str, Any]:
        task_id = task["task_id"]
        results = {"task_id": task_id, "passed": True, "details": {}}
        
        all_examples = task["train"] + task["test"]

        # Test positive cases
        positive_results = []
        for example in all_examples:
            res, err = self.execute_validator(validator_code, example["input"], example["output"])
            positive_results.append(res is True)
            if err:
                results["passed"] = False
                results["details"]["positive_error"] = err
                return results
        
        results["details"]["positive_tests_passed"] = all(positive_results)
        if not all(positive_results):
            results["passed"] = False

        # Test negative cases (augmentations)
        negative_results = []
        for example in all_examples:
            augmentations = {
                "shuffled": shuffle_colors(example["output"]),
                "transposed": transpose_grid(example["output"]),
                "random_cell": change_random_cell(example["output"]),
            }
            for aug_name, aug_output in augmentations.items():
                if aug_output == example["output"]: continue # Skip if augmentation had no effect
                
                res, err = self.execute_validator(validator_code, example["input"], aug_output)
                negative_results.append(res is False)
                if err:
                    results["passed"] = False
                    results["details"][f"negative_error_{aug_name}"] = err
                    return results

        results["details"]["negative_tests_passed"] = all(negative_results)
        if not all(negative_results):
            results["passed"] = False
            
        return results


async def main():
    parse_flags()

    validators_file = validators_file_flag()
    task_subset = task_subset_flag()
    parallelism = parallelism_flag()

    print(f"Loading validators from {validators_file}...")
    validators_df = pd.read_parquet(validators_file)
    
    task_loader = get_task_loader()
    tasks = {t["task_id"]: t for t in task_loader.get_subset_tasks(task_subset)}
    
    tester = ValidatorTester()

    async def run_test(row):
        task_id = row["task_id"]
        if task_id not in tasks:
            return {"task_id": task_id, "passed": False, "details": {"error": "Task not found in subset."}}
        
        task = tasks[task_id]
        validator_code = row["validator_code"]
        return await tester.test_validator(task, validator_code)

    limited_tester = limit_parallelism(run_test, parallelism)
    
    test_tasks = [limited_tester(row) for _, row in validators_df.iterrows()]
    
    all_results = await asyncio.gather(*test_tasks)

    passed_count = sum(1 for r in all_results if r["passed"])
    print(f"\n--- Validation Summary ---")
    print(f"{passed_count} / {len(all_results)} validators passed.")
    
    for result in all_results:
        if not result["passed"]:
            print(f"  - Task {result['task_id']}: FAILED")
            for key, value in result["details"].items():
                print(f"    - {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
