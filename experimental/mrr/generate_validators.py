import asyncio
import re
import traceback
from typing import Any, Callable, Coroutine, Dict, List, Optional, ParamSpec, TypeVar

import pandas as pd
from dotenv import load_dotenv

from experimental.flags import flag, parse_flags
from experimental.mrr.llms import invoke_llm_async
from llm_python.utils.task_loader import TaskData, get_task_loader

load_dotenv()


output_file_flag = flag(
    name="output_file",
    type=str,
    default="validators.parquet",
    help="Path to the output parquet file.",
)
llm_parallelism_flag = flag(
    name="llm_parallelism",
    type=int,
    default=8,
    help="Number of parallel LLM calls to make.",
)
task_subset_flag = flag(
    name="task_subset",
    type=str,
    default="arc-prize-2024/evaluation",
    help="The task subset to use.",
)
num_tasks_flag = flag(
    name="num_tasks",
    type=int,
    default=None,
    help="Number of tasks to process.",
)


P = ParamSpec("P")
R = TypeVar("R")


def limit_parallelism(
    func: Callable[P, Coroutine[Any, Any, R]], parallelism: int
) -> Callable[P, Coroutine[Any, Any, R]]:
    """
    Takes an async function and a parallelism limit, and returns a new function
    that limits the number of concurrent executions of the original function.
    """
    semaphore = asyncio.Semaphore(parallelism)

    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        async with semaphore:
            return await func(*args, **kwargs)

    return wrapper


def create_validator_prompt(task_data: TaskData) -> str:
    prompt = """
You are an expert programmer specializing in abstract reasoning.
You will be given a series of input/output examples for an abstract reasoning task.
Your goal is to write a Python function `validate(input_grid, output_grid)` that determines if a given `output_grid` is a correct transformation of the `input_grid` according to the rules demonstrated by the examples.

The grids are represented as lists of lists of integers, where each integer is a color from 0-9.

The `validate` function should return `True` if the transformation is correct, and `False` otherwise.

Here are the examples for the task:
"""

    def format_grid(grid: List[List[int]]) -> str:
        return str(grid)

    for i, example in enumerate(task_data["train"]):
        prompt += f"--- Example {i + 1} ---\n"
        prompt += f"Input: {format_grid(example['input'])}\n"
        prompt += f"Output: {format_grid(example['output'])}\n\n"

    prompt += """
Please provide the `validate` function inside a Python code block.
```python
def validate(input_grid: list[list[int]], output_grid: list[list[int]]) -> bool:
    # Your implementation here
```
"""
    return prompt


def parse_llm_output(output: str) -> Optional[str]:
    match = re.search(r"```python\n(.*?)\n```", output, re.DOTALL)
    if match:
        return match.group(1).strip()
    print("Could not find python code block in LLM output.")
    return None


async def process_task(task_id: str, task_data: TaskData, limited_generator: Callable) -> Optional[Dict]:
    print(f"Processing task {task_id}...")
    prompt = create_validator_prompt(task_data)
    try:
        response = await limited_generator(prompt)
        if response is None:
            print(f"LLM failed to produce a response for task {task_id}.")
            return None

        validator_code = parse_llm_output(response)
        if validator_code is None:
            print(f"Failed to parse LLM response for task {task_id}.")
            return None

        print(f"Successfully generated validator for task {task_id}.")
        print(validator_code)
        return {"task_id": task_id, "validator_code": validator_code}

    except Exception as e:
        print(f"Error processing task {task_id}: {e}")
        traceback.print_exc()
        return None


async def main():
    parse_flags()

    output_file = output_file_flag()
    parallelism = llm_parallelism_flag()
    task_subset = task_subset_flag()
    num_tasks = num_tasks_flag()

    task_loader = get_task_loader()
    tasks = task_loader.get_subset_tasks(task_subset)
    
    if num_tasks is not None:
        tasks = tasks[:num_tasks]

    limited_generator = limit_parallelism(invoke_llm_async, parallelism)

    async_tasks = [process_task(task_id, task_data, limited_generator) for task_id, task_data in tasks]
    results = await asyncio.gather(*async_tasks)

    successful_results = [res for res in results if res is not None]
    output_df = pd.DataFrame(successful_results)

    if not output_df.empty:
        print(f"Writing {len(output_df)} validators to {output_file}...")
        output_df.to_parquet(output_file)
        print("Done.")
    else:
        print("No successful results to write.")


if __name__ == "__main__":
    asyncio.run(main())
