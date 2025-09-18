import asyncio
import json
import traceback
from dataclasses import asdict
from datetime import datetime
from typing import Any, Callable, Coroutine, List, Optional, ParamSpec, TypeVar

import pandas as pd
from dotenv import load_dotenv

from experimental.flags import flag, parse_flags
from experimental.mrr.llms import invoke_llm_async
from experimental.mrr.minimal_arc_tester import (
    SingleSolverInput,
    evaluate_solver,
    score_attempts,
    split_multi_test,
)

from llm_python.utils.task_loader import Grid, TaskExample, get_task_loader

load_dotenv()


solver_parallelism_flag = flag(
    name="solver_parallelism",
    type=int,
    default=1,
    help="Number of parallel solver instances to run.",
)
descriptions_file_flag = flag(
    name="descriptions_file",
    type=str,
    help="Path to the parquet file with algorithm descriptions.",
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


def create_transductive_prompt(
    examples: List[TaskExample],
    test_input: Grid,
    description: Optional[str] = None,
) -> str:
    prompt = """
You are an expert in solving abstract reasoning problems.
You will be provided with several examples of pairs of colored grids in ascii format, each pair consisting of an input grid and the corresponding output grid. Each number represents one of 10 colors (0-9).
Your goal is to determine what the transformation rule is based on the examples, and then apply that rule to a new test input to produce the correct output.
"""
    if description:
        prompt += "\nHere is a description of the transformation required:\n"
        prompt += description + "\n"

    prompt += """
Before answering, briefly explain:
  a) Your reasoning for what the transformation rule is in english.
  b) The procedure you will use to transform the test input grid to produce the output grid.

At the end, output your final answer in the same ascii format as the examples with spaces between each cell value in a column and newlines to seperate rows, enclosed in <answer>...</answer> tags.

Here are the examples:
"""

    def format_grid(grid: Grid) -> str:
        return "\n".join(" ".join(str(cell) for cell in row) for row in grid)

    for i, example in enumerate(examples):
        if not example["input"] or not example["output"]:
            raise ValueError("Example input or output grid is empty.")
        prompt += f"--- Example {i + 1} ---\n"
        prompt += "Input:\n"
        prompt += format_grid(example["input"]) + "\n\n"
        prompt += "Output:\n"
        prompt += format_grid(example["output"]) + "\n\n"
        prompt += "\n\n"

    prompt += "--- Test Case ---\n"
    prompt += "Input:\n"
    prompt += format_grid(test_input) + "\n\n"

    return prompt


def parse_llm_output(output: str) -> Grid | None:
    try:
        answer_start = output.index("<answer>") + len("<answer>")
        answer_end = output.index("</answer>")
        answer_content = output[answer_start:answer_end].strip()
        lines = answer_content.splitlines()
        grid = [
            [int(cell) for cell in line.split() if cell.isdigit()] for line in lines
        ]
        # Check for valid NxM grid: non-empty and all rows same length
        if not grid or any(len(row) != len(grid[0]) for row in grid):
            print("Parsed grid is not a valid NxM grid.")
            return None
        return grid
    except (ValueError, IndexError) as e:
        print(f"Error parsing LLM output: {e}")
        return None


async def main():
    parse_flags()

    descriptions = {}
    descriptions_file = descriptions_file_flag()
    if descriptions_file:
        print(f"Loading descriptions from {descriptions_file}...")
        df = pd.read_parquet(descriptions_file)
        for _, row in df.iterrows():
            if row["task_id"] not in descriptions:
                descriptions[row["task_id"]] = []
            descriptions[row["task_id"]].append(row["description"])
        print(f"Loaded descriptions for {len(descriptions)} tasks.")

    async def solver(input: SingleSolverInput) -> Grid:
        description = None
        if input.task_id in descriptions:
            # For now, just use the first description if multiple exist
            description = descriptions[input.task_id][0]

        prompt = create_transductive_prompt(
            input.examples, input.test_input, description
        )

        try:
            response = await invoke_llm_async(prompt)
            if response is None:
                print(f"LLM failed to produce a response for task {input.task_id}.")
                return [[0]]
            grid = parse_llm_output(response)
            if grid is None:
                print(f"Failed to parse LLM response for task {input.task_id}.")
                return [[0]]
            print(f"Response from LLM for task {input.task_id}:\n{response}\n")
            print(f"Extracted grid for task {input.task_id}:\n{grid}\n")
            return grid
        except Exception as e:
            print(f"Error invoking LLM for task {input.task_id}: {e}")
            traceback.print_exc()
            return [[0]]

    wrapped_solver = split_multi_test(
        limit_parallelism(solver, solver_parallelism_flag())
    )
    task_loader = get_task_loader()
    tasks = task_loader.get_subset_tasks("arc-prize-2024/evaluation")[
        :100
    ]  # Limit to 10 tasks for testing
    result = await evaluate_solver(wrapped_solver, tasks)
    print(score_attempts(result))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"transductive_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump([asdict(attempt) for attempt in result], f, indent=2)
    print(f"Result dumped to {filename}")


if __name__ == "__main__":
    asyncio.run(main())
