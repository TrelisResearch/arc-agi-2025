import asyncio
import traceback
from typing import Any, Callable, Coroutine, List, ParamSpec, TypeVar

from dotenv import load_dotenv
from experimental.flags import flag, parse_flags
from experimental.mrr.heuristic_segmenter import (
    TokenizedObject,
    create_prompt,
    detokenize_objects,
    parse_llm_output,
    reconstruct_grid,
    segment_grid,
    tokenize_objects,
)
from experimental.mrr.llms import invoke_llm, invoke_llm_async
from experimental.mrr.minimal_arc_tester import (
    SingleSolverInput,
    SolverInput,
    evaluate_solver,
    split_multi_test,
)
import numpy as np

from llm_python.utils.numpy import convert_numpy_types
from llm_python.utils.task_loader import Grid

load_dotenv()


def grid_to_tokenized(grid: Grid) -> List[TokenizedObject]:
    segmented_objects = segment_grid(np.array(grid))
    tokenized_objects = tokenize_objects(segmented_objects)
    return tokenized_objects


def tokens_to_grid(tokens: List[TokenizedObject]) -> Grid:
    segmented_objects = detokenize_objects(tokens)
    return convert_numpy_types(reconstruct_grid(segmented_objects))

solver_parallelism_flag = flag(
    name="solver_parallelism",
    type=int,
    default=1,
    help="Number of parallel solver instances to run."
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

async def main():
    parse_flags()

    async def solver(input: SingleSolverInput) -> Grid:
        tokenized_examples = [
            (
                grid_to_tokenized(example["input"]),
                grid_to_tokenized(example["output"]),
            )
            for example in input.examples
            if example["output"]
        ]

        if max(len(inp) for inp, _ in tokenized_examples) > 20:
            print(
                f"Warning: Example with {max(len(inp) for inp, _ in tokenized_examples)} objects exceeds token limit, skipping."
            )
            return [[0]]  # Return a dummy grid on failure

        tokenized_test_input = grid_to_tokenized(input.test_input)

        prompt = create_prompt(tokenized_examples, tokenized_test_input)
        # print(f"Prompt for task {input.task_id}:\n{prompt}\n")

        try:
            response, reasoning = await invoke_llm_async(prompt)
            if response is None:
                print(f"LLM failed to produce a response for task {input.task_id}.")
                return [[0]]
            tokens = parse_llm_output(response)
            if tokens is None:
                print(f"Failed to parse LLM response for task {input.task_id}.")
                return [[0]]
            grid = tokens_to_grid(tokens)
            print(f"Response from LLM for task {input.task_id}:\n{response}\n")
            print(f"Generated grid for task {input.task_id}:\n{grid}\n")
            return grid
        except Exception as e:
            print(f"Error invoking LLM for task {input.task_id}: {e}")
            traceback.print_exc()
            return [[0]]

    wrapped_solver = split_multi_test(limit_parallelism(solver, solver_parallelism_flag()))
    result = await evaluate_solver(wrapped_solver, "arc-prize-2025/evaluation")

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
