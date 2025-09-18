import asyncio
import traceback
from typing import Any, Callable, Coroutine, Optional, ParamSpec, TypeVar

import pandas as pd
from dotenv import load_dotenv

from experimental.flags import flag, parse_flags
from experimental.mrr.llms import invoke_llm_async
from llm_python.datasets.io import read_soar_parquet, write_soar_parquet

load_dotenv()


input_file_flag = flag(
    name="input_file",
    type=str,
    required=True,
    help="Path to the input SOAR parquet file.",
)
output_file_flag = flag(
    name="output_file",
    type=str,
    required=True,
    help="Path to the output SOAR parquet file.",
)
llm_parallelism_flag = flag(
    name="llm_parallelism",
    type=int,
    default=8,
    help="Number of parallel LLM calls to make.",
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


def create_english_prompt(code: str) -> str:
    return f"""
You are an expert programmer. Your task is to explain the following code in plain English.
This code defines a transformation function that takes a 2D input grid of colors with the values 0-9, and produces an output grid.
Describe the algorithm and the process it follows in english. Your description should be high level, a couple of paragraphs, such that a human or LLM could follow the description to implement or perform the same algorithm.
The description should not describe the specific code or refer to the code, but rather general steps a human could follow, and should be written directed at the user, as a set of instructions.

Code:
```python
{code}
```

Explanation:
"""


async def generate_description(code: str) -> Optional[str]:
    prompt = create_english_prompt(code)
    try:
        response = await invoke_llm_async(prompt)
        if response is None:
            print("LLM failed to produce a response.")
            return None
        return response
    except Exception as e:
        print(f"Error invoking LLM: {e}")
        traceback.print_exc()
        return None


async def process_row(row, limited_generator):
    description = await limited_generator(row["code"])
    if description is None:
        return None
    return {
        "task_id": row["task_id"],
        "row_id": row.get("row_id"),  # Use .get for safer access
        "code": row["code"],
        "description": description,
    }


async def main():
    parse_flags()

    input_file = input_file_flag()
    output_file = output_file_flag()
    parallelism = llm_parallelism_flag()

    print(f"Reading from {input_file}...")
    df = read_soar_parquet(input_file)
    # df = df.head(5)
    print(f"Read {len(df)} rows.")

    limited_generator = limit_parallelism(generate_description, parallelism)

    tasks = [process_row(row, limited_generator) for _, row in df.iterrows()]
    results = await asyncio.gather(*tasks)

    # Filter out None results from failed operations
    print(f"Processing complete. {len(results)} results obtained.")
    successful_results = [res for res in results if res is not None]
    print(
        f"{len(successful_results)} rows successfully processed, {len(results) - len(successful_results)} failed."
    )

    output_df = pd.DataFrame(successful_results)

    print(f"Writing {len(output_df)} rows to {output_file}...")
    output_df.to_parquet(output_file, index=False)
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
