import openai
from experimental.flags import flag
from typing import Any, Optional, Tuple, cast

# Define the flag for the base URL
base_url_flag = flag(
    name="base_url",
    type=str,
    help="The base URL for the OpenAI API.",
    default="https://api.openai.com/v1",
)

# Define the flag for the model name
model_name_flag = flag(
    name="model",
    type=str,
    help="The model to use for the completion.",
    default="gpt-3.5-turbo",
)

reasoning_effort_flag = flag(
    name="reasoning_effort",
    type=str,
    help="The level of reasoning effort to use (none, low, medium, high).",
    default="medium",
)


def invoke_llm(prompt: str) -> Optional[str]:
    """
    Invokes an LLM with a given prompt.

    This function requires that `experimental.flags.parse_flags()` has been called
    previously in the main execution block to populate the flag values.

    Args:
        prompt: The prompt to send to the LLM.

    Returns:
        A tuple containing:
        - The response string from the model.
        - An optional reasoning string (currently returns None).
    """
    client = openai.OpenAI(base_url=base_url_flag())

    completion = client.chat.completions.create(
        model=model_name_flag(),
        reasoning_effort=cast(Any, reasoning_effort_flag()),
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )

    response_content = completion.choices[0].message.content

    return response_content


async def invoke_llm_async(prompt: str) -> Optional[str]:
    """
    Invokes an LLM with a given prompt asynchronously.

    This function requires that `experimental.flags.parse_flags()` has been called
    previously in the main execution block to populate the flag values.

    Args:
        prompt: The prompt to send to the LLM.

    Returns:
        A tuple containing:
        - The response string from the model.
        - An optional reasoning string (currently returns None).
    """
    client = openai.AsyncOpenAI(base_url=base_url_flag())

    completion = await client.chat.completions.create(
        model=model_name_flag(),
        reasoning_effort=cast(Any, reasoning_effort_flag()),
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )

    response_content = completion.choices[0].message.content

    return response_content
