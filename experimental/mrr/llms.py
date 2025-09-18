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

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

_async_api_client: Optional[openai.AsyncOpenAI] = None


def get_async_client() -> openai.AsyncOpenAI:
    """
    Creates and returns an asynchronous OpenAI client using the base URL from the flag.

    Returns:
        An instance of `openai.AsyncOpenAI` configured with the specified base URL.
    """
    global _async_api_client
    if _async_api_client is None:
        _async_api_client = openai.AsyncOpenAI(base_url=base_url_flag())
    return _async_api_client


_api_client: Optional[openai.OpenAI] = None


def get_client() -> openai.OpenAI:
    """
    Creates and returns a synchronous OpenAI client using the base URL from the flag.

    Returns:
        An instance of `openai.OpenAI` configured with the specified base URL.
    """
    global _api_client
    if _api_client is None:
        _api_client = openai.OpenAI(base_url=base_url_flag())
    return _api_client


def invoke_llm(prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
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
    client = get_client()

    completion = client.chat.completions.create(
        model=model_name_flag(),
        reasoning_effort=cast(Any, reasoning_effort_flag()),
        messages=[
            {
                "role": "system",
                "content": system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT,
            },
            {"role": "user", "content": prompt},
        ],
    )

    response_content = completion.choices[0].message.content

    return response_content


async def invoke_llm_async(
    prompt: str, system_prompt: Optional[str] = None
) -> Optional[str]:
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
    client = get_async_client()

    completion = await client.chat.completions.create(
        model=model_name_flag(),
        reasoning_effort=cast(Any, reasoning_effort_flag()),
        messages=[
            {
                "role": "system",
                "content": system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT,
            },
            {"role": "user", "content": prompt},
        ],
    )

    response_content = completion.choices[0].message.content

    return response_content
