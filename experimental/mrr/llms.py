import openai
from experimental.flags import flag
from typing import Optional, Tuple

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


def invoke_llm(prompt: str) -> Tuple[Optional[str], Optional[str]]:
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

    print(model_name_flag())

    completion = client.chat.completions.create(
        model=model_name_flag(),
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )

    response_content = completion.choices[0].message.content
    reasoning = None  # Placeholder for optional reasoning string

    return response_content, reasoning


async def invoke_llm_async(prompt: str) -> Tuple[Optional[str], Optional[str]]:
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

    print(model_name_flag())

    completion = await client.chat.completions.create(
        model=model_name_flag(),
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )

    response_content = completion.choices[0].message.content
    reasoning = None  # Placeholder for optional reasoning string

    return response_content, reasoning
