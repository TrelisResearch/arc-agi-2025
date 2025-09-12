from dotenv import load_dotenv
from experimental.flags import parse_flags
from experimental.mrr.llms import invoke_llm

load_dotenv()

def main():
    """
    Example of how to use the invoke_llm function.

    To run this example:
    uv run python -m experimental.mrr.test_llms --model <your-model> --base_url <your-base-url>
    """
    parse_flags()
    prompt = "What is the capital of France?"
    response, reasoning = invoke_llm(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    if reasoning:
        print(f"Reasoning: {reasoning}")


if __name__ == "__main__":
    main()
