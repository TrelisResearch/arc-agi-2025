import argparse
import logging
import select
import sys
from pathlib import Path
from time import time
from typing import Optional

import debugpy
import pandas as pd
from dotenv import load_dotenv

from llm_python.datasets.collector import SoarDatasetCollector
from llm_python.datasets.io import read_soar_parquet
from llm_python.datasets.schema import ProgramSample
from llm_python.utils.api_client import ARCAPIClient
from llm_python.utils.arc_tester import ArcTester
from llm_python.utils.prompt_utils import create_compound_prompt, extract_python_code
from llm_python.utils.shutdown import ensure_system_exit
from llm_python.utils.task_loader import get_task_loader
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

logger = logging.getLogger(Path(__file__).name)


def invoke_llm(api_client: ARCAPIClient, system_prompt: str, user_prompt: str):
    response: Optional[str] = None
    result = api_client.call_chat_completions_api(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    if not result["success"]:
        logger.debug(f"API call failed: {result['error']}")

    response = result["response"]
    if hasattr(response, "choices") and len(response.choices) > 0:
        message = response.choices[0].message
        return getattr(message, "content", "") if hasattr(message, "content") else ""


def generate_compound_programs(
    dataset_collector: SoarDatasetCollector,
    api_client: ARCAPIClient,
    executor: ArcTester,
    dataset: pd.DataFrame,
    attempts: int,
    max_workers: int = 8,
) -> None:
    """
    Generate compound programs by refining existing ones in the dataset.
    This function processes each program in the dataset, generates refined versions
    using the provided API client, and evaluates them using the executor.
    The results are collected and saved using the dataset collector.
    """
    task_loader = get_task_loader()

    def run_attempt(i: int):
        logger.info(f"Starting attempt {i + 1} of {attempts}")
        sample = dataset.sample(n=1, random_state=int(time())).iloc[0]
        task_data = task_loader.get_task(sample["task_id"])
        inspiration = (
            dataset[dataset["task_id"] != sample["task_id"]]
            .sample(n=1, random_state=int(time()))
            .iloc[0]
        )

        system_prompt, user_prompt = create_compound_prompt(
            task_data=task_loader.get_task(sample["task_id"]),
            original_program=sample["code"],
            reference_program=inspiration["code"],
        )

        logger.debug(f"Compound prompt:\nSystem: {system_prompt}\nUser: {user_prompt}")
        response = invoke_llm(api_client, system_prompt, user_prompt)
        if response is None:
            logger.info("No response from LLM API")
            return
        logger.debug(f"LLM response:\n{response}")
        program = extract_python_code(response)
        if program is None:
            logger.info("No valid Python code extracted from LLM response")
            return
        logger.debug(f"Extracted program:\n{program}")
        execution_result = executor.test_program(program, task_data)
        if any(execution_result.train_errors):
            logger.info(
                f"Program execution failed with errors: {execution_result.train_errors}"
            )
            return

        succesfully_collected = dataset_collector.collect(
            ProgramSample(
                task_id=sample["task_id"],
                reasoning="",
                code=program,
                correct_train_input=execution_result.correct_train_input,
                correct_test_input=execution_result.correct_test_input,
                predicted_train_output=execution_result.train_outputs,
                predicted_test_output=execution_result.test_outputs,
                model=api_client.model,
                is_transductive=False,
                row_id="",  # Will be assigned by collector
                refined_from_id=sample["row_id"],
                compound_inspiration_id=inspiration["row_id"],
            )
        )
        if not succesfully_collected:
            logger.info("Failed to collect program sample, rejected by collector.")
            return
        logger.info(
            f"Successfully collected refined program for task {sample['task_id']}"
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor_pool:
        executor_pool.map(run_attempt, range(attempts))


def main():
    parser = argparse.ArgumentParser(
        description="Run ARC tasks with all-attempts, rolling execution, and voting-based evaluation"
    )
    parser.add_argument(
        "--dataset",
        default="arc-prize-2025",
        help="Dataset to use (e.g., arc-prize-2025, arc-agi-1, arc-agi-2)",
    )
    parser.add_argument(
        "--subset",
        default="training",
        help="Dataset subset to use. Supports: (1) Traditional subsets like 'training', 'evaluation' (2) HuggingFace datasets like 'username/dataset-name' (3) Parquet files/directories like '/path/to/data.parquet'",
    )
    parser.add_argument("--model", default="gpt-4.1-mini", help="Model to use")
    parser.add_argument(
        "--base-url", type=str, help="Base URL for OpenAI-compatible API endpoint"
    )
    parser.add_argument(
        "--max_workers", type=int, default=1, help="Maximum number of parallel workers"
    )
    parser.add_argument(
        "--max-tokens", type=int, help="Maximum tokens for model responses"
    )
    parser.add_argument(
        "--temperature", type=float, help="Temperature for model responses"
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="low",
        help="Reasoning effort for OpenAI models",
    )
    parser.add_argument(
        "--qwen-no-think",
        action="store_true",
        help="Disable thinking for Qwen models (Note: Not supported by DashScope commercial models)",
    )
    parser.add_argument(
        "--compound-dataset",
        type=str,
        required=True,
        help="Compound dataset: HuggingFace dataset or parquet file containing draft programs to refine. Uses programs with at least one (but not all) correct training examples. Enables refinement prompts with draft code.",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        help="Number of attempts to generate a compound program",
        required=True,
    )
    parser.add_argument(
        "--remote-debug",
        action="store_true",
        help="Wait 30s for a remote debugger to attach",
    )
    parser.add_argument(
        "--log-level",
        type=lambda s: s.upper(),
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    if args.remote_debug:
        print("Debug mode: waiting 30 seconds for debugger to attach...")
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for remote debugger to attach on port 5678...")
        print("Press Enter to continue immediately, or wait 30 seconds...")
        rlist, _, _ = select.select([sys.stdin], [], [], 30)
        if rlist:
            sys.stdin.readline()
        else:
            print("Continuing after 30 seconds...")

    dataset_output_folder = Path(__file__).parent / "datasets" / "compounded"
    dataset_output_folder.mkdir(parents=True, exist_ok=True)
    dataset_collector = SoarDatasetCollector(
        args.model, output_dir=dataset_output_folder, flush_every=50
    )
    api_client = ARCAPIClient(
        model=args.model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        qwen_no_think=args.qwen_no_think,
        api_timeout=600,  # API timeout enforced by OpenAI client
    )

    executor = ArcTester()
    dataset = read_soar_parquet(args.compound_dataset)

    try:
        generate_compound_programs(
            dataset_collector=dataset_collector,
            api_client=api_client,
            executor=executor,
            dataset=dataset,
            attempts=args.attempts,
            max_workers=args.max_workers,
        )

    except KeyboardInterrupt:
        print("Interrupted by user, shutting down...")

    dataset_collector.flush()


if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except Exception as e:
        print(f"An unexpected error occurred in main: {e}", file=sys.stderr)
        exit_code = 1
    finally:
        ensure_system_exit(exit_code)
