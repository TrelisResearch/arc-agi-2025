import argparse
import logging
import select
import sys
from pathlib import Path
from time import time
from typing import Optional, Dict

import debugpy
import pandas as pd
from dotenv import load_dotenv

from llm_python.datasets.collector import SoarDatasetCollector
from llm_python.datasets.io import read_soar_parquet
from llm_python.datasets.schema import ProgramSample
from llm_python.utils.api_client import ARCAPIClient
from llm_python.utils.arc_tester import ArcTester
from llm_python.utils.prompt_utils import create_rewrite_prompt, extract_python_code
from llm_python.utils.shutdown import ensure_system_exit
from llm_python.utils.task_loader import get_task_loader
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

logger = logging.getLogger(Path(__file__).name)


def invoke_llm(api_client: ARCAPIClient, system_prompt: str, user_prompt: str):
    result = api_client.call_chat_completions_api(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    if not result["success"]:
        logger.info(f"API call failed: {result['error']}")
        return None

    # print(f"Result: {result}\n\n")

    response = result["response"]
    if hasattr(response, "choices") and len(response.choices) > 0:
        message = response.choices[0].message
        content = getattr(message, "content", "") if hasattr(message, "content") else ""
        return content
    else:
        logger.info(f"No choices in API response")
        return None


def rewrite_programs(
    dataset_collector: SoarDatasetCollector,
    api_client: ARCAPIClient,
    executor: ArcTester,
    dataset: pd.DataFrame,
    attempts: int,
    max_workers: int = 8,
    limit: Optional[int] = None,
    debug: bool = False,
) -> Dict[str, int]:
    """
    Rewrite programs from the dataset to improve their style and readability.
    This function processes each program in the dataset, generates rewritten versions
    using the provided API client, and evaluates them to ensure they produce correct outputs.
    The results are collected and saved using the dataset collector.
    """
    task_loader = get_task_loader()
    
    # Limit dataset if specified
    if limit:
        dataset = dataset.head(limit)
        logger.info(f"Limited dataset to first {limit} samples")
    
    logger.info(f"Processing {len(dataset)} tasks, {attempts} attempts each = {len(dataset) * attempts} total attempts")
    
    # Statistics tracking
    from threading import Lock
    stats_lock = Lock()
    stats = {
        'total_attempts': 0,
        'original_failed': 0,
        'llm_failed': 0,
        'no_code_extracted': 0,
        'rewritten_failed': 0,
        'incorrect_outputs': 0,
        'collection_failed': 0,
        'successful': 0
    }

    def run_attempt(task_attempt_tuple):
        task_idx, attempt_num = task_attempt_tuple
        sample = dataset.iloc[task_idx]
        task_data = task_loader.get_task(sample["task_id"])
        
        with stats_lock:
            stats['total_attempts'] += 1

        # Test original program first to get expected outputs
        original_execution_result = executor.test_program(sample["code"], task_data)
        if any(original_execution_result.train_errors):
            logger.info(f"❌ FAIL: Original program execution failed with errors: {original_execution_result.train_errors}")
            with stats_lock:
                stats['original_failed'] += 1
            return

        system_prompt, user_prompt = create_rewrite_prompt(
            task_data=task_data,
            original_program=sample["code"],
        )

        if debug:
            logger.info(f"=== SYSTEM PROMPT ===\n{system_prompt}\n")
            logger.info(f"=== USER PROMPT ===\n{user_prompt}\n")
        else:
            logger.debug(f"Rewrite prompt:\nSystem: {system_prompt}\nUser: {user_prompt}")
            
        response = invoke_llm(api_client, system_prompt, user_prompt)
        if response is None:
            logger.info("❌ FAIL: No response from LLM API")
            with stats_lock:
                stats['llm_failed'] += 1
            return
        logger.info(f"Got LLM response, length: {len(response) if response else 0}")
        
        if debug:
            logger.info(f"=== LLM RESPONSE ===\n{response}\n")
        else:
            logger.debug(f"LLM response:\n{response}")
        
        rewritten_program = extract_python_code(response)
        if rewritten_program is None:
            logger.info("❌ FAIL: No valid Python code extracted from LLM response")
            with stats_lock:
                stats['no_code_extracted'] += 1
            return
            
        if debug:
            logger.info(f"=== EXTRACTED PROGRAM ===\n{rewritten_program}\n")
        else:
            logger.debug(f"Extracted rewritten program:\n{rewritten_program}")
        
        # Test rewritten program
        execution_result = executor.test_program(rewritten_program, task_data)
        if any(execution_result.train_errors):
            logger.info(f"❌ FAIL: Rewritten program execution failed with errors: {execution_result.train_errors}")
            with stats_lock:
                stats['rewritten_failed'] += 1
            return

        # Verify rewritten program produces correct outputs (matching ground truth)
        expected_train_outputs = [example["output"] for example in task_data["train"]]
        expected_test_outputs = [example["output"] for example in task_data["test"]]
        
        # Check training outputs match ground truth
        train_correct = True
        for i, (expected, actual) in enumerate(zip(expected_train_outputs, execution_result.train_outputs)):
            if actual is None or expected != actual:
                logger.info(f"❌ FAIL: Rewritten program train output {i+1} doesn't match ground truth")
                train_correct = False
                break
        
        if not train_correct:
            with stats_lock:
                stats['incorrect_outputs'] += 1
            return

        # Check test outputs match ground truth
        test_correct = True
        for i, (expected, actual) in enumerate(zip(expected_test_outputs, execution_result.test_outputs)):
            if actual is None or expected != actual:
                logger.info(f"❌ FAIL: Rewritten program test output {i+1} doesn't match ground truth")
                test_correct = False
                break
                
        if not test_correct:
            with stats_lock:
                stats['incorrect_outputs'] += 1
            return

        successfully_collected = dataset_collector.collect(
            ProgramSample(
                task_id=sample["task_id"],
                reasoning="",
                code=rewritten_program,
                correct_train_input=execution_result.correct_train_input,
                correct_test_input=execution_result.correct_test_input,
                predicted_train_output=execution_result.train_outputs,
                predicted_test_output=execution_result.test_outputs,
                model=api_client.model,
                is_transductive=False,
                row_id="",  # Will be assigned by collector
                refined_from_id=sample["row_id"],
                # Don't include compound_inspiration_id for rewritten programs
            )
        )
        if not successfully_collected:
            logger.info("❌ FAIL: Failed to collect program sample, rejected by collector.")
            with stats_lock:
                stats['collection_failed'] += 1
            return
        
        logger.info(f"✅ SUCCESS: Collected rewritten program for task {sample['task_id']}")
        with stats_lock:
            stats['successful'] += 1

    # Create list of (task_idx, attempt_num) tuples for all tasks and attempts
    task_attempts = [(task_idx, attempt_num) 
                    for task_idx in range(len(dataset))
                    for attempt_num in range(attempts)]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor_pool:
        executor_pool.map(run_attempt, task_attempts)
    
    # Print summary statistics
    logger.info("\n" + "="*50)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*50)
    logger.info(f"Total attempts: {stats['total_attempts']}")
    logger.info(f"✅ Successful: {stats['successful']} ({100*stats['successful']/max(stats['total_attempts'],1):.1f}%)")
    logger.info(f"❌ Failed breakdown:")
    logger.info(f"  - Original program failed: {stats['original_failed']}")
    logger.info(f"  - LLM API failed: {stats['llm_failed']}")
    logger.info(f"  - No code extracted: {stats['no_code_extracted']}")
    logger.info(f"  - Rewritten program failed: {stats['rewritten_failed']}")
    logger.info(f"  - Incorrect outputs: {stats['incorrect_outputs']}")
    logger.info(f"  - Collection failed: {stats['collection_failed']}")
    total_failed = stats['total_attempts'] - stats['successful']
    logger.info(f"  - Total failed: {total_failed} ({100*total_failed/max(stats['total_attempts'],1):.1f}%)")
    logger.info("="*50)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Rewrite ARC programs to improve style and readability while maintaining correctness"
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
        "--rewrite-dataset",
        type=str,
        required=True,
        help="Path to parquet file containing programs to rewrite",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        help="Number of rewrite attempts per task",
        required=True,
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit processing to first N tasks from dataset",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full prompts, responses, and extracted programs",
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

    dataset_output_folder = Path(__file__).parent / "datasets" / "rewritten"
    dataset_output_folder.mkdir(parents=True, exist_ok=True)
    dataset_collector = SoarDatasetCollector(
        args.model, output_dir=dataset_output_folder
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
    dataset = read_soar_parquet(args.rewrite_dataset)

    stats = rewrite_programs(
        dataset_collector=dataset_collector,
        api_client=api_client,
        executor=executor,
        dataset=dataset,
        attempts=args.attempts,
        max_workers=args.max_workers,
        limit=args.limit,
        debug=args.debug,
    )

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