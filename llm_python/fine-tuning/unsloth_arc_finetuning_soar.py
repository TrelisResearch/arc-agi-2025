#!/usr/bin/env python
"""
Generated from: unsloth_arc_finetuning_soar.ipynb
"""

# ---------------------------------------------------------------------
# Cell 1
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Config (loaded from YAML with environment variable overrides)
# ---------------------------------------------------------------------
import yaml
import os
import argparse
import sys
from pathlib import Path

def load_config_from_yaml(config_path="config.yaml"):
  """Load configuration from YAML file with environment variable overrides."""
  if not Path(config_path).exists():
      print(f"Config file {config_path} not found! Using default values.")
      return {}

  with open(config_path, 'r') as f:
      config = yaml.safe_load(f)

  print(f"✅ Loaded config from: {config_path}")
  return config

# Detect if we're running as script vs notebook and handle config path accordingly
if '__file__' in globals():
    # Script mode - use argparse for --config parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    args, unknown = parser.parse_known_args()
    config_path = args.config
else:
    # Notebook mode - use default config file
    config_path = "config.yaml"

# Load configuration
config = load_config_from_yaml(config_path)

# Extract values with fallbacks and environment variable overrides
test_run = config.get('test_run', False)
is_kaggle = config.get('is_kaggle', False)

# Override if running on Kaggle
if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None:
    is_kaggle = True

# Environment variable overrides
model_save_dir = os.environ.get('MODEL_SAVE_DIR', 
                              os.environ.get('LOCAL_MODEL_DIR',
                                            config.get('model_save_dir', "/kaggle/working/")))

execution_mode = os.environ.get('FINE_TUNING_MODE', config.get('execution_mode', 'full'))

# Data source configuration
data_config = config.get('data', {})
data_source = os.environ.get('DATA_SOURCE', data_config.get('source', 'huggingface'))
parquet_path = os.environ.get('ARC_PROGRAMS_PARQUET',
                              data_config.get('parquet', {}).get('path', '../datasets/inference/'))

# Dataset configuration
train_slug = data_config.get('dataset_slug', None)
if data_source == 'parquet' or train_slug is None:
    data_source = 'parquet'  # Default to parquet now
    train_slug = None
    print(f"📊 Data source: parquet ({parquet_path})")
else:
    print(f"📊 Data source: huggingface ({train_slug})")

# Model configuration with environment override
model_config = config.get('model', {})
model_slug = os.environ.get('MODEL_SLUG', model_config.get('slug', "Qwen/Qwen3-4B"))
model_max_length = model_config.get('max_length', 32768)
lora_rank = model_config.get('lora_rank', 128)

# Training configuration
training_config = config.get('training', {})
batch_size_global = 1 if is_kaggle else training_config.get('batch_size_global', 4)
enable_thinking = training_config.get('enable_thinking', False)

# Handle max_rows with test_run override
if test_run:
  overrides = config.get('overrides', {})
  max_rows = overrides.get('test_run_max_rows', 128)
else:
  max_rows = training_config.get('max_rows')  # None for all rows

# Print loaded configuration
print(f"Config loaded:")
print(f"  config_path: {config_path}")
print(f"  test_run: {test_run}")
print(f"  execution_mode: {execution_mode}")
print(f"  data_source: {data_source}")
print(f"  model_slug: {model_slug}")
print(f"  batch_size_global: {batch_size_global}")
print(f"  max_rows: {max_rows}")
print(f"  model_save_dir: {model_save_dir}")
if train_slug:
    print(f"  train_slug: {train_slug}")
else:
    print(f"  parquet_path: {parquet_path}")
print("-" * 50)

# ---------------------------------------------------------------------
# Cell 2
# ---------------------------------------------------------------------
# Check if env variable exists
if is_kaggle:
    report_to = "none"
else:
    report_to = ["wandb"]

print(f"Report to: {report_to}")

# ---------------------------------------------------------------------
# Cell 3
# ---------------------------------------------------------------------
import os

if not config.get("training", {}).get("multi_gpu", False):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # single GPU

# ---------------------------------------------------------------------
# Cell 4
# ---------------------------------------------------------------------
from huggingface_hub import HfFolder, login

if not is_kaggle:
    # Call this at the top of your script / notebook
    if HfFolder.get_token() is None:   # no token cached or in $HF_TOKEN
        login()                        # interactive prompt

# ---------------------------------------------------------------------
# Cell 5
# ---------------------------------------------------------------------
if not is_kaggle:
    import os
    os.environ["HF_HOME"] = "/workspace"
    os.environ["HF_HUB_CACHE"] = "/workspace/hub" # (recommended) override just the repo cache
    print(os.environ["HF_HOME"])

import unsloth
import os
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_slug,
    max_seq_length = model_max_length,   # Context length - can be longer, but uses more memory
    load_in_4bit = False,     # 4bit uses much less memory
    load_in_8bit = False,    # A bit more accurate, uses 2x memory
)

# ---------------------------------------------------------------------
# Cell 6
# ---------------------------------------------------------------------
print(f"model.max_seq_length: {model.max_seq_length}")

# ---------------------------------------------------------------------
# Cell 7
# ---------------------------------------------------------------------
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128. could consider 128.
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",
                     ],
    lora_alpha = 64,  # Best to choose alpha = rank or rank*2. EXCEPT if using rslora, in which case set it as sqrt(max matrix dimension). 64 is good for Qwen 4B
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    # use_gradient_checkpointing = False, # Hard to know if this really turns it off.
    random_state = 3407,
    use_rslora = True,   # We support rank stabilized LoRA
)

# ---------------------------------------------------------------------
# Cell 8
# ---------------------------------------------------------------------
# %%
print(f"tokenizer.padding_side: {tokenizer.padding_side}")
# %%

# ---------------------------------------------------------------------
# Cell 9
# ---------------------------------------------------------------------
# Import utils using standard project root detection
from pathlib import Path
import sys

if not is_kaggle:
    # Find project root by looking for pyproject.toml
    project_root = next(
      (parent for parent in [Path.cwd()] + list(Path.cwd().parents)
       if (parent / "pyproject.toml").exists()),
      Path.cwd()
    )
    
    # Add project root to path for consistent imports
    sys.path.insert(0, str(project_root))
    
    print(f"📁 Project root: {project_root}")

# Import from llm_python with consistent root-level imports
from llm_python.utils.task_loader import TaskLoader
from llm_python.utils.scoring import GridScorer
from llm_python.utils.arc_tester import ArcTester
from llm_python.utils.prompt_utils import create_arc_prompt, extract_python_code
from llm_python.utils.metrics_utils import calculate_task_metrics, format_metrics_display, metrics_to_percentages
from llm_python.utils.timeout_utils import execute_with_timeout
from llm_python.utils.prompt_loader import PromptLoader

# Initialize utility instances
prompt_loader = PromptLoader()
scorer = GridScorer()
print("✅ Utils imported and initialized successfully")


# ---------------------------------------------------------------------
# Cell 10
# ---------------------------------------------------------------------
import re

def clean_multiple_newlines(code: str) -> str:
    """Remove multiple consecutive newlines and replace with at most one empty line."""
    # Pattern to match multiple consecutive newlines with optional whitespace
    # This handles cases like \n\n\n, \n  \n\n, \n\t\n\n\n etc.
    pattern = r'\n(\s*\n)+'
    # Replace with at most one empty line (two newlines)
    cleaned = re.sub(pattern, '\n\n', code)
    return cleaned

def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text using the provided tokenizer."""
    return len(tokenizer.encode(text))

def should_filter_code(code: str, tokenizer, max_tokens: int = 2000) -> bool:
    """Check if code should be filtered based on token count."""
    return count_tokens(code, tokenizer) > max_tokens

print("✅ Added code cleaning and filtering functions")


# ---------------------------------------------------------------------
# Cell 11
# ---------------------------------------------------------------------
# Test cases
test_code = """def solve(grid):
  # First comment


  # Second comment after multiple empty lines
  rows = len(grid)
  cols = len(grid[0])
  
  # Another comment
    
    
  return grid"""

print("ORIGINAL CODE:")
print(repr(test_code))
print("\nORIGINAL CODE (formatted):")
print(test_code)

cleaned = clean_multiple_newlines(test_code)
print("\n" + "="*50)
print("CLEANED CODE:")
print(repr(cleaned))
print("\nCLEANED CODE (formatted):")
print(cleaned)

print("\n" + "="*50)
print("CHANGES SUMMARY:")
print(f"Original length: {len(test_code)} chars")
print(f"Cleaned length: {len(cleaned)} chars")
print(f"Characters removed: {len(test_code) - len(cleaned)}")

# ---------------------------------------------------------------------
# Cell 12
# ---------------------------------------------------------------------
from pathlib import Path
import json
from typing import Optional
from datasets import load_dataset, DatasetDict, Dataset

from llm_python.utils.task_loader import get_task_loader
from llm_python.datasets.parquet_utils import parquet_to_dataset

# ---------------------------------------------------------------------
# Prompt management using utils (replacing hard-coded prompts)
# ---------------------------------------------------------------------

# Use prompt_loader to get SOAR prompts from utils
SYSTEM_PROMPT = prompt_loader.get_system_message("soar")
INITIAL_TURN_PROMPT = prompt_loader.get_initial_turn_prompt("soar")

print(f"✅ Using SOAR prompts from utils:")
print(f"   System prompt: {len(SYSTEM_PROMPT)} chars")
print(f"   Initial turn prompt: {len(INITIAL_TURN_PROMPT)} chars")

def hf_train_dataset_to_chat_dataset(dataset_slug, split="train", max_rows=None):
  """
  Faster path:
    1) Server-side slice to avoid downloading full split.
    2) Pre-filter cheap/invalid rows BEFORE expensive prompt/tokenizer work.
    3) Map to build chat fields.
  """
  effective_split = f"{split}[:{max_rows}]" if max_rows else split
  ds_raw = load_dataset(dataset_slug, split=effective_split)

  # Create a single TaskLoader instance to reuse
  task_loader = get_task_loader()

  # ---- Pre-filter: keep only rows with valid grids and acceptable code length
  def keep_example(ex):
      # Guard: task must exist
      try:
          task_loader.get_task(ex["task_id"])
      except FileNotFoundError:
          return False

      # Guard: code length after cleaning
      cleaned = clean_multiple_newlines(ex["code"])
      if should_filter_code(cleaned, tokenizer, max_tokens=2000):
          return False

      return True

  ds_kept = ds_raw.filter(keep_example, desc=f"pre-filter ({effective_split})", load_from_cache_file=False)

  # ---- Build chat fields
  def to_chat(example):
      task_id = example["task_id"]
      task_data = task_loader.get_task(task_id)

      original_code = example["code"]
      cleaned_code = clean_multiple_newlines(original_code)
      cleaned_flag = int(cleaned_code != original_code)

      # Use predicted outputs if present; else fall back to ground-truth grids
      train_outputs = example.get(
          "predicted_train_output",
          [ex["output"] for ex in task_data["train"]],
      )
      test_outputs = example.get(
          "predicted_test_output",
          [ex["output"] for ex in task_data["test"]],
      )

      task_data_for_prompt = {
          "train": [
              {"input": ex["input"], "output": out}
              for ex, out in zip(task_data["train"], train_outputs)
          ],
          "test": [
              {"input": ex["input"], "output": out}
              for ex, out in zip(task_data["test"], test_outputs)
          ],
      }

      # Use create_arc_prompt from utils
      system_content, user_content = create_arc_prompt(task_data_for_prompt, prompt_loader, "soar")

      messages = [
          {"role": "system", "content": system_content},
          {"role": "user", "content": user_content},
          {"role": "assistant", "content": f"```python\n{cleaned_code}\n```"},
      ]

      # Apply chat template
      text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
      prompt_text = tokenizer.apply_chat_template(
          messages[:-1],
          tokenize=False,
          add_generation_prompt=True,
          enable_thinking=enable_thinking,
      )

      return {
          "messages": messages,
          "text": text,
          "prompt": prompt_text,
          "train_input": [ex["input"] for ex in task_data_for_prompt["train"]],
          "train_output": train_outputs,
          "test_input": [ex["input"] for ex in task_data_for_prompt["test"]],
          "test_output": test_outputs,
          "task_id": task_id,
          "cleaned_newlines": cleaned_flag,  # for stats
      }

  ds = ds_kept.map(to_chat, desc="build train chat fields", load_from_cache_file=False)

  # ---- Stats (robust; no reliance on closures/caching)
  total_raw = ds_raw.num_rows
  kept = ds_kept.num_rows
  retained = ds.num_rows
  cleaned_count = sum(ds["cleaned_newlines"]) if "cleaned_newlines" in ds.column_names else 0

  print(f"\n📊 Training data cleaning statistics:")
  print(f"   Total examples (raw slice): {total_raw}")
  print(f"   Removed in pre-filter: {total_raw - kept}")
  print(f"   Examples retained: {retained}")
  print(f"   Examples with cleaned newlines (retained): {cleaned_count}")

  # Optionally drop the helper stats column:
  # ds = ds.remove_columns(["cleaned_newlines"])

  return ds

def build_dataset() -> DatasetDict:
  """Build dataset from either HuggingFace or parquet source."""
  if data_source == 'parquet':
      # Load raw parquet data

      max_incorrect_per_task = data_config.get('parquet', {}).get('max_incorrect_per_task', 4)
      raw_ds = parquet_to_dataset(parquet_path, max_rows, max_incorrect_per_task)

      # Create a single TaskLoader instance to reuse
      task_loader = get_task_loader()

      # Define chat conversion function (same as in hf_train_dataset_to_chat_dataset)
      def to_chat(example):
          task_id = example["task_id"]
          task_data = task_loader.get_task(task_id)

          original_code = example["program"]  # parquet uses 'program' not 'code'
          cleaned_code = clean_multiple_newlines(original_code)
          cleaned_flag = int(cleaned_code != original_code)

          # Use ground-truth outputs from task data (parquet doesn't have predicted outputs)
          train_outputs = [ex["output"] for ex in task_data["train"]]
          test_outputs = [ex["output"] for ex in task_data["test"]]

          task_data_for_prompt = {
              "train": [
                  {"input": ex["input"], "output": out}
                  for ex, out in zip(task_data["train"], train_outputs)
              ],
              "test": [
                  {"input": ex["input"], "output": out}
                  for ex, out in zip(task_data["test"], test_outputs)
              ],
          }

          # Use create_arc_prompt from utils
          system_content, user_content = create_arc_prompt(task_data_for_prompt, prompt_loader, "soar")

          messages = [
              {"role": "system", "content": system_content},
              {"role": "user", "content": user_content},
              {"role": "assistant", "content": f"```python\n{cleaned_code}\n```"},
          ]

          # Apply chat template
          text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
          prompt_text = tokenizer.apply_chat_template(
              messages[:-1],
              tokenize=False,
              add_generation_prompt=True,
              enable_thinking=enable_thinking,
          )

          return {
              "messages": messages,
              "text": text,
              "prompt": prompt_text,
              "train_input": [ex["input"] for ex in task_data_for_prompt["train"]],
              "train_output": train_outputs,
              "test_input": [ex["input"] for ex in task_data_for_prompt["test"]],
              "test_output": test_outputs,
              "task_id": task_id,
              "cleaned_newlines": cleaned_flag,
          }

      train_ds = raw_ds.map(to_chat, desc="build parquet chat fields", load_from_cache_file=False)
  else:  # HuggingFace
      train_ds = hf_train_dataset_to_chat_dataset(train_slug, split="train", max_rows=max_rows)

  return DatasetDict(train=train_ds)

# ---------------------------------------------------------------------
# Build the dataset
# ---------------------------------------------------------------------
data = build_dataset()

# ---------------------------------------------------------------------
# Cell 13
# ---------------------------------------------------------------------
import numpy as np
from statistics import median

def length_stats(dataset, name=""):
    """
    Return min / median / max tokenised length for a 🤗 Dataset split that has a
    single 'text' column. Uses the same tokenizer already in memory.
    """
    # Tokenise in batches → list of list[int] → list[int] lengths
    lengths = dataset.map(
        lambda batch: {
            "len": [len(ids) for ids in tokenizer(batch["text"],
                                                  add_special_tokens=False
                                                 )["input_ids"]]
        },
        batched=True,
        remove_columns=dataset.column_names,   # drop 'text'
        keep_in_memory=True,
    )["len"]

    print(f"{name:>11}:  min={min(lengths):>4}  "
          f"median={int(median(lengths)):>4}  max={max(lengths):>4}")

# ── run for both splits ────────────────────────────────────────────────────────
length_stats(data["train"],       "train")

# ---------------------------------------------------------------------
# Cell 14
# ---------------------------------------------------------------------
import random

# Configuration for pre-training tests
NUM_TEST_EXAMPLES = 32  # Number of random examples to test
RANDOM_SEED = 42  # For reproducible results

def run_pre_training_data_integrity_tests(dataset_split="train", num_examples=NUM_TEST_EXAMPLES):
    """
    Test ground-truth code from dataset on random examples to validate data quality.
    
    Args:
        dataset_split: Which split to test (should be "train" since validation has no ground-truth code)
        num_examples: Number of random examples to test
    """
    print(f"🧪 Running Pre-Training Data Integrity Tests")
    print(f"📊 Testing {num_examples} random examples from {dataset_split} split")
    print("=" * 60)
    
    # Set seed for reproducible sampling
    random.seed(RANDOM_SEED)
    
    # Get the dataset split
    dataset = data[dataset_split]
    
    # Randomly sample examples
    total_examples = len(dataset)
    if num_examples > total_examples:
        print(f"⚠️  Requested {num_examples} examples but only {total_examples} available. Testing all.")
        sample_indices = list(range(total_examples))
    else:
        sample_indices = random.sample(range(total_examples), num_examples)
    
    # Initialize tracking variables
    results = []
    executor = ArcTester(timeout=5.0, executor_type="unrestricted")
    
    print(f"\n🔍 Testing {len(sample_indices)} examples...\n")
    
    for i, idx in enumerate(sample_indices):
        example = dataset[idx]
        task_id = example.get("task_id", f"idx_{idx}")
        if "code" in example:
          code = example["code"]  # HuggingFace format
        elif "program" in example:
          code = example["program"]  # Parquet format
        else:
          code = ""
        
        print(f"[{i+1}/{len(sample_indices)}] Testing {task_id}")
        
        # Initialize results for this example
        example_result = {
            "task_id": task_id,
            "index": idx,
            "code": code,
            "train_results": [],
            "test_results": [],
            "train_success": 0,
            "test_success": 0,
            "code_executed": False,
            "errors": []
        }
        
        # Test on training examples
        train_correct = 0
        for t_idx, (train_in, train_out) in enumerate(zip(example["train_input"], example["train_output"])):
            try:
                predicted_output, error, timed_out = executor.execute_program_with_timeout(code, train_in)
                
                if predicted_output is not None:
                    example_result["code_executed"] = True
                    score_result = scorer.score_grid(predicted_output, train_out)
                    is_correct = score_result["correct"]
                    
                    if is_correct:
                        train_correct += 1
                    
                    example_result["train_results"].append({
                        "index": t_idx,
                        "correct": is_correct,
                        "predicted": predicted_output,
                        "expected": train_out,
                        "timed_out": timed_out
                    })
                else:
                    example_result["train_results"].append({
                        "index": t_idx,
                        "correct": False,
                        "error": error,
                        "timed_out": timed_out
                    })
                    if error:
                        example_result["errors"].append(f"Train {t_idx}: {error}")
                        
            except Exception as e:
                example_result["train_results"].append({
                    "index": t_idx,
                    "correct": False,
                    "error": str(e)
                })
                example_result["errors"].append(f"Train {t_idx}: {str(e)}")
        
        # Test on test examples
        test_correct = 0
        for t_idx, (test_in, test_out) in enumerate(zip(example["test_input"], example["test_output"])):
            try:
                predicted_output, error, timed_out = executor.execute_program_with_timeout(code, test_in)
                
                if predicted_output is not None:
                    example_result["code_executed"] = True
                    score_result = scorer.score_grid(predicted_output, test_out)
                    is_correct = score_result["correct"]
                    
                    if is_correct:
                        test_correct += 1
                    
                    example_result["test_results"].append({
                        "index": t_idx,
                        "correct": is_correct,
                        "predicted": predicted_output,
                        "expected": test_out,
                        "timed_out": timed_out
                    })
                else:
                    example_result["test_results"].append({
                        "index": t_idx,
                        "correct": False,
                        "error": error,
                        "timed_out": timed_out
                    })
                    if error:
                        example_result["errors"].append(f"Test {t_idx}: {error}")
                        
            except Exception as e:
                example_result["test_results"].append({
                    "index": t_idx,
                    "correct": False,
                    "error": str(e)
                })
                example_result["errors"].append(f"Test {t_idx}: {str(e)}")
        
        # Calculate success rates for this example
        example_result["train_success"] = train_correct / len(example["train_input"]) if example["train_input"] else 0
        example_result["test_success"] = test_correct / len(example["test_input"]) if example["test_input"] else 0
        
        # Print summary for this example
        total_train = len(example["train_input"])
        total_test = len(example["test_input"])
        
        print(f"  ✅ Train: {train_correct}/{total_train} ({example_result['train_success']:.1%})")
        print(f"  ✅ Test:  {test_correct}/{total_test} ({example_result['test_success']:.1%})")
        
        if example_result["errors"]:
            print(f"  ❌ Errors: {len(example_result['errors'])}")
        if not example_result["code_executed"]:
            print(f"  ⚠️  Code never executed successfully")
        print()
        
        results.append(example_result)
    
    return results

# Run the tests
data_integrity_results = run_pre_training_data_integrity_tests("train", NUM_TEST_EXAMPLES)

# ---------------------------------------------------------------------
# Cell 15
# ---------------------------------------------------------------------
def analyze_data_integrity_results(results):
    """
    Analyze and display comprehensive statistics from the data integrity tests.
    """
    print("=" * 60)
    print("📈 PRE-TRAINING DATA INTEGRITY RESULTS ANALYSIS")
    print("=" * 60)
    
    if not results:
        print("❌ No results to analyze!")
        return
    
    # Overall statistics
    total_examples = len(results)
    examples_with_executable_code = sum(1 for r in results if r["code_executed"])
    examples_with_errors = sum(1 for r in results if r["errors"])
    
    # Training performance statistics
    train_success_rates = [r["train_success"] for r in results]
    perfect_train = sum(1 for rate in train_success_rates if rate == 1.0)
    partial_train = sum(1 for rate in train_success_rates if 0 < rate < 1.0)
    failed_train = sum(1 for rate in train_success_rates if rate == 0.0)
    
    # Test performance statistics  
    test_success_rates = [r["test_success"] for r in results]
    perfect_test = sum(1 for rate in test_success_rates if rate == 1.0)
    partial_test = sum(1 for rate in test_success_rates if 0 < rate < 1.0)
    failed_test = sum(1 for rate in test_success_rates if rate == 0.0)
    
    # Calculate overall metrics
    avg_train_success = sum(train_success_rates) / len(train_success_rates) if train_success_rates else 0
    avg_test_success = sum(test_success_rates) / len(test_success_rates) if test_success_rates else 0
    
    # Count total grids tested
    total_train_grids = sum(len(r["train_results"]) for r in results)
    total_test_grids = sum(len(r["test_results"]) for r in results)
    correct_train_grids = sum(sum(tr["correct"] for tr in r["train_results"]) for r in results)
    correct_test_grids = sum(sum(tr["correct"] for tr in r["test_results"]) for r in results)
    
    print(f"\n🎯 OVERALL PERFORMANCE:")
    print(f"   Examples tested: {total_examples}")
    print(f"   Code executable: {examples_with_executable_code}/{total_examples} ({examples_with_executable_code/total_examples:.1%})")
    print(f"   Examples with errors: {examples_with_errors}/{total_examples} ({examples_with_errors/total_examples:.1%})")
    
    print(f"\n📊 TRAINING GRIDS PERFORMANCE:")
    print(f"   Average success rate: {avg_train_success:.1%}")
    print(f"   Perfect examples (100%): {perfect_train}/{total_examples} ({perfect_train/total_examples:.1%})")
    print(f"   Partial examples (>0% <100%): {partial_train}/{total_examples} ({partial_train/total_examples:.1%})")
    print(f"   Failed examples (0%): {failed_train}/{total_examples} ({failed_train/total_examples:.1%})")
    print(f"   Grid-level accuracy: {correct_train_grids}/{total_train_grids} ({correct_train_grids/total_train_grids:.1%})")
    
    print(f"\n🎯 TEST GRIDS PERFORMANCE:")
    print(f"   Average success rate: {avg_test_success:.1%}")
    print(f"   Perfect examples (100%): {perfect_test}/{total_examples} ({perfect_test/total_examples:.1%})")
    print(f"   Partial examples (>0% <100%): {partial_test}/{total_examples} ({partial_test/total_examples:.1%})")
    print(f"   Failed examples (0%): {failed_test}/{total_examples} ({failed_test/total_examples:.1%})")
    print(f"   Grid-level accuracy: {correct_test_grids}/{total_test_grids} ({correct_test_grids/total_test_grids:.1%})")
    
    # Detailed breakdown by example
    print(f"\n📋 DETAILED BREAKDOWN BY EXAMPLE:")
    print("-" * 60)
    
    for i, result in enumerate(results):
        task_id = result["task_id"]
        train_rate = result["train_success"]
        test_rate = result["test_success"]
        executed = "✅" if result["code_executed"] else "❌"
        error_count = len(result["errors"])
        
        print(f"[{i+1:2d}] {task_id}")
        print(f"     Train: {train_rate:5.1%} | Test: {test_rate:5.1%} | Executed: {executed} | Errors: {error_count}")
        
        if result["errors"] and len(result["errors"]) <= 3:  # Show first few errors
            for error in result["errors"][:3]:
                print(f"     Error: {error}")
        elif len(result["errors"]) > 3:
            print(f"     Errors: {result['errors'][0]} ... (+{len(result['errors'])-1} more)")
    
    # Quality assessment
    print(f"\n🔍 DATASET QUALITY ASSESSMENT:")
    print("-" * 60)
    
    if avg_train_success > 0.9:
        print("✅ EXCELLENT: Ground-truth code performs very well on training examples")
    elif avg_train_success > 0.7:
        print("✅ GOOD: Ground-truth code performs well on training examples")
    elif avg_train_success > 0.5:
        print("⚠️  MODERATE: Ground-truth code has mixed performance on training examples")
    else:
        print("❌ POOR: Ground-truth code has low performance on training examples")
    
    if avg_test_success > 0.9:
        print("✅ EXCELLENT: Ground-truth code generalizes very well to test examples")
    elif avg_test_success > 0.7:
        print("✅ GOOD: Ground-truth code generalizes well to test examples")
    elif avg_test_success > 0.5:
        print("⚠️  MODERATE: Ground-truth code has mixed generalization to test examples")
    else:
        print("❌ POOR: Ground-truth code has poor generalization to test examples")
    
    if examples_with_executable_code == total_examples:
        print("✅ EXCELLENT: All ground-truth code is executable")
    elif examples_with_executable_code / total_examples > 0.9:
        print("✅ GOOD: Most ground-truth code is executable")
    else:
        print("⚠️  ISSUE: Some ground-truth code is not executable")
    
    print("\n" + "=" * 60)
    
    return {
        "total_examples": total_examples,
        "executable_rate": examples_with_executable_code / total_examples,
        "avg_train_success": avg_train_success,
        "avg_test_success": avg_test_success,
        "perfect_train_rate": perfect_train / total_examples,
        "perfect_test_rate": perfect_test / total_examples,
        "train_grid_accuracy": correct_train_grids / total_train_grids if total_train_grids > 0 else 0,
        "test_grid_accuracy": correct_test_grids / total_test_grids if total_test_grids > 0 else 0
    }

# Analyze the results
summary_stats = analyze_data_integrity_results(data_integrity_results)


# ---------------------------------------------------------------------
# Cell 16
# ---------------------------------------------------------------------
def examine_failure(results, example_index):
  """Examine a specific failing example in detail with grid visualization."""
  if example_index >= len(results):
      print(f"❌ Invalid index {example_index}. Only {len(results)} examples available.")
      return

  result = results[example_index]
  dataset_example = data["train"][result['index']]  # Get the original dataset example

  print(f"\n🔍 DETAILED EXAMINATION: Example {example_index + 1}")
  print(f"Task ID: {result['task_id']}")
  print(f"Dataset Index: {result['index']}")
  print("=" * 70)

  print(f"\n📝 GROUND TRUTH CODE:")
  print("-" * 30)
  print(result['code'])

  print(f"\n📊 EXECUTION SUMMARY:")
  print(f"Code executed successfully: {result['code_executed']}")
  print(f"Train success rate: {result['train_success']:.1%}")
  print(f"Test success rate: {result['test_success']:.1%}")
  print(f"Number of errors: {len(result['errors'])}")

  if result['errors']:
      print(f"\n❌ ERRORS:")
      for i, error in enumerate(result['errors']):
          print(f"  {i+1}. {error}")

  # Load original task for ground truth comparison
  task_loader = get_task_loader()
  try:
      original_task = task_loader.get_task(result['task_id'])
  except Exception as e:
      print(f"❌ Could not load original task: {e}")
      return

  def print_grid(grid, title):
      """Helper to print a grid nicely."""
      print(f"\n{title}:")
      if grid is None:
          print("  None")
          return
      for row in grid:
          print("  " + " ".join(f"{cell:2d}" for cell in row))

  # Examine training examples
  print(f"\n🏋️ TRAINING EXAMPLES:")
  print("=" * 50)

  for i, train_result in enumerate(result['train_results']):
      print(f"\nTrain Example {i + 1}: {'✅ CORRECT' if train_result['correct'] else '❌ INCORRECT'}")
      print("-" * 40)

      # Input (should be same from dataset and original)
      dataset_input = dataset_example["train_input"][i]
      original_input = original_task["train"][i]["input"]
      print_grid(dataset_input, "Input (from dataset)")
      if dataset_input != original_input:
          print_grid(original_input, "Input (from original) - MISMATCH!")

      # Expected output (from dataset - might be predicted)
      dataset_expected = dataset_example["train_output"][i]
      original_expected = original_task["train"][i]["output"]
      print_grid(dataset_expected, "Expected (from dataset)")
      if dataset_expected != original_expected:
          print_grid(original_expected, "Expected (ground truth) - DIFFERENT!")

      # Predicted output (from code execution)
      if 'predicted' in train_result:
          print_grid(train_result['predicted'], "Predicted (from code)")

      if 'error' in train_result:
          print(f"\n❌ Execution Error: {train_result['error']}")

  # Examine test examples
  print(f"\n🧪 TEST EXAMPLES:")
  print("=" * 50)

  for i, test_result in enumerate(result['test_results']):
      print(f"\nTest Example {i + 1}: {'✅ CORRECT' if test_result['correct'] else '❌ INCORRECT'}")
      print("-" * 40)

      # Input
      dataset_input = dataset_example["test_input"][i]
      original_input = original_task["test"][i]["input"]
      print_grid(dataset_input, "Input (from dataset)")
      if dataset_input != original_input:
          print_grid(original_input, "Input (from original) - MISMATCH!")

      # Expected output
      dataset_expected = dataset_example["test_output"][i]
      original_expected = original_task["test"][i].get("output")  # Might be None
      print_grid(dataset_expected, "Expected (from dataset)")
      if original_expected and dataset_expected != original_expected:
          print_grid(original_expected, "Expected (ground truth) - DIFFERENT!")
      elif not original_expected:
          print("Expected (ground truth): No ground truth available")

      # Predicted output
      if 'predicted' in test_result:
          print_grid(test_result['predicted'], "Predicted (from code)")

      if 'error' in test_result:
          print(f"\n❌ Execution Error: {test_result['error']}")

# Check for failing examples
failed_examples = [i for i, r in enumerate(data_integrity_results)
                if r['train_success'] < 1.0 or r['test_success'] < 1.0 or not r['code_executed']]

print(f"\n🔍 FAILING EXAMPLES SUMMARY:")
if failed_examples:
  print(f"Found {len(failed_examples)} examples with issues: {failed_examples}")
  print("To examine a specific failure, run: examine_failure(data_integrity_results, index)")
  print("Example: examine_failure(data_integrity_results, 0)")
else:
  print("🎉 No failing examples found! All ground-truth code works perfectly.")

print(f"\n✅ Pre-training data integrity tests complete!")
print(f"📋 Summary stats saved in 'summary_stats' variable")
print(f"📊 Detailed results saved in 'data_integrity_results' variable")

# ---------------------------------------------------------------------
# Cell 17
# ---------------------------------------------------------------------
examine_failure(data_integrity_results, 21)

# ---------------------------------------------------------------------
# Cell 18
# ---------------------------------------------------------------------
# Determine dataset name for run naming
if data_source == 'parquet':
    dataset_name = "parquet-programs"
    print(f"Using parquet data source: {dataset_name}")
else:
    # Use HuggingFace dataset name
    if train_slug is None:
        from datetime import datetime
        dataset_name = f"local_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        dataset_name = train_slug.split('/')[-1]  # Get name part after last slash
    print(f"Using HuggingFace dataset: {dataset_name}")

run_name = f"{model_slug.split('/')[-1]}_ds-{dataset_name}"

if test_run:
    run_name = run_name + "_test"   # or "-test"

    
print(f"Run name will be {run_name}")

# ---------------------------------------------------------------------
# Cell 19
# ---------------------------------------------------------------------
import torch, subprocess, os, gc, time

def _print_gpu(prefix=""):
    alloc = torch.cuda.memory_allocated() / 2**20  # MiB
    reserved = torch.cuda.memory_reserved() / 2**20
    print(f"{prefix}CUDA‑alloc={alloc:.0f} MiB | reserved={reserved:.0f} MiB")

def _nvidia_smi():
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free",
             "--format=csv,noheader,nounits"]).decode().strip()
        print("nvidia-smi (used/free MiB):", smi)
    except Exception:
        pass  # nvidia-smi not always available


TEMPLATES = {
    "llama": (
        "<|start_header_id|>user<|end_header_id|>\n\n",
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
    ),
    "gemma": (
        "<start_of_turn>user\n",
        "<start_of_turn>model\n",
    ),
    "qwen-coder": (
        "<|im_start|>user\n",
        "<|im_start|>assistant\n", # this is actually how you properly allow the model to keep reasoning!
    ),
    "qwen": (
        "<|im_start|>user\n",
        "<|im_start|>assistant\n<think>\n\n</think>\n\n", # this is actually how you properly allow the model to keep reasoning!
    ),
    "mistral": (
        "[INST]",
        "[/INST]",
    )
}

# instruction_tag, response_tag = TEMPLATES["qwen-coder"]   # ← change if needed and comment out below

model_slug_lower = model_slug.lower()

if "qwen" in model_slug_lower:
    if "coder" in model_slug_lower:
        instruction_tag, response_tag = TEMPLATES["qwen-coder"]
    elif "soar-qwen" in model_slug_lower:
        instruction_tag, response_tag = TEMPLATES["qwen-coder"]
    else:
        instruction_tag, response_tag = TEMPLATES["qwen"]
else:
    raise ValueError(f"Unsupported model slug for Qwen template: {model_slug}")

# ---------------------------------------------------------------------
# Cell 20
# ---------------------------------------------------------------------
print(f"Response tag selected: {response_tag}")

# ---------------------------------------------------------------------
# Cell 21
# ---------------------------------------------------------------------
if not is_kaggle:
    import wandb

    # 1. Log in (will prompt for your API key in notebook)
    wandb.login()

    # 2. Initialize a run and set the project
    wandb.init(
        project="arc-agi-2025",   # set your project name
        entity="trelis",  # optional: your W&B username or team
        name=run_name  # optional: custom run name
    )


# ---------------------------------------------------------------------
# Cell 22
# ---------------------------------------------------------------------
from trl import SFTTrainer, SFTConfig
import math
from torch.optim.lr_scheduler import LambdaLR

setattr(model, "_flag_for_generation", True)

if config.get("training", {}).get("multi_gpu", False):
    num_gpus = max(1, torch.cuda.device_count())
else:
    num_gpus = 1

target_batch_size = 32  # already int
per_device_batch_size = int(batch_size_global)  # ensure int from config
grad_accum = int(target_batch_size // (per_device_batch_size * num_gpus))  # ensure int division result
print(f"GPUs used: {num_gpus}, per_device: {per_device_batch_size}, grad_accum: {grad_accum}")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=data["train"],
    args=SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=2,
        learning_rate=1e-4,
        logging_strategy="steps",
        logging_steps=0.0125,         # keep as FRACTION of an epoch
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="constant", # ignored after we inject
        seed=3407,
        report_to=report_to,
        logging_dir=f"./logs/{run_name}",
        remove_unused_columns=True,
        save_strategy="steps",
        save_steps=0.5,              # keep as FRACTION of an epoch
        save_total_limit=4,
        prediction_loss_only=False,
        hub_model_id=f"Trelis/{run_name}-trainer",  # ← this sets the repo to push to
        hub_strategy="all_checkpoints",         # when to push (end, every_save, checkpoint, all_checkpoints)
        hub_private_repo=True,             # optional: make it private
        push_to_hub=not is_kaggle
    )
)

# --- derive counts (without mutating args) ---
train_dl = trainer.get_train_dataloader()
ga = trainer.args.gradient_accumulation_steps
updates_per_epoch = max(1, math.ceil(len(train_dl) / ga))
total_updates = updates_per_epoch * trainer.args.num_train_epochs

def _effective_interval(val):
    """Use fractions (<1) as fraction-of-epoch; ints as-is. Do NOT mutate args."""
    if isinstance(val, float) and 0.0 < val < 1.0:
        return max(1, int(round(val * total_updates)))

# Internal interval used ONLY for LR dips (trainer keeps fractions)
effective_save_interval = _effective_interval(trainer.args.save_steps)

# Build save marks in optimizer-step indices
save_marks = list(range(effective_save_interval, total_updates + 1, effective_save_interval))

# 10% of ONE epoch for warmup/dip windows
window = max(1, int(round(0.1 * updates_per_epoch)))
min_frac = 0.1

def lr_multiplier(step_idx: int) -> float:
    # initial warmup
    if step_idx < window:
        return (step_idx + 1) / float(window)
    # dip before save, recover after
    for s in save_marks:
        if (s - window) <= step_idx < s:      # down-ramp
            pos = step_idx - (s - window)
            return 1.0 - (1.0 - min_frac) * ((pos + 1) / float(window))
        if s <= step_idx < (s + window):      # up-ramp
            pos = step_idx - s
            return min_frac + (1.0 - min_frac) * ((pos + 1) / float(window))
    return 1.0

# Inject optimizer & custom scheduler (Unsloth-safe)
trainer.create_optimizer()
optimizer = trainer.optimizer
scheduler = LambdaLR(optimizer, lr_lambda=lambda step: lr_multiplier(step))

trainer.optimizer = optimizer
trainer.lr_scheduler = scheduler
trainer.create_optimizer = lambda *a, **k: trainer.optimizer
trainer.create_scheduler = lambda *a, **k: trainer.lr_scheduler

print(f"[setup] updates/epoch={updates_per_epoch} total_updates={total_updates} "
      f"save_steps(raw)={trainer.args.save_steps} "
      f"effective_save_interval(steps)={effective_save_interval} "
      f"output_dir={trainer.args.output_dir}")

# ---------------------------------------------------------------------
# Cell 23
# ---------------------------------------------------------------------
# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# ---------------------------------------------------------------------
# Cell 24
# ---------------------------------------------------------------------
from unsloth.chat_templates import train_on_responses_only # or run the code above if not using unsloth

# TO SUPPORT REASONING, WE NEED TO DYNAMICALLY APPLY THE RIGHT MASKING, NOT YET IMPLEMENTED
# masks everything between the instruction_part and response_part
trainer = train_on_responses_only(
    trainer,
    instruction_part = instruction_tag,
    response_part = response_tag,
    # force_match=False # comment out to set true for a cleaner masking
)

# ---------------------------------------------------------------------
# Cell 25
# ---------------------------------------------------------------------
print(tokenizer.decode(trainer.train_dataset[0]["input_ids"]))

# ---------------------------------------------------------------------
# Cell 26
# ---------------------------------------------------------------------
print(tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[0]["labels"]]).replace(tokenizer.pad_token, " "))

# ---------------------------------------------------------------------
# Cell 27
# ---------------------------------------------------------------------
trainer_stats = trainer.train()

# ---------------------------------------------------------------------
# Cell 28
# ---------------------------------------------------------------------
# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# ---------------------------------------------------------------------
# Cell 29
# ---------------------------------------------------------------------
print(trainer_stats)

# ---------------------------------------------------------------------
# Cell 30
# ---------------------------------------------------------------------
if execution_mode == "full":
    print(f"Current working directory: {os.getcwd()}")
    print(f"Trainer output dir: {trainer.args.output_dir}")
    print(f"Checkpoints exist: {os.listdir(trainer.args.output_dir)}")
    trainer.push_to_hub()

# ---------------------------------------------------------------------
# Cell 31
# ---------------------------------------------------------------------
# Checkpoint processing based on execution mode
import os, re, torch
from unsloth import FastLanguageModel

ROOT = "trainer_output"
RUN_NAME = run_name

if execution_mode == "final_only":
    print("🔧 Final-only mode: Only processing the last checkpoint")
    
    # Find all checkpoints and get the latest one
    ckpts = []
    for d in os.listdir(ROOT):
        m = re.fullmatch(r"checkpoint-(\d+)", d)
        if m and os.path.isdir(os.path.join(ROOT, d)):
            ckpts.append((int(m.group(1)), os.path.join(ROOT, d)))
    
    if not ckpts:
        print("❌ No checkpoints found!")
    else:
        # Sort and get the last checkpoint
        ckpts.sort(key=lambda x: x[0])
        final_step, final_path = ckpts[-1]
        
        print(f"📦 Processing final checkpoint: {final_step}")
        
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=final_path,
                load_in_4bit=False,
            )
            
            # Merge and save locally
            merged_model = model.merge_and_unload()
            final_model_path = os.path.join(model_save_dir, f"{RUN_NAME}-final")
            
            print(f"💾 Saving final model to: {final_model_path}")
            merged_model.save_pretrained(final_model_path)
            tokenizer.save_pretrained(final_model_path)
            
            print(f"✅ Final model saved successfully")
            
            # Clean up
            del merged_model, model, tokenizer
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
                
        except Exception as e:
            print(f"❌ Error processing final checkpoint: {e}")

else:
    print("🔄 Full mode: Processing all checkpoints")
    
    # Original behavior - process all checkpoints
    ckpts = []
    for d in os.listdir(ROOT):
        m = re.fullmatch(r"checkpoint-(\d+)", d)
        if m and os.path.isdir(os.path.join(ROOT, d)):
            ckpts.append((int(m.group(1)), os.path.join(ROOT, d)))
    ckpts.sort(key=lambda x: x[0])  # ascending; use reverse=True for newest first

    print(f"Found {len(ckpts)} checkpoints:", [s for s, _ in ckpts])

    for step, path in ckpts:
        try:
            print(f"\n=== STEP {step} === Loading {path}")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = path,
                load_in_4bit = False,
                # device_map = "auto",   # uncomment if you have GPU available
            )
            repo_id = f"Trelis/{RUN_NAME}-c{step}"
            
            if is_kaggle:
                # Save locally when running locally
                local_path = os.path.join(model_save_dir, f"{RUN_NAME}-c{step}")
                print(f"Saving locally at {local_path}")
                merged_model = model.merge_and_unload()
                merged_model.save_pretrained(local_path)
                tokenizer.save_pretrained(local_path)
                del merged_model
            else:
                print(f"Pushing to {repo_id} …")
                # If you trained with LoRA, keep merge_and_unload(); if full-finetune, drop this line.
                model = model.merge_and_unload()
                model.push_to_hub(repo_id)
                tokenizer.push_to_hub(repo_id)

            # tidy up between checkpoints
            del model
            del tokenizer
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        except Exception as e:
            print(f"[WARN] Skipping checkpoint {step}: {e}")
            
print(f"\n✅ Checkpoint processing complete for {execution_mode} mode")
