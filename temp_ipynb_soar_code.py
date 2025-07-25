from pathlib import Path
import json
from typing import Optional
from datasets import load_dataset, DatasetDict

# ---------------------------------------------------------------------
# Config (examples)
# ---------------------------------------------------------------------

# Set max_rows flag to limit train size. None for all
max_rows = None

# CASE 1: single slug with both splits
train_slug = "Trelis/synth_arc-agi-1_all_training_20250724_131808"
val_slug = None

# # CASE 2: two different slugs
# train_slug = "Trelis/synth_arc-agi-1_shortest_training_10_20250724_091954"
# val_slug   = "Trelis/synth_arc-agi-1_shortest_training_10_20250724_091954"

include_reasoning = False  # See note in original code

# SOAR-style prompts (self-contained)
SYSTEM_PROMPT = "You are an AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks by reasoning and generating Python code."

INITIAL_TURN_PROMPT = """You are an AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks by generating Python code.
Your goal is to analyze input-output grid pairs. The outputs were produced by applying a transformation rule to the inputs. Implement the transformation rules as a Python function.
You should only write the implemented the transformation in code.
You must write code in triple backticks (```python and then ```). You must write a function called 'transform' which takes a single argument, the input grid as 'list[list[int]]', and returns the transformed grid (also as 'list[list[int]]').
You should make sure that you implement a version of the transformation which works in general (at least for all given input-output pairs and test input pairs).
The number in the input grid can be mapped to the following colors: 0:Black; 1:Blue; 2:Red; 3:Green; 4:Yellow; 5:Grey; 6:Pink; 7:Orange; 8:Purple; 9:Brown
Now, solve the following ARC-AGI task:
# Task to solve:
{task_content}"""

def format_grid(grid):
    """Format a grid as a string using the SOAR approach (with outer brackets and no commas)"""
    return str(grid).replace('[', '[[').replace(']', ']]').replace(',', '')

def format_task_content(train_input, train_output, test_input):
    """Format task content using the SOAR approach"""
    task_content = ""
    
    # Add training examples
    for i, (inp, out) in enumerate(zip(train_input, train_output), 1):
        # Get grid shapes
        input_shape = f"{len(inp[0])} by {len(inp)}"
        output_shape = f"{len(out[0])} by {len(out)}"
        
        # Format grids using SOAR approach
        input_str = format_grid(inp)
        output_str = format_grid(out)
        
        task_content += f"## Input {i} (grid shape: {input_shape}):\n{input_str}\n"
        task_content += f"## Output {i} (grid shape: {output_shape}):\n{output_str}\n\n"
    
    # Add test examples
    for i, inp in enumerate([test_input], 1):
        # Get grid shape
        input_shape = f"{len(inp[0])} by {len(inp)}"
        
        # Format grid using SOAR approach
        input_str = format_grid(inp)
        
        task_content += f"## Test Input {i} (grid shape: {input_shape}):\n{input_str}\n"
    
    return task_content

def hf_dataset_to_chat_dataset(dataset_slug: str, split: str = "train"):
    """
    Convert a HF split into the chat/prompt format.
    """
    ds = load_dataset(dataset_slug, split=split, keep_in_memory=True)

    def create_chat_messages(example):
        task_content = format_task_content(example["train_input"],
                                          example["train_output"],
                                          example["test_input"])

        user_content = INITIAL_TURN_PROMPT.format(task_content=task_content)

        assistant_content = ""
        if include_reasoning and example.get("reasoning", "").strip():
            assistant_content += f"<think>{example['reasoning'].strip()}</think>"
        assistant_content += f"Final answer:\n```python\n{example['code']}\n```"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]

        prompt_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        if not include_reasoning:
            prompt_text += "<think>\n\n</think>\n\n"

        return {
            "messages": messages,
            "text": text,
            "prompt": prompt_text,
            "train_input": example["train_input"],
            "train_output": example["train_output"],
            "test_input": example["test_input"],
            "test_output": example["test_output"],
            "reasoning": example.get("reasoning", ""),
            "code": example["code"],
            "task_id": example.get("task_id", ""),
        }

    ds = ds.map(create_chat_messages, desc=f"build chat + prompt fields ({split})")
    return ds

def build_dataset(train_slug: str,
                  val_slug: Optional[str] = None,
                  train_split: str = "train",
                  val_split: str = "validation") -> DatasetDict:
    """
    Build a DatasetDict with 'train' and 'validation' keys.
    - If val_slug is None, both splits are loaded from train_slug.
    - Otherwise, load train_split from train_slug and val_split from val_slug.
    """
    
    # Load and filter
    train_ds = hf_dataset_to_chat_dataset(train_slug, split=train_split)
    if max_rows:
        train_ds = train_ds.select(range(min(len(train_ds), max_rows)))

    # Validation logic
    if val_slug is None:
        # Use the same slug, but a different split
        try:
            val_ds = hf_dataset_to_chat_dataset(train_slug, split=val_split)
        except Exception as e:
            raise ValueError(
                f"Could not load split '{val_split}' from '{train_slug}'. "
                f"Pass an explicit val_slug or choose a valid split.\nOriginal error: {e}"
            )
    else:
        val_ds = hf_dataset_to_chat_dataset(val_slug, split=val_split)

    if max_rows:
        val_ds = val_ds.select(range(min(len(train_ds), max_rows)))

    return DatasetDict(train=train_ds, validation=val_ds)

# ---------------------------------------------------------------------
# Build the dataset
# ---------------------------------------------------------------------
data = build_dataset(train_slug, val_slug)  # val_slug may be None 