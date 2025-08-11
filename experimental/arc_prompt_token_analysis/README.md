# ARC Task Prompt Token Length Analysis

This folder contains scripts and results for analyzing token lengths of ARC (Abstract Reasoning Corpus) task prompts using the Qwen3-4B tokenizer.

## Files

### Scripts
- **`calculate_prompt_tokens.py`** - Main analysis script that:
  - Loads ARC tasks from all datasets and splits
  - Uses the same prompt creation utilities as the main runner (`run_arc_tasks_soar.py`)
  - Applies Qwen3-4B tokenizer with proper chat templating (`enable_thinking=True`)
  - Calculates token counts for each task prompt
  - Generates CSV results and example templated prompt

### Results
- **`token_analysis_results_qwen/`** - Analysis results using Qwen3-4B tokenizer:
  - **`arc_prompt_token_analysis.csv`** - Detailed token counts for all 920 tasks
  - **`example_templated_prompt.txt`** - Example showing exact prompt format sent to LLM

## Usage

```bash
# Run from the project root
cd experimental/arc_prompt_token_analysis
uv run python calculate_prompt_tokens.py
```

## Key Findings

### Token Statistics (Qwen3-4B Tokenizer)
- **Total tasks analyzed**: 920
- **Mean tokens per prompt**: 3,917
- **Median tokens per prompt**: 2,801
- **Token range**: 483 to 19,400 tokens

### Dataset Breakdown
- **ARC-AGI-1 Training**: 400 tasks, mean 2,701 tokens, **max 18,733 tokens**
- **ARC-AGI-1 Evaluation**: 400 tasks, mean 4,232 tokens, **max 18,733 tokens**
- **ARC-AGI-2 Evaluation**: 120 tasks, mean 6,915 tokens, **max 19,400 tokens**

### Token Distribution Percentiles
- 10th percentile: 1,007 tokens
- 25th percentile: 1,629 tokens
- 50th percentile: 2,801 tokens
- 75th percentile: 5,088 tokens
- 90th percentile: 8,321 tokens
- 95th percentile: 10,693 tokens
- 99th percentile: 15,041 tokens

## Implementation Details

### Chat Template Format
The script uses the Qwen3-4B tokenizer with proper chat templating:
```
<|im_start|>system
[System message]<|im_end|>
<|im_start|>user
[Task prompt with examples]<|im_end|>
<|im_start|>assistant
```

### Prompt Structure
Each prompt contains:
- System message (SOAR format)
- Training examples (input/output grids)
- Test input (and expected output for training tasks)
- Proper grid formatting matching SOAR approach

### Key Parameters
- `enable_thinking=True` - Enables thinking mode in Qwen tokenizer
- `add_generation_prompt=True` - Adds assistant prompt for generation
- Uses same prompt utilities as main SOAR runner for consistency

## Notes

- ARC-AGI-2 tasks are generally more complex (longer prompts)
- Some tasks approach 20k tokens, requiring models with large context windows
- Token counts are specific to Qwen3-4B tokenizer (other tokenizers will vary)
- The longest tasks may challenge models with smaller context limits (8k-16k tokens)