# ARC-AGI Task Runner

A comprehensive tool for testing OpenAI-compatible language models on ARC-AGI tasks using the Chat Completions API. Supports reasoning models (o3, Gemini Flash, Qwen) with multi-turn feedback, independent attempts, and detailed analysis.

Folders:
- `fine-tuning` - ipynb notebooks for fine-tunign as well as a logs folder for tensorboard logs.
- `tests` misc test scripts

**Videos**
[Part 5: Comments on discrete vs continuous + Ablating Sampling vs Feedback](https://share.descript.com/view/W3OEzy9PW9A)
[Part 4 - Visualisation + Testing out Feedback and Images](https://share.descript.com/view/zfBfDlP20uA)
[Part 3: o3-tools - inspecting the prompts (day 3)](https://share.descript.com/view/eEJtRvt1XlM)
[Part 2: Part 2: Running code locally](https://share.descript.com/view/V9EjCb9cMZB)
[Part 1: Running o3 with remote code interpreter tools](https://share.descript.com/view/RmRclePaxMP)

**Measuring Performance:**
Objective: Define a test that is a representative measure of performance while also being fast to run. Currently using o4-mini on arc-agi-1 mit-medium. Will graduate to hard once we score 15+/20 consistently.

**Runpod One-click-template**
Runpod One-click-template [here](https://console.runpod.io/deploy?template=agyu4xrpgl&ref=jmfkcdio) - swap out the model name if using a fine-tuned model.


## Features

- Run ARC-AGI tasks with any OpenAI-compatible language model API
- Support for custom API endpoints (Claude, Qwen, DeepSeek, local models, etc.)
- Multi-turn execution with training examples feedback
- **Robust timeout handling** with automatic retries (1200s timeout for reasoning models, 3 attempts per turn)
- Comprehensive scoring including pixel accuracy and binary correctness
- Budget tracking with token usage and cost estimation
- Detailed logging of all runs for analysis
- Support for different datasets and task subsets
- Text-only mode for fast, efficient execution

> **Note:** In testing, o3 looped does not solve any of the longest ARC-AGI problems (tested on 5). Shortest and medium tasks are solved much more reliably.

## Setup

1. Install dependencies with uv:
```bash
uv sync
```

2. Ensure you have the `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_key_here
```

### UV Project Discovery Note

**Important**: When running `uv` commands from this `llm-python/` subdirectory, `uv` automatically searches upward and discovers the root `pyproject.toml` file in the repository root.

This means:
- Commands like `uv venv`, `uv sync`, and `uv run` will use the root project configuration
- The Python version requirement (`requires-python = ">=3.12"`) from the root will be respected
- All dependencies from the root `pyproject.toml` will be available

To create an isolated environment that ignores the root configuration:
```bash
uv venv --no-config
```

## Usage

### Basic Usage

Run the shortest training tasks from ARC-AGI-1:
```bash
uv run python run_arc_tasks_soar.py --dataset arc-agi-1 --subset shortest_training_10
```

### All-Attempts Evaluation Mode (`run_arc_tasks_soar.py`)

**Refactored system** with all-attempts execution, parallel processing, and voting-based evaluation:

```bash
# Run with all-attempts mode (default 8 attempts per task)
uv run python run_arc_tasks_soar.py --dataset arc-agi-1 --subset shortest_10

# High parallelization for speed  
uv run python run_arc_tasks_soar.py --dataset arc-agi-1 --subset shortest_10 --max_attempts 8 --max_workers 20

# Gemini Flash via OpenRouter with reasoning
uv run python run_arc_tasks_soar.py --dataset arc-agi-1 --subset shortest_10 --model google/gemini-2.5-flash --base-url https://openrouter.ai/api/v1 --reasoning_effort low

# Run repeated tests with statistics
uv run python run_arc_tasks_soar.py --dataset arc-agi-1 --subset shortest_10 --repeat-runs 3
```

**Key Features:**
- **True Parallelization**: Parallelizes at the attempt level - all attempts across all tasks run simultaneously
- **Real-time Task Summaries**: Displays brief statistics for each task as it completes (test-correct, train-perfect, train-partial counts)
- **Real-time Logging**: Individual task logs are saved immediately when each task completes, not at the end of the run
- **Efficient Resource Usage**: Workers process individual attempts independently for maximum throughput
- **Voting Algorithms**: Weighted majority (frequency + accuracy) and train-majority voting for robust evaluation
- **Transduction Filtering**: Automatically removes hardcoded/cheating responses
- **Sampling Parameter Logging**: Comprehensive logging of all sampling parameters (temperature, top_p, top_k, min_p) used in API calls
- **Adaptive Sampling Parameters**: Automatic detection of endpoint type with appropriate defaults:
  - **TCP endpoints** (containing ":"): Uses `min_p=0.05` in `extra_body`
  - **Other endpoints**: Uses `top_k=50` and `top_p=0.9` defaults
- **Optimized File I/O**: 30-second timeout for file operations with detailed error logging for debugging
- **Independent Multiple Runs**: Each repeated run is completely isolated with no shared state - results loaded from files for aggregation
- **Robust State Management**: Explicit garbage collection and cleanup between runs prevents state spillover

**When to use:**
- For comprehensive evaluation with statistical rigor
- When you want oracle upper bounds and pass@k metrics
- For maximum parallelization efficiency with high worker counts
- For systematic comparison of multiple attempts per task
- When you need real-time progress updates as tasks complete

### Advanced Usage

```bash
# Run with different models and reasoning efforts
uv run python run_arc_tasks_soar.py --dataset arc-agi-1 --subset shortest_training_10 --model google/gemini-2.5-flash --base-url https://openrouter.ai/api/v1 --reasoning_effort low

# Run with higher reasoning effort (8k tokens)
uv run python run_arc_tasks_soar.py --dataset arc-agi-1 --subset middle_training_10 --model google/gemini-2.5-flash --base-url https://openrouter.ai/api/v1 --reasoning_effort medium

# RunPod: Use direct TCP to avoid Cloudflare 524 timeouts
uv run python run_arc_tasks_soar.py --dataset arc-agi-1 --subset shortest_training_10 --model Qwen/Qwen3-4B --base-url http://157.66.254.42:15712/v1

# Run tasks in parallel with high worker count for faster execution
uv run python run_arc_tasks_soar.py --dataset arc-agi-2 --subset shortest_training_30 --max_workers 20

# Run the same test 3 times with completely independent runs and aggregate statistics
uv run python run_arc_tasks_soar.py --dataset arc-agi-1 --subset shortest_training_10 --repeat-runs 3

# Disable thinking for Qwen models (sets enable_thinking=false in chat_template_kwargs)
uv run python run_arc_tasks_soar.py --dataset arc-agi-1 --subset shortest_training_10 --model Qwen/Qwen3-4B --base-url http://localhost:8000/v1 --qwen-no-think

# Set specific token limit for responses (overrides reasoning effort defaults)
uv run python run_arc_tasks_soar.py --dataset arc-agi-1 --subset shortest_training_10 --model gpt-4.1-mini --max-tokens 2000

# Available options:
#   --dataset: arc-agi-1 or arc-agi-2
#   --subset: shortest_training_1, shortest_training_10, shortest_training_30, shortest_evaluation_1, shortest_evaluation_10, shortest_evaluation_30, etc.
#   --model: Model name (default: gpt-4.1-mini) - works with any OpenAI-compatible API
#   --base-url: Custom API endpoint URL (default: OpenAI) - enables Claude, Qwen, local models, etc.
#   --reasoning_effort: Reasoning effort level: low (2k tokens), medium (8k tokens), high (32k tokens) - for Gemini; other models may vary
#   --max-tokens: Maximum tokens for model responses (overrides reasoning effort defaults)
#   --limit: Limit number of tasks to run
#   --max_attempts: Maximum number of attempts per task (default: 8)
#   --max_workers: Number of parallel workers (default: 1, efficient up to 50+)
#   --repeat-runs: Number of times to repeat the entire test (default: 1, max: 10)
#   --qwen-no-think: Disable thinking for Qwen models (sets enable_thinking=false in chat_template_kwargs)
```

### Reasoning Effort Support

For compatible models that support reasoning, control reasoning token allocation:

**Gemini Models (via OpenRouter):**
- `--reasoning_effort low`: 2,000 reasoning tokens (default)
- `--reasoning_effort medium`: 8,000 reasoning tokens  
- `--reasoning_effort high`: 32,000 reasoning tokens
- Uses optimal `extra_body={"reasoning": {"max_tokens": X}}` parameter structure
- Reasoning content captured in logs for analysis

**Other Reasoning Models:**
- Uses standard `max_tokens` parameter for reasoning allocation
- Works with OpenRouter and other compatible APIs automatically

**Token Control Priority:**
- `--max-tokens` parameter overrides all automatic reasoning effort settings
- Without `--max-tokens`: reasoning effort controls token allocation automatically
- With `--max-tokens`: your specified limit takes precedence for all models

**Example Results:** Gemini Flash gets 7/10 correct on shortest training tasks with 2k reasoning tokens, 5/10 correct on medium difficulty tasks with 8k reasoning tokens.

### Reasoning Content Capture

The tool automatically captures and standardizes reasoning content from models that provide it:

- **All Models**: Reasoning content is standardized to the `reasoning` field in logs for consistency
- **Gemini models** (via OpenRouter): 
  - Original `reasoning` field preserved and standardized
  - Additional `reasoning_details` field with structured reasoning data
  - Example: *"**Examining Grid Transformations** I've been scrutinizing the input and output grid examples..."*
- **Qwen models**: 
  - Via TCP endpoints: `reasoning_content` → standardized to `reasoning` field
  - Via OpenRouter: `reasoning` field preserved
  - Automatically disabled with `--qwen-no-think` flag (sets `enable_thinking=false`)
- **o1/o3 models** (via OpenAI): Hidden reasoning tokens captured when available
- **Other models**: Standard content logging

All reasoning data is preserved in logs for analysis. The code extraction searches both content and reasoning fields, ensuring no code is missed regardless of where models place their solutions.

### Training Data Generation

Generate fine-tuning datasets from logged program attempts using hindsight relabeling. The script now creates **Hugging Face datasets** instead of JSONL files and pushes them directly to Hugging Face Hub.

```bash
# Create HF dataset from Gemini and Qwen3-4B logs (includes reasoning by default)
uv run python generate_training_data.py --model "google/gemini-2.5-flash,qwen/qwen3-4b" --dataset "arc-agi-1" --subset "middle_training_10" --clean-code --hf-private

# With validation split and custom name
uv run python generate_training_data.py --model "google/gemini-2.5-flash" --validation --clean-code --hf-dataset-name "arc_gemini_training_v1" --hf-private

# Basic usage (includes reasoning by default)
uv run python generate_training_data.py --dataset "arc-agi-1" --subset "middle_training_10" --clean-code --hf-private

# Disable reasoning content if desired
uv run python generate_training_data.py --dataset "arc-agi-1" --subset "middle_training_10" --clean-code --no-reasoning --hf-private
```

**Key Changes:**
- ✅ **Hugging Face datasets**: Direct push to HF Hub instead of JSONL files
- ✅ **Competition format**: Uses flat structure with all expected fields (`reasoning`, `code`, `train_input`, `predicted_train_output`, etc.)
- ✅ **Reasoning included by default**: Automatically captures model reasoning traces (use `--no-reasoning` to disable)
- ✅ **Auto-naming**: Default format `synth_{dataset}_{subset}_DATETIME`
- ✅ **Trelis organization**: Pushes to `Trelis/` by default (configurable with `--hf-org`)
- ✅ **Private datasets**: Use `--hf-private` flag for private repositories

**Deduplication Logic:**
- **Test-correct programs**: Deduplicated by cleaned code string matching (after comment removal)
- **Test-incorrect programs**: Deduplicated by output behavior similarity across training examples  
- **Hindsight relabeling**: Programs that fail tests use their actual outputs as ground truth

**Key Features:**
- Automatic transduction/cheating detection and filtering
- Code cleaning with comment/whitespace removal (optional `--clean-code`)
- Validation of training examples for consistency
- Debug mode (`--debug`) for detailed deduplication information

### Available Subsets

- `shortest_training_1`: Single shortest training task
- `shortest_training_10`: 10 shortest training tasks
- `shortest_training_30`: 30 shortest training tasks
- `shortest_evaluation_1`: Single shortest evaluation task
- `shortest_evaluation_10`: 10 shortest evaluation tasks
- `shortest_evaluation_30`: 30 shortest evaluation tasks
- ... and similarly for `middle` and `longest`
- `grid_size_distributed_30_training`: 30 training tasks evenly distributed by grid size
- `grid_size_distributed_30_evaluation`: 30 evaluation tasks evenly distributed by grid size

**Example:**

Run the 10 shortest evaluation tasks from ARC-AGI-2:
```bash
uv run python o3-tools/run_arc_tasks.py --dataset arc-agi-2 --subset shortest_evaluation_10
```

Run the 30 longest training tasks from ARC-AGI-1:
```bash
uv run python o3-tools/run_arc_tasks.py --dataset arc-agi-1 --subset longest_training_30 --model o3
```

Run 30 grid size distributed evaluation tasks from ARC-AGI-2:
```bash
uv run python o3-tools/run_arc_tasks.py --dataset arc-agi-2 --subset grid_size_distributed_30_evaluation
```

Run 30 tasks in parallel with 15 workers for 15x speedup:
```bash
uv run python o3-tools/run_arc_tasks.py --dataset arc-agi-2 --subset shortest_training_30 --max_workers 15
```

## Task Evolution Visualization

The `visualize_task_evolution.py` tool creates detailed visual analysis of how models learn and evolve their solutions across multiple turns.

### Features

- **Turn-by-turn visualization**: Shows input → expected → predicted for every training and test example
- **Clean, organized charts**: One chart per turn for easy comparison
- **Color-coded results**: Green for solved examples, red for failures
- **ARC-style grids**: Proper color mapping and grid formatting
- **Comprehensive metrics**: Accuracy percentages and solve counts for each turn

### Usage

```bash
# Visualize evolution from a log file
uv run python visualize_task_evolution.py [LOG_FILE] [--dataset arc-agi-1|arc-agi-2]

# Examples:
uv run python visualize_task_evolution.py 20250709_095758_577922_8698928896_aa300dc3.json
uv run python visualize_task_evolution.py my_log_file.json --dataset arc-agi-2
```

### Output Structure

**Charts are saved to `plots/` directory:**
- `turn_1_[task_id]_[log_stem].png` - Turn 1 visualization
- `turn_2_[task_id]_[log_stem].png` - Turn 2 visualization  
- `turn_3_[task_id]_[log_stem].png` - Turn 3 visualization
- ... etc.

**Each chart shows:**
- **3 columns**: Input | Expected Output | Predicted Output
- **N+1 rows**: N training examples + 1 test example
- **Performance metrics**: Training solve rate, test accuracy
- **Visual feedback**: Grid dimensions, accuracy percentages

### Example Output

```
Visualizing task evolution for: aa300dc3
Found 8 turns with valid programs
Creating individual visualizations in plots/...
  Creating visualization for Turn 1...
    Saved: plots/turn_1_aa300dc3_20250709_095758_577922_8698928896_aa300dc3.png
  ...

Evolution Summary:
----------------------------------------------------------------------
Turn   Test Acc   Train Solved Train Avg Acc File                
----------------------------------------------------------------------
1      91.0%      3/4          98.0%        turn_1_aa300dc3_...png
2      82.0%      0/4          83.0%        turn_2_aa300dc3_...png
3      91.0%      3/4          97.0%        turn_3_aa300dc3_...png
...

Created 8 turn visualizations for task aa300dc3
All plots saved in: plots/
```

### Use Cases

- **Debug model learning**: See exactly how the model's understanding evolves
- **Identify patterns**: Spot when models oscillate between solutions
- **Visual analysis**: Compare predicted vs expected outputs side-by-side
- **Progress tracking**: Monitor training vs test performance across turns

## Training Data Generation

The `generate_training_data.py` tool extracts programs from log files to create fine-tuning training data in JSONL format.

### Data Quality & Validation

**Fixed Grid Serialization Issue (23 Jul 2025)**: Previous versions had a 0.3% validation failure rate due to empty rows being lost during serialization. This has been completely fixed:

- **Problem**: Programs outputting grids with empty rows (e.g., `[[], [8, 8, 8]]`) would lose the empty rows during format/parse cycle
- **Solution**: Empty rows now use `[EMPTY_ROW]` marker to preserve structure
- **Result**: 100% validation success rate on all training examples

**Validation Script**: Use `validate_hf_dataset.py` to verify HF dataset quality:
```bash
# Validate a HF dataset
uv run python validate_hf_dataset.py Trelis/synth_arc-agi-1_middle_training_10_20250724_081021

# Validate with verbose output
uv run python validate_hf_dataset.py Trelis/synth_arc-agi-1_middle_training_10_20250724_081021 --verbose

# Validate only first 10 rows for quick testing
uv run python validate_hf_dataset.py Trelis/synth_arc-agi-1_middle_training_10_20250724_081021 --limit 10

# Validate validation split
uv run python validate_hf_dataset.py Trelis/my_dataset_name --split validation
```

**What the validation script checks:**
- ✅ **Format validation**: All grids are proper 2D lists with integer values 0-9
- ✅ **Execution consistency**: Programs actually produce the claimed `predicted_train_output`
- ✅ **Data integrity**: All field lengths match and no corrupted data
- ✅ **Type safety**: No boolean/tuple/string values in grid cells

**Multiple Models Support**: You can filter by multiple models using either comma-separated values (`--model "model1,model2"`) or repeated arguments (`--model "model1" --model "model2"`).

### Key Features

- **Full parallel processing**: Uses all CPU cores for maximum speed (6-10x faster)
- **Balanced datasets**: Balances difficulty by ensuring 50/50 split of programs with/without correct examples
- **Stratified validation splits**: Task-level validation with balanced difficulty distribution
- **Strict quality control**: Re-executes programs, validates 2D grid formats, ensures consistency
- **Smart error handling**: Drops individual failed examples but rejects programs with format violations
- **Code cleaning**: Optional aggressive comment stripping with `--clean-code` flag (up to 58% size reduction)
- **Automatic deduplication**: Removes duplicate programs within each task based on test correctness and output similarity
- **Transduction/cheating detection**: Identifies and removes programs that hardcode answers instead of implementing transformations

### Usage

```bash
# Generate training data from the last 100 log files (automatically uses all cores - 2)
uv run python generate_training_data.py --limit 100

# Generate from all log files with validation split
uv run python generate_training_data.py --validation --output 16-jul-lorge.jsonl

# Generate with clean code (strips comments and whitespace)
uv run python generate_training_data.py --limit 100 --clean-code --output clean_training.jsonl

# Filter by model and include reasoning for correct solutions
uv run python generate_training_data.py --model "google/gemini-2.5-flash" --reasoning --output gemini_with_reasoning.jsonl

# Filter by multiple models (comma-separated)
uv run python generate_training_data.py --model "google/gemini-2.5-flash,gpt-4.1-mini,o4-mini" --output multi_model_data.jsonl

# Filter by multiple models (repeated arguments)
uv run python generate_training_data.py --model "google/gemini-2.5-flash" --model "gpt-4.1-mini" --output multi_model_data.jsonl

# Filter by date range and model
uv run python generate_training_data.py --model "google/gemini-2.5-flash" --date-from "20250721" --date-to "20250721" --output gemini_july_21.jsonl

# Specify custom output filename
uv run python generate_training_data.py --limit 500 --output my_training_data.jsonl

# Generate with validation split and custom name
uv run python generate_training_data.py --limit 1000 --validation --output arc_training_data.jsonl

# Combine reasoning with other options
uv run python generate_training_data.py --limit 500 --reasoning --validation --output reasoning_training.jsonl

# Filter by dataset and subset to target specific task groups
uv run python generate_training_data.py --dataset "arc-agi-1" --subset "shortest_training_10" --output shortest_tasks.jsonl

# Filter by dataset only (includes both training and evaluation tasks from that dataset)
uv run python generate_training_data.py --dataset "arc-agi-2" --output arc_agi_2_only.jsonl

# Combine dataset/subset filtering with model filtering
uv run python generate_training_data.py --dataset "arc-agi-1" --subset "all_training" --model "google/gemini-2.5-flash" --output gemini_arc1_training.jsonl

# Combine multiple models with dataset/subset filtering
uv run python generate_training_data.py --dataset "arc-agi-1" --subset "all_training" --model "google/gemini-2.5-flash,o4-mini" --output multi_model_arc1_training.jsonl

# Filter by subset only (includes that subset from both datasets)
uv run python generate_training_data.py --subset "middle_training_10" --validation --hf-private

# Disable deduplication if you want all programs (including duplicates)
uv run python generate_training_data.py --limit 100 --no-dedup --hf-private

# Enable debug mode to see detailed transduction detection info
uv run python generate_training_data.py --limit 100 --debug --hf-private

# Disable transduction filtering if you want to keep all programs (including cheating ones)
uv run python generate_training_data.py --limit 100 --no-transduction-filter --hf-private

# New HF-specific options:
#   --hf-dataset-name: Custom dataset name (default: synth_{dataset}_{subset}_DATETIME)
#   --hf-org: Organization to push to (default: "Trelis")
#   --hf-private: Make dataset private (recommended)
#   --reasoning: Include reasoning content (default: True)
#   --no-reasoning: Disable reasoning content inclusion
```

### Filtering Options

**Model/Dataset/Subset Filtering:**
- `--model`: Filter by model name(s). Supports multiple models via comma-separated list or repeated arguments (e.g., `"google/gemini-2.5-flash,gpt-4.1-mini"` or multiple `--model` flags)
- `--dataset`: Filter by dataset (e.g., `"arc-agi-1"`, `"arc-agi-2"`)
- `--subset`: Filter by subset (e.g., `"all_training"`, `"shortest_evaluation_10"`)

**Date Range Filtering:**
- `--date-from`: Filter files from this date onwards (format: `YYYYMMDD`)
- `--date-to`: Filter files up to this date (format: `YYYYMMDD`)
- `--pattern`: Filter log files by filename pattern (e.g., `"20250721_112639"`)

### Reasoning Content (`--reasoning`)

The `--reasoning` flag enables inclusion of model reasoning traces for programs that correctly solve the test output:

#### **What It Does**
- **Checks test correctness**: For each program, executes it on the actual test input and compares with expected output
- **Includes reasoning for correct solutions**: Only programs that solve the test correctly get reasoning content included
- **Wraps reasoning in think tags**: Reasoning is formatted as `<think>...reasoning content...</think>` before the program code
- **Works with all reasoning models**: Automatically extracts reasoning from Gemini, Qwen, o1/o3, and other models

#### **Reasoning Sources**
- **Gemini models**: Extracts from `choices[0]['message']['reasoning']` field
- **Qwen models**: Extracts from `reasoning_content` or `reasoning` fields  
- **o1/o3 models**: Extracts hidden reasoning when available
- **Other models**: Searches various common reasoning field names

#### **Example Output Format**

**Without `--reasoning`:**
```json
{
  "role": "assistant",
  "content": "Final answer:\n```python\ndef transform(grid):\n    return processed_grid\n```"
}
```

**With `--reasoning` (for correct solutions):**
```json
{
  "role": "assistant", 
  "content": "<think>\nLet me analyze the training examples to understand the pattern...\nI notice that in each case, the transformation involves...\nBased on this analysis, I should implement...\n</think>\n\nFinal answer:\n```python\ndef transform(grid):\n    return processed_grid\n```"
}
```

#### **Use Cases**
- **Reasoning-aware fine-tuning**: Train models to think through problems step-by-step
- **Quality filtering**: Focus on solutions that actually work (test-correct programs)
- **Chain-of-thought training**: Learn both reasoning patterns and implementation patterns
- **Debugging model logic**: Understand how models arrive at correct solutions

#### **Selection Criteria**
Programs get reasoning content included only if:
- ✅ `--reasoning` flag is specified
- ✅ Program correctly solves the test input → expected test output  
- ✅ Original log contains reasoning content from the model
- ✅ Program executes successfully without errors

**Note**: Since only test-correct programs get reasoning, the percentage with reasoning will typically be lower than the overall accuracy rate, as it's filtered by both correctness and reasoning availability.

### Simple Dataset Generation (No Code/Reasoning)

For creating datasets without code or reasoning content, use the simplified `create_simple_dataset.py` script:

```bash
# Create simple dataset from arc-agi-1 training tasks
uv run python create_simple_dataset.py arc-agi-1 all_training --save-local --validation

# Create dataset and push to Hugging Face  
uv run python create_simple_dataset.py arc-agi-2 shortest_training_30 --hf-private

# Custom dataset name and organization
uv run python create_simple_dataset.py arc-agi-1 random_split_1_training --hf-dataset-name "simple_baseline_v1" --hf-org "YourOrg"
```

**Key differences from full training data generation:**
- **Input**: Takes dataset and subset names directly (not log files)
- **Simplified content**: All `reasoning`, `code`, `predicted_*`, and `correct_*` fields are empty/blank
- **Same structure**: Maintains competition format with all expected columns and types
- **No execution**: No program running or validation required
- **Faster**: Much simpler processing without code analysis

**Use cases:**
- Baseline datasets for fine-tuning experiments
- Template datasets with proper structure but empty content
- Testing dataset loading/processing pipelines
- Creating placeholder datasets before adding real content

### Code Cleaning (`--clean-code`)

The `--clean-code` flag enables aggressive comment stripping and code cleanup to produce compact, professional training data:

#### **What It Does**
- **Removes comment-only lines**: Lines that start with `#` are eliminated entirely
- **Strips inline comments**: Text after `#` on code lines is removed (respects strings)
- **Eliminates blank lines**: Reduces excessive whitespace and empty lines
- **Normalizes formatting**: Produces clean, compact code without documentation overhead

#### **Benefits**
- **Significant size reduction**: Typically achieves 50-60% reduction in character count
- **Token efficiency**: Fewer tokens needed for LLM training and inference
- **Cleaner training data**: Focuses on executable logic without comment clutter
- **Professional formatting**: Produces code that looks hand-cleaned

#### **Safety Features**
- **Pre-validation cleaning**: Code is cleaned **before** program validation to catch issues early
- **Compilation testing**: Each cleaned program is verified with `compile()` to ensure syntax validity
- **Graceful fallback**: If cleaning breaks the code, the original version is preserved
- **Zero failures tolerated**: Programs that fail cleaning are kept in original form
- **Detailed reporting**: Shows cleaning statistics and any failures

#### **Example Results**
```
Code cleaning results:
  - 423 programs processed
  - 0 cleaning failures (kept original)
  - 7,306 → 3,076 characters
  - 57.9% size reduction achieved
```

#### **Before vs After**

**Original code (112 lines):**
```python
def transform(grid):
    rows_in = len(grid)
    cols_in = len(grid[0]) # Always 9

    rows_out = 7
    cols_out = 9 # Always 9

    # Step 1: Initialize transformed_grid by copying relevant parts
    # The first row of the output grid is always the first row of input
    for c_out in range(cols_out):
        transformed_grid[0][c_out] = grid[0][c_out]
```

**Cleaned code (55 lines):**
```python
def transform(grid):
    rows_in = len(grid)
    cols_in = len(grid[0])
    rows_out = 7
    cols_out = 9
    for c_out in range(cols_out):
        transformed_grid[0][c_out] = grid[0][c_out]
```

#### **Use Cases**
- **Fine-tuning optimization**: Reduce token usage and training costs
- **Production training data**: Clean, professional code for model training
- **Storage efficiency**: Smaller files for faster loading and transfer
- **Focus on logic**: Remove documentation to emphasize executable patterns

### Output

**Datasets are pushed directly to Hugging Face Hub:**

- **Without validation**: Single `train` split in HF dataset
- **With validation**: Both `train` and `validation` splits in HF dataset
- **Dataset URLs**: Printed in console output (e.g., `https://huggingface.co/datasets/Trelis/synth_arc-agi-1_middle_training_10_20250724_080500`)

**Note**: Dataset is first balanced (50/50 difficulty split), then validation uses different tasks than training.

### Quality Control

**Programs included if they:**
- ✅ Execute successfully on ≥1 training example  
- ✅ Fail on ≥1 training example (learning opportunity)
- ✅ Return proper 2D grids (`[[...], [...]]`)
- ✅ All cell values are integers 0-9

**Programs rejected if they:**
- ❌ Solve all examples (no room for improvement)
- ❌ Return invalid formats (integers, 1D lists, etc.)
- ❌ Contain invalid cell values (booleans, tuples, floats, strings, integers <0 or >9)
- ❌ Fail to execute on all examples

**Error handling:** Execution failures drop individual examples; format violations reject entire programs.

**Validation improvements:** As of January 2025, strict cell value validation prevents malformed training data from boolean values (`True`/`False`), tuple coordinates `(1, 2)`, invalid integers, and other non-grid data types that previously passed validation.

### Process

1. **Extract** programs from log files (parallel)
2. **Validate** by re-executing programs (parallel)  
3. **Filter** by quality criteria
4. **Balance** dataset (50/50 difficulty split)
5. **Split** into training/validation sets (optional)
6. **Generate** JSONL training examples (parallel)

### Dataset Format

Each dataset row contains the **competition format** with all required fields:
- **`reasoning`**: Model's reasoning trace (if available and `--reasoning` used)
- **`code`**: Generated Python program
- **`train_input/train_output`**: Original training examples
- **`predicted_train_output`**: Program's actual outputs on training inputs
- **`correct_train_input`**: Boolean list indicating which training examples were solved correctly
- **`test_input/test_output`**: Original test examples  
- **`predicted_test_output`**: Program's actual outputs on test inputs
- **`correct_test_input`**: Boolean list indicating which test examples were solved correctly
- **`task_id/model/generation`**: Metadata fields

This format is compatible with the competition and enables training models that learn from partially-correct solutions.

### Example Output

```
Found 2,847 total programs
Qualified programs: 423
Generated 381 training examples
Programs with at least one originally correct answer: 298/381 (78.2%)
  Task breakdown: 67 with correct examples, 22 with no correct examples
  Balanced dataset: dropped 45 excess correct-example tasks
  Balanced breakdown: 22 with correct examples, 22 with no correct examples
Saved training data to: training_data/training_data_train.jsonl (154 examples from 40 tasks)
Saved validation data to: training_data/training_data_val.jsonl (24 examples from 4 tasks)
```

**Performance**: Uses parallel processing with `total_cores - 2` workers for optimal speed. Typically achieves **6-10x speedup** compared to single-threaded processing. Progress updates appear every 100 log files and every 50 programs during validation.

### Program Deduplication (`--no-dedup`)

The tool automatically deduplicates programs within each task to create higher-quality training data:

#### **How Deduplication Works**
1. **Test-correct deduplication**: If multiple programs correctly solve the test case, only the first one is kept (since they're all equivalent in terms of correctness)
2. **Output-similarity deduplication**: For programs that don't solve the test correctly, deduplication is based on output similarity across all training examples

#### **Deduplication Logic**
- Programs are grouped by task ID
- Within each task, programs that correctly solve the test are deduplicated (keep only first)
- Remaining programs are deduplicated based on their output signatures (combination of outputs on all training examples)
- Programs with identical output patterns are considered duplicates (only first is kept)

#### **Benefits**
- **Reduces training data redundancy**: Eliminates functionally identical programs
- **Improves training efficiency**: Fewer duplicate patterns mean more diverse learning examples
- **Maintains solution diversity**: Keeps programs with different approaches (different outputs)
- **Preserves correctness**: Always keeps at least one test-correct program per task when available

#### **Example Output**
```
📊 Deduplication Summary:
  Tasks processed: 45
  Programs before: 234
  Programs after: 156
  Test-correct deduped: 23
  Output-similarity deduped: 55
  Total deduplication: 78 programs (33.3%)
```

#### **Disable Deduplication**
Use `--no-dedup` flag to keep all programs including duplicates:
```bash
uv run python generate_training_data.py --no-dedup --output all_programs.jsonl
```

### Transduction/Cheating Detection

Automatically detects and removes programs that hardcode answers instead of implementing genuine transformations.

**Detection methods:**
- Programs with lines >200 characters (likely hardcoded arrays)
- Programs containing exact output values as strings in the code

**Debug mode** (`--debug`): Shows detailed info about detected cheating per task
**Disable filtering** (`--no-transduction-filter`): Keep all programs including cheating ones

**Example output:**
```
🛡️ Transduction/Cheating Filter Results:
  Programs rejected for cheating: 45
  Tasks with cheating programs: 12
  📊 Rejection categories:
    Hardcoded outputs: 38
    Long lines (>200 chars): 7
```

### Use Cases

- **Fine-tune models**: Use the JSONL files directly with OpenAI's fine-tuning API
- **Improve reasoning**: Train on partially-correct solutions to learn better pattern recognition
- **Domain adaptation**: Adapt general models to ARC-specific reasoning patterns
- **Validation**: Use the validation split to monitor training progress and prevent overfitting

## Repeated Runs with Statistical Analysis

The tool supports running the same test multiple times with **completely independent runs** to calculate robust performance statistics.

### Key Features

- **Independent execution**: Each run creates a fresh runner instance with no shared state
- **File-based aggregation**: Results are saved to individual files and loaded for final statistics
- **Robust state management**: Explicit garbage collection and cleanup between runs prevents state spillover
- **Comprehensive validation**: Task data integrity validation and thread-safe execution
- **Graceful failure handling**: Failed runs don't affect other runs - statistics calculated from successful runs only

### Usage Examples

```bash
# Run the same test 3 times with statistical analysis
uv run python -m llm-python.run_arc_tasks --dataset arc-agi-1 --subset shortest_training_1 --repeat-runs 3 --model google/gemini-2.5-flash --reasoning_effort medium --base-url https://openrouter.ai/api/v1

# Run 5 times with parallelization for faster execution
uv run python -m llm-python.run_arc_tasks --dataset arc-agi-2 --subset shortest_evaluation_10 --repeat-runs 5 --max_workers 10

# Test model consistency with many runs
uv run python -m llm-python.run_arc_tasks --dataset arc-agi-1 --subset shortest_training_30 --repeat-runs 10 --model o4-mini
```

### Statistical Output

**Individual Run Results:**
```
Run  Attempted  Turn 1 Only  All Turns   Turn 1 Rate   All Turns Rate
1    10         3            6           30.0%         60.0%
2    10         2            7           20.0%         70.0%
3    10         4            5           40.0%         50.0%
```

**Aggregate Statistics:**
```
Turn 1 Only Success Rate:
  Mean: 30.0%
  Std Dev: 10.0%
  95% CI: [10.4%, 49.6%]

All Turns Success Rate:
  Mean: 60.0%
  Std Dev: 10.0%
  95% CI: [40.4%, 79.6%]
```

### Success Metrics Explained

- **Turn 1 Only**: Tasks solved correctly on the first conversation turn (`turns_used == 1`)
- **All Turns**: Tasks solved correctly by the end of all turns (any `turns_used`)
- **API Failures Excluded**: Timeout failures are removed from both numerator and denominator
- **Confidence Intervals**: 95% CI calculated using ±1.96 standard deviations

### File Outputs

**Individual Runs:**
- Task logs: `{timestamp}_{thread_id}_{task_id}_run{N}.json`
- Run summaries: `{timestamp}_summary_{dataset}_{subset}_run{N}.json`

**Aggregate Results:**
- Combined statistics: `{timestamp}_aggregate_summary_{dataset}_{subset}_{N}runs.json`

### Use Cases

- **Model evaluation**: Assess consistency and reliability across multiple runs
- **Performance benchmarking**: Get statistically robust performance estimates
- **A/B testing**: Compare different models, reasoning efforts, or configurations
- **Confidence intervals**: Understand the uncertainty in your performance measurements



## Parallelization

The tool supports parallel execution to dramatically reduce wall-clock time for large task sets:

### Key Features

- **Thread-based parallelization**: Up to 128 concurrent workers
- **Thread-safe execution**: Safe cost tracking, progress reporting, and file I/O
- **Rate limiting**: Optional delays to respect API rate limits
- **Progress tracking**: Real-time progress updates for parallel execution
- **Robust error handling**: Individual task failures don't crash the entire batch

### Usage Examples

```bash
# Run 10 tasks with 5 workers (2x speedup)
uv run python -m llm-python.run_arc_tasks --max_workers 5

# Run 30 tasks with high parallelization (30x speedup)
uv run python -m llm-python.run_arc_tasks --subset shortest_training_30 --max_workers 30

# Run with rate limiting to avoid hitting API limits
uv run python -m llm-python.run_arc_tasks --max_workers 10 --rate_limit_delay 0.2

# Conservative parallel execution with 5 workers and 0.5s delay
uv run python -m llm-python.run_arc_tasks --max_workers 5 --rate_limit_delay 0.5
```

### Performance Benefits

- **Sequential**: 30 tasks × 10 seconds each = 5 minutes total
- **10 workers**: 30 tasks ÷ 10 workers × 10 seconds = 30 seconds total  
- **30 workers**: 30 tasks ÷ 30 workers × 10 seconds = 10 seconds total

**Actual speedup depends on**:
- API response times
- Network latency
- OpenAI rate limits
- Task complexity

### Rate Limiting Guidelines

OpenAI has rate limits that vary by model and tier. Recommended settings:

- **Conservative**: `--max_workers 5 --rate_limit_delay 0.5`
- **Moderate**: `--max_workers 10 --rate_limit_delay 0.2` 
- **Aggressive**: `--max_workers 20 --rate_limit_delay 0.1`
- **Maximum**: `--max_workers 30` (only if you have high rate limits)

**Note**: Start conservative and increase workers/reduce delays if you don't hit rate limits.

## Timeout Handling

The tool includes robust timeout handling to prevent hanging on API calls and file operations:

### Default Timeouts

- **API timeout**: 1200 seconds (20 minutes) for reasoning models, 120 seconds (2 minutes) for Qwen no-think mode
- **Client timeout**: 2400 seconds (40 minutes) - maximum per HTTP request
- **Program execution timeout**: 0.5 seconds per program execution
- **Attempt timeout**: 350 seconds total per attempt (includes buffer beyond API timeout)
- **File I/O timeout**: 30 seconds for file write operations
- **Worker timeout**: 600 seconds (10 minutes) total for parallel execution

### Worker Performance & Timeouts

Increasing workers generally speeds up a run through parallelization, but risks hitting the timeouts (in which case you may wish to increase them). Decreasing workers results in faster per-request processing (but with less parallelization) so this can be useful for seeing a few results earlier - although that can also be achieved via the `--limit` parameter.

### Key Features

- **3 retry attempts** per turn with 2-second backoff between retries
- **Separate timeout failure tracking** - doesn't count as regular task failures
- **Complete conversation preservation** during retries
- **Detailed error logging** with specific failure reasons and complete API response data
- **Explicit timeout debugging** - shows exactly which operations are timing out

### How It Works

For each turn in a multi-turn conversation:

1. **Initial attempt**: API call with model-specific timeout (1200s for reasoning models, 120s for Qwen no-think)
2. **Retry 1**: If timeout, wait 2 seconds and retry
3. **Retry 2**: If timeout again, wait 2 seconds and final retry
4. **Timeout failure**: If all 3 attempts fail, mark as timeout failure

### Console Output Examples

```bash
# Normal execution
Turn 1/8...
  💰 Turn cost: $0.004146 (input: 1089, output: 4321)

# Timeout with retries
Turn 2/8...
  🔄 Turn 2 attempt 1/3...
  ⏰ Turn 2 attempt 1 failed (TimeoutError: Operation timed out after 1200 seconds), retrying in 2s...
  🔄 Turn 2 attempt 2/3...
  ✅ Turn 2 successful on attempt 2

# Complete timeout failure
Turn 3/8...
  🔄 Turn 3 attempt 1/3...
  ⏰ Turn 3 attempt 1 failed (HTTPStatusError: 524 A timeout occurred), retrying in 2s...
  🔄 Turn 3 attempt 2/3...
  ⏰ Turn 3 attempt 2 failed (HTTPStatusError: 524 A timeout occurred), retrying in 2s...
  🔄 Turn 3 attempt 3/3...
  ❌ Turn 3 failed after 3 attempts: HTTPStatusError: 524 A timeout occurred
```

### Summary Statistics

Timeout failures are tracked separately from regular failures:

```bash
==================================================
SUMMARY
==================================================
Total tasks attempted: 30
Successful API calls: 27/30 (90.0%)
Failed API calls: 1/30 (3.3%) ❌
Timeout failures: 2/30 (6.7%) ⏰
Tasks solved correctly: 8/30 (26.7%)
```

### Why These Timeout Values?

The API timeout is set to 1200 seconds (20 minutes) for reasoning models because:
- **Medium reasoning effort** can take 2-4 minutes for complex tasks
- **High reasoning effort** can take 4-8 minutes for difficult tasks
- **Very high reasoning effort** can take 8-15 minutes for extremely difficult tasks
- **Buffer time** accounts for network latency, API processing, and RunPod response handling
- **Balance** between patience and preventing indefinite hangs

### Log File Structure

Timeout failures are logged with special metadata:

```json
{
  "task_id": "example_task",
  "api_success": false,
  "timeout_failure": true,
  "task_failure_reason": "API timeout after retries",
  "turns_used": 3,
  "multiturn_data": {
    "conversation_history": [...],
    "all_responses": [...],
    "turn_details": [...],
    "total_turns": 3
  }
}
```

## Scoring Metrics

The tool provides two core scoring metrics to evaluate model performance:

### 1. **Binary Correctness**
- **What it measures**: Whether the predicted output grid exactly matches the expected output
- **Values**: True/False (perfect match or not)
- **Use case**: Identifies tasks that are completely solved correctly

### 2. **Pixel Accuracy** 
- **What it measures**: Percentage of individual pixels that match between predicted and expected output
- **Formula**: `correct_pixels / total_pixels`
- **Values**: 0.0 to 1.0 (0% to 100%)
- **Use case**: Measures partial correctness when solutions aren't perfect
- **Example**: If 7 out of 9 pixels match → 77.8% pixel accuracy

### **Why These Metrics Matter**

- **Binary Correctness**: Shows the "solve rate" - what percentage of tasks are completely correct
- **Pixel Accuracy**: Reveals how close imperfect solutions are to being correct  

**Example Results Interpretation**:
```
Tasks solved correctly: 4/10 (40.0%)     # 40% perfect solutions
Pixel accuracy: 85/90 (94.4%)            # Very close to correct on average
```

This shows a model that writes mostly-correct solutions.

## Task Failure Analysis

The `task_failure_reason` field in log files tracks **why tasks failed to complete successfully**. Understanding these failure types helps analyze model performance:

### **Successful Completion**
```json
"task_failure_reason": ""  // Empty string = task completed successfully
```

### **Python Execution Failures**
When the generated Python code fails to run:
```json
"task_failure_reason": "NameError: name 'numpy' is not defined"
"task_failure_reason": "IndexError: list index out of range"
"task_failure_reason": "Program exceeded timeout of 0.1s"
"task_failure_reason": "Invalid output format: some_invalid_output"
```

### **Task Completion Failures**
When the task process itself fails (code may have executed fine):
```json
"task_failure_reason": "All attempts failed"        // Independent mode: all attempts gave wrong answers
"task_failure_reason": "Max turns reached"          // Multi-turn mode: ran out of turns before solving
"task_failure_reason": "API timeout after retries" // API call failed after 3 attempts
"task_failure_reason": "Task setup failed: ..."     // Setup/initialization error
```

### **Important Distinction**
- **"All attempts failed"** ≠ execution error - means code ran but gave incorrect results
- **Python execution errors** = actual code runtime failures
- **Empty string** = complete success (correct answer achieved)

## Output

Results are saved in multiple directories:

**Log files in `logs/` directory:**
- Individual task results: `{timestamp}_{task_id}.json` (full attempt details)
- Summary reports: `{timestamp}_summary_{dataset}_{subset}.json` (minimal metrics only)

**Summary File Optimization:**
- **Minimal size**: Summary files contain only essential metrics (~5KB) instead of full attempt details (~160MB)
- **Essential data**: Task IDs, final metrics, cost summary, and aggregate statistics
- **Individual details**: Full attempt details are preserved in separate task-specific log files
- **Performance**: Dramatically reduces file I/O time and disk contention during high-concurrency runs

**Visualization plots in `plots/` directory:**
- Turn-by-turn evolution charts: `turn_{N}_{task_id}_{log_stem}.png`
- Generated by running `visualize_task_evolution.py` on log files

Each individual task log includes:
- Complete program code generated by the model
- Execution results and any errors encountered
- Both scoring metrics (binary correctness and pixel accuracy)
- Predicted vs actual output grids for comparison
- Token usage breakdown and estimated costs
- Turn usage statistics (when multi-turn is enabled)
- Full API response for detailed analysis
- Sampling parameters used (temperature, top_p, top_k, min_p)
- Complete prompt (system and user messages)

Summary reports aggregate across all tasks and include:
- Overall task solve rate and pixel accuracy across the subset
- Total costs and token usage for the entire run
- Turn usage patterns and performance comparisons

## Log File Formats

### Individual Task Log (`{timestamp}_{task_id}.json`)

**Reasoning Model Example (Gemini Flash):**
```json
{
  "task_id": "007bbfb7",
  "model": "google/gemini-2.5-flash", 
  "reasoning_effort": "low",
  "api_type": "chat_completions_independent_attempts",
  "program": "def transform(grid):\n    output_grid = [[0 for _ in range(9)] for _ in range(9)]...",
  "task_failure_reason": "",
  "timed_out": false,
  "tokens_used": 4373,
  "turns_used": 1,
  "request_cost": 0.007654,
  "raw_response": {
    "content": "The observed pattern is as follows:\nThe input grid is 3x3...",
    "reasoning": "**Examining Grid Transformations**\n\nI've been scrutinizing the input and output grid examples to discern the core transformation logic...",
    "usage": {
      "prompt_tokens": 1451,
      "completion_tokens": 2922,
      "reasoning_tokens": null
    }
  },
  "sampling_params": {
    "top_p": 0.9,
    "top_k": 50
  },
  "full_prompt": {
    "system": "You are an AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks...",
    "user": "You are an AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks..."
  },
  "score": {
    "correct": true,
    "pixel_accuracy": 1.0,
    "total_pixels": 81,
    "correct_pixels": 81,
    "error": null
  },
  "predicted_output": [[7,0,7], [0,7,0], [7,0,7]],
  "actual_output": [[7,0,7], [0,7,0], [7,0,7]]
}
```

**Non-Reasoning Model Example (gpt-4o-mini):**
```json
{
  "task_id": "6150a2bd",
  "model": "gpt-4o-mini", 
  "reasoning_effort": "N/A",
  "api_type": "responses_api",
  "program": "def transform(grid):\n    return [row[::-1] for row in grid[::-1]]",
  "task_failure_reason": "",
  "timed_out": false,
  "tokens_used": 542,
  "turns_used": 1,
  "request_cost": 0.000405,
  "raw_response": { /* Full API response */ },
  "sampling_params": {
    "top_p": 0.9,
    "top_k": 50
  },
  "full_prompt": {
    "system": "You are an AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks...",
    "user": "You are an AI assistant specialized in solving Abstract Reasoning Corpus (ARC-AGI) tasks..."
  },
  "score": {
    "correct": true,
    "pixel_accuracy": 1.0,
    "total_pixels": 9,
    "correct_pixels": 9,
    "error": null
  },
  "predicted_output": [[0,0,4], [0,8,6], [5,3,6]],
  "actual_output": [[0,0,4], [0,8,6], [5,3,6]]
}
```

### Summary Report (`{timestamp}_summary_{dataset}_{subset}.json`)

```json
{
  "timestamp": "20250703_135331",
  "dataset": "arc-agi-1",
  "subset": "shortest_10", 
  "model": "o4-mini",
  "reasoning_effort": "low",
  "api_type": "responses_api",
  "total_tasks": 10,
  "correct_tasks": 4,
  "task_accuracy": 0.4,
  "total_pixels": 90,
  "correct_pixels": 85,
  "pixel_accuracy": 0.944,
  "total_turns_used": 15,
  "avg_turns_used": 1.5,
  "total_tokens": 35847,
  "total_cost": 0.196734,
  "results": [ /* Array of individual task results */ ]
}
```

## Example Console Output

### Sequential Execution
```
Running 10 tasks from arc-agi-1/shortest_10
Model: o4-mini
API: Responses API (multi-turn, max 3 turns)
Input mode: Text-only
Parallelization: DISABLED (sequential execution)
--------------------------------------------------

Processing task: 6150a2bd
  💰 Cost: $0.019646 (input: 1089 @ $1.1, output: 4321 @ $4.4)
  ✅ Perfect solution found!
```

### Parallel Execution
```
Running 10 tasks from arc-agi-1/shortest_10
Model: o4-mini
API: Responses API (multi-turn, max 3 turns)
Input mode: Text-only
Parallelization: ENABLED (5 workers)
--------------------------------------------------
Starting parallel execution with 5 workers...
Progress: 3/10 tasks completed (30.0%)
Progress: 7/10 tasks completed (70.0%)
Progress: 10/10 tasks completed (100.0%)

Parallel execution completed. All 10 tasks processed.
```

### Summary Output

**Reasoning Model Example (o4-mini):**
```
==================================================
SUMMARY
==================================================
Dataset: arc-agi-1
Subset: shortest_10
Model: o4-mini
Reasoning effort: medium
API: Responses (multi-turn, max 3 turns)
Multi-turn enabled: True
Tasks solved correctly: 4/10 (40.0%)
Pixel accuracy: 85/90 (94.4%)
Total turns used: 15
Average turns per task: 1.5
Total tokens used: 35,847
Total cost: $0.196734
```

**Non-Reasoning Model Example (gpt-4o-mini):**
```
==================================================
SUMMARY
==================================================
Dataset: arc-agi-1
Subset: shortest_10
Model: gpt-4o-mini
API: Responses (single-shot)
Multi-turn enabled: False
Tasks solved correctly: 3/10 (30.0%)
Pixel accuracy: 78/90 (86.7%)
Total turns used: 10
Average turns per task: 1.0
Total tokens used: 12,456
Total cost: $0.009845
```

This example shows:
- **40% solve rate**: 4 out of 10 tasks solved perfectly
- **94.4% pixel accuracy**: Very close to correct solutions on average
- **Turn usage**: Model used an average of 1.5 conversation turns per task
- **Cost tracking**: Detailed token usage and cost calculation for budget management

## Analyzing Results

### Key Fields Explained

**Individual Task Logs:**
- `task_id`: ARC task identifier
- `model`: OpenAI model used (e.g., "o4-mini", "gpt-4o-mini")
- `reasoning_effort`: Reasoning effort level for reasoning models ("low", "medium", "high") or "N/A" for non-reasoning models
- `program`: Generated Python code 
- `task_failure_reason`: Reason why task failed - includes Python execution errors, "All attempts failed", "Max turns reached", "API timeout after retries", etc. (empty if successful)
- `request_cost`: Cost for this specific task in USD
- `turns_used`: Number of conversation turns used for this task
- `sampling_params`: All sampling parameters used (temperature, top_p, top_k, min_p)
- `full_prompt`: Complete system and user messages used
- `score.correct`: Boolean - whether output exactly matches expected
- `score.pixel_accuracy`: Fraction of pixels that match (0.0 to 1.0)
- `predicted_output` vs `actual_output`: Compare model's solution to ground truth

**Summary Reports:**
- `task_accuracy`: Fraction of tasks solved perfectly 
- `pixel_accuracy`: Overall pixel-level accuracy across all tasks
- `total_cost`: Total USD spent on this run
- `results[]`: Contains all individual task data for deeper analysis

### Performance Analysis

Compare different configurations:
```bash
# Find most expensive runs
grep '"total_cost"' logs/*summary*.json | sort -k2 -n
```

## Testing

Test individual components:
```bash
uv run python utils/task_loader.py  # Test task loading functionality
uv run python scoring.py      # Test scoring functionality
```

Test utils modules:
```bash
# Run all utils tests
uv run python -m pytest utils/tests/ llm-python/utils/test_scoring.py -v

# Run specific utils tests
uv run python -m pytest utils/tests/test_prompt_utils.py -v
uv run python -m pytest utils/tests/test_timeout_utils.py -v
uv run python -m pytest utils/tests/test_voting_utils.py -v
uv run python -m pytest utils/tests/test_transduction.py -v
uv run python -m pytest llm-python/utils/test_scoring.py -v
```

Quick API test:
```bash
# Test with a single task
uv run python -m llm-python.run_arc_tasks --dataset arc-agi-1 --subset shortest_1 --model gpt-4o-mini --max_turns 1
```

## Cost Management

The tool automatically tracks costs with high precision:
- **Token usage**: Input/output tokens per request with detailed breakdowns
- **Model-specific pricing**: Accurate rates for all 29 supported OpenAI models
- **Running totals**: Cumulative costs across all tasks in a session
- **6-decimal precision**: Shows costs down to $0.000001 for accurate budget tracking

**Cost accumulation**: Costs accumulate across all tasks in a session but reset between script runs.

Current pricing (as of 2025, $/1M tokens):

**Reasoning Models:**
- **o3-pro**: Input $20.00, Output $80.00
- **o3-deep-research**: Input $10.00, Output $40.00  
- **o3**: Input $2.00, Output $8.00
- **o3-mini**: Input $1.10, Output $4.40
- **o4-mini**: Input $1.10, Output $4.40
- **o4-mini-deep-research**: Input $2.00, Output $8.00
- **o1-pro**: Input $150.00, Output $600.00
- **o1**: Input $15.00, Output $60.00
- **o1-mini**: Input $1.10, Output $4.40

**GPT-4 Models:**
- **gpt-4.5-preview**: Input $75.00, Output $150.00
- **gpt-4.1**: Input $2.00, Output $8.00
- **gpt-4.1-mini**: Input $0.40, Output $1.60
- **gpt-4.1-nano**: Input $0.10, Output $0.40
- **gpt-4o**: Input $2.50, Output $10.00
- **gpt-4o-mini**: Input $0.15, Output $0.60

**Google Models:**
- **google/gemini-2.5-flash**: Input $0.30, Output $2.50
- **google/gemini** (other models): Input $0.30, Output $2.50

**Specialized Models:**
- **computer-use-preview**: Input $3.00, Output $12.00
- **codex-mini**: Input $1.50, Output $6.00

## All-Attempts Execution

The refactored system uses **all-attempts execution** with voting-based evaluation:

- **Direct prompting**: Each attempt uses the same initial prompt (no feedback)
- **All attempts executed**: Always runs all N attempts per task for statistical rigor
- **Parallel execution**: Workers process tasks independently for maximum throughput
- **Oracle metrics**: Shows upper bound potential if best attempt could be selected
- **Pass@2 voting**: Uses weighted-majority and train-majority voting for robustness
- **Transduction filtering**: Automatically detects and filters out hardcoded/cheating responses
- **Local execution**: All code is executed locally for immediate scoring

**Key Point**: We execute code locally and use voting algorithms to select the best solutions from multiple independent attempts.

## File Structure

```
llm-python/
├── run_arc_tasks_soar.py       # Main script (all-attempts, voting-based evaluation)
├── utils/                       # Utility modules
│   ├── __init__.py
│   ├── task_loader.py          # Load ARC tasks and subsets
│   ├── scoring.py              # Grid scoring and program execution (0.5s timeout)
│   ├── prompt_utils.py         # Prompt creation and code extraction
│   ├── timeout_utils.py        # Timeout handling utilities
│   ├── voting_utils.py         # Voting algorithms and prediction processing
│   ├── metrics_utils.py        # Metrics calculation and formatting
│   ├── transduction.py         # Transductive cheating detection
│   ├── test_scoring.py         # Tests for scoring utilities
│   └── tests/                  # Tests for utility modules
│       ├── __init__.py
│       ├── test_task_loader.py
│       ├── test_prompt_utils.py
│       ├── test_timeout_utils.py
│       ├── test_voting_utils.py
│       └── test_transduction.py
├── prompt_loader.py            # Load and manage prompt templates
├── generate_training_data.py   # Extract training data from logs
├── visualize_task_evolution.py # Create task evolution visualizations
├── create_simple_dataset.py    # Create simple datasets without code/reasoning
├── create_grid_size_distributed_subset.py # Create grid-size distributed subsets
├── validate_hf_dataset.py      # Validate Hugging Face datasets
├── tests/                      # Main test scripts
│   ├── test_arc_visual_with_api.py
│   ├── test_execution_diff.py
│   ├── test_generation_flow.py
│   ├── test_multiple_examples.py
│   ├── test_multiturn_reasoning.py
│   ├── test_openrouter_qwen_direct.py
│   ├── test_reasoning_persistence.py
│   ├── test_task_loader.py
│   ├── test_validation.py
│   └── results/                # Test results
├── logs/                       # Results and summaries
├── training_data/              # Generated training files
├── plots/                      # Task evolution visualizations
├── fine-tuning/                # Fine-tuning notebooks and logs
│   ├── unsloth_arc_finetuning_soar.ipynb
│   ├── generate_soar_data.ipynb
│   ├── logs/
│   └── README.md
├── prompt-strings/             # Prompt template files
│   ├── code-request/
│   ├── initial-turn/
│   ├── subsequent-turn/
│   └── system/
└── README.md                   # This file
```

## Cost Tracking

**Important**: All costs are calculated using the correct Responses API token field names (`input_tokens`/`output_tokens`) with accurate model-specific pricing rates.

## Cleanup

```bash
# Clean up old log files
uv run python cleanup_logs.py
```

## Model Support

Works with any OpenAI-compatible Chat Completions API:

- **✅ OpenAI models**: gpt-4o, gpt-4o-mini, gpt-4-turbo, etc.
- **✅ Claude (via API)**: claude-3-5-sonnet, claude-3-haiku, etc. (with --base-url)
- **✅ Local models**: Any model running with OpenAI-compatible server (vLLM, Ollama, etc.)
- **✅ Other APIs**: Qwen, DeepSeek, Gemini (with compatible endpoints)

### Setup Examples
```bash
# OpenAI (default)
uv run python -m llm-python.run_arc_tasks --model gpt-4o-mini

# Claude via compatible endpoint
uv run python -m llm-python.run_arc_tasks --model claude-3-haiku --base-url https://api.anthropic.com/v1

# Local model
uv run python -m llm-python.run_arc_tasks --model llama-3.1-8b --base-url http://localhost:8000/v1
```

## Implementation Notes

- **Execution timeout**: Program execution has a 0.5 second timeout for robust evaluation
- **Function interface**: All programs must define a `transform` function that takes a grid (2D list) and returns the transformed grid
- **Grid format**: All grids are represented as 2D lists of integers (0-9)
- **API architecture**: Uses the Chat Completions API for broad compatibility with OpenAI-compatible endpoints
- **Cost accuracy**: Uses standard prompt_tokens/completion_tokens for cost calculation
- **Pixel counting**: Fixed pixel accuracy calculation to include failed executions in totals
- **Utils organization**: Modular utility functions with comprehensive test coverage
- **Issue tracking**: Covers API failures, code extraction, execution errors, and timeouts. Does not track serialization, file I/O, or infrastructure failures
- **Adaptive sampling parameters**: Automatically detects endpoint type and applies appropriate defaults:
  - **TCP endpoints** (containing ":"): Uses `min_p=0.05` in `extra_body`
  - **Other endpoints**: Uses `top_k=50` and `top_p=0.9` defaults
- **Sampling parameter logging**: Comprehensive logging of all sampling parameters used in API calls
- **True parallelization**: Parallelizes at the attempt level for maximum efficiency - all attempts run simultaneously
- **Real-time feedback**: Displays task completion summaries as they finish for immediate progress tracking

## Additional Notes

- You can control the maximum number of turns using --max_turns (default: 3). This is especially useful for limiting cost and runaway conversations.
- **API Compatibility**: Works with any endpoint that implements OpenAI's Chat Completions API format
- **Custom Endpoints**: Use --base-url to connect to Claude, local models, or other compatible APIs
- **Parallelization**: Use `--max_workers` (1-128) to run tasks in parallel. Start with 5 workers and increase gradually while monitoring for rate limit errors. Use `--rate_limit_delay` to add delays between requests if needed.
- **Cost Control**: Parallel execution accumulates costs faster but maintains the same per-task costs. Monitor total spending especially when using expensive models like o3 with many workers.
- **Thread Safety**: All file I/O, progress tracking, and cost accumulation is thread-safe. Individual task logs use unique filenames with thread IDs to prevent conflicts.

## create_grid_size_distributed_subset.py

This script creates a new subset of ARC-AGI problems by selecting tasks from the evaluation sets of arc-agi-1 and arc-agi-2, distributed evenly by grid size. For each task, the grid size is defined as the sum of the number of cells in the first input and first output grid (from the first training example). The script selects 30 tasks from each evaluation set, spaced evenly across the range of grid sizes, and copies them into a new subset directory for balanced benchmarking.

Usage:
```
uv run o3-tools/create_grid_size_distributed_subset.py
```

The script will output a manifest of selected tasks and their grid sizes.

## Progress / Build Notes

**Levers for Improvement:**
[ ] Run some baseline performance tests on Qwen3 4B.
[ ] Use the hindsight relabelling trick.
[ ] Use the reversal trick.
[ ] Use the intermediation approach.
[ ] Ablate the costs of solving if we do o4-mini (low) versus o4-mini (high). Is it possibly better to use o4-mini (low) with 8 max turns versus o4-mini (high) with 4 max turns? Consider costs across three runs. (best to develop a script for doing this that calculates means etc.).
[ ] PROBLEM: WE ARE CHEATING BY ALLOWING THE MODEL TO CONTINUE IF THE TRAINING EXAMPLES ARE ALL CORRECT, BUT THE TEST IS WRONG. THERE'S AN ABLATION TO TEST FOR THE CASE WHERE WE - BEFORE STOPPING - ASK THE MODEL TO SEE IF IT SHOULD MAKE THE PROGRAM MORE GENERAL. Fix the cheating issue whereby there is more sampling/turns if the training examples are all correct, but the test is wrong. Only applies to feedback.
[ ] MCTS-type ablation: Sample for half of the max_turns, and then feedback for the other remaining turns (stop of course if the test is solved). Not worth it as sampling seems better than feedback.
[ ] Swap to chat completions endpoint so as to allow for openai-style endpoint usage (enable other models, incl. reasoning). THIS IS NOT GOING TO SUPPORT OPENAI REASONING MODELS, WHICH DONT' DISCLOSE THE REASONING TRACE, AND SO YOU MUST USE THE RESPONSES API TO USE REASONING WITH OPENAI MODELS. OTHERS (CLAUDE, QWEN, GEMINI?, DEEPSEEK?) RESPOND WITH <think> TAGS.
[ ] Apply a limit to oscillation within feedback roll-outs.
[ ] Put in simpler images (particularly relevant when we fine-tune because the model will know the format to expect).
[ ] Start with strict prompt, only then fall back to partial attempt. DELAY.
[ ] Use a code interpreter tool rather than running code from an extract code block.
[ ] Overfitting checks are probably needed because sometimes all training problems are solved but then the test fails. Could just rotate or do simple checks like that.
[ ] Allow the code environment to persist for a given task. [not relevant until we do the code sandbox locally.]

Cleanups:
  [x] Drop the MDL / compression calculation altogether from scripts here.
  [x] Strip out "tools calls made" etc. as there are no tool calls. There are only turns used.
  [x] Automatically support non-reasoning or reasoning models (no flags required).
  [x] Improve logging:
    [x] Manually inspect the prompt.
    [x] Inspect how wrong grids are passed back to the model (or failed runs of the code produced).
    [x] in our logging / logs, it would be best to save not just the final responses, but the ones before thta too - so I can inspect what the code output is and what is being passed back in.
  [x] Run tests on low levels of reasoning effort.

Completed:
[x] Describing grids ablation: Get the model to also describe the input grid and the output grid with code (so, return three code blocks), and provide feedback on those too. DONE AND IN A DEDICATED BRANCH.
[x] Port the scripts to an openai style endpoint. Run Qwen and try to calibrate.
[x] Review of some samples.
[x] Add guidance around output grid sizes, if wrong. (Enhanced: now tells model target dimensions upfront + general reminders)
[x] Create a script that automatically will do a run three times and calculate the mean and std dev (for the number correct on one turn, and the number correct on more than one turn).
[x] Ablate feedback of max 8 turns versus sampling for max 8 turns.
[x] Refine prompting:
  [x] Examine the correct tasks for what happened. Examine also some wrong tasks.
  [x] Adjust the soft prompt so that it encourages finding an improvement! check that. Sometimes there is no attempt to improve when some training grids pass. Perhaps try a prompt that encourages generalisation to the other training grids.
  [x] Review prompts for when a training example is solved (at least one, but not all).
  [x] Add a note that if all training examples are solved, then the program is overfitting.
[x] Try inputting images of the problem as well as just the problem itself.
[x] Test out having the model attempt a partial transformation, if it cannot determine a complete rule that solves the problem.
[x] Run on ARC AGI 1 Eval set. MIT splits. Starting with Easy, then Medium, then Hard, then Expert (if needed). Answers the question of whether refinement helps.
[x] Bringing the sandbox to be local:
  [x] Just run the code locally each time, rather than use the remote code interpreter.
    - Print, after each tool call, the result in terms of pixel match average on all training examples AND number of training examples solved out of those present.
[x] Include the test grid, it adds information.
[x] When providing code, also provide a summary of the rationale behind what is being done. (not in the reasoning). [Test this out in a clean test script to go in a tests folder.]
[x] Check whether the code sandbox on openai is ephemeral or not. Yes, with `auto` the same container is used and variables persist.
[x] prompt so that the model keeps reasoning until it finds a python program that solves (for the tool use case). don't include the test examples in the prompt.
[x] **Simplified scoring**: Removed complex compression-based calculations and focused on core metrics.

## ARC Task Runner: All-Attempts Evaluation (run_arc_tasks_soar.py)

**Refactored system** with true parallelization at the attempt level and voting-based evaluation:

- **True Parallelization:** All attempts across all tasks run simultaneously for maximum efficiency
- **Real-time Task Summaries:** Displays brief statistics for each task as it completes (test-correct, train-perfect, train-partial counts)
- **Voting Algorithms (Both Pass@2):**
  - **Weighted majority voting:** Uses pattern frequency + 1000×train_accuracy, returns top 2 patterns
  - **Train-majority voting:** Among best-training-accuracy attempts, majority vote for top 2 patterns
- **Oracle Metric:** Shows upper bound performance - if ANY attempt got test correct across all attempts
- **Transduction filtering:** Filters out hardcoded/cheating responses before voting
- **Comprehensive logging:** All attempts, full prompts, and voting decisions stored in detailed JSON logs

Key features: true parallelization at attempt level, real-time progress reporting, robust error handling, oracle upper bounds, and consistent pass@2 evaluation metrics for thorough assessment.