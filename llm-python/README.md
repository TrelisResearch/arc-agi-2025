# ARC-AGI Task Runner

A comprehensive tool for testing OpenAI-compatible language models on ARC-AGI tasks using the Chat Completions API. Supports reasoning models (o3, Gemini Flash, Qwen) with all-attempts evaluation, parallel processing, and voting-based selection.

## Key Folders and Files:

**Main Scripts:**
- `run_arc_tasks_soar.py` - **Main task runner** with all-attempts evaluation, parallel processing, and voting-based selection
- `read_log_stats.py` - **Log analysis tool** to retrospectively read and display statistics from log directories
- `create_grids_only_dataset.py` - **Grids-only dataset creator** for creating datasets with just grid data (no code or reasoning)

**Core Folders:**
- `utils/` - Core utility modules for task loading, scoring, prompts, and metrics
- `fine-tuning/` - Jupyter notebooks for fine-tuning as well as a logs folder for tensorboard logs
- `tests/` - Miscellaneous test scripts  
- `archive/` - Legacy scripts and tools moved for reference

## Features

- **OpenAI-compatible API support**: Works with OpenAI, Claude, Qwen, DeepSeek, local models, etc.
- **All-attempts evaluation**: Parallel execution with voting-based selection for robust results
- **Reasoning model support**: Optimized for o3, Gemini Flash, Qwen with configurable reasoning effort
- **Multiple datasets**: Support for ARC-AGI-1/1r/2 with various task subsets and difficulty levels

## Setup

1. Install dependencies with uv:
```bash
uv sync
```

2. Ensure you have the `.env` file with your OpenAI API key (or OpenRouter API key if using OpenRouter):
```
OPENAI_API_KEY=your_key_here
```

### UV Project Discovery Note

**Important**: When running `uv` commands from this `llm-python/` subdirectory, `uv` automatically searches upward and discovers the root `pyproject.toml` file in the repository root.

This means:
- Commands like `uv venv`, `uv sync`, and `uv run` will use the root project configuration
- The Python version requirement (`requires-python = ">=3.11,<3.13"`) from the root will be respected
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

Create a grids-only dataset for evaluation tasks:
```bash
uv run python create_grids_only_dataset.py arc-agi-1 all_evaluation --save-local
```

### All-Attempts Evaluation Mode (`run_arc_tasks_soar.py`)

**Current system** with all-attempts execution, parallel processing, and voting-based evaluation:

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
- **Direct Prompting**: Each attempt uses the same initial prompt (no feedback between attempts)
- **True Parallelization**: All attempts across all tasks run simultaneously for maximum efficiency
- **Real-time Task Summaries**: Displays brief statistics for each task as it completes (test-correct, train-perfect, train-partial counts)
- **Real-time Logging**: Individual task logs are saved immediately when each task completes, not at the end of the run
- **Health Monitoring**: Health reports every 100 attempts showing execution success rates, timeout rates, and average execution times
- **Secure Docker Execution**: All generated code runs in isolated Docker containers for security (can be disabled with `--unsafe-executor` flag for testing)
- **Voting Algorithms (Pass@2)**: 
  - **Weighted majority voting**: Uses pattern frequency + 1000×train_accuracy, returns top 2 patterns
  - **Train-majority voting**: Among best-training-accuracy attempts, majority vote for top 2 patterns
- **Oracle Metrics**: Shows upper bound potential if best attempt could be selected
- **Transduction Filtering**: Automatically detects and filters out hardcoded/cheating responses
- **Local Execution**: All code is executed locally for immediate scoring
- **Sampling Parameter Logging**: Comprehensive logging of all sampling parameters (temperature, top_p, top_k, min_p) used in API calls
- **Adaptive Sampling Parameters**: Automatic detection of endpoint type with appropriate defaults:
  - **TCP endpoints** (containing ":"): Uses `min_p=0.05` in `extra_body`
  - **Other endpoints**: Uses `top_k=50` and `top_p=0.9` defaults
- **Optimized File I/O**: 30-second timeout for file operations with detailed error logging for debugging
- **Independent Multiple Runs**: Each repeated run is completely isolated with no shared state - results loaded from files for aggregation
- **Robust State Management**: Explicit garbage collection and thread-safe cleanup between runs prevents state spillover and race conditions

**When to use:**
- For comprehensive evaluation with statistical rigor
- When you want oracle upper bounds and pass@k metrics
- For maximum parallelization efficiency with high worker counts
- For systematic comparison of multiple attempts per task
- When you need real-time progress updates as tasks complete

### Grids-Only Dataset Creation (`create_grids_only_dataset.py`)

The `create_grids_only_dataset.py` script creates datasets containing only grid data without any generated code or reasoning:

```bash
# Create grids-only dataset for evaluation tasks
uv run python create_grids_only_dataset.py arc-agi-1 all_evaluation --save-local

# Create grids-only dataset for training tasks
uv run python create_grids_only_dataset.py arc-agi-1 all_training --save-local

# Push to Hugging Face (default behavior)
uv run python create_grids_only_dataset.py arc-agi-2 shortest_evaluation_30

# Create with validation split
uv run python create_grids_only_dataset.py arc-agi-1 all_training --validation
```

**Features:**
- **Grid Data Only**: Contains `train_input`, `train_output`, `test_input`, `test_output` fields
- **No Code/Reasoning**: `code` and `reasoning` fields are empty
- **No Predictions**: `predicted_*` fields are empty lists
- **No Correctness Flags**: All `correct_*` fields are False
- **Hugging Face Integration**: Automatically pushes to HF Hub with descriptive naming
- **Validation Splits**: Optional train/validation split creation

**Use Cases:**
- Creating clean datasets for fine-tuning without generated content
- Extracting raw ARC task data for analysis
- Preparing datasets for models that don't need code/reasoning examples

### Log Analysis (`read_log_stats.py`)

The `read_log_stats.py` script provides retrospective analysis of completed runs:

```bash
# Analyze a single run directory
uv run python read_log_stats.py logs/20250728_114716

# Analyze multiple runs (for repeated run experiments)
uv run python read_log_stats.py logs/20250728_113731 logs/20250728_114716 logs/20250728_115648

# Auto-discover all runs from a specific date
uv run python read_log_stats.py --pattern 20250728

# Show verbose output with file details
uv run python read_log_stats.py logs/20250728_114716 --verbose
```

**Features:**
- **Single Run Analysis**: Detailed statistics for individual runs
- **Multi-Run Aggregation**: Mean, standard deviation, and confidence intervals across multiple runs
- **Auto-Discovery**: Find all runs from a specific date pattern
- **Comprehensive Metrics**: All core metrics including Pass@2, Oracle, and error rates
- **Cost Analysis**: Token usage and cost breakdowns

### Health Monitoring

The tool automatically monitors execution health during long runs and displays periodic reports:

```bash
🏥 Health [100 attempts]: Success 78% | Timeout 5% | ExecErr 17% | AvgTime 0.31s
```

**Health Metrics Explained:**
- **Success %**: Programs that execute without timeout/error
- **Timeout %**: Programs that exceed the 0.5s execution limit  
- **ExecErr %**: Programs with runtime errors
- **AvgTime**: Average total time per attempt (API + execution + processing)

**Resource Management**: Docker containers are automatically managed on a per-execution basis for optimal performance and security isolation.

### Security and Execution Safety

By default, the tool runs all generated code in **isolated Docker containers** for maximum security. This prevents generated code from:
- Accessing or modifying your files
- Making unauthorized network requests  
- Executing system commands
- Stealing environment variables or API keys

#### Unsafe Executor Mode (`--unsafe-executor`)

For testing purposes in isolated environments, you can disable Docker sandboxing with the `--unsafe-executor` flag. **This is dangerous and should only be used when:**

✅ **Safe to use:**
- Testing in disposable VMs or containers
- Running in isolated development environments
- Debugging executor issues when Docker is problematic

❌ **Never use in production or on your main system:**
- Generated code runs directly on your system
- Full access to your filesystem and network
- Potential for data theft or system compromise
- No protection against malicious code execution

**Warning signs when using `--unsafe-executor`:**
```bash
⚠️  WARNING: Using unrestricted executor - generated code will run directly on your system!
Executor: unrestricted (timeout: 0.5s) ⚠️  UNSAFE MODE
```

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

# Use unrestricted executor (UNSAFE - for testing only in isolated environments)
uv run python run_arc_tasks_soar.py --dataset arc-agi-1 --subset shortest_training_10 --unsafe-executor

# Available options:
#   --dataset: arc-agi-1, arc-agi-1r, or arc-agi-2
#   --subset: shortest_training_1, shortest_training_10, shortest_training_30, shortest_evaluation_1, shortest_evaluation_10, shortest_evaluation_30, etc.
#   --model: Model name (default: gpt-4.1-mini)
#   --base-url: Custom API endpoint URL (default: OpenAI) - enables Claude, Qwen, local models, etc.
#   --reasoning_effort: Reasoning effort level: low (2k tokens), medium (8k tokens), high (32k tokens) - for Gemini; other models may vary
#   --max-tokens: Maximum tokens for model responses (overrides reasoning effort defaults)
#   --limit: Limit number of tasks to run
#   --max_attempts: Maximum number of attempts per task (default: 8)
#   --max_workers: Number of parallel workers (default: 1, efficient up to 50+)
#   --repeat-runs: Number of times to repeat the entire test (default: 1, max: 10)
#   --qwen-no-think: Disable thinking for Qwen models (sets enable_thinking=false in chat_template_kwargs)
#   --unsafe-executor: ⚠️ UNSAFE: Use unrestricted executor (no Docker sandboxing) - SECURITY RISK!
#   --prompt_version: Version of prompts to use (default: soar)
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


## File Structure

```
llm-python/
├── run_arc_tasks_soar.py       # Main script (all-attempts, voting-based evaluation)
├── read_log_stats.py           # Log analysis tool for retrospective statistics
├── utils/                       # Utility modules
│   ├── __init__.py
│   ├── task_loader.py          # Load ARC tasks and subsets
│   ├── scoring.py              # Grid scoring and program execution (0.5s timeout)
│   ├── prompt_utils.py         # Prompt creation and code extraction
│   ├── prompt_loader.py        # Load and manage prompt templates
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
├── generate_training_data.py   # Extract training data from logs
├── create_grids_only_dataset.py # Create grids-only datasets (no code/reasoning)
├── validate_hf_dataset.py      # Validate Hugging Face datasets
├── experiment_notes.md         # Development notes and experiments
├── archive/                    # Legacy scripts (moved for reference)
│   ├── run_arc_tasks.py        # Original task runner (deprecated)
│   ├── visualize_task_evolution.py # Task evolution visualizations
│   ├── create_simple_dataset.py # Simple dataset creation
│   ├── create_grid_size_distributed_subset.py # Grid-size distributed subsets
│   ├── analyze_pattern_learning.py # Pattern analysis tools
│   ├── test_openrouter_direct.py # OpenRouter testing
│   └── [other archived tools]  # Various development and analysis scripts
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
├── debug_images/               # Debug visualization outputs
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

