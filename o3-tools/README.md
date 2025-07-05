# o3-tools

A tool for testing OpenAI o3/o4 models on ARC-AGI tasks with and without code interpreter tools.

>[!TIP]
> This branch contains an approach where the remote code interpreter is used to solve the tasks.

**Todo**
[x] Reorganize data so that training and evaluation problems are split, because they are so different in terms of difficulty.

**Measuring Performance:**
Objective: Define a test that is a representative measure of performance while also being fast to run.

**Levers:**
- FAST:
[x] Include the test grid, it adds information.
[x] When providing code, also provide a summary of the rationale behind what is being done. (not in the reasoning). [Test this out in a clean test script to go in a tests folder.]
[x] Check whether the code sandbox on openai is ephemeral or not. Yes, with `auto` the same container is used and variables persist.

- MEDIUM:
[ ] Bringing the sandbox to be local:
  [ ] Describe the code interpreter as a tool that the openai resopnese api can use.
  - When the tool is called, the code interpreter output / results need to be passed back as a tool response. We'll write a small test script to go in the tests folder to check this out. and test it with gpt-4.1-nano.
  - We'll need to update code for how loggings works and to apply a max number of turns. In fact we'll need to handle looping of the code ourselves, until "Final Answer: " is present.
  [ ] Stop looping at thresholds:
    - Stop after 4 | 16 tool calls if no valid grid sizes created.
    - Stop after 8 | 32 tool calls if no training problems solved.
[ ] Build a priority list based on # (or percentage) of training grids solved.

[ ] Try inputting images of the problem as well as just the problem itself.

- SLOW:
[ ] Add pixel accuracy as a priority list metric (e.g. metric = f(pixel accuracy, training problems solved)).
    - Add f(..., gzip) as a metric.

Other ideas:
[ ] Overfitting checks are probably needed because sometimes all training problems are solved but then the test fails. Could just rotate or do simple checks like that.
[ ] Allow the code environment to persist for a given task. [not relevant until we do the code sandbox locally.]

Completed:
[x] prompt so that the model keeps reasoning until it finds a python program that solves (for the tool use case). don't include the test examples in the prompt.
[x] Use gzip for the program too (strip comments), possibly this removes the need for having the alpha and beta parameters. Also, make sure we're including all train examples for that task in the data description length.
[x] **Fixed residual calculation**: Changed from broken "actual-value-if-wrong" to proper difference-based approach (`residual = actual - predicted`) that allows perfect reconstruction.
[x] **Replaced MDL with residual reduction**: Changed from complex MDL (program + residuals) to pure pattern learning measurement using residual reduction percentage.

For Kaggle / low compute competition:
[ ] Testing out a baseline with Qwen.
[ ] Potentially distilling from o3 down to Qwen if needed.

## Features

- Run ARC-AGI tasks with OpenAI models (currently using gpt-4o-mini or o4-mini)
- Support for code interpreter tools via function calling
- Comprehensive scoring including pixel accuracy and residual reduction pattern learning
- Budget tracking with token usage and cost estimation
- Detailed logging of all runs for analysis
- Support for different datasets and task subsets

> **Note:** In testing, o3 looped (with tools enabled) does not solve any of the longest ARC-AGI problems (tested on 5). Shortest and medium tasks are solved much more reliably.

## Setup

1. Install dependencies with uv:
```bash
uv sync
```

2. Ensure you have the `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_key_here
```

## Usage

### Basic Usage

Run the shortest task from ARC-AGI-1:
```bash
uv run python o3-tools/run_arc_tasks.py
```

### Advanced Usage

```bash
# Run 10 shortest training tasks from ARC-AGI-2 with tools enabled
uv run python run_arc_tasks.py --dataset arc-agi-2 --subset shortest_training_1 --tools

# Run 30 shortest evaluation tasks from ARC-AGI-1 with model selection and a limit of 5
uv run python run_arc_tasks.py --dataset arc-agi-1 --subset shortest_evaluation_30 --model gpt-4.1-mini --limit 5

# Run tasks in parallel with 10 workers for faster execution
uv run python run_arc_tasks.py --dataset arc-agi-2 --subset shortest_training_30 --max_workers 10

# Run tasks in parallel with rate limiting to respect API limits
uv run python run_arc_tasks.py --dataset arc-agi-1 --subset shortest_evaluation_10 --max_workers 5 --rate_limit_delay 0.5

# Available options:
#   --dataset: arc-agi-1 or arc-agi-2
#   --subset: shortest_training_1, shortest_training_10, shortest_training_30, shortest_evaluation_1, shortest_evaluation_10, shortest_evaluation_30, etc.
#   --model: OpenAI model name (default: gpt-4.1-nano)
#   --tools: Enable code interpreter tools
#   --limit: Limit number of tasks to run
#   --max_tool_calls: Maximum number of tool calls allowed for the model (default: 64, only applies if --tools is set)
#   --reasoning_effort: Reasoning effort for the model (low, medium, high; default: medium, only applies to o3/o4/o1 models)
#   --max_workers: Number of parallel workers (default: 1, max: 30)
#   --rate_limit_delay: Delay between API calls in seconds (default: 0.0)
```

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

Run the 10 shortest evaluation tasks from ARC-AGI-2 with tools enabled:
```bash
uv run python o3-tools/run_arc_tasks.py --dataset arc-agi-2 --subset shortest_evaluation_10 --tools
```

Run the 30 longest training tasks from ARC-AGI-1:
```bash
uv run python o3-tools/run_arc_tasks.py --dataset arc-agi-1 --subset longest_training_30 --model o3
```

Run 30 grid size distributed evaluation tasks from ARC-AGI-2:
```bash
uv run python o3-tools/run_arc_tasks.py --dataset arc-agi-2 --subset grid_size_distributed_30_evaluation --tools
```

Run 30 tasks in parallel with 15 workers for 15x speedup:
```bash
uv run python o3-tools/run_arc_tasks.py --dataset arc-agi-2 --subset shortest_training_30 --max_workers 15
```

## Parallelization

The tool supports parallel execution to dramatically reduce wall-clock time for large task sets:

### Key Features

- **Thread-based parallelization**: Up to 30 concurrent workers
- **Thread-safe execution**: Safe cost tracking, progress reporting, and file I/O
- **Rate limiting**: Optional delays to respect API rate limits
- **Progress tracking**: Real-time progress updates for parallel execution
- **Robust error handling**: Individual task failures don't crash the entire batch

### Usage Examples

```bash
# Run 10 tasks with 5 workers (2x speedup)
uv run python run_arc_tasks.py --max_workers 5

# Run 30 tasks with maximum parallelization (up to 30x speedup)
uv run python run_arc_tasks.py --subset shortest_training_30 --max_workers 30

# Run with rate limiting to avoid hitting API limits
uv run python run_arc_tasks.py --max_workers 10 --rate_limit_delay 0.2

# Conservative parallel execution with 5 workers and 0.5s delay
uv run python run_arc_tasks.py --max_workers 5 --rate_limit_delay 0.5
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

## Scoring Metrics

The tool provides three complementary scoring metrics to evaluate model performance:

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

### 3. **Residual Reduction (Pattern Learning Score)**

- **What it measures**: How much of the transformation pattern the program learned compared to no program at all
- **Formula**: `reduction = (null_residual_bytes - program_residual_bytes) / null_residual_bytes`
- **Core Principle**: Measures the **pattern learning ability** by comparing residuals
  - **Program Residuals**: Compressed differences for ALL training examples that the program failed to capture
  - **Null Baseline**: Compressed differences for null program (predicts grid of zeros with correct output dimensions)
- **Training-Based Calculation**: Uses only training examples to measure learning
- **Components**:
  - `program_residual_bytes`: Gzip-compressed size of residuals from program execution on training
  - `null_residual_bytes`: Gzip-compressed size of residuals from null program (all zeros) on training
  - `residual_reduction`: Percentage improvement over null baseline (0.0 to 1.0)
  - `pattern_learning_score`: Same as residual_reduction × 100 (0% to 100%)
  - `training_executions`: Number of training examples the program executed successfully
  - `training_correct`: Number of training examples the program got exactly right
- **Benefits of residual reduction**:
  - ✅ **Pure pattern learning**: Measures only how well the program learned the transformation
  - ✅ **No complexity bias**: Doesn't penalize longer but more accurate programs
  - ✅ **Intuitive scale**: 0% = no learning, 100% = perfect pattern learning
  - ✅ **Baseline comparison**: Always compared to outputting all zeros (meaningful null baseline)
- **Interpretation**: Higher scores are better (0% = no better than all zeros, 100% = perfect pattern learning)
- **Example**: A program with 35 residual bytes vs null baseline 150 bytes → 76.7% pattern learning

### **Why These Metrics Matter**

- **Binary Correctness**: Shows the "solve rate" - what percentage of tasks are completely correct
- **Pixel Accuracy**: Reveals how close imperfect solutions are to being correct  
- **Pattern Learning**: Measures how well the program learned the transformation pattern, independent of code complexity

**Example Results Interpretation**:
```
Tasks solved correctly: 4/10 (40.0%)     # 40% perfect solutions
Pixel accuracy: 85/90 (94.4%)            # Very close to correct on average
Average pattern learning: 76.7%          # Programs learned most of the patterns
```

This shows a model that writes mostly-correct solutions with strong pattern learning ability.

## Output

Results are saved in the `logs/` directory:

- Individual task results: `{timestamp}_{task_id}.json`
- Summary reports: `{timestamp}_summary_{dataset}_{subset}.json`

Each individual task log includes:
- Complete program code generated by the model
- Execution results and any errors encountered
- All three scoring metrics (binary correctness, pixel accuracy, pattern learning)
- Predicted vs actual output grids for comparison
- Token usage breakdown and estimated costs
- Tool usage statistics (when tools are enabled)
- Full API response for detailed analysis

Summary reports aggregate across all tasks and include:
- Overall task solve rate and pixel accuracy across the subset
- Average pattern learning scores, training success rates, and residual analysis
- Total costs and token usage for the entire run
- Tool usage patterns and performance comparisons

## Log File Formats

### Individual Task Log (`{timestamp}_{task_id}.json`)

```json
{
  "task_id": "6150a2bd",
  "model": "o4-mini", 
  "use_tools": true,
  "api_type": "responses_api",
  "program": "def transform(grid):\n    return [row[::-1] for row in grid[::-1]]",
  "execution_error": "",
  "timed_out": false,
  "tokens_used": 1189,
  "tool_calls_count": 1,
  "request_cost": 0.004146,
  "raw_response": { /* Full API response */ },
  "score": {
    "correct": true,
    "pixel_accuracy": 1.0,
    "total_pixels": 9,
    "correct_pixels": 9,
    "error": null
  },
  "residual_reduction": {
    "program_residual_bytes": 35,
    "null_residual_bytes": 150,
    "residual_reduction": 0.767,
    "pattern_learning_score": 76.7,
    "training_examples_count": 3,
    "training_successes": 3,
    "training_errors": []
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
  "use_tools": true,
  "api_type": "responses_api",
  "total_tasks": 10,
  "correct_tasks": 4,
  "task_accuracy": 0.4,
  "total_pixels": 90,
  "correct_pixels": 85,
  "pixel_accuracy": 0.944,
  "avg_pattern_learning_score": 76.7,
  "avg_program_residual_bytes": 42.5,
  "avg_null_residual_bytes": 135.8,
  "training_success_rate": 0.933,
  "good_pattern_learners": 7,
  "excellent_pattern_learners": 4,
  "total_tool_calls": 12,
  "avg_tool_calls": 1.2,
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
API: Responses API (single-shot)
Tools: ENABLED (code interpreter - OpenAI runs code internally, model can iterate)
Parallelization: DISABLED (sequential execution)
--------------------------------------------------

Processing task: 6150a2bd
  💰 Cost: $0.019646 (input: 1089 @ $1.1, output: 4321 @ $4.4)
  🧠 Pattern learning: 76.7% (35 vs 150 bytes)
  ✅ Program executed on all 3 training examples
  🎯 Training accuracy: 3/3 (100%) correct outputs
  ✅ Perfect solution found!
```

### Parallel Execution
```
Running 10 tasks from arc-agi-1/shortest_10
Model: o4-mini
API: Responses API (single-shot)
Tools: ENABLED (code interpreter - OpenAI runs code internally, model can iterate)
Parallelization: ENABLED (5 workers)
--------------------------------------------------
Starting parallel execution with 5 workers...
Progress: 3/10 tasks completed (30.0%)
Progress: 7/10 tasks completed (70.0%)
Progress: 10/10 tasks completed (100.0%)

Parallel execution completed. All 10 tasks processed.
```

### Summary Output
```
==================================================
SUMMARY
==================================================
Dataset: arc-agi-1
Subset: shortest_10
Model: o4-mini
API: Responses (single-shot)
Tools enabled: True
Tasks solved correctly: 4/10 (40.0%)
Pixel accuracy: 85/90 (94.4%)
Average pattern learning: 76.7%
Training success rate: 93.3% (28/30)
Programs with >50% pattern learning: 7/10
Programs with >80% pattern learning: 4/10
Average program residual: 42.5 bytes
Average null baseline: 135.8 bytes
Total tool calls made: 12
Average tool calls per task: 1.2
Total tokens used: 35,847
Total cost: $0.196734
```

This example shows:
- **40% solve rate**: 4 out of 10 tasks solved perfectly
- **94.4% pixel accuracy**: Very close to correct solutions on average
- **76.7% pattern learning**: Programs learned most of the transformation patterns
- **93.3% training success**: Programs executed successfully on training examples
- **70% good learners**: 7 out of 10 programs showed >50% pattern learning
- **Tool usage**: Model used code interpreter 1.2 times per task on average
- **Cost tracking**: Detailed token usage and cost calculation for budget management

## Analyzing Results

### Key Fields Explained

**Individual Task Logs:**
- `task_id`: ARC task identifier
- `program`: Generated Python code 
- `execution_error`: Any runtime errors (empty if successful)
- `request_cost`: Cost for this specific task in USD
- `tool_calls_count`: Number of code interpreter calls made
- `score.correct`: Boolean - whether output exactly matches expected
- `score.pixel_accuracy`: Fraction of pixels that match (0.0 to 1.0)
- `residual_reduction.pattern_learning_score`: Percentage pattern learning (0-100%)
- `residual_reduction.training_successes`: Number of training examples that executed successfully
- `predicted_output` vs `actual_output`: Compare model's solution to ground truth

**Summary Reports:**
- `task_accuracy`: Fraction of tasks solved perfectly 
- `pixel_accuracy`: Overall pixel-level accuracy across all tasks
- `avg_pattern_learning_score`: Average pattern learning percentage (higher = better)
- `training_success_rate`: Fraction of training examples that executed successfully
- `good_pattern_learners`: Number of programs with >50% pattern learning
- `total_cost`: Total USD spent on this run
- `results[]`: Contains all individual task data for deeper analysis

### Performance Analysis

Compare different configurations:
```bash
# Compare tools vs no-tools performance
grep '"use_tools": true' logs/*summary*.json | grep task_accuracy
grep '"use_tools": false' logs/*summary*.json | grep task_accuracy

# Find most expensive runs
grep '"total_cost"' logs/*summary*.json | sort -k2 -n
```

## Testing

Test individual components:
```bash
uv run python task_loader.py  # Test task loading functionality
uv run python scoring.py      # Test scoring and MDL calculation
```

Quick API test:
```bash
# Test with a single task (fast and cheap)
uv run python run_arc_tasks.py --dataset arc-agi-1 --subset shortest_1 --model gpt-4o-mini

# Test with tools enabled (slower but more capable)
uv run python run_arc_tasks.py --dataset arc-agi-1 --subset shortest_1 --model gpt-4o-mini --tools
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

**Specialized Models:**
- **computer-use-preview**: Input $3.00, Output $12.00
- **codex-mini**: Input $1.50, Output $6.00

## Important: Tool Behavior

We use **only the Responses API** with two modes:

**With `--tools` enabled**:
- OpenAI's built-in code interpreter runs internally
- Model can execute code, see results, and iterate within the same API call
- Model has access to a live Python environment and can debug interactively
- More expensive but potentially more accurate

**Without `--tools` (default)**:
- Model outputs final code as text
- We extract and execute the code locally using subprocess
- Model cannot see execution results or iterate
- Less expensive but requires model to write correct code in one shot

**Key Point**: In both cases, we execute the final code locally to score it. The difference is whether the model gets to use a code interpreter during problem-solving.

## File Structure

```
o3-tools/
├── run_arc_tasks.py             # Main script (Responses API only)
├── task_loader.py               # Load ARC tasks and subsets
├── scoring.py                   # Grid scoring and MDL calculation
├── cleanup_logs.py             # Clean up log files
├── logs/                       # Results and summaries
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

- **✅ o4-mini**: Reasoning model, higher cost but better performance
- **✅ o3**: Most powerful reasoning model (expensive!)
- **✅ gpt-4o-mini**: Fast, cost-effective baseline
- **✅ o3-mini**: Available for testing

### Model-Specific Notes
- **o3/o4 models**: Use `max_completion_tokens` instead of `max_tokens`
- **o3/o4 models**: Don't support `temperature=0`, use default
- **Cost difference**: o4-mini is ~7x more expensive than gpt-4o-mini

## Implementation Notes

- **Execution timeout**: Program execution has a 0.1 second timeout as specified in requirements
- **Function interface**: All programs must define a `transform` function that takes a grid (2D list) and returns the transformed grid
- **Grid format**: All grids are represented as 2D lists of integers (0-9)
- **API architecture**: Uses only the Responses API - Chat Completions API has been removed
- **Cost accuracy**: Fixed cost calculation to use correct Responses API field names
- **Pixel counting**: Fixed pixel accuracy calculation to include failed executions in totals

## Additional Notes

- You can control the maximum number of tool calls the model can make per task using --max_tool_calls (default: 64). This is especially useful for limiting cost and runaway tool loops when --tools is enabled.
- You can also set the reasoning effort for the model using --reasoning_effort (choices: low, medium, high; default: medium). This may affect the model's thoroughness and cost.
- **Parallelization**: Use `--max_workers` (1-30) to run tasks in parallel. Start with 5 workers and increase gradually while monitoring for rate limit errors. Use `--rate_limit_delay` to add delays between requests if needed.
- **Cost Control**: Parallel execution accumulates costs faster but maintains the same per-task costs. Monitor total spending especially when using expensive models like o3 with many workers.
- **Thread Safety**: All file I/O, progress tracking, and cost accumulation is thread-safe. Individual task logs use unique filenames with thread IDs to prevent conflicts.

## create_grid_size_distributed_subset.py

This script creates a new subset of ARC-AGI problems by selecting tasks from the evaluation sets of arc-agi-1 and arc-agi-2, distributed evenly by grid size. For each task, the grid size is defined as the sum of the number of cells in the first input and first output grid (from the first training example). The script selects 30 tasks from each evaluation set, spaced evenly across the range of grid sizes, and copies them into a new subset directory for balanced benchmarking.

Usage:
```
uv run o3-tools/create_grid_size_distributed_subset.py
```

The script will output a manifest of selected tasks and their grid sizes.