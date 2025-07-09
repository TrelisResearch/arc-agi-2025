# o3-tools

A tool for testing OpenAI o3/o4 models on ARC-AGI tasks with and without code interpreter tools.

**Todo**
[x] Reorganize data so that training and evaluation problems are split, because they are so different in terms of difficulty.

**Measuring Performance:**
Objective: Define a test that is a representative measure of performance while also being fast to run.

**Levers:**
- FAST:
[x] Test out having the model attempt a partial transformation, if it cannot determine a complete rule that solves the problem.
[x] Run on ARC AGI 1 Eval set. MIT splits. Starting with Easy, then Medium, then Hard, then Expert (if needed). Answers the question of whether refinement helps.

- MEDIUM:
[x] Refine prompting:
  [x] Examine the correct tasks for what happened. Examine also some wrong tasks.
  [x] Adjust the soft prompt so that it encourages finding an improvement! check that. Sometimes there is no attempt to improve when some training grids pass. Perhaps try a prompt that encourages generalisation to the other training grids.
  [x] Review prompts for when a training example is solved (at least one, but not all).
  [x] Add a note that if all training examples are solved, then the program is overfitting.
[ ] Try inputting images of the problem as well as just the problem itself.
[ ] Try sampling on improved examples. Potentially building a priority list.
[ ] Build a priority list based on # (or percentage) of training grids solved. Ideally you have an id and converstaion history for each candidate incomplete program (so you can reuse LLM cache).

- SLOW:
[ ] Add pixel accuracy as a priority list metric (e.g. metric = f(pixel accuracy, training problems solved)).
    - Add f(..., gzip) as a metric.

Other ideas:
[ ] Put in simpler images (particularly relevant when we fine-tune because the model will know the format to expect).
[ ] Add guidance around output grid sizes, if wrong.
[ ] Start with strict prompt, only then fall back to partial attempt. DELAY.
[ ] Swap to chat completions endpoint so as to allow for openai-style endpoint usage (enable other models, incl. reasoning). THIS IS NOT GOING TO SUPPORT OPENAI REASONING MODELS, WHICH DONT' DISCLOSE THE REASONING TRACE, AND SO YOU MUST USE THE RESPONSES API TO USE REASONING WITH OPENAI MODELS. OTHERS (CLAUDE, QWEN, GEMINI?, DEEPSEEK?) RESPOND WITH <think> TAGS.
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
[x] Bringing the sandbox to be local:
  [x] Just run the code locally each time, rather than use the remote code interpreter.
    - Print, after each tool call, the result in terms of pixel match average on all training examples AND number of training examples solved out of those present.
[x] Include the test grid, it adds information.
[x] When providing code, also provide a summary of the rationale behind what is being done. (not in the reasoning). [Test this out in a clean test script to go in a tests folder.]
[x] Check whether the code sandbox on openai is ephemeral or not. Yes, with `auto` the same container is used and variables persist.
[x] prompt so that the model keeps reasoning until it finds a python program that solves (for the tool use case). don't include the test examples in the prompt.
[x] **Simplified scoring**: Removed complex compression-based calculations and focused on core metrics.

For Kaggle / low compute competition:
[ ] Testing out a baseline with Qwen.
[ ] Potentially distilling from o3 down to Qwen if needed.

## Features

- Run ARC-AGI tasks with OpenAI models (currently using gpt-4o-mini or o4-mini)
- Support for reasoning models (o3, o4, o1)
- Multi-turn execution with training examples feedback
- **Robust timeout handling** with automatic retries (300s timeout, 3 attempts per turn)
- Comprehensive scoring including pixel accuracy and binary correctness
- Budget tracking with token usage and cost estimation
- Detailed logging of all runs for analysis
- Support for different datasets and task subsets

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

## Usage

### Basic Usage

Run the shortest task from ARC-AGI-1:
```bash
uv run python o3-tools/run_arc_tasks.py
```

### Advanced Usage

```bash
# Run 10 shortest training tasks from ARC-AGI-2 with multi-turn execution enabled
uv run python run_arc_tasks.py --dataset arc-agi-2 --subset shortest_training_1

# Run with custom max turns for multi-turn execution  
uv run python run_arc_tasks.py --dataset arc-agi-1 --subset shortest_training_1 --max_turns 5

# Run 30 shortest evaluation tasks from ARC-AGI-1 with model selection and a limit of 5
uv run python run_arc_tasks.py --dataset arc-agi-1 --subset shortest_evaluation_30 --model gpt-4.1-mini --limit 5

# Run tasks in parallel with 10 workers for faster execution
uv run python run_arc_tasks.py --dataset arc-agi-2 --subset shortest_training_30 --max_workers 10

# Run tasks in parallel with rate limiting to respect API limits
uv run python run_arc_tasks.py --dataset arc-agi-1 --subset shortest_evaluation_10 --max_workers 5 --rate_limit_delay 0.5

# Available options:
#   --dataset: arc-agi-1 or arc-agi-2
#   --subset: shortest_training_1, shortest_training_10, shortest_training_30, shortest_evaluation_1, shortest_evaluation_10, shortest_evaluation_30, etc.
#   --model: OpenAI model name (default: gpt-4.1-mini)
#   --limit: Limit number of tasks to run
#   --max_turns: Maximum number of turns for multi-turn execution (default: 3)
#   --reasoning_effort: Reasoning effort for the model (low, medium, high; default: low, only applies to o3/o4/o1 models)
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

## Debug Images for Vision Models

When using vision-capable models (o3, o4, gpt-4o variants), the script automatically generates debug images alongside the regular text-based prompts. This provides visual feedback to help understand model performance.

### Features

- **Automatic detection**: Vision capabilities are auto-detected based on model name
- **Training images**: Visual representation of all training examples with input→output transformations
- **Feedback images**: Side-by-side comparison of expected vs predicted outputs after each turn
- **ARC color palette**: Proper 0-9 color mapping matching official ARC visualization
- **Clear labeling**: Arrows, dimensions, and accuracy indicators for easy interpretation

### Debug Image Generation

**Training Phase (Turn 1):**
- Shows all training examples as input→output pairs
- Includes grid dimensions and clear visual arrows
- Saved as `debug_images/[timestamp]_[task_id]_turn1_training.png`

**Feedback Phase (Turn 2+):**
- Compares expected outputs with model predictions
- Shows accuracy percentages and pixel-level differences
- Color-coded: green borders for correct, red for incorrect
- Saved as `debug_images/[timestamp]_[task_id]_turn[N]_feedback.png`

### Model Support

**Vision-enabled models:**
- `o3` (full OpenAI o3)
- `o4` and `o4-mini` 
- `gpt-4o`, `gpt-4o-mini`
- `gpt-4-vision-preview`

**Text-only models:**
- `o3-mini` (reasoning without vision)
- `gpt-4`, `gpt-3.5-turbo`
- All other non-vision models

### Usage

Debug images are automatically generated when using vision models:

```bash
# This will generate debug images (o4 supports vision)
uv run python run_arc_tasks.py --model o4-mini

# This will NOT generate debug images (o3-mini is text-only)
uv run python run_arc_tasks.py --model o3-mini
```

**Debug images are saved to:**
```
o3-tools/debug_images/
├── 20250109_143022_abc123_turn1_training.png
├── 20250109_143035_abc123_turn2_feedback.png
└── 20250109_143048_abc123_turn3_feedback.png
```

### Example Debug Images

**Training Image (Turn 1):**
- Visual grid representations of all training examples
- Input grids on the left, output grids on the right
- Arrows showing transformations
- Grid dimensions displayed clearly

**Feedback Image (Turn 2+):**
- Three columns: Input | Expected | Predicted
- Visual comparison of model predictions vs ground truth
- Accuracy percentages for each example
- Color-coded borders indicating success/failure

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

## Timeout Handling

The tool includes robust timeout handling to prevent hanging on API calls that take too long:

### Key Features

- **300-second timeout** per API call (5 minutes)
- **3 retry attempts** per turn with 2-second backoff between retries
- **Separate timeout failure tracking** - doesn't count as regular task failures
- **Complete conversation preservation** during retries
- **Detailed timeout logging** with clear failure reasons

### How It Works

For each turn in a multi-turn conversation:

1. **Initial attempt**: API call with 300-second timeout
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
  ⏰ Turn 2 attempt 1 timed out, retrying in 2s...
  🔄 Turn 2 attempt 2/3...
  ✅ Turn 2 successful on attempt 2

# Complete timeout failure
Turn 3/8...
  🔄 Turn 3 attempt 1/3...
  ⏰ Turn 3 attempt 1 timed out, retrying in 2s...
  🔄 Turn 3 attempt 2/3...
  ⏰ Turn 3 attempt 2 timed out, retrying in 2s...
  🔄 Turn 3 attempt 3/3...
  ❌ Turn 3 failed after 3 attempts: TimeoutError
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

### Why 300 Seconds?

The timeout is set to 300 seconds (5 minutes) because:
- **Medium reasoning effort** can take 2-4 minutes for complex tasks
- **High reasoning effort** can take 4-8 minutes for difficult tasks
- **Buffer time** accounts for network latency and API processing
- **Balance** between patience and preventing indefinite hangs

### Log File Structure

Timeout failures are logged with special metadata:

```json
{
  "task_id": "example_task",
  "api_success": false,
  "timeout_failure": true,
  "execution_error": "API timeout after retries",
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

## Output

Results are saved in multiple directories:

**Log files in `logs/` directory:**
- Individual task results: `{timestamp}_{task_id}.json`
- Summary reports: `{timestamp}_summary_{dataset}_{subset}.json`

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

Summary reports aggregate across all tasks and include:
- Overall task solve rate and pixel accuracy across the subset
- Total costs and token usage for the entire run
- Turn usage patterns and performance comparisons

## Log File Formats

### Individual Task Log (`{timestamp}_{task_id}.json`)

**Reasoning Model Example (o4-mini):**
```json
{
  "task_id": "6150a2bd",
  "model": "o4-mini", 
  "reasoning_effort": "medium",
  "api_type": "responses_api",
  "program": "def transform(grid):\n    return [row[::-1] for row in grid[::-1]]",
  "execution_error": "",
  "timed_out": false,
  "tokens_used": 1189,
  "turns_used": 2,
  "request_cost": 0.004146,
  "raw_response": { /* Full API response */ },
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

**Non-Reasoning Model Example (gpt-4o-mini):**
```json
{
  "task_id": "6150a2bd",
  "model": "gpt-4o-mini", 
  "reasoning_effort": "N/A",
  "api_type": "responses_api",
  "program": "def transform(grid):\n    return [row[::-1] for row in grid[::-1]]",
  "execution_error": "",
  "timed_out": false,
  "tokens_used": 542,
  "turns_used": 1,
  "request_cost": 0.000405,
  "raw_response": { /* Full API response */ },
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
API: Responses API (single-shot)
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
API: Responses API (single-shot)
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
- `execution_error`: Any runtime errors (empty if successful)
- `request_cost`: Cost for this specific task in USD
- `turns_used`: Number of conversation turns used for this task
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
uv run python task_loader.py  # Test task loading functionality
uv run python scoring.py      # Test scoring functionality
```

Quick API test:
```bash
# Test with a single task
uv run python run_arc_tasks.py --dataset arc-agi-1 --subset shortest_1 --model gpt-4o-mini --max_turns 1
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

## Important: Multi-Turn Behavior

We use **only the Responses API** with multi-turn mode:

- **Multi-turn local execution** with training feedback (up to `--max_turns`, default 3)
- Model writes code, we test it locally on test input immediately  
- If incorrect, we run it on training examples and provide detailed feedback
- Model can see training results and iterate to improve the solution
- Uses encrypted reasoning traces to maintain context between turns
- More expensive but potentially more accurate through iteration

**Key Point**: In both cases, we execute the final code locally to score it. The difference is whether the model gets multiple conversation turns with training feedback to improve its solution.

## File Structure

```
o3-tools/
├── run_arc_tasks.py             # Main script (Responses API only)
├── task_loader.py               # Load ARC tasks and subsets
├── scoring.py                   # Grid scoring
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

- You can control the maximum number of turns using --max_turns (default: 3). This is especially useful for limiting cost and runaway conversations.
- You can also set the reasoning effort for the model using --reasoning_effort (choices: low, medium, high; default: medium). This may affect the model's thoroughness and cost.
- **Automatic Model Detection**: The script automatically detects reasoning models (o3, o4, o1) vs non-reasoning models (GPT-4 series). Reasoning effort is only sent to models that support it, preventing API errors.
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