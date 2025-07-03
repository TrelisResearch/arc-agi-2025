# o3-tools

A tool for testing OpenAI o3/o4 models on ARC-AGI tasks with and without code interpreter tools.

**Todo**
[ ] Reorganize data so that training and evaluation problems are split, because they are so different in terms of difficulty.

Improvements:
[ ] Try inputting images of the problem as well as just the problem itself.
[ ] Simplify gzip to just compress a list of numbers (even with no new lines) vs just zeros.
[ ] Use pixel accuracy OR gzip OR training problems correct to re-sample.
[x] prompt so that the model keeps reasoning until it finds a python program that solves (for the tool use case). don't include the test examples in the prompt.
[x] Use gzip for the program too (strip comments), possibly this removes the need for having the alpha and beta parameters. Also, make sure we're including all train examples for that task in the data description length.
[x] **Fixed residual calculation**: Changed from broken "actual-value-if-wrong" to proper difference-based approach (`residual = actual - predicted`) that allows perfect reconstruction.
[x] **Replaced MDL with residual reduction**: Changed from complex MDL (program + residuals) to pure pattern learning measurement using residual reduction percentage.

For Kaggle / low compute:
[ ] Run on Qwen 3 using OpenRouter as an API endpoint.
[ ] Add in a tool to run python code.
    [ ] Allow access to specific grids via defined variables (to save context).

**Open questions:**
- What happens if the grid output isn't the right size? how is pixel accuracy and residual reduction calculated?
- If there is no output grid, how does that affect residual reduction and pixel score aggregation?

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
uv run python run_arc_tasks.py
```

### Advanced Usage

```bash
# Run 10 shortest tasks from ARC-AGI-2 with tools enabled
uv run python run_arc_tasks.py --dataset arc-agi-2 --subset shortest_10 --tools

# Run specific subset with model selection
uv run python run_arc_tasks.py --dataset arc-agi-1 --subset shortest_100 --model gpt-4o-mini --limit 5

# Available options:
#   --dataset: arc-agi-1 or arc-agi-2
#   --subset: shortest_1, shortest_10, shortest_100, etc.
#   --model: OpenAI model name (default: gpt-4o-mini)
#   --tools: Enable code interpreter tools
#   --limit: Limit number of tasks to run
#   --max_tool_calls: Maximum number of tool calls allowed for the model (default: 64, only applies if --tools is set)
```

### Available Subsets

- `shortest_1`: Single shortest task
- `shortest_10`: 10 shortest tasks  
- `shortest_100`: 100 shortest tasks

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

## create_grid_size_distributed_subset.py

This script creates a new subset of ARC-AGI problems by selecting tasks from the evaluation sets of arc-agi-1 and arc-agi-2, distributed evenly by grid size. For each task, the grid size is defined as the sum of the number of cells in the first input and first output grid (from the first training example). The script selects 30 tasks from each evaluation set, spaced evenly across the range of grid sizes, and copies them into a new subset directory for balanced benchmarking.

Usage:
```
uv run o3-tools/create_grid_size_distributed_subset.py
```

The script will output a manifest of selected tasks and their grid sizes.