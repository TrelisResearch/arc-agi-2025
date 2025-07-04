# Experiment Notes

---

## 2025-07-04

### Experiment 1: High Reasoning Effort
- **Experiment:** Run o3 model on 30 evenly distributed tasks (by length) from arc-agi-2 evaluation set
- **Subset:** TBD (evenly distributed by length)
- **Dataset:** arc-agi-2 (evaluation)
- **Model:** o3
- **Reasoning effort:** High
- **Workers:** 10 (to avoid API limits)
- **Tools enabled:** True
- **Purpose:** Test performance with high reasoning effort on tasks evenly distributed across different lengths

### Results:
- **Tasks attempted:** 20/30 (incomplete run)
- **Tasks solved correctly:** 2/20 (10.0%)

### Notes:
- Run was incomplete (only 20 of 30 tasks completed)
- Using 10 workers to manage API rate limits

### Experiment 2: Medium Reasoning Effort (Retry)
- **Experiment:** Retry with 30 tasks using medium reasoning effort
- **Subset:** grid_size_distributed_30_evaluation
- **Dataset:** arc-agi-2 (evaluation)
- **Model:** o3
- **Reasoning effort:** Medium
- **Workers:** 10 (to avoid API limits)
- **Tools enabled:** True
- **Purpose:** Compare performance with medium vs high reasoning effort

### Results:
- **Tasks attempted:** 28/30 (93.3% completion)
- **Tasks solved correctly:** 1/28 (3.6%) - task db0c5428 with perfect solution
- **Tasks with executable code:** 2/28 (7.1%)
  - db0c5428: 100% training accuracy (perfect)
  - 7ed72f31: 0% training accuracy, 92.3% pixel accuracy
- **Tasks with no code found:** 26/28 (92.9%)
- **API issues:** Several prompt violations and JSON parsing errors requiring retries
- **Cost range:** $0.56 - $3.54 per task

### Notes:
- Significantly lower success rate compared to high reasoning effort (3.6% vs 10.0%)
- Most tasks failed to produce executable code
- API stability issues with prompt violations and parsing errors
- High cost variation between tasks ($0.56 - $3.54)

---

## 2025-07-03

- **Experiment:** Run o3 model on shortest 30 tasks from evaluation set of arc-agi-2
- **Subset:** shortest_evaluation_30
- **Dataset:** arc-agi-2 (evaluation)
- **Model:** o3
- **API:** Responses (single-shot)
- **Tools enabled:** True
- **Purpose:** Test performance on shortest tasks using summaries of reasoning in assistant responses

### Results:
- **Tasks solved correctly:** 3/30 (10.0%)
- **Pixel accuracy:** 1179/5439 (21.7%)
- **Average pattern learning:** 65.8%
- **Training execution rate:** 100.0% (22/22)
- **Training correctness rate:** 90.9% (20/22)
- **Programs with >50% pattern learning:** 7/8
- **Programs with >80% pattern learning:** 0/8
- **Average program residual:** 38.2 bytes
- **Average null baseline:** 118.0 bytes
- **Total tool calls made:** 958
- **Average tool calls per task:** 31.9
- **Total tokens used:** 14,446,749
- **Total cost:** $31.987884

### Notes:
- Training correctness is only measured where code is returned that runs (8/30 tasks)
- Interpretation: likely close on 5/30 additional tasks beyond the 3/30 correctly solved
- **Next planned experiment:** Run 30 tasks with high reasoning level, evenly distributed across task lengths. Ablate leaving out the reasoning summaries.

### Improved Failure Tracking (2025-01-04):
- **Problem:** Failed tasks were logged to files but silent in console output, causing confusion
- **Issues:** 
  - Progress counter incremented for failed tasks, making them appear successful
  - No clear distinction between API failures and successful tasks with execution issues
  - Summary statistics didn't track failed tasks separately
- **Improvements Made:**
  - **Explicit console logging:** Failed tasks now print `❌ TASK FAILED: {task_id}` with error details
  - **Enhanced progress tracking:** Shows `✅ COMPLETED` vs `❌ FAILED` status with task IDs
  - **Detailed summary stats:** Separates successful API calls from complete failures
  - **Failed task listing:** Summary includes list of all failed tasks with error messages
  - **Better data tracking:** Added `api_success` field to distinguish failure types

--- 

## 2024-07-03

- **Experiment:** Run o3 model on 30 grid-size-distributed problems from the evaluation set of arc-agi-2
- **Subset:** grid_size_distributed_30
- **Dataset:** arc-agi-2 (evaluation)
- **Command:**
  ```
  uv run python o3-tools/run_arc_tasks.py --dataset arc-agi-2 --subset grid_size_distributed_30 --model o3 --tools
  ```
- **Reasoning level:** Medium (default)
  - **Max tool calls:** 64 (default)
  - **Purpose:** Benchmark o3 model performance across a range of grid sizes in the evaluation set

