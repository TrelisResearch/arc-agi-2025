# Experiment Notes

## 2025 10th July

### Ablating sampling versus feedback (depth of 8).
Run two times - once with independent attempts, and once with feedback.

**Baseline scoring is ~8/20 (40% +/-8% with feedback, 28% +/- 11% without feedback) from yesterday.**

**Results:**

gpt-4.1-mini:
```bash
uv run python run_arc_tasks.py --dataset arc-agi-1 --subset mit-easy --repeat-runs 3 --max_workers 10 --max_turns 8 --model gpt-4.1-mini [--independent-attempts]
```
- Independent attempts: .../20
- Feedback: .../20

o4-mini:
```bash
uv run python run_arc_tasks.py --dataset arc-agi-1 --subset mit-medium --repeat-runs 3 --max_workers 10 --max_turns 8 --model o4-mini [--independent-attempts]
```
- Independent attempts: .../20
- Feedback: .../20

Full script
```bash
uv run python run_arc_tasks.py --dataset arc-agi-1 --subset mit-easy --repeat-runs 3 --max_workers 10 --max_turns 8 --model gpt-4.1-mini --independent-attempts
uv run python run_arc_tasks.py --dataset arc-agi-1 --subset mit-medium --repeat-runs 3 --max_workers 10 --max_turns 8 --model o4-mini --independent-attempts
uv run python run_arc_tasks.py --dataset arc-agi-1 --subset mit-medium --repeat-runs 3 --max_workers 10 --max_turns 8 --model o4-mini
```

**Notes:**
...

### Investigating how to improve output grid sizes
Note that this is a problem with gpt-4.1-mini but also o4-mini (and maybe o3?).

In the feedback loop, prompting has been improved to give feedback if grid sizes are wrong. Also, the first turn prompt now explicitly says what size output grid should be produced.

Wasn't able to get a clear ablation improvement. But seems reasonable the prompts won't hurt.

>[!TIP]
>Note that `arc-agi-2/shortest_evaluation_1` has an output grid size different than the input and is good for testing.

### Notes on o4-mini (low) run from yesterday:
- It seems that task 27a77e38 in the arc-agi-1 mit-medium split can be fluked?!?
- There's some, very limited, evidence that higher pixel match results in a better chance of the next iteration with feedback being correct.
- Re oscillation: There is evidence of oscillation, even with prompting - with o4-mini. BTW, maybe this doesn't happen with o3?
- Open Q: When is it good to refine versus when good to search?
- There is a strong correlation between pixel accuracy on the training tasks and the test tasks ![image train-v-test-pixel-accuracy](image train-v-test-pixel-accuracy.png)

## 2025 9th July

Actions:
- Adjusted the prompts to encourage a more step by step approach based on training examples correct.

Test #1:
- GPT 4.1 mini
  - Quick test of new prompting.
  - Seems like model oscillates between the same responses.
  - Fixed prompt to say don't repeat the same transformation... need to re-run. Seems not to repeat any more now (although re-run was with images).

Test #2:
Objective: To try to get a better signal to noise ratio on o3, by using the arc agi 1 splits.
- Run on mit-medium: Getting roughly 10+/20 correct, so now we have perhaps higher signal to noise?

Test #3:
Objective: Does iteration / feedback help?
- Based on Test #2, we go on mit-medium arc-agi-1 from 4/20 up to 7/20 with o4-mini when going from single turn to more than one turn (max of eight).
- About 4/20 tasks hit API issues with o3, so there were only 16 tasks run. 11/20 correct with feedback, 8/20 correct with one turn.
- Both run with low levels of reasoning effort.

Test #4:
Objective: See if adding image information helps the score further.
- Tested o4-mini on a low level of reasoning on mit-medium arc-agi-1.
- Results (3 runs): 
  - o4-mini scores 9/19 (one timeout failure) with feedback, 7/19 without feedback.
- Tested o4-mini on a low level of reasoning on mit-medium arc-agi-1.
- Results (3 runs) with images: 
  - 9/19 with feedback, 7/19 without feedback.
  - 6/19 (feedback), 3/19 without feedback. With one api failure.
  - 8/19 (feedback), 6/19 without feedback. With one api failure.
  - Averages WITH IMAGES:
    - 28% +/- 11% without feedback.
    - 40% +/- 8% with feedback. (although I have never seen a case where feedback scores lower).
- Results with no images:
  - 7/20 with feedback, 4/20 without feedback.
  - 6/20 with feedback, 3/20 without feedback.
  - 9/20 with feedback, 4/20 without feedback.
  - Averages WITHOUT IMAGES:
    - 18% +/- 3% without feedback.
    - 37% +/- 8% with feedback.

---

## 2025-07-06

### Experiment 1: Optimized Prompts with Partial Solutions (8 Max Turns)
- **Experiment:** Run o3 model with enhanced prompts that encourage partial transformation solutions instead of model refusal
- **Model:** o3
- **Reasoning effort:** Low
- **Max turns:** 8 (increased from default 3)
- **Prompt optimizations:** 
  - Softened language to reduce model refusal
  - Explicitly encourage partial solutions when complete rules aren't clear
  - Added validation examples showing good vs bad implementations
  - Enhanced training feedback with encouraging language for partial progress
- **Purpose:** Test whether optimized prompts reduce model refusal and improve solution attempts, even for partial transformations
- **Status:** Completed

### Results (Low Reasoning Effort):
WARNING: THESE TESTS ARE ON THE SHORTEST 30, NOT THE EVENLY DISTRIBUTED 30 FROM THE PAST DAYS OF TESTING!!!
- **Dataset:** arc-agi-2/shortest_evaluation_30
- **Tasks attempted:** 30/30 (100.0% completion)
- **Tasks solved correctly:** 1/30 (3.3%) - task 1ae2feb7 solved perfectly [a different task than correct yesterday.]
- **Pixel accuracy:** 200/5439 (3.7%)
- **Average turns per task:** 7.8 (close to max 8 turns limit)
- **Total tokens used:** 4,505,039
- **Total cost:** $17.01
- **Execution:** Parallel with 10 workers
- **API stability:** 100% successful API calls, no timeout failures

### Notes:
- **Problem addressed:** Previous experiments showed models refusing to attempt solutions or providing placeholder code like `return transformed_grid`
- **Key difference:** **Softened prompt approach** - model is explicitly encouraged to give its best attempt at a partial program if it can't find a transformation that satisfies all training examples
- **Prompt changes made:**
  - Modified initial prompt to emphasize "attempt a solution" vs "solve perfectly"
  - Updated code request prompt to discourage refusal
  - Enhanced training feedback to be more encouraging about partial progress
  - Added language encouraging partial solutions when complete rules aren't clear
- **Timeout observations:** 300-second timeout appears to cause medium reasoning effort to timeout on o3 model
- **Performance:** Very low solve rate (3.3%) but high turn usage (7.8 avg) suggests model is engaging but struggling with actual solutions
- **Cost efficiency:** ~$0.57 per task average, similar to previous experiments but with more turns
- **Behavioral change:** Model no longer refuses tasks but attempts partial transformations - however, these attempts still have very low accuracy
- **Next steps:** May need to refine prompting strategy - start with strict requirements, then fall back to partial attempts
- Quite a few problems have got one or a few training examples correct, so there is some sign of progress.

Other note: There is a case where all training examples are correct, but the test is wrong...



---

## 2025-07-05

### Experiment 1: Multi-turn with 8 max turns
- **Experiment:** Run o3 model on 30 grid size distributed tasks with extended multi-turn capability
- **Subset:** grid_size_distributed_30_evaluation
- **Dataset:** arc-agi-2 (evaluation) 
- **Model:** o3
- **Reasoning effort:** Medium
- **Workers:** 10 (parallel execution)
- **Tools enabled:** True (multi-turn local execution)
- **Max turns:** 8 (increased from default 3)
- **Purpose:** Test whether additional turns improve solve rate on challenging distributed tasks

### Results:
- **Tasks attempted:** 30/30 (100.0% completion)
- **Tasks solved correctly:** 2/30 (6.7%) - tasks 7ed72f31 and 36a08778
- **Pixel accuracy:** 580/12175 (4.8%)
- **Average pattern learning:** 74.2%
- **Training execution rate:** 100.0% (8/8)
- **Training correctness rate:** 87.5% (7/8)
- **Programs with >50% pattern learning:** 2/2
- **Programs with >80% pattern learning:** 1/2
- **Average program residual:** 47.0 bytes
- **Average null baseline:** 198.5 bytes
- **Total tokens used:** 2,398,560
- **Total cost:** $10.51

### Notes:
- Using new multi-turn implementation with local code execution and training feedback
- Extended max turns to allow more iterative improvement attempts
- Running in parallel with 10 workers for efficiency
- **Comparison to previous experiments:** 6.7% solve rate vs July 4th medium reasoning (3.6%) and high reasoning (10.0%) - **BUT NOTE:** July 4th experiments allowed up to 64 tool calls vs today's 8-turn limit, making cost/resource investment very different
- **Strong pattern learning:** High pattern learning scores (74.2% average) suggest model understands patterns but struggles with execution
- **Cost efficiency:** ~$0.35 per task average, ranging from $0.06 to $1.92 - much more cost-controlled than July 4th experiments
- **Resource-constrained performance:** 6.7% solve rate with 8-turn limit may represent good efficiency compared to unlimited tool calls

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

