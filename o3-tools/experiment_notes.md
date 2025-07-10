# Experiment Notes

## 2025 10th July

### Repeat of the feedback versus sampling experiment, with a better dataset

**Commentary:**
- The error on running the shortest 100 arc-agi-1 evaluation problems appears a bit lower than just running MIT-easy, although the single turn results aren't within each other's confidence bounds suggesting that 3 runs is too few.
- Sampling is within error of feedback, but the cost is 3x higher for feedback (traces are longer per task, as they are accumulated)... suggesting that feedback is currently no better than sampling from scratch.
- IDEA: To improve, perhaps it's worth running the full 400 arc-agi-1 evaluation problems on gpt-4.1 and saving the ones that a) are correct in one turn and b) are correct in multiple turns. That larger subset might then be suitable for running ablations on gpt-4.1-mini and allow for a stronger signal. The cost of that is probably about $15, and I'm guessing would generate about 50-100 correct tasks. To further augment this dataset, we could remove tasks that are solved with a single attempt of gpt-4.1-nano. This should give a range of difficulty that is roughly calibrated to gpt-4.1-mini.

**Results:**

AGGREGATE STATISTICS - with independent attempts (i.e. sampling):
----------------------------------------------------------------------
```bash
uv run python run_arc_tasks.py --dataset arc-agi-1 --subset shortest_evaluation_100 --repeat-runs 3 --max_workers 25 --max_turns 8 --model gpt-4.1-mini --independent-attempts
```
Attempt 1 Only Success Rate:
  Mean: 11.7%
  Std Dev: 0.6%
  95% CI: [10.5%, 12.8%]

All Attempts Success Rate:
  Mean: 22.3%
  Std Dev: 2.1%
  95% CI: [18.3%, 26.4%]
Note that cost of the entire run is about $1.70 per run.

AGGREGATE STATISTICS - with feedback:
----------------------------------------------------------------------
```bash
uv run python run_arc_tasks.py --dataset arc-agi-1 --subset shortest_evaluation_100 --repeat-runs 3 --max_workers 25 --max_turns 8 --model gpt-4.1-mini
```
Turn 1 Only Success Rate:
  Mean: 13.0%
  Std Dev: 1.7%
  95% CI: [9.6%, 16.4%]

All Turns Success Rate:
  Mean: 18.7%
  Std Dev: 2.9%
  95% CI: [13.0%, 24.3%]
Note that the cost of the entire run is about $4.50 per run, 3x the cost of search.

### Figuring out how to get error down on measurement

Plan is first to see how gpt-4.1-mini and nano score on the shortest arc-agi-1 tasks, 100 of them.
==================================================
SUMMARY
==================================================
```bash
uv run python run_arc_tasks.py --dataset arc-agi-1 --subset shortest_evaluation_100 --repeat-runs 1 --max_workers 10 --max_turns 8 --model gpt-4.1-mini --independent-attempts
```
Dataset: arc-agi-1
Subset: shortest_evaluation_100
Model: gpt-4.1-mini
API: Responses (independent attempts, max 8 attempts)
Total tasks attempted: 100
Successful API calls: 100/100 (100.0%)
Tasks solved correctly: 24/100 (24.0%)
Pixel accuracy: 722/5676 (12.7%)
Total attempts used: 654
Average attempts per task: 6.5
Total tokens used: 2,086,479
Total cost: $1.700422

==================================================
SUMMARY
==================================================
```bash
uv run python run_arc_tasks.py --dataset arc-agi-1 --subset shortest_evaluation_100 --repeat-runs 1 --max_workers 10 --max_turns 8 --model gpt-4.1-nano --independent-attempts
```
Dataset: arc-agi-1
Subset: shortest_evaluation_100
Model: gpt-4.1-nano
API: Responses (independent attempts, max 8 attempts)
Total tasks attempted: 100
Successful API calls: 100/100 (100.0%)
Tasks solved correctly: 5/100 (5.0%)
Pixel accuracy: 111/5676 (2.0%)
Total attempts used: 780
Average attempts per task: 7.8
Total tokens used: 2,498,011
Total cost: $0.442855

### Ablating sampling versus feedback (depth of 8).
Run two times - once with independent attempts, and once with feedback.

**Baseline scoring is ~8/20 (40% +/-8% with feedback, 28% +/- 11% without feedback) from yesterday.**

**Commentary:**
- These results are showing that there is a HUGE amount of noise. Because the “Attempt 1 Only Success Rate” and “Turn 1 Only Success Rate” should be equivalent, and they are falling outside their 95% confidence bounds, indicating that just doing 3 runs is not capturing the mean of the distribution…
- I re-ran the tests with 10 runs of each, and now the results are within each other's confidence bounds for single turn/attempt, which is good. However, the error is so high that distinguishing the two runs is very difficult.
- IMPLICATION: Ablating some kind of MCTS will be almost impossible to see in the noise...

**Results:**

gpt-4.1-mini:
Re-run this time with 10 runs of each.

AGGREGATE STATISTICS - with independent attempts (i.e. sampling):
----------------------------------------------------------------------
```bash
uv run python run_arc_tasks.py --dataset arc-agi-1 --subset mit-easy --repeat-runs 10 --max_workers 10 --max_turns 8 --model gpt-4.1-mini --independent-attempts
```
Attempt 1 Only Success Rate:
  Mean: 23.0%
  Std Dev: 6.7%
  95% CI: [9.8%, 36.2%]

All Attempts Success Rate:
  Mean: 35.0%
  Std Dev: 4.1%
  95% CI: [27.0%, 43.0%]

AGGREGATE STATISTICS - with feedback:
----------------------------------------------------------------------
```bash
uv run python run_arc_tasks.py --dataset arc-agi-1 --subset mit-easy --repeat-runs 10 --max_workers 10 --max_turns 8 --model gpt-4.1-mini
```
Turn 1 Only Success Rate:
  Mean: 22.5%
  Std Dev: 4.2%
  95% CI: [14.2%, 30.8%]

All Turns Success Rate:
  Mean: 32.5%
  Std Dev: 4.9%
  95% CI: [23.0%, 42.0%]

AGGREGATE STATISTICS - with independent attempts (i.e. sampling):
----------------------------------------------------------------------
```bash
uv run python run_arc_tasks.py --dataset arc-agi-1 --subset mit-easy --repeat-runs 3 --max_workers 10 --max_turns 8 --model gpt-4.1-mini --independent-attempts
```
Attempt 1 Only Success Rate:
  Mean: 33.3%
  Std Dev: 2.9%
  95% CI: [27.7%, 39.0%]

All Attempts Success Rate:
  Mean: 38.3%
  Std Dev: 2.9%
  95% CI: [32.7%, 44.0%]

AGGREGATE STATISTICS - with feedback:
----------------------------------------------------------------------
```bash
uv run python run_arc_tasks.py --dataset arc-agi-1 --subset mit-easy --repeat-runs 3 --max_workers 10 --max_turns 8 --model gpt-4.1-mini
```
Turn 1 Only Success Rate:
  Mean: 26.7%
  Std Dev: 2.9%
  95% CI: [21.0%, 32.3%]

All Turns Success Rate:
  Mean: 31.7%
  Std Dev: 2.9%
  95% CI: [26.0%, 37.3%]

o4-mini:

AGGREGATE STATISTICS - with independent attempts (i.e. sampling):
----------------------------------------------------------------------
```bash
uv run python run_arc_tasks.py --dataset arc-agi-1 --subset mit-medium --repeat-runs 3 --max_workers 10 --max_turns 8 --model o4-mini --independent-attempts
```
Attempt 1 Only Success Rate:
  Mean: 13.3%
  Std Dev: 5.8%
  95% CI: [2.0%, 24.6%]

All Attempts Success Rate:
  Mean: 43.3%
  Std Dev: 2.9%
  95% CI: [37.7%, 49.0%]

AGGREGATE STATISTICS - with feedback:
----------------------------------------------------------------------
```bash
uv run python run_arc_tasks.py --dataset arc-agi-1 --subset mit-medium --repeat-runs 3 --max_workers 10 --max_turns 8 --model o4-mini
```
Turn 1 Only Success Rate:
  Mean: 22.9%
  Std Dev: 4.2%
  95% CI: [14.7%, 31.2%]

All Turns Success Rate:
  Mean: 38.8%
  Std Dev: 5.0%
  95% CI: [28.9%, 48.6%]


**Notes:**
...

### MIT Difficulty vs Task Size Analysis

Analyzed overlap between MIT difficulty splits (easy/medium/hard) and shortest evaluation tasks in arc-agi-1 to understand the relationship between task difficulty and task size.

**Key Findings:**

1. **MIT-Easy tasks correlate strongly with smallest tasks**
   - **30%** of MIT-easy tasks are in the 100 shortest evaluation tasks
   - The **single shortest task** (`66e6c45b`) is classified as MIT-easy
   - **20%** of the 10 shortest tasks are MIT-easy

2. **MIT-Hard tasks also appear in shortest tasks** 
   - **35%** of MIT-hard tasks are in the 100 shortest evaluation tasks (highest percentage!)
   - But **no overlap** with the 10 shortest tasks
   - This suggests some hard tasks are small but not the smallest

3. **MIT-Medium has moderate overlap**
   - **20%** of MIT-medium tasks in shortest_evaluation_100
   - **No overlap** with the 10 shortest tasks
   - Overlaps start appearing at the 30-task level

**Key Insights:**

🔍 **Task difficulty ≠ Task size**: The MIT difficulty classification doesn't directly correlate with grid size. Some hard tasks are actually quite compact.

🎯 **The shortest task is easy**: Task `66e6c45b` (96 total cells) is both the shortest and classified as MIT-easy.

📈 **Overlap increases with subset size**: 
- shortest_1: Only MIT-easy overlap
- shortest_10: Only MIT-easy overlap  
- shortest_30: MIT-easy + MIT-medium + MIT-hard
- shortest_100: All three MIT categories well represented

**Implications:**
- **Small tasks can still be hard** - complexity isn't just about size
- **MIT-easy tasks tend to be smaller** - but not exclusively
- **Good test sets**: The shortest evaluation tasks provide a mix of difficulties, making them useful for testing across complexity levels

This analysis shows that task size and cognitive difficulty are orthogonal dimensions in ARC-AGI tasks!

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

