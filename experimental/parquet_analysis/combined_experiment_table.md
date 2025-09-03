# Combined Analysis: Baseline 1 + Each Refinement Approach

This table shows the combined results when Baseline 1 is merged with each other approach.
For combined columns, we take the MAX success count from either Baseline 1 or the other approach.

## Experiment Descriptions

**Individual Runs:**
- **Baseline 1**: No refinement (Sept 2)
- **Baseline 2**: No refinement (Sept 3)
- **Refine + Diff**: Refinement with output diffs
- **Refine + Full**: Refinement with full outputs
- **Refine Only**: Refinement with program only

**Combined Runs (Baseline 1 + X):**
- **Baseline 1 + Baseline 2**: Best of either approach per task
- **Baseline 1 + Refine + Diff**: Best of either approach per task
- **Baseline 1 + Refine + Full**: Best of either approach per task
- **Baseline 1 + Refine Only**: Best of either approach per task

## Results Table

Numbers show count of all-correct programs. Combined columns show MAX(Baseline 1, Other).

| Task ID | Baseline 1 | Baseline 2 | Refine + Diff | Refine + Full | Refine Only | B1+Baseline 2 | B1+Refine + Diff | B1+Refine + Full | B1+Refine Only |
|---|---|---|---|---|---|---|---|---|---|
| 1c0d0a4b |  | 1 |  | 2 | 2 | 1 |  | 2 | 2 |
| 2037f2c7 | 1 |  |  |  |  | 1 | 1 | 1 | 1 |
| 42918530 |  |  |  | 1 | 3 |  |  | 1 | 3 |
| 458e3a53 | 3 |  | 1 | 2 | 6 | 3 | 3 | 3 | 6 |
| 551d5bf1 |  |  |  |  | 1 |  |  |  | 1 |
| 5d588b4d | 2 | 2 | 2 |  | 1 | 2 | 2 | 2 | 2 |
| 62ab2642 | 1 |  | 1 |  |  | 1 | 1 | 1 | 1 |
| 7447852a |  |  |  |  | 1 |  |  |  | 1 |
| 78e78cff | 1 |  |  |  |  | 1 | 1 | 1 | 1 |
| 7c8af763 | 1 | 3 | 1 | 1 | 2 | 3 | 1 | 1 | 2 |
| 7d1f7ee8 |  | 2 |  |  |  | 2 |  |  |  |
| 8f2ea7aa |  | 4 | 3 | 2 | 1 | 4 | 3 | 2 | 1 |
| 94be5b80 |  |  | 1 |  |  |  | 1 |  |  |
| 9def23fe | 1 | 1 |  |  | 2 | 1 | 1 | 1 | 2 |
| 9f41bd9c |  | 1 |  |  |  | 1 |  |  |  |
| a57f2f04 |  |  |  | 1 | 1 |  |  | 1 | 1 |
| a644e277 |  |  |  |  | 1 |  |  |  | 1 |
| a934301b |  | 1 | 10 | 5 | 4 | 1 | 10 | 5 | 4 |
| b782dc8a | 7 | 5 | 8 | 1 | 5 | 7 | 8 | 7 | 7 |
| cf133acc |  |  | 1 |  |  |  | 1 |  |  |
| e729b7be |  | 2 | 1 | 3 | 2 | 2 | 1 | 3 | 2 |
| f35d900a |  |  | 1 |  |  |  | 1 |  |  |
| f8c80d96 |  |  |  | 1 |  |  |  | 1 |  |
| ff2825db |  |  | 1 |  |  |  | 1 |  |  |
| **TOTAL TASKS** | 8 | 10 | 12 | 10 | 14 | **14** | **15** | **15** | **17** |

## Summary Statistics

### Individual Experiments

**Baseline 1**:
- Tasks with ≥1 all-correct: 8
- Total all-correct programs: 17

**Baseline 2**:
- Tasks with ≥1 all-correct: 10
- Total all-correct programs: 22

**Refine + Diff**:
- Tasks with ≥1 all-correct: 12
- Total all-correct programs: 31

**Refine + Full**:
- Tasks with ≥1 all-correct: 10
- Total all-correct programs: 19

**Refine Only**:
- Tasks with ≥1 all-correct: 14
- Total all-correct programs: 32

### Combined Results (Baseline 1 + Each Approach)

**Baseline 1 + Baseline 2**:
- Tasks with ≥1 all-correct: **14** (+6 vs Baseline 1 alone)
- Breakdown: 4 from B1 only, 6 from other only, 4 from both

**Baseline 1 + Refine + Diff**:
- Tasks with ≥1 all-correct: **15** (+7 vs Baseline 1 alone)
- Breakdown: 3 from B1 only, 7 from other only, 5 from both

**Baseline 1 + Refine + Full**:
- Tasks with ≥1 all-correct: **15** (+7 vs Baseline 1 alone)
- Breakdown: 5 from B1 only, 7 from other only, 3 from both

**Baseline 1 + Refine Only**:
- Tasks with ≥1 all-correct: **17** (+9 vs Baseline 1 alone)
- Breakdown: 3 from B1 only, 9 from other only, 5 from both

## Improvement Analysis

How much does combining Baseline 1 with each approach improve over Baseline 1 alone?

**Baseline 2**: 8 → 14 tasks (+6 tasks, 75.0% improvement)
**Refine + Diff**: 8 → 15 tasks (+7 tasks, 87.5% improvement)
**Refine + Full**: 8 → 15 tasks (+7 tasks, 87.5% improvement)
**Refine Only**: 8 → 17 tasks (+9 tasks, 112.5% improvement)
