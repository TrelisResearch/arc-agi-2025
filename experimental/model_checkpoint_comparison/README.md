# Model Checkpoint Performance Comparison

This analysis compares the performance of three different model configurations across their training checkpoints using data from the experiment notes.

## Models Analyzed

1. **Best Length Filtered (Constant LR)** - `Trelis/Qwen3-4B_dsarc-agi-1-train-programs-best-length-filtered-250_20250811-221545_cst`
   - Training with constant learning rate on best length filtered data
   - Checkpoints: c904, c1808, c2712, c3614

2. **Best Length Filtered (Annealing)** - `Trelis/Qwen3-4B_dsarc-agi-1-train-programs-best-length-filtered-250_20250811-155856`
   - Training with learning rate annealing on best length filtered data  
   - Checkpoints: c904, c1808, c2712, c3616

3. **50 Correct 200 Partial (Constant LR)** - `Trelis/Qwen3-4B_dsarc-programs-50-full-200-partial_20250807-211749`
   - Training with constant learning rate on 50 correct + 200 partial programs
   - Checkpoints: c1057, c2114, c3171, c4228

4. **50 Correct 200 Partial (No Test Transduction)** - Same model as #3 but evaluated without transductive filtering
   - Only c3171 checkpoint tested for comparison
   - Shows impact of transductive filtering on performance

## Evaluation Setup

- **Dataset**: arc-agi-1 evaluation set (400 tasks)
- **Runs**: 3 runs per checkpoint for statistical significance
- **Max Attempts**: 8 attempts per task
- **Metric**: Weighted Voting Pass@2 accuracy (%)
- **Error Bars**: 2σ (~95% confidence interval)

## Key Results

### Performance Summary
- **Best Length Filtered (Annealing)**: Peak performance at early checkpoint c904 (15.6%), gradual decline
- **50 Correct 200 Partial**: Steady improvement peaking at c3171 (16.7%), then slight decline
- **Best Length Filtered (Constant LR)**: More stable performance around 11-13%

### Best Performing Checkpoints
1. **c3171 (50 Correct 200 Partial)**: 16.7% ± 1.0% - Highest accuracy with transductive filtering
2. **c904 (Best Length Filtered Annealing)**: 15.6% ± 2.4% - High performance but higher variance
3. **c1808 (Best Length Filtered Annealing)**: 15.4% ± 0.4% - Good balance of performance and consistency
4. **c3171 (No Test Transduction)**: 15.1% ± 0.4% - High consistency, shows 1.6pp impact of transductive filtering

## Files

- `plot_checkpoint_comparison.py` - Python script to generate the performance comparison plot
- `checkpoint_performance_comparison.png` - Main visualization
- `checkpoint_performance_comparison.pdf` - PDF version of the plot

## Usage

```bash
# Generate the plot
uv run python plot_checkpoint_comparison.py
```

## Observations

1. **Training Data Impact**: The 50 Correct 200 Partial model shows the most improvement over training, suggesting this data composition may be more effective for learning.

2. **Learning Rate Strategy**: Annealing LR shows higher peak performance but less stability compared to constant LR.

3. **Transductive Filtering Impact**: Comparing c3171 with/without transductive filtering shows:
   - **+1.6 percentage points** improvement with transductive filtering (15.1% → 16.7%)
   - Higher variance with transductive filtering (±1.0% vs ±0.4%)
   - Demonstrates significant value of test-time transductive approaches

4. **Overfitting**: Most models show performance degradation at later checkpoints, indicating potential overfitting.

5. **Consistency vs Performance**: There's a trade-off between peak performance and consistency across runs.

6. **Constant LR Stability**: The constant LR model maintains relatively stable performance (11.3%-13.3%) across all checkpoints.