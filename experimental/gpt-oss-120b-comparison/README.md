# GPT-OSS-120B Performance Comparison

This experiment compares the performance of the `openai/gpt-oss-120b` model across three different ARC-AGI subsets to understand its generalization capabilities.

## Experiment Overview

The model was tested with consistent parameters across all subsets:
- **Model**: `openai/gpt-oss-120b` via OpenRouter
- **Max Tokens**: 64,000
- **Attempts**: 1 per task
- **Unsafe Executor**: Enabled
- **Max Workers**: 32

## Test Subsets

| Subset | Dataset | Task Count | Description |
|--------|---------|------------|-------------|
| ARC-AGI-2 Unique Training | arc-agi-2 | 233 tasks | Training tasks from ARC-AGI-2 |
| ARC-AGI-2 All Evaluation | arc-agi-2 | 120 tasks | Evaluation tasks from ARC-AGI-2 |
| ARC-AGI-1 All Evaluation | arc-agi-1 | 400 tasks | Evaluation tasks from ARC-AGI-1 |

## Results Summary

### Pass@2 (Weighted Voting) Performance:
- **ARC-AGI-1 All Evaluation**: **29.8%** (400 tasks)
- **ARC-AGI-2 Unique Training**: **12.4%** (233 tasks)
- **ARC-AGI-2 All Evaluation**: **1.7%** (120 tasks)

## Key Findings

1. **Significant Dataset Difficulty Gap**: ARC-AGI-2 appears substantially more challenging than ARC-AGI-1
   - 28.1 percentage point gap between ARC-AGI-1 and ARC-AGI-2 evaluation sets

2. **Overfitting Evidence**: Large gap between training and evaluation performance on ARC-AGI-2
   - 10.7 percentage point drop from unique training (12.4%) to evaluation (1.7%)

3. **Evaluation Set Challenge**: ARC-AGI-2 evaluation tasks are particularly difficult
   - Only 1.7% success rate despite strong performance on other subsets

4. **Model Capability**: GPT-OSS-120B shows strong performance on ARC-AGI-1 evaluation (29.8%)
   - Suggests the model has significant reasoning capabilities when tasks align with its training distribution

## Cost Analysis

- **Total Cost**: $2.78 across all three experiments
- **Token Usage**: 6.6M tokens total
- **Cost per Task**: ~$0.003 average

## Files Generated

- `performance_comparison.py` - Script to generate the comparison chart
- `performance_comparison.png` - High-resolution bar chart (300 DPI)
- `performance_comparison.pdf` - Vector format chart for publications

## Commands Used

```bash
# ARC-AGI-2 Unique Training Tasks
uv run python -m llm_python.run_arc_tasks_soar --dataset arc-agi-2 --subset unique_training_tasks --repeat-runs 1 --max_workers 32 --max_attempts 1 --model openai/gpt-oss-120b --base-url https://openrouter.ai/api/v1 --unsafe-executor --max-tokens 64000

# ARC-AGI-2 All Evaluation
uv run python -m llm_python.run_arc_tasks_soar --dataset arc-agi-2 --subset all_evaluation --repeat-runs 1 --max_workers 32 --max_attempts 1 --model openai/gpt-oss-120b --base-url https://openrouter.ai/api/v1 --unsafe-executor --max-tokens 64000

# ARC-AGI-1 All Evaluation  
uv run python -m llm_python.run_arc_tasks_soar --dataset arc-agi-1 --subset all_evaluation --repeat-runs 1 --max_workers 32 --max_attempts 1 --model openai/gpt-oss-120b --base-url https://openrouter.ai/api/v1 --unsafe-executor --max-tokens 64000
```

## Implications

These results suggest that:
1. ARC-AGI-2 represents a significant step up in difficulty from ARC-AGI-1
2. Models may struggle to generalize from ARC-AGI-2 training to evaluation tasks
3. GPT-OSS-120B shows promise on ARC-AGI-1 level tasks but needs improvement for ARC-AGI-2
4. Further investigation into ARC-AGI-2's increased complexity is warranted