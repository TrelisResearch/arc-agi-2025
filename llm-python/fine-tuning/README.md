# Fine-tuning

Contains fine-tuning notebooks for ARC-AGI task training:

## Notebooks
- **Qwen3-ARC-AGI-2-23-jul-B.ipynb**: Stable fine-tuning notebook with basic metrics. (note does need a little cleanuup by Ronan)
- **Qwen3-ARC-AGI-2-23-jul-C.ipynb**: Enhanced version with custom program evaluation metrics (**NOT TESTED YET**) - but Recommend using this notebook.

## Custom Metrics (Version C)
Version C adds custom metrics during fine-tuning to evaluate program generation quality:
- `pct_valid_programs`: % of validation rows with extractable Python programs
- `pct_rows_at_least_one_correct`: % solving ≥1 training example correctly  
- `pct_rows_all_correct`: % solving all training examples correctly
- `avg_examples_correct_per_row`: Average training examples solved per row
- `pct_examples_correct_overall`: Overall training example accuracy

These metrics provide direct measurement of the model's ability to generate working Python programs that solve ARC reasoning tasks, beyond standard loss metrics.

## TensorBoard Logs
To view training logs:
```bash
uv venv
uv pip install tensorboard
tensorboard --logdir logs
```

Open browser to http://localhost:6006/ or via runpod proxy: https://<pod-id>-6006.proxy.runpod.net

## Runpod One-click Template

**Runpod One-click-template FOR FINE-TUNING**
Runpod One-click-template [here](https://console.runpod.io/deploy?template=ifyqsvjlzj) - swap out the model name if using a fine-tuned model.

OR FOR BLACKWELL (not working):

[HERE](https://console.runpod.io/deploy?template=njbyqyiuty&ref=jmfkcdio)