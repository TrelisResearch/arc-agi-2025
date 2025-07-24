# Fine-tuning

Contains fine-tuning notebooks for ARC-AGI task training:

## Notebooks
- **unsloth_arc_finetuning.ipynb**: Latest notebook, ready to integrate custom metrics as a next step.

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