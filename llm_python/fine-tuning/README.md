# Fine-tuning

Contains fine-tuning notebooks for ARC-AGI task training:

## Notebooks

- **unsloth_arc_finetuning_soar.ipynb**: A fine-tuning notebook that uses the SOAR prompt. Recommended for now, unless results show another prompt to perform better.

Archived:
- **unsloth_arc_finetuning.ipynb**: A notebook with an incorrect v1 prompt that over-constrains the grid outputs.

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
You can start a GPU instance using [this Runpod template](https://console.runpod.io/deploy?template=bh0rvngapk&ref=jmfkcdio), which will install .toml dependencies AND unsloth, required for the fine-tuning notebook, via an on-start script.

To run the template, you'll need to set - in Runpod:
- `GITHUB_PAT`: a github personal access token with access to this arc-agi-2025 repo. YOU MUST GRANT CONTENTS: READ AND WRITE PERMISSION!
- `HUGGING_FACE_HUB_TOKEN`: a hugging face token with access the Trelis org [token](https://huggingface.co/settings/tokens)