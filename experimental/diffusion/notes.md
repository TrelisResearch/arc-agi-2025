**Todo**
[x] Implement cosine noise scheduler. Move to a uniform noise distribution.
[x] Remove weight decay if duplicated? Was not duplicated, so added it back.
[x] Use optimizer updates not epochs.
[x] Remove input encoder and simplify positional encodings to use coordinate embeddings.
[x] Just do self attention between task id, timestep, input cells and noised output cells.
[x] Log info on accuracy.
[x] Generate diffusion charts every x optimizer updates.

Command to run all three model sizes:
```bash
nohup bash -c 'PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/smol_config.json > smol-sc.log 2>&1 ; \
PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/mediom_config.json > mediom-sc.log 2>&1 ; \
PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/lorge_config.json > lorge-sc.log 2>&1' &
```

**Results at 7M params with separate diffusion head training**
- Adding noise to the inputs seems to hurt diffusion performance but help grid size prediction.
- Going from 0->9 augmentations helps diffusion and grid size. Going to 39 augmentations hurts diffusion but helps grid size.

**Results at 7M params with integrated diffusion head training**
- Size prediction is much better.
- Adding noise to inputs seems to hurt size and diffusion performance.
- Integrating the size head seems to hurt diffusion performance. Makes sense as this is now multi-objective.

**Results at 90M params with integrated diffusion head**
- ...

**Things to consider adding:**
- Self-conditioning. The idea is to pass previous logits as an additional input 50% of the time during training.
- Classifier-free guidance.

Things still to understand:
- EMA
- Label smoothing

**Clarifying Questions**
- Alpha and beta
Beta is the chance a cell gets repainted at step t. Alpha is the chance a cell stays the same! alpha_bar is the chance a cell survives without any change from start to finish.

- What is the interpertation of cross entropy? Would it be useful to also plot some kind of pixel accuracy metric? If so what and how?
Yes we plot by noise bucket and also total.

- During training and forward passing on one batch, will each datapoint in a batch get a different randomly sampled timestamp? Are timestamps sampled uniformly?
Yes! and uniform should be fine.

- How many timesteps to use?
It's a bit unclear, empirically people seem to use 64-1000. Perhaps lower vocab leads to lower required steps for training.

- How many epochs or gradient updates to use?
The logic seems to be, bigger model, more gradient updates, U. And you adjust learning rate linearly with batch size, as a larger batch size smooths gradient updates. The rough numbers seem to be: 10k updates for 100M params, 100k updates for 1B params. Perhaps for a 2M param model, use 3k updates, for 20M use 7k updates.

- What noise distribution should I use? Uniform? OR matching typical arc tasks?
Uniform is mroe robust and simpler.

- During inference, what normally happens? We start with a noised input and denoise repeatedly? We don't add any intermediate noise do we?
Correct!

## Possible Ablations
- Simplify: Remove input_grid_dropout and pixel noise.
- Increase LR. Grad norm is stable and <1. In the past, this likely helped performance.
- Use separate encodings for task augmentations (rotate, flip, recolor).
- One-hot encode train vs test examples.

## Daily Notes
### Oct 1st 2025
#### Model Results on AA1 Eval (Measured only on first test grid!)
smol 145 mins training time on H200: 
- best model: Pass@2: 13/284 (4.6%)
- best model (rerun): Pass@2: 12/284 (4.2%)
- best model (rerun); 32 steps only [instead of 128 used for training]: Pass@2: 13/284 (4.6%)
- final model: Pass@2: 12/284 (4.2%)

mediom 496 mins training time on H200:
- best: Pass@2: 23/284 (8.1%)
- final: Pass@2: 25/284 (8.8%)

lorge (still training)...

#### Majority Voting Stats (now grading, with partial credit, on all output test grids)
smol:
- simple: 3.5%
- sample-40x-augs: 4.5%
Note: Kind of makes sense based on the HRM uplift at 40x sampling.

mediom - 32 steps:
- simple: 7.4%
- sample-40x-augs: 10.4%

mediom - 128 steps:
- simple: ... todo

```bash
uv run python experimental/diffusion/evaluate.py --config experimental/diffusion/configs/mediom_config.json --num-steps 32 --stats --maj && uv run python experimental/diffusion/evaluate.py --config experimental/diffusion/configs/mediom_config.json --num-steps 32 --stats
```

Possible improvements:
- Roll forward on two top grid size predictions. Maybe give a little boost, although grid accuracy is already good.