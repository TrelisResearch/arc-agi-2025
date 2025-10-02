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
nohup bash -c 'PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/smol_config.json > smol-v1.log 2>&1 ; \
PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/mediom_config.json > mediom-v1.log 2>&1 ; \
PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/lorge_config.json > lorge-v1.log 2>&1' &
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

**Training:**
[ ] Set lorge aux loss to 0.07 or 0.08 (up from 0.05).
- Simplify:
 [ ] Remove input_grid_dropout.
 [x] Remove pixel noise.
- Increase LR. Grad norm is stable and <1. In the past, this likely helped performance.
- One-hot encode train vs test examples.
[x] Use separate encodings for task augmentations (rotate, flip, recolor).

**Inference**
- Roll forward on two top grid size predictions. Maybe give a little boost, although grid accuracy is already good.

## Daily Notes
### Oct 2nd 2025
### AA2 4X LR Ablation
Aiming to train 4x longer to see if it helps results.

```bash
nohup bash -c 'PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/smol_config.json > smol-v1_4x-lr.log 2>&1' &
```

```bash
nohup bash -c 'PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/mediom_config.json > mediom-v1_4x-lr.log 2>&1' &
```

#### Testing speedups

Baseline is 150s.

speedups-i (vectorize masks) gets to 135s.

Can add --profile to a training run to get the profile. I reduced creating lots of zero tensors and also some tensor copies by moving device. Hard to know what speedup that gave, if anything.

### Oct 1st 2025
#### Testing splitting up inputs instead of creating new task embeddings
```bash
PYTHONUNBUFFERED=1 nohup uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/smol_config.json > smol-sep.log &
```

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

#### Majority Voting Stats on v0 model (now grading, with partial credit, on all output test grids)
BEST MODEL RESULTS:
smol:
- simple (best model): 3.5%
- sample-40x-augs (best model): 4.5%

mediom - 32 steps:
- simple (best model): 7.4%
- sample-40x-augs (best model): 10.4%

lorge - 32 steps:
- sample (best model): 35/303 (11.6%)

FINAL MODEL RESULTS:
smol - 32 steps:
- simple (final model): 4.9%
- sample-40x-augs (final model): 5.3%

mediom - 32 steps:
- simple (final model): 8.1%
- sample-40x-augs (final model): 10.6%

lorge - 32 steps:
- simple (final model): 45/303 (14.9%)
- sample-40x-augs (final model): 49/303 (16.2%)

Note: Kind of makes sense based on the HRM uplift at 40x sampling.

```bash
uv run python experimental/diffusion/evaluate.py --config experimental/diffusion/configs/lorge_config.json --num-steps 32 --model-path experimental/diffusion/outputs/lorge/final_model.pt --maj --stats
```

#### AA1 Eval results on v1 model baseline (separate augmentation inputs)

smol - 32 steps (best model, according to val. loss):
- simple: 8.6%
- sample-40x-augs: 10.7%

smol - 32 steps (final model, not best):
- sample-40x-augs:  20.2%

smol - 128 steps:
- sample-40x-augs: 19.5%

### AA2 Eval results on v1 model baseline
Note: I had intended in increase bsz to 128 and to increase gradient steps, but forgot to pull the update.

```bash
nohup bash -c 'PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/smol_config.json --eval-limit 0 > smol-v1.log 2>&1 ; \
PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/mediom_config.json --eval-limit 0 > mediom-v1.log 2>&1' &
```

```bash
uv run python experimental/diffusion/evaluate.py --config experimental/diffusion/configs/smol_config.json --num-steps 32 --maj --model-path experimental/diffusion/outputs/smol/final_model.pt
```
smol - 32 steps (final model, not best):
- simple: 
- sample-40x-augs: 

```bash
uv run python experimental/diffusion/evaluate.py --config experimental/diffusion/configs/mediom_config.json --num-steps 32 --maj --model-path experimental/diffusion/outputs/mediom/final_model.pt
```
mediom - 32 steps (final model, not best):
- simple: 
- sample-40x-augs: Correct Sizes: 125/172 (72.7%)