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

**Clarifying Questions**
- Alpha and beta
Beta is the chance a cell gets repainted at step t. Alpha is the chance a cell stays the same! alpha_bar is the chance a cell survives without any change from start to finish.

- What is the interpretation of cross entropy? Would it be useful to also plot some kind of pixel accuracy metric? If so what and how?
Yes we plot by noise bucket and also total. And we now record accuracy as well in wandb.

- During training and forward passing on one batch, will each datapoint in a batch get a different randomly sampled timestamp? Are timestamps sampled uniformly?
Yes! and uniform should be fine.

- How many timesteps to use?
It's a bit unclear, empirically people seem to use 64-1000. Perhaps lower vocab leads to lower required steps for training. WE USE 128.

- How many epochs or gradient updates to use?
The logic seems to be, bigger model, more gradient updates, U. And you adjust learning rate linearly with batch size, as a larger batch size smooths gradient updates. The rough numbers seem to be: 10k updates for 100M params, 100k updates for 1B params. Perhaps for a 2M param model, use 3k updates, for 20M use 7k updates. ACTUALLY WE JUST USE 96K EVERYWHERE FOR NOW.

- What noise distribution should I use? Uniform? OR matching typical arc tasks?
Uniform is more robust and simpler.

- During inference, what normally happens? We start with a noised input and denoise repeatedly? We don't add any intermediate noise do we?
Correct!

## Daily Notes
### Oct 4th 2025
#### Running huoge on B200
```bash
nohup bash -c 'PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/huoge_config.json > huoge-v1-boost-b200.log 2>&1' &
```

Running with less aggressive dropouts - I did run this reducing sc and embedding dropout for larger models, but saw no big difference early on on the huoge model:
```bash
nohup bash -c 'PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/huoge_config.json > huoge-v1-boost-b200_low-dropout.log 2>&1' &
```

#### Running models on aa1 and aa2

```bash
nohup bash -c 'PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/mediom_config.json > mediom-v1-boost.log 2>&1 ; \
PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/lorge_config.json > lorge-v1-boost.log 2>&1 ; \
PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/huoge_config.json > huoge-v1-boost.log 2>&1
PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/giont_config.json > giont-v1-boost.log 2>&1' &
```
- smol: 0% (although a similar model has scored 0.4% before)
- mediom: 1.7%
- lorge: still running
- huoge: still running

```bash
nohup bash -c 'PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/smol_config_aa1.json > smol-v1-aa1.log 2>&1 ; \
PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/mediom_config_aa1.json > mediom-v1-aa1.log 2>&1 ; \
PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/lorge_config_aa1.json > lorge-v1-aa1.log 2>&1 ; \
PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/huoge_config_aa1.json > huoge-v1-aa1.log 2>&1 ; \
PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/giont_config_aa1.json > giont-v1-aa1.log 2>&1' &
```
Scoring - all `--num-steps 32 --maj`:
- smol: 13.8%
- mediom: 23.1%
- lorge: still running

### Oct 3rd 2025
Have started some runs on aa1 and aa2 although they are running for cst forward pass steps, not optimizer steps. Still it can give some sense of performance.


#### Training huoge on aa2 boost strategy
Moving to train a 270M model.

The training data will have 39 augmentations for aa2 training tasks and 390 augmentations for each evaluation task (where only train, not test grids are included).

Note that we are now seeing three tasks at least partially solved:
- 981571dc solved fully by mediom and lorge
- 269e22fb partially solved by lorge
- 71e489b6 partially solved by smol boost

Run all of those commands in series on the same gpu:
```bash
nohup bash -c 'PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/smol_config.json > smol-v1-boost.log 2>&1 ; \
PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/huoge_config.json > huoge-v1-boost.log 2>&1' &
```

#### Support pre-training and LoRA
LoRA tuning on 1 or 10 tasks on a pre-trained model seems not to work, at least yet.

#### AA2-hard Results
Trained on training-hard + evaluation.

smol:
Scores zeros.

mediom:
- simple (halfway model, 32 steps): 0%
- simple (best model, 32 steps): 0.83%
- simple (final model, 128 steps): 0.83%
- simple (final model, 32 steps): 0.83%
- sample-40x-augs (final model, 32 steps): 0.83%

```bash
uv run experimental/diffusion/evaluate.py --config experimental/diffusion/configs/mediom_config.json --num-steps 32 --model-path experimental/diffusion/outputs/mediom/halfway_model.pt --stats && \
uv run experimental/diffusion/evaluate.py --config experimental/diffusion/configs/mediom_config.json --num-steps 32 --model-path experimental/diffusion/outputs/mediom/best_model.pt && \
uv run experimental/diffusion/evaluate.py --config experimental/diffusion/configs/mediom_config.json --num-steps 32 --model-path experimental/diffusion/outputs/mediom/final_model.pt && \
uv run experimental/diffusion/evaluate.py --config experimental/diffusion/configs/mediom_config.json --num-steps 32 --maj --model-path experimental/diffusion/outputs/mediom/final_model.pt
```

We did get one aa2 eval task correct on mediom (still running more evals): https://arcprize.org/play?task=981571dc

hard:
- simple (halfway model, 32 steps): 0%
- simple (best model, 32 steps): 0.42%
- simple (final model, 128 steps): 0%
- simple (final model, 32 steps): 0.83%
- sample-40x-augs (final model, 32 steps): 1.2%


This is the task that our lorge aa2-hard (30x30) model is getting partially correct: https://arcprize.org/play?task=269e22fb

```bash
uv run experimental/diffusion/evaluate.py --config experimental/diffusion/configs/lorge_config.json --num-steps 32 --model-path experimental/diffusion/outputs/lorge/halfway_model.pt --stats && \
uv run experimental/diffusion/evaluate.py --config experimental/diffusion/configs/lorge_config.json --num-steps 32 --model-path experimental/diffusion/outputs/lorge/best_model.pt && \
uv run experimental/diffusion/evaluate.py --config experimental/diffusion/configs/lorge_config.json --num-steps 32 --model-path experimental/diffusion/outputs/lorge/final_model.pt && \
uv run experimental/diffusion/evaluate.py --config experimental/diffusion/configs/lorge_config.json --num-steps 32 --maj --model-path experimental/diffusion/outputs/lorge/final_model.pt
```

### AA2-hard 20x20 Results
Trained on training-hard + evaluation, but only 20x20 grids.

```bash
uv run experimental/diffusion/evaluate.py --config experimental/diffusion/configs/smol_config.json --num-steps 32 --model-path experimental/diffusion/outputs/smol/halfway_model.pt --stats && \
uv run experimental/diffusion/evaluate.py --config experimental/diffusion/configs/smol_config.json --num-steps 32 --model-path experimental/diffusion/outputs/smol/best_model.pt && \
uv run experimental/diffusion/evaluate.py --config experimental/diffusion/configs/smol_config.json --num-steps 32 --model-path experimental/diffusion/outputs/smol/final_model.pt && \
uv run experimental/diffusion/evaluate.py --config experimental/diffusion/configs/smol_config.json --num-steps 32 --maj --model-path experimental/diffusion/outputs/smol/final_model.pt && \
uv run experimental/diffusion/evaluate.py --config experimental/diffusion/configs/mediom_config.json --num-steps 32 --model-path experimental/diffusion/outputs/mediom/halfway_model.pt --stats && \
uv run experimental/diffusion/evaluate.py --config experimental/diffusion/configs/mediom_config.json --num-steps 32 --model-path experimental/diffusion/outputs/mediom/best_model.pt && \
uv run experimental/diffusion/evaluate.py --config experimental/diffusion/configs/mediom_config.json --num-steps 32 --model-path experimental/diffusion/outputs/mediom/final_model.pt && \
uv run experimental/diffusion/evaluate.py --config experimental/diffusion/configs/mediom_config.json --num-steps 32 --maj --model-path experimental/diffusion/outputs/mediom/final_model.pt && \
uv run experimental/diffusion/evaluate.py --config experimental/diffusion/configs/lorge_config.json --num-steps 32 --model-path experimental/diffusion/outputs/lorge/halfway_model.pt --stats && \
uv run experimental/diffusion/evaluate.py --config experimental/diffusion/configs/lorge_config.json --num-steps 32 --model-path experimental/diffusion/outputs/lorge/best_model.pt && \
uv run experimental/diffusion/evaluate.py --config experimental/diffusion/configs/lorge_config.json --num-steps 32 --model-path experimental/diffusion/outputs/lorge/final_model.pt && \
uv run experimental/diffusion/evaluate.py --config experimental/diffusion/configs/lorge_config.json --num-steps 32 --maj --model-path experimental/diffusion/outputs/lorge/final_model.pt
```

smol:
Scores zeros.

mediom:
Scores zeros.

hard:
Scores zeros.

### Oct 2nd 2025
### Restrict to 20x20 grids and train on training-hard + evaluation from aa2.

Run all of those commands in series on the same gpu:
```bash
nohup bash -c 'PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/smol_config.json > smol-v1_20x20.log 2>&1 ; \
PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/mediom_config.json > mediom-v1__20x20.log 2>&1 ; \
PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/lorge_config.json > lorge-v1_20x20.log 2>&1' &
```

### Go back to original LR and restrict training data to training-hard subset
```bash
nohup bash -c 'PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/smol_config.json > smol-v1_training-hard.log 2>&1' &
```

```bash
nohup bash -c 'PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/mediom_config.json > mediom-v1_training-hard.log 2>&1' &
```

```bash
export HF_TOKEN=
nohup bash -c 'PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/lorge_config.json > lorge-v1_training-hard.log 2>&1' &
```

v1 model still getting zero correct when using the final model with 32 steps, even with --maj.

Try 128 steps:


### AA2 4X LR Ablation
Aiming to train 4x longer to see if it helps results.

```bash
nohup bash -c 'PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/smol_config.json > smol-v1_4x-lr.log 2>&1' &
```

```bash
nohup bash -c 'PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/mediom_config.json > mediom-v1_4x-lr.log 2>&1' &
```

```bash
export HF_TOKEN=
nohup bash -c 'PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/lorge_config.json > lorge-v1_4x-lr.log 2>&1' &
```

This was very unstable so I stopped the run.

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