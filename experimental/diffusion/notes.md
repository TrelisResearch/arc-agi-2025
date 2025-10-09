# Diffusion Notes

**Notes on Schedulers**
- What type of noise is being used?
v1 uses a uniform noise kernel. Tokens are replaced at random with uniformly sampled values across 0-9.

- What type of scheduler is being used in our v1 model? How does that compare to image and text diffusion models?
Possibly linear was used originally but is too aggressive early on and not aggressive enough at the end.

- What is the implication of using fewer steps?
Faster inference, and larger denoising steps.

- What are some possible improvements that could be made to the noise schedule?
We were embedding timestamps, but it's probably more fundamental to embed noise levels instead.

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
### Oct 9th 2025
#### Kicking off a long run
Will run for 1000000 optimizer steps.

#### v7 - fixed SC
So far, it seems like the v7 approach to SC is weaker than v6. Possibly just doing time-aligned is better than where we currently inference the same step twice. I'll need to re-run evaluation whereby I pass 128 steps, to see if that's the issue, and shorter steps doesn't work well without doing time steps.

```bash
nohup bash -c 'PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/smol_config_aa1.json > smol-v7.log 2>&1' &
```
Scores --maj with 48k steps (same time, not temporal): 7.6%
same time 2x attempts:  6.0%
temporal 2x attempts: 6.0%

so I had to re-train it with temporal (And that seems to have worked):
Scores --maj with 48k steps: 

re-run deep:
```bash
nohup bash -c 'PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/smol_config_aa1_deep.json > smol_config_aa1-v7-deep.log 2>&1' &
```

#### v6 - with cst LR
- uses cst learning rate
- uses ema by default everywhere
- uses 48k optimizer steps

Testing first on toiny, should take about 40 mins (logs say v4 instead of v6) - btw this is wrong because it's aa2 not aa1.
```bash
nohup bash -c 'PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/toiny_config.json > toiny-v4.log 2>&1' &
```
Scores --maj with 48k steps: 12.0%

Testing smol
```bash
nohup bash -c 'PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/smol_config_aa1.json > smol_config_aa1-v6.log 2>&1' &
```

and testing with a deeper network:
```bash
nohup bash -c 'PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/smol_config_aa1_deep.json > smol_config_aa1-v6-deep.log 2>&1' &
```

### Oct 8th 2025
#### Train LoRA and Embeddings of aa1 model to see if we can score on aa2!
```bash
nohup bash -c 'PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/smol_config_lora.json > smol-v3-reverted-lora.log 2>&1 ; \
PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/smol_config_aa1.json > smol-v3-reverted.log 2>&1 ; \
PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/mediom_config.json > mediom-v3-reverted.log 2>&1' &
```
smol scores 16.1% maj. roughly in line with the previous score.
mediom scores 0.83% with maj., which is quite a bit lower than 2.5% on v3. Seems a bit odd, and could be that ema is scoring lower? I re-ran ema. I've no idea why the scoring is lower than before. NOTE FOR THE FUTURE. MOVING ON AS V3-REVERTED IS FINE ON aa1.

re-test with `no-ema`: 0.8% as well.

```bash
nohup bash -c 'PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/smol_config_lora.json > smol-v3-reverted-lora.log 2>&1' &
```
Scores 0.83% with majority vote. Training is stable. Was trained for 8k steps.

#### Add evaluations during training so we can chart progress.
- We can now configure how often to run evaluations, and see that progress in wandb.
- We'll also use ema weights going forward in evaluate, which we weren't doing but should improve performance.

#### Reverting to Self Conditioning
Notes on self conditioning:
- Currently, training uses 50% SC, 25% nothing, 25% zeros in place of conditioning (weird?)

Checking out evaluation of the smol-v3 model, which should score ~ 16.5%:
```bash
uv run experimental/diffusion/evaluate.py --config experimental/diffusion/outputs/aa1/config.json --model-path experimental/diffusion/outputs/aa1/final_model.pt --limit 0
```
This scores 16.9%, which is excellent and what we want.

With `--maj`, updated since the last time (where we got 20.2%), we get: 19.8%, which is very similar, seems fine.

And now that ema is working, we are scoring without maj: 16.6% (no obvious boost).

### Oct 7th 2025
#### Try using prior grid prediction as feedback
- blend predicted grid with ground truth before noising. Sample evenly from both.
```bash
nohup bash -c 'PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/smol_config_aa1.json > smol-v5.log 2>&1 ; \
PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/mediom_config.json > mediom-v5.log 2>&1' &
```
Scores 11.2% without maj, on final_model.
Scores 6.9% without maj, on halfway model.



### Oct 6th 2025
#### Running v4 with an MLP in place of a linear layer for the self-conditioning
```bash
nohup bash -c 'PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/smol_config_aa1.json > smol-v4.log 2>&1 ; \
PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/mediom_config.json > mediom-v4.log 2>&1' &
```
Final model 2x attempt scores
smol aa1: 12.25% | 
mediom aa2: 1.2% | or 2.5% with --maj [compare this to 1.7% for --maj on v1 model]

mediom got 2 perfect and 2 partial.

Notes:
- I also ran the best model on aa1 and it scored only 1.8%, indicating that more training helps the score EVEN if the validation loss is falling. This means the validation loss alone is not all that useful. Perhaps it indicates overfitting when it falls, but there is something beneficial happening with more training that is still not seen.

#### Fixing up augmentations and running v3
Fixes:
- Instead of having an input for flips and one for rotates, we just have one with eight possible values (incl. the original) for the eight possible values of the dihedral group d4.
- Evaluation versus Training dataset weighting is now controled by `eval_weight`, and there is a pytorch sampler that allows for this.
- Validation is now done using evaluation test examples, a max of 128.

```bash
nohup bash -c 'PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/smol_config_aa1.json > smol-v3.log 2>&1' &
```
Scoring (no --maj): 16.5% | with --maj: 20.2%

#### Running v2 toiny with the snr input
```bash
nohup bash -c 'PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/toiny_config_aa1.json > toiny-v2.log 2>&1' &
```
Scoring (no --maj): 14.1% | with --maj: 13.0% (seems odd, I didn't dig in)

#### Running v2 smol with the snr input
```bash
nohup bash -c 'PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/smol_config_aa1.json > smol-v2.log 2>&1' &
```
Scoring (no --maj): 14.1% | with --maj: 18.2%.

### Oct 4th 2025
#### Running the v1 model on aa1 and aa2

aa1 results, all using majority voting of 40 attempts:
>  w/ wrong scheduler (although the scheduler issue appears to have been minor, the fix probably would boost the score up to around 15% on smol). This was fixed on commit 3a4c2f303ea7b7d769133a65c951d050b9c40965, after which the repo moved to using noise instead of timesteps as inputs.
aa1:
- smol: 13.8%. 
- mediom: 23.1%.
- lorge: 23.5%.

aa2:
> also with wrong scheduler, so these results probably should be a bit, maybe 20%-25% higher relative.
- smol: 0% (although a similar model has scored 0.4% before)
- mediom: 1.7%
- lorge: 2.1% (best model [no maj, 32 steps]: 1.2%)
- huoge [stopped at 2/3rds of the intended optimizer steps, not know why]: Scores 0%.

General Notes:
- *Val curves differ for aa1 and aa2* Unclear why the val/accuracy curves move upwards with model size for aa2 (as one would expect), but fall for aa1 - even though training loss curves fall for aa1. In both cases, the weighting of train to eval data in the training mix is 50-50. The validation dataset is taken at random from the mixed dataset used for training. PERHAPS THIS IS DUE TO A BAD CHOICE FOR VALIDATION DATA SPLIT, SEE THE BULLET TWO BELOW. Essentially, validation and training curves were somewhat reflective of each other (minus a distribution and sampling shift). Now the validation should be reflective of test output performance.
- *Tasks are solved with few initial diffusion steps* The tasks that are solved, when allowing 32 steps, are solved within the first or first few diffusion steps. I'm unsure if this indicates we have a sub-optimal noise scheduler. Apparently cosine does make sense here. One change I've made is, instead of embedding timestep, I'm now embedding alpha_bar_s, which is the level of noise at that given timestep. Hopefully this gets the model to more progressively de-noise.
- *Validation losses are not meaningful* Since I sample so many augmentations the validation split is contaminated with examples that are in training. Oddly, the training and validation loss are not the same, which is perhaps just because i) some validation examples don't appear in training (by chance), and ii) the distributions are not the same and therefore the trainer may be weighting performance towards certain permutations.

Improvements:
- Embed the noise level rather than an integer timestamp (to which a sinusoidal embedding was applied). DONE FOR V2.
- Consider training for longer, because the final checkpoint always seems to perform best. Doing this now.
- We sample augmentations at random, which gives a very slightly non-uniform distribution of augmentations because there are overlapping combos (rotate 180 + flip horizontal is equivalent to flip diagonal). Probably doesn't have a huge effect. DONE.
- Save the model optimizer state and more frequent checkpoints, to allow for training restarts. NOT DONE YET.

**aa2 results**
```bash
nohup bash -c 'PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/mediom_config.json > mediom-v1-boost.log 2>&1 ; \
PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/lorge_config.json > lorge-v1-boost.log 2>&1 ; \
PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/huoge_config.json > huoge-v1-boost.log 2>&1
PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/giont_config.json > giont-v1-boost.log 2>&1' &
```
- smol: 0% (although a similar model has scored 0.4% before)
- mediom: 1.7%
- lorge: 2.1% [1.2% with 128 steps, no maj]. 1.2% [unchanged after SC and scheduler fix].
- huoge [stopped at 263299/384000]: 0% running best model.

On lorge, seeing this task correct: `71e489b6` and `981571dc` (the symetric complex pattern). Note that the task is correct after just a few diffusion steps and then stays the same from step 26 down to 0. For `981571dc`, the solution appears to be diffused out almost immediately.

**aa1 results**
```bash
nohup bash -c 'PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/smol_config_aa1.json > smol-v1-aa1.log 2>&1 ; \
PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/mediom_config_aa1.json > mediom-v1-aa1.log 2>&1 ; \
PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/lorge_config_aa1.json > lorge-v1-aa1.log 2>&1 ; \
PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/huoge_config_aa1.json > huoge-v1-aa1.log 2>&1 ; \
PYTHONUNBUFFERED=1 uv run experimental/diffusion/pipeline.py --config experimental/diffusion/configs/giont_config_aa1.json > giont-v1-aa1.log 2>&1' &
```
Scoring - all `--num-steps 32 --maj` - WITH A BROKEN INFERENCE SCHEDULER (WAS NOT CORRECTLY MAPPING TIMESTEPS TO THE ORIGINAL 128 STEPS):
- smol: 13.8% | with --num-steps 32 and without --maj: 9.1%.
- mediom: 23.1%.
- lorge: 23.5% | with --num-steps 32 and without --maj: 17.6% | scores 19% with scheduler fixed and without majority voting.
- huoge: don't plan to run this.

Scoring - all `--num-steps 32 --maj` - with scheduler fixed for inference BUT SC is wrong!:
- smol: 9.0% [but self-conditioning is wrong!]
- mediom: ...
- lorge: ...

Scoring - all with --num-steps 32 and without --maj - with scheduler fixed AND SC fixed:
- smol: 10.1%
- mediom: ...
- lorge: ...

So maybe that fix does have some influence.

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

### Oct 9th 2025
#### Architectural cleanup (Part 1)
Removed dead code and consolidated self-conditioning (SC) logic:
- **Removed ARCDiffusionSampler class** (~195 lines): This class had a blend-and-re-noise variant that was never called. All inference goes through DiffusionInference.sample_with_steps with temporal SC (previous step's probs).
- **Fixed "no-SC" to be truly zero**: Removed the `elif` branch that added `sc_proj(zeros)`, which was injecting bias even when SC was disabled.
- **Moved sc_dropout_prob from model to trainer**: Consolidated all SC logic in the trainer, not split between model and trainer. This parameter now lives in the `training` section of config files, not the `model` section.
- **Created smol_config_aa1_deep.json**: 4x deeper network with same parameter count: 192d × 16 layers (vs 384d × 4 layers). Tests depth vs width trade-off.

Files modified:
- `src/model.py`: Removed sc_dropout_prob parameter from DiffusionDenoiser and ARCDiffusionModel
- `src/training.py`: Removed ARCDiffusionSampler class, added sc_dropout_prob to ARCDiffusionTrainer
- `evaluate.py`: Removed ARCDiffusionSampler import
- `configs/*.json`: Moved sc_dropout_prob from "model" to "training" section

#### Self-conditioning improvements (Part 2)
Improved SC mechanism for better stability and effectiveness:

**1. Enhanced SC projection head:**
- Replaced `nn.Linear(10, d_model)` with `nn.Sequential(nn.Linear(10, d_model), nn.LayerNorm(d_model))`
- Added learnable scalar gate `nn.Parameter(torch.tensor(0.3))` to allow model to downweight SC when harmful
- LayerNorm keeps scale consistent across timesteps

**2. Switched from probabilities to log-probs:**
- Changed SC input from `softmax(logits)` to tempered, centered log-probs
- Temperature = 1.5 for stability at high noise
- Center by subtracting per-cell mean: `log_probs - log_probs.mean(dim=-1, keepdim=True)`
- Log-probs preserve margin info better than saturated probabilities

**3. Fixed to same-timestep two-pass SC:**
- **Training & validation**: Already used two-pass (pass-1: no SC → get logits → pass-2: with SC)
- **Inference**: Changed from temporal SC (using previous step's probs) to same-timestep two-pass
- Each denoising step now does: pass-1 (no SC) → create SC input → pass-2 (with SC) → update x_t

**4. Deterministic sampling:**
- Changed inference to use `argmax` at all timesteps (was: sampling for t>0, argmax for t=0)
- Better for ARC tasks where diversity is not needed

**5. Applied masks to SC input:**
- Zero SC features outside valid regions before projection

Files modified:
- `src/model.py`: Updated SC projection head, added gate, masked SC input
- `src/training.py`: Changed to log-probs with temperature and centering, added SC_TEMPERATURE constant, fp32 stability
- `evaluate.py`: Changed to same-timestep two-pass, deterministic sampling, fixed constructor call, added SC_TEMPERATURE constant, fp32 stability

**Additional stability improvements:**
- Hoisted temperature to `SC_TEMPERATURE = 1.5` constant at top of training.py and evaluate.py for consistency
- Wrapped log_softmax computation in fp32 for stability: `logits.float() → log_softmax → .to(dtype)` to prevent NaNs at high noise
- Removed `sc_dropout_prob` from `DiffusionInference._load_model()` constructor call (was causing crash)

**Fixed denoising progression visualizations:**
- `create_denoising_progression_visualization` was using single-pass with no SC
- Also had a bug: was passing timestep index where model expects logsnr
- Updated to use same two-pass SC as training/inference
- Now properly computes logsnr from timestep before passing to model
- Visualizations now accurately reflect actual model behavior