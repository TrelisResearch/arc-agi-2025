**Todo**
[x] Implement cosine noise scheduler. Move to a uniform noise distribution.
[ ] Remove weight decay if duplicated?
[ ] Use optimizer updates not epochs.

**Things to consider adding:**
- Self-conditioning.
The idea is to pass previous logits as an additional input 50% of the time during training.

Things still to understand:
- EMA
- Label smoothing

**Clarifying Questions**
- Alpha and beta
Beta is the chance a cell gets repainted at step t. Alpha is the chance a cell stays the same! alpha_bar is the chance a cell survives without any change from start to finish.

- What is the interpertation of cross entropy? Would it be useful to also plot some kind of pixel accuracy metric? If so what and how?


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