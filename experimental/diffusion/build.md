ARC Diffusion Approach — High-Level Summary
Goal
Predict an output grid from an input grid for an ARC task using a discrete diffusion model that supports variable grid sizes (up to 30×30) and task conditioning.
Data & Representations
Max canvas: 30×30.
Vocabulary: {0..9, PAD} → K = 11 (PAD clamps unused cells).
Size labels: (H, W) with 1..30.
Task conditioning: one or more task IDs → learned embeddings (sum/mean if multiple).
Masks: M is 1 on [0:H, 0:W], else 0.
Overall Flow
Predict size (H, W) from (input_grid, task_tokens).
Diffuse the output grid on the full 30×30 canvas with categorical diffusion, but:
Compute loss only where M=1.
Clamp tokens outside [0:H,0:W] to PAD at every sampling step.
(We do not feed a “train vs test” flag; generalization is driven by the input structure and task tokens.)
Model Components
Backbone: small Transformer / ViT over 900 tokens.
Inputs per step: x_t (noisy output tokens), positional encodings, time embedding, task embeddings, and an encoding of the input grid (added or cross-attended).
Size head: tiny classifier predicting (H, W) from a pooled conditioner (e.g., encoder features of the input grid + task tokens).
Output head: per-cell logits over K classes.
Diffusion Setup (Discrete / D3PM-style)
Noise kernel: uniform-mix (at each step, keep token with prob 1−β_t, else replace with a random token).
Steps: T = 32 (start) — 64 if needed.
Schedule: cosine or linear; default cosine with β_min=1e-4, β_max=2e-2.
Target parameterization: predict p(x₀ | x_t, cond); train with cross-entropy to the clean token at each cell.
Training
For each example (task_tokens, input_grid, output_grid, H, W):
Build x₀ by padding the output with PAD outside [0:H,0:W]; build mask M.
Sample t ∈ {1..T} and corrupt x₀ → x_t.
Denoiser forward: logits = fθ(x_t, input_grid, task_tokens, t).
Grid loss: CE(logits, x₀) masked by M (ignore PAD region).
Size loss: CE for H and W from the size head (weight ~0.2).
Classifier-free guidance (CFG) training: drop task/input cond with prob p=0.15.
Optional stabilizers: self-conditioning (feed previous x̂₀), color-permutation + joint rotations/reflections on (input, output).
Inference (Sampling)
Predict size: Ĥ, Ŵ = size_head(input_grid, task_tokens).
Make mask M̂ for [0:Ĥ,0:Ŵ]; init x_T (random tokens or all PAD).
For t = T…1:
logits = fθ(x_t, input_grid, task_tokens, t).
CFG at test: combine cond/uncond logits (scale 1.5–3.0).
Sample/argmax to get x_{t−1}.
Clamp: set x_{t−1}[~M̂] = PAD.
Return x̂₀[:Ĥ, :Ŵ].
Defaults to start with
T=32, cosine schedule (1e-4 → 2e-2).
CFG drop p=0.15; guidance = 2.0.
Transformer ~8 layers, d_model≈384, 6 heads.
AdamW lr=3e-4, batch size ≈ 64 (grids are tiny).
No “train/test” flag; if desired, pass input-derived complexity scalars (e.g., #components, entropy) for logging or to adapt sampler settings (not the mapping).
Evaluation Notes
Report accuracy on exact match within [0:Ĥ,0:Ŵ].
Track size accuracy (H,W) separately.
Ablate: with/without CFG; T=32 vs T=64; with/without self-conditioning.
This is the simplest robust path: predict size → PAD-masked categorical diffusion, conditioned on task tokens + input grid, with hard clamping outside the predicted rectangle.

Other notes:
- Training: CUDA H200.
- Logging: wandb (see how wandb is used in the unsloth finetuning notebook).
- Training data, arc-prize-2025 training (use train and test examples), arc-prize-2025 evaluation (use train examples only). We'll evaluate on arc-prize-2025 evaluation split. This data is in the data/ folder in root.