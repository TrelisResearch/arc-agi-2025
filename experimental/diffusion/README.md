# ARC Diffusion Model

A discrete diffusion model for solving ARC (Abstraction and Reasoning Corpus) tasks.

## Quick Start

```bash
# Train on MPS (Mac)
uv run python train_with_config.py --config configs/mps_config.json

# Train on GPU
uv run python train_with_config.py --config configs/gpu_config.json

# Train on CPU (testing)
uv run python train_with_config.py --config configs/cpu_config.json

# Run inference
uv run python run_inference.py --model-path outputs/mps/best_model.pt --dataset arc-prize-2024 --subset evaluation --limit 5 --output results.json

# View results
# Open viewer.html in browser, upload results.json
```

## Known Issues

### 🚨 Data Augmentation + Size Prediction
**Problem:** Data augmentation (especially rotations) corrupts size prediction labels.

**Details:** When we rotate a 3×5 grid to 5×3, the model sees the rotated grid but still gets told the size should be 3×5. This creates contradictory training data that confuses the size predictor.

**Status:** Augmentation temporarily disabled (`"augment": false`) until this is fixed.

**Solutions to investigate:**
1. Fix augmentation to properly update height/width labels after rotations
2. Use only flips (not rotations) since they preserve dimensions
3. Consider dropping size prediction entirely if the current approach is too problematic

### 💡 Alternative: Drop Size Prediction?
**Question:** Could we drop size prediction altogether?

**Answer:** Technically yes, but it would hurt performance:
- Size prediction helps the model know how much of the 30×30 padded grid to use
- Without it, we'd need to rely on PAD token masking alone
- Many ARC tasks have predictable size patterns that are valuable signals

**Better approach:** Fix the augmentation issue rather than removing a useful capability.

## Architecture

- **Model:** Transformer-based discrete diffusion
- **Conditioning:** Input grid + task ID
- **Guidance:** Classifier-free guidance (CFG)
- **Size Prediction:** Unified head (when working properly)
- **Vocab:** 0-9 (ARC colors) + 10 (PAD token)