### LoRA merge helper

Merge two PEFT LoRA adapters into a single adapter (linear or TIES), optionally bake into full model weights, and then use the result with Unsloth.

#### Why this exists
- Unsloth can be finicky when loading multiple adapters and doing merges in place. The reliable path is: merge with vanilla PEFT → produce a single merged adapter → load that one adapter (or a baked full model) with Unsloth.

### Requirements
- Python env with `torch`, `transformers`, `peft`, `huggingface_hub`.
- Prefer `uv` for running/installing.

Install (if needed):
```bash
uv add torch transformers peft huggingface_hub
```

### Usage
Run the CLI to merge two adapters. Supports `--method linear` or `--method ties`.

Minimal example (your Qwen/Qwen3-4B adapters):
```bash
uv run experimental/lora-merges/merge-options.py \
  --base-id Qwen/Qwen3-4B \
  --a1-repo Trelis/Qwen3-4B_dsarc-programs-50-full-200-incorrect_20250808-134330-trainer \
  --a1-sub checkpoint-2874 \
  --a1-name incorrect2874 \
  --a2-repo Trelis/Qwen3-4B_dsarc-programs-50-full-200-partial_20250807-211749-trainer \
  --a2-sub checkpoint-2114 \
  --a2-name partial2114 \
  --method ties \
  --density 0.3 \
  --sign total \
  --out adapters/qwen3-4b_incorrect2874__partial2114_ties \
  --bake-dir models/qwen3-4b_incorrect2874__partial2114_ties_baked
```

Options (selected):
- `--method {linear,ties}`: linear blend vs TIES merge.
- `--weights w1 w2`: adapter weights (default 1.0 1.0). For linear, these are the blend weights; for TIES they weight the surviving entries.
- `--density d` (TIES): fraction kept (0<d≤1). Typical sweep 0.2–0.5.
- `--sign {frequency,total}` (TIES): how to resolve sign conflicts. `total` keeps the stronger side; `frequency` drops perfect ties.
- `--out DIR`: where to save the merged adapter (LoRA files). Loadable with `PeftModel.from_pretrained`.
- `--bake-dir DIR`: also create a baked full model (no adapter dependency).
- `--push-adapter-repo` / `--push-baked-repo`: optionally push to the Hub (requires token); add `--private` to create private repos.

### Using the result with Unsloth
Load the single merged adapter (recommended for modularity):
```python
from unsloth import FastLanguageModel
from peft import PeftModel

base_id = "Qwen/Qwen3-4B"
merged_dir = "adapters/qwen3-4b_incorrect2874__partial2114_ties"
merged_name = "incorrect2874__partial2114_ties"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_id,
    load_in_4bit=False,
    fast_inference=True,
    max_lora_rank=64,
    gpu_memory_utilization=0.3,
)

model = PeftModel.from_pretrained(
    model,
    model_id=merged_dir,
    adapter_name=merged_name,
)
model.set_adapter(merged_name)
```

Or load the baked full model (no adapter needed):
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="models/qwen3-4b_incorrect2874__partial2114_ties_baked",
    load_in_4bit=False,
    fast_inference=True,
)
```

### Recommendations (short)
- **Start point**: TIES with equal weights, density in 0.3–0.5. If you need to preserve unique skills on conflicts, use `--sign total`; if you prefer to drop conflicts, `--sign frequency`.
- **Linear baseline**: Quick, often fine when tasks are very similar. Try weights like `--weights 0.7 0.3` if one adapter should dominate.
- **Density**: lower (≈0.2–0.3) = more pruning (less interference, riskier to drop useful signal). Higher (≈0.4–0.5) = gentler pruning, better when tasks are similar.
- **Unsloth gotcha**: avoid loading two adapters into an Unsloth-wrapped model to merge; do the merge here, then load the single adapter or baked model.
- **Quantization**: do not bake (`merge_and_unload`) from a 4-bit/8-bit base.

### Notes
- This script loads the base with `trust_remote_code` by default (Qwen variants often require it). You can disable with `--trust-remote-code` omitted and pass `--trust-remote-code` only when needed.
- Tokenizer is only needed when pushing the baked full model to the Hub.

