# Manual Task Augmentation

Two approaches for creating augmented ARC tasks:

## Approach 1: Validated Augmentation
**Script**: `run_augmentation.py`

Creates augmented tasks guaranteed to have working programs:
- Supports noise (replace black cells) and geometric (flip/rotate/recolor) transformations
- Validates existing programs still work on augmented variants
- Only keeps successful augmentations
- Expands parquet with new program samples

**Usage**:
```bash
# Noise augmentation
uv run python run_augmentation.py --parquet-path /path/to/parquet --augmentation-type noise --noise-percentage 0.1

# Geometric augmentation
uv run python run_augmentation.py --parquet-path /path/to/parquet --augmentation-type geometric
```

## Approach 2: Simple Augmentation
**Script**: `simple_noise_augmentation.py`

Creates noise augmented tasks without validation:
- Augments all tasks (no program requirement)
- No validation or parquet expansion

**Usage**:
```bash
uv run python simple_noise_augmentation.py --num-augmentations 5 --noise-percentage 0.05
```

## Output
- Augmented tasks: `data/manual/arc-agi_augmented_*.json`
- Task naming: `{original_id}_aug_{variant_number}`