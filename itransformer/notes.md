# iTransformer (iterative Transformer) Notes

## October 7th 2025
### v2
- remove embedding for step number
- switch to back-prop after each segment, to encourage later steps to learn
- turn off noise in inputs

Run toiny:
```bash
PYTHONUNBUFFERED=1 nohup uv run itransformer/pipeline.py --config itransformer/configs/toiny_config_aa1.json > toiny_config_aa1_v2.log &
```

Run smol:
```bash
PYTHONUNBUFFERED=1 nohup uv run itransformer/pipeline.py --config itransformer/configs/smol_config_aa1.json > smol_config_aa1_v2.log &
```


### v1 
- remove learnable start token, so we just always start from black now.
- ramp noise up to 50% of inputs, and at a 10% level; up from 10% of grids at a 5% level.

Cancelled the runs as loss curves were the same as v0.

### v0 itransformer design

Run toiny:
```bash
PYTHONUNBUFFERED=1 nohup uv run itransformer/pipeline.py --config itransformer/configs/toiny_config_aa1.json > toiny_config_aa1.log &
```
Scores, without --maj: 0.6%. 

📏 SIZE PREDICTION:
  Correct: 281/419 (67.1%)

Run smol:
```bash
PYTHONUNBUFFERED=1 nohup uv run itransformer/pipeline.py --config itransformer/configs/smol_config_aa1.json > smol_config_aa1.log &
```
Scores, without --maj: 1.2%
📏 SIZE PREDICTION:
  Correct: 284/419 (67.8%)