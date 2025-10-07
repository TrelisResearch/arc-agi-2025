# iTransformer (iterative Transformer) Notes

## October 7th 2025
### v0 itransformer design

Run toiny:
```bash
PYTHONUNBUFFERED=1 nohup uv run itransformer/pipeline.py --config itransformer/configs/toiny_config_aa1.json > toiny_config_aa1.log &
```

Run smol:
```bash
PYTHONUNBUFFERED=1 nohup uv run itransformer/pipeline.py --config itransformer/configs/smol_config_aa1.json > smol_config_aa1.log &
```