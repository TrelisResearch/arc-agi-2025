# RunPod Pod Management

Create and manage RunPod pods using YAML/JSON templates.

## Setup

```bash
export RUNPOD_API_KEY="your-api-key-here"
```

Or add it to your `.env` file.

## Usage

```bash
uv run runpod/create_pod.py <template> [options] -- [extra_args...]
```

## Examples

Basic SGLang server:
```bash
uv run runpod/create_pod.py sglang -- --model-path Qwen/Qwen3-4B
```

SGLang with reasoning parser:
```bash
uv run runpod/create_pod.py sglang -- --model-path Qwen/Qwen3-4B --reasoning-parser qwen3
```

Skip health check:
```bash
uv run runpod/create_pod.py sglang --no-health-check -- --model-path Qwen/Qwen2.5-72B
```

The script automatically handles pod creation, health checks, and cleanup on Ctrl+C.
