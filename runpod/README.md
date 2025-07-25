# RunPod Pod Management

Create and manage RunPod pods using YAML/JSON templates.

## Setup

```bash
export RUNPOD_API_KEY="your-api-key-here"
```

Or add it to your `.env` file.

## Usage

**Recommended (TCP with testing):**
```bash
uv run runpod/create_pod_tcp.py sglang-tcp -- --model-path <model> [options]
```
*Waits for container + tests endpoint = ready-to-use experience*

**Skip endpoint testing (faster setup):**
```bash
uv run runpod/create_pod_tcp.py sglang-tcp --no-health-check -- --model-path <model> [options]
```
*Shows connection info immediately, no waiting*

**Legacy (HTTP only):**
```bash
uv run runpod/create_pod.py <template> [options] -- [extra_args...]
```

## Examples

**Recommended - TCP with automatic testing:**
```bash
uv run runpod/create_pod_tcp.py sglang-tcp -- --model-path Qwen/Qwen3-4B
```

**With reasoning parser:**
```bash
uv run runpod/create_pod_tcp.py sglang-tcp -- --model-path Qwen/Qwen3-4B --reasoning-parser qwen3
```

**Skip health check for faster startup:**
```bash
uv run runpod/create_pod_tcp.py sglang-tcp --no-health-check -- --model-path Qwen/Qwen2.5-72B
```

## Advanced Options

**Debug mode (shows full pod info):**
```bash
uv run runpod/create_pod_tcp.py sglang-tcp --debug -- --model-path Qwen/Qwen3-4B
```

**Skip health check for faster startup:**
```bash
uv run runpod/create_pod_tcp.py sglang-tcp --no-health-check -- --model-path Qwen/Qwen3-4B
```

The scripts automatically handle pod creation, health checks, and cleanup on Ctrl+C.
