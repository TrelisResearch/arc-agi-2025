# RunPod Pod Management

This directory contains scripts and templates for creating and managing RunPod pods with configurable health checks.

## Setup

```bash
export RUNPOD_API_KEY="your-api-key-here"
```

Or add it to your `.env` file.

## Usage

```bash
uv run create_pod.py <template> [--no-health-check] [--debug] -- [docker_args...]
```

### Examples

```bash
# Create an HTTP pod with default health checks
uv run create_pod.py sglang -- --model-path Qwen/Qwen3-4B

# Create a TCP pod with OpenAI endpoint testing
uv run create_pod.py sglang -- --model-path Qwen/Qwen3-4B --reasoning-parser qwen3

# Create a TCP pod with LORA adapter loaded (requires --disable-radix-cache)
uv run create_pod.py sglang -- --model-path Qwen/Qwen3-4B --lora-paths ckpt-1057=Trelis/Qwen3-4B_dsarc-programs-50-full-200-partial_20250807-211749-trainer/checkpoint-1057 --max-loras-per-batch 1 --disable-radix-cache

# Create a TCP pod with multiple LORA adapters loaded (requires --disable-radix-cache)
```bash
uv run runpod/create_pod.py sglang -- --model-path Qwen/Qwen3-4B \
  --lora-paths \
    ckpt-1057=Trelis/Qwen3-4B_dsarc-programs-50-full-200-partial_20250807-211749-trainer/checkpoint-1057 \
    ckpt-2114=Trelis/Qwen3-4B_dsarc-programs-50-full-200-partial_20250807-211749-trainer/checkpoint-2114 \
    ckpt-3171=Trelis/Qwen3-4B_dsarc-programs-50-full-200-partial_20250807-211749-trainer/checkpoint-3171 \
  --max-loras-per-batch 3 \
  --max-loaded-loras 3
```

# Skip health checks entirely
uv run create_pod.py sglang --no-health-check -- --model-path Qwen/Qwen3-4B

# Enable debug output
uv run create_pod.py sglang --debug -- --model-path Qwen/Qwen3-4B
```

## Health Check Configuration

Templates can include a `healthCheck` section to configure how the pod's readiness is verified. This section is automatically stripped before sending the configuration to RunPod.

### Health Check Types

1. **HTTP** (`type: "http"`)
   - Tests HTTP endpoint at specified path
   - Default path: `/health`
   - Good for: Web services, APIs with health endpoints

2. **OpenAI** (`type: "openai"`)
   - Tests OpenAI-compatible API endpoint with a dummy request
   - Requires model path from command line arguments
   - Good for: SGLang, vLLM, other inference libraries

3. **TCP** (`type: "tcp"`)
   - Basic TCP connectivity check
   - Tests if port is accepting connections
   - Good for: Database connections, basic service validation

4. **None** (`type: "none"`)
   - Skips health checks entirely
   - Pod is considered ready immediately after startup

### Configuration Options

```yaml
healthCheck:
  type: "http"        # Required: http, openai, tcp, or none
  path: "/health"     # Optional: Path for HTTP checks (default: /health)
  timeout: 300        # Optional: Timeout in seconds (default varies by type)
```

### Default Behavior

If no `healthCheck` section is provided:
- **TCP ports**: Uses OpenAI health check (tests `/v1` endpoint)
- **HTTP ports**: Uses HTTP health check with `/health` path

## Template Examples

### HTTP Service with Custom Health Endpoint
```yaml
name: "my-web-service"
# ... other config ...
ports:
  - "8080/http"
healthCheck:
  type: "http"
  path: "/api/health"
  timeout: 180
```

### OpenAI-Compatible Inference Server
```yaml
name: "sglang-server"
# ... other config ...
ports:
  - "8080/tcp"
healthCheck:
  type: "openai"
  timeout: 600
```

### Database with TCP Check
```yaml
name: "postgres-db"
# ... other config ...
ports:
  - "5432/tcp"
healthCheck:
  type: "tcp"
  timeout: 120
```

### Skip Health Checks
```yaml
name: "debug-pod"
# ... other config ...
healthCheck:
  type: "none"
```

## Features

- **Unified Script**: Handles both HTTP and TCP ports in a single script
- **Configurable Health Checks**: Template-based health check configuration
- **Auto-detection**: Smart defaults based on port types
- **Direct + Proxy**: Tests both direct IP connections and RunPod proxy
- **Graceful Shutdown**: Ctrl+C properly cleans up pods
- **Debug Mode**: Show detailed pod information with `--debug`

## Environment

Requires `RUNPOD_API_KEY` environment variable to be set.
