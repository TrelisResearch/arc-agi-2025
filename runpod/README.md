# RunPod Pod Management

This directory contains scripts and templates for creating and managing RunPod pods with configurable health checks.

## Setup

### Required Environment Variables

```bash
export RUNPOD_API_KEY="your-api-key-here"
```

Or add it to your `.env` file.

### RunPod Secrets (for Hugging Face models)

For private Hugging Face models, you need to configure a RunPod secret:

1. Go to your [RunPod Settings](https://www.runpod.io/console/user/settings)
2. Add a new secret named `HUGGING_FACE_HUB_TOKEN` with your HF token
3. The templates will automatically use this secret via `{{ RUNPOD_SECRET_HUGGING_FACE_HUB_TOKEN }}`

Alternatively, you can modify the template to use a local environment variable instead of a RunPod secret.

## Available Templates

- **`sglang`**: SGLang inference server (recommended for most models)
- **`vllm`**: vLLM inference server (optimized for OSS models)

## Usage

### Basic Pod Creation

```bash
uv run create_pod.py <template> [--no-health-check] [--debug] -- [docker_args...]
```

### Automated Pod Creation + ARC Task Running

```bash
uv run runpod/create_pod_and_run_tasks.py <dataset> <model_path> [options]
```

This combined script will:
1. Create a RunPod pod with the specified model
2. Wait for the OpenAI endpoint to be ready
3. Automatically run ARC tasks with configured settings
4. Keep the pod running for reuse until you press Ctrl+C

#### Arguments:
- `dataset`: Either `arc-agi-1` or `arc-agi-2`
- `model_path`: Full Hugging Face model path (e.g., `Trelis/Qwen3-4B_...`)

#### Options:
- `--template`: RunPod template to use (default: `sglang`, can also use `vllm`)
- `--subset`: Dataset subset to run (default: `all_evaluation`)
- `--skip-tasks`: Only create the pod without running ARC tasks
- `--no-health-check`: Skip health check after pod creation

#### Task Runner Configuration:
The script automatically runs tasks with these settings:
- Subset: `all_evaluation` (configurable with `--subset`)
- Attempts: 8
- Repeat runs: 3
- Max workers: 32
- Max tokens: 1000
- Includes `--unsafe-executor` and `--qwen-no-think` flags

#### Examples:

```bash
# Run arc-agi-1 evaluation with default sglang template
uv run runpod/create_pod_and_run_tasks.py arc-agi-1 Trelis/Qwen3-4B_dsarc-agi-1-train-programs-best-length-filtered-250_20250811-155856-c904

# Run arc-agi-2 evaluation with vllm template
uv run runpod/create_pod_and_run_tasks.py arc-agi-2 Trelis/Your-Model --template vllm

# Run with specific subset
uv run runpod/create_pod_and_run_tasks.py arc-agi-1 Trelis/Your-Model --subset shortest_1

# Just create pod without running tasks (for manual testing)
uv run runpod/create_pod_and_run_tasks.py arc-agi-1 Trelis/Your-Model --skip-tasks
```

#### What happens after task completion:
- The script reports success/failure of the task runs
- The pod remains running so you can:
  - Run additional experiments with the same endpoint
  - Manually test the model
  - Avoid model reload time for subsequent runs
- Press Ctrl+C to terminate and delete the pod when done

### Examples

#### SGLang (recommended for most models)

```bash
# Create a basic TCP pod with SGLang
uv run create_pod.py sglang -- --model-path Qwen/Qwen3-4B

# Create a TCP pod with LORA adapter
uv run create_pod.py sglang -- --model-path Qwen/Qwen3-4B --lora-paths ckpt-1057=Trelis/my-lora-adapter --max-loras-per-batch 1 --disable-radix-cache

# Create a TCP pod with multiple LORA adapters  
uv run create_pod.py sglang -- --model-path Qwen/Qwen3-4B --lora-paths ckpt-1057=Trelis/lora-1,ckpt-2114=Trelis/lora-2 --max-loras-per-batch 2 --disable-radix-cache
```

#### vLLM (optimized for OSS models)

```bash
# Create a basic TCP pod with vLLM
uv run create_pod.py vllm -- --model microsoft/DialoGPT-medium

# Create a vLLM pod with tensor parallel processing
uv run create_pod.py vllm -- --model meta-llama/Llama-2-7b-hf --tensor-parallel-size 2

# Create a vLLM pod with custom settings
uv run create_pod.py vllm -- --model huggingface-hub/CodeLlama-7b-Python-hf --max-model-len 4096 --dtype float16
```

#### General Options

```bash
# Skip health checks entirely (works with any template)
uv run create_pod.py sglang --no-health-check -- --model-path Qwen/Qwen3-4B
uv run create_pod.py vllm --no-health-check -- --model microsoft/DialoGPT-medium

# Enable debug output (works with any template)
uv run create_pod.py sglang --debug -- --model-path Qwen/Qwen3-4B
uv run create_pod.py vllm --debug -- --model microsoft/DialoGPT-medium
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
