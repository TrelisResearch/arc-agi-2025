# Python Code Executors

This module provides secure Python code execution through multiple backends with a common interface.

## Features

- **Class-based architecture** with a common `BaseExecutor` interface
- **Unrestricted execution** via subprocess (fast, but less secure)
- **Docker sandbox execution** for isolation (secure, but requires Docker)
- **Context manager support** for automatic cleanup
- **Native Python object returns** via pickle serialization
- **Timeout support** for long-running code
- **Comprehensive error handling** with exception propagation

## Quick Start

### Unrestricted Executor (Subprocess)

```python
from unrestricted_executor import UnrestrictedExecutor

# Method 1: Manual setup/cleanup
executor = UnrestrictedExecutor()
executor.setup()

result, error = executor.execute_code("return 2 + 2")
print(f"Result: {result}")  # Output: Result: 4

executor.cleanup()

# Method 2: Context manager (recommended)
with UnrestrictedExecutor() as executor:
    result, error = executor.execute_code("""
numbers = [1, 2, 3, 4, 5]
return sum(numbers) * 2
""")
    print(f"Result: {result}")  # Output: Result: 30
```

### Docker Sandbox Executor (Isolated)

```python
from docker_sandbox import DockerSandboxExecutor

# Requires Docker to be installed and running
with DockerSandboxExecutor() as executor:
    result, error = executor.execute_code("""
import os
return {'working_dir': os.getcwd(), 'user': os.getenv('USER', 'unknown')}
""")
    
    if error:
        print(f"Error: {error}")
    else:
        print(f"Container info: {result}")
```

## Installation

### Basic Requirements

```bash
# For the base executor (subprocess-based)
# No additional dependencies required

# For legacy compatibility
pip install pickle5  # Optional, for older Python versions
```

### Docker Executor Requirements

```bash
# Install Docker dependencies
pip install docker requests

# Make sure Docker is installed and running
docker --version
```

### Development Dependencies

```bash
pip install -r requirements.txt
```

## Architecture

### Base Interface

All executors implement the `BaseExecutor` abstract base class:

```python
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

class BaseExecutor(ABC):
    @abstractmethod
    def execute_code(self, code: str, timeout: Optional[float] = None) -> Tuple[Any, Optional[Exception]]:
        """Execute Python code and return (result, error)."""
        pass
    
    @abstractmethod
    def setup(self) -> None:
        """Initialize the executor."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources."""
        pass
```

### Executor Implementations

1. **UnrestrictedExecutor**: Runs code in a subprocess on the host system
   - ✅ Fast startup and execution
   - ✅ Full access to host Python environment
   - ❌ No isolation (security risk)
   - ❌ Can affect host system

2. **DockerSandboxExecutor**: Runs code in an isolated Docker container
   - ✅ Complete isolation and security
   - ✅ Consistent environment
   - ✅ Automatic container lifecycle management
   - ❌ Slower startup (builds image, starts container)
   - ❌ Requires Docker installation

## Code Format

All executors expect code that can be wrapped in a function and should return a value:

```python
# Good - returns a value
code = "return 42"
code = "x = 10; y = 20; return x + y"
code = """
import math
numbers = [1, 4, 9, 16]
return [math.sqrt(n) for n in numbers]
"""

# Bad - no return statement
code = "print('hello')"  # Returns None
code = "x = 42"          # Returns None
```

## Error Handling

Executors return a tuple of `(result, error)`:

```python
result, error = executor.execute_code("return 1 / 0")
if error:
    print(f"Execution failed: {error}")
    # error will be: ZeroDivisionError('division by zero')
else:
    print(f"Success: {result}")
```

## Timeouts

Set execution timeouts to prevent infinite loops:

```python
# 5-second timeout
result, error = executor.execute_code("""
import time
time.sleep(10)  # This will timeout
return "done"
""", timeout=5.0)

if error:
    print(error)  # Will be a timeout error
```

## Docker Configuration

### Custom Image

You can customize the Docker image by modifying:

- `Dockerfile`: Base image and system dependencies
- `sandbox_requirements.txt`: Python packages installed in container
- `sandbox_server.py`: FastAPI server running in container

### Port Configuration

```python
# Use custom port
executor = DockerSandboxExecutor(container_port=9000)
```

## Testing

Run the test suite using pytest:

```bash
# From the python-sandbox directory (recommended)
cd python-sandbox
uv run pytest -v

# Or from the workspace root
uv run pytest python-sandbox/tests/ -v

# Run specific test classes
uv run pytest tests/test_executors.py::TestUnrestrictedExecutor -v

# Run with minimal output
uv run pytest --tb=line
```

The test suite includes:
- ✅ Basic executor functionality
- ✅ Error handling and edge cases  
- ✅ Legacy function compatibility
- ✅ Factory function testing
- ✅ Context manager support
- ✅ Docker executor tests (skipped if Docker not available)

Run examples:

```bash
python3 example_usage.py
```

## Legacy Support

The original function-based API is still supported:

```python
from unrestricted_executor import execute_python_code_with_result

result, error = execute_python_code_with_result("return 'legacy'")
```

## Security Considerations

### Unrestricted Executor
- Runs on host system with full permissions
- Can access files, network, system resources
- Suitable for trusted code only
- Use for development, testing, or trusted environments

### Docker Executor
- Runs in isolated container
- Limited access to host system
- Network access can be controlled
- Suitable for untrusted code execution
- Recommended for production use with external code

## Performance

### Unrestricted Executor
- Setup: ~1ms (minimal)
- Execution: ~10-50ms per call
- Memory: Low overhead

### Docker Executor
- Initial setup: ~1-5s (image build + container start)
- Execution: ~100-500ms per call (HTTP overhead)
- Memory: Higher overhead (container runtime)
- Subsequent calls: Fast (container reused)

## Troubleshooting

### Docker Issues

```bash
# Check Docker is running
docker ps

# Build image manually
cd python-sandbox
docker build -t python-sandbox .

# Test container manually
docker run -p 8000:8000 python-sandbox
curl http://localhost:8000/health
```

### Common Errors

1. **Import Error**: Missing dependencies
   ```bash
   pip install docker requests
   ```

2. **Docker Connection Error**: Docker not running
   ```bash
   sudo systemctl start docker  # Linux
   # or start Docker Desktop
   ```

3. **Port Conflicts**: Port already in use
   - The executor automatically finds available ports
   - Check with `docker ps` for running containers

4. **Permission Errors**: Docker permissions
   ```bash
   sudo usermod -aG docker $USER  # Add user to docker group
   newgrp docker                  # Refresh group membership
   ```
