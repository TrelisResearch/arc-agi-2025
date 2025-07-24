# Sandbox

Internal Python code execution library.

## Usage

```python
from sandbox import create_executor

# Auto-select best available executor (tries Docker, falls back to unrestricted)
with create_executor() as executor:
    result, error = executor.execute_code("return 2 + 2")
    if error:
        print(f"Error: {error}")
    else:
        print(f"Result: {result}")

# Or explicitly choose executor type
with create_executor("unrestricted") as executor:
    result, error = executor.execute_code("return 'fast execution'")

with create_executor("docker") as executor:
    result, error = executor.execute_code("return 'secure execution'")
```
