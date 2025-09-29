"""
Timeout utilities for ARC-AGI task processing.

IMPORTANT: This implementation uses soft cancellation (future.cancel()) rather than 
hard process termination for timeout enforcement. This is intentional to prevent 
GPU resource leaks when working with LLM APIs:

- Hard kills can leave HTTP connections open on the GPU server
- Abandoned requests continue consuming GPU memory/VRAM  
- Connection pools become exhausted over time
- GPU servers can become unresponsive due to resource leaks

The trade-off is that timeouts are not strictly enforced - background threads may
continue running beyond the timeout. However, this prevents GPU resource exhaustion
and maintains system stability during long-running parallel workloads.

For strict timeout enforcement, consider server-side timeouts or connection-level
timeouts in the HTTP client rather than process-level killing.
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Callable


def execute_with_timeout(func: Callable, *args, timeout: float = 30, **kwargs) -> Any:
    """Execute a function with a hard timeout without blocking on thread shutdown.

    Uses a single-use ThreadPoolExecutor. On timeout, we cancel the future and
    shutdown the executor with wait=False (and cancel_futures=True when available)
    so we do not block even if the underlying call keeps running. This prevents
    the caller from hanging beyond the specified timeout.

    Note: The underlying task thread may continue running in the background and
    will be cleaned up when it returns; we prefer responsiveness over blocking.
    """
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(func, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except FuturesTimeoutError:
        # Best-effort cancellation and non-blocking shutdown
        future.cancel()
        try:
            executor.shutdown(wait=False, cancel_futures=True)  # type: ignore[call-arg]
        except TypeError:
            # Older Python: cancel_futures not supported
            executor.shutdown(wait=False)
        raise TimeoutError(f"Function execution exceeded timeout of {timeout}s")
    except Exception:
        # Non-timeout failure: shutdown without blocking
        try:
            executor.shutdown(wait=False, cancel_futures=True)  # type: ignore[call-arg]
        except TypeError:
            executor.shutdown(wait=False)
        raise
    else:
        # Normal completion; ensure cleanup
        executor.shutdown(wait=True)