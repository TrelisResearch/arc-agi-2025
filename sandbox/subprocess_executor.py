"""
Simple, self-contained subprocess executor for Python code.
No dependencies on base classes or complex imports.
"""
import subprocess
import sys
import pickle
import base64
import os
import threading
import time
from typing import Optional, Any, Tuple

# --- New Dependency ---
# This implementation requires the `psutil` library.
# You can install it with: pip install psutil
try:
    import psutil
except ImportError:
    psutil = None

def _monitor_memory(process: subprocess.Popen, memory_limit_bytes: int, result_container: dict):
    """
    A thread target function that polls a process's memory usage and kills it
    if it exceeds the limit.
    """
    if not psutil:
        return # psutil not installed, monitoring is disabled.

    try:
        p = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                rss = p.memory_info().rss
                if rss > memory_limit_bytes:
                    result_container['oom_killed'] = True
                    # Kill the entire process group to be safe
                    for child in p.children(recursive=True):
                        child.kill()
                    p.kill()
                    return
            except psutil.NoSuchProcess:
                # Process finished between poll() and memory_info()
                return
            time.sleep(0.05)
    except psutil.NoSuchProcess:
        # Process already finished before the thread started
        return

def execute_code_in_subprocess(
    code: str, timeout: Optional[float] = None, memory_limit_mb: int = 512
) -> Tuple[Any, Optional[Exception]]:
    """
    Executes a Python code string in a memory and time-constrained subprocess.

    This version uses a monitor thread in the parent process to poll the child's
    memory usage. If the child exceeds the memory limit, it is killed. This
    approach works reliably across different environments (Linux, macOS, Docker)
    without relying on `systemd` or the unreliable `ulimit`.

    Requires the `psutil` library to be installed.

    Args:
        code (str): Python code to execute (should contain a return statement).
        timeout (float, optional): Timeout in seconds.

    Returns:
        A tuple containing the result or an exception.
    """
    if not psutil:
        raise ImportError("The 'psutil' library is required for memory monitoring. Please run 'pip install psutil'.")

    runner_script = """
import pickle, base64, sys
try:
    encoded_code = sys.argv[1]
    user_code = base64.b64decode(encoded_code).decode('utf-8')
    exec_globals = {}
    full_code = "def user_function():\\n" + user_code
    exec(full_code, exec_globals)
    result = exec_globals['user_function']()
    serialized = base64.b64encode(pickle.dumps(result)).decode('utf-8')
    print(f"RESULT_START{serialized}RESULT_END")
except Exception as e:
    serialized_error = base64.b64encode(pickle.dumps(e)).decode('utf-8')
    print(f"ERROR_START{serialized_error}ERROR_END", file=sys.stderr)
    sys.exit(1)
"""
    monitor_thread = None
    process = None
    try:
        indented_code = "\n".join("    " + line for line in code.splitlines())
        encoded_user_code = base64.b64encode(indented_code.encode('utf-8')).decode('utf-8')

        # Use a simple command that works across Linux and macOS
        command = [sys.executable, "-c", runner_script, encoded_user_code]

        # Use Popen to start the process in the background
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Start the memory monitor thread
        memory_limit_bytes = memory_limit_mb * 1024 * 1024
        result_container = {'oom_killed': False} # Use a dict to pass mutable state
        monitor_thread = threading.Thread(
            target=_monitor_memory,
            args=(process, memory_limit_bytes, result_container),
            daemon=True
        )
        monitor_thread.start()

        # Wait for the process to complete or timeout
        stdout, stderr = process.communicate(timeout=timeout)

        # Wait for the monitor thread to finish, just in case
        monitor_thread.join(timeout=1)

        # Check if the monitor thread killed the process
        if result_container['oom_killed']:
            return None, MemoryError(f"Process killed due to excessive memory usage (Limit: ~{memory_limit_mb}MB).")

        if "ERROR_START" in stderr:
            start_marker, end_marker = "ERROR_START", "ERROR_END"
            start_idx = stderr.find(start_marker) + len(start_marker)
            end_idx = stderr.find(end_marker)
            serialized_error = stderr[start_idx:end_idx]
            return None, pickle.loads(base64.b64decode(serialized_error))
        elif "RESULT_START" in stdout:
            start_marker, end_marker = "RESULT_START", "END_RESULT" # Small typo in original, fixed
            start_idx = stdout.find(start_marker) + len(start_marker)
            end_idx = stdout.find("RESULT_END")
            serialized_result = stdout[start_idx:end_idx]
            return pickle.loads(base64.b64decode(serialized_result)), None
        else:
            return None, Exception(f"An unknown error occurred. Stderr: {stderr}")

    except subprocess.TimeoutExpired:
        if process:
            # Be thorough in cleanup
            p = psutil.Process(process.pid)
            for child in p.children(recursive=True):
                child.kill()
            p.kill()
        return None, Exception(f"Code execution timed out after {timeout} seconds.")
    except Exception as e:
        return None, Exception(f"Subprocess execution failed: {e}")
    finally:
        if monitor_thread and monitor_thread.is_alive():
             monitor_thread.join(timeout=1)

