"""
Simple, self-contained subprocess executor for Python code.
No dependencies on base classes or complex imports.
"""

import sys
import pickle
import base64
import os
import threading
import time
from typing import Optional, Any, Tuple

from sandbox.ipc_sync_fork_server_client import IPCForkServerClient

try:
    import psutil
except ImportError:
    psutil = None

from sandbox.concurrency import ConcurrencyGate


class ExecutionTimeout(Exception):
    """Exception raised when code execution exceeds the time limit."""

    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Code execution timed out after {timeout_seconds} seconds.")


# Untrusted execution can runaway with memory quicker than we can kill processes, so we need a hard limit on concurrency here.
CPU_COUNT = os.cpu_count() or 1
MAX_CONCURRENT_PROCESSES = max(1, CPU_COUNT // 4)
sandbox_process_gate = ConcurrencyGate(MAX_CONCURRENT_PROCESSES)


class ProcessMonitor:
    """
    A singleton class that runs a single background thread to monitor all
    subprocess memory and execution time.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            self.processes_to_monitor = {}
            self.monitor_lock = threading.Lock()
            self.monitor_thread = threading.Thread(target=self._run, daemon=True)
            self.monitor_thread.start()
            self._initialized = True

    def add_process(
        self,
        pid: int,
        memory_limit_bytes: int,
        timeout: Optional[float],
    ):
        """Adds a process to be monitored."""
        with self.monitor_lock:
            self.processes_to_monitor[pid] = {
                "pid": pid,
                "memory_limit_bytes": memory_limit_bytes,
                "timeout": timeout,
                "start_time": time.time(),
                "oom_killed": False,
                "timeout_killed": False,
            }

    def get_process_info(self, pid: int) -> Optional[dict]:
        """Gets monitoring info for a specific process."""
        with self.monitor_lock:
            return self.processes_to_monitor.get(pid)

    def remove_process(self, pid: int):
        """Removes a process from monitoring."""
        with self.monitor_lock:
            if pid in self.processes_to_monitor:
                del self.processes_to_monitor[pid]

    def _run(self):
        """The main monitoring loop that runs in a background thread."""
        if not psutil:
            return  # psutil not installed, monitoring is disabled.

        while True:
            with self.monitor_lock:
                pids = list(self.processes_to_monitor.keys())

            for pid in pids:
                with self.monitor_lock:
                    info = self.processes_to_monitor.get(pid)
                    if not info:
                        continue

                    try:
                        p = psutil.Process(pid)
                        # If process is done, no need to monitor
                        if not p.is_running():
                            continue

                        # 1. Check for timeout
                        if info["timeout"] is not None:
                            elapsed = time.time() - info["start_time"]
                            if elapsed > info["timeout"]:
                                info["timeout_killed"] = True
                                for child in p.children(recursive=True):
                                    child.kill()
                                p.kill()
                                continue  # Move to next process

                        # 2. Check for memory usage
                        rss = p.memory_info().rss
                        if rss > info["memory_limit_bytes"]:
                            info["oom_killed"] = True
                            for child in p.children(recursive=True):
                                child.kill()
                            p.kill()

                    except psutil.NoSuchProcess:
                        # Process finished and was removed between getting keys and processing
                        continue

            time.sleep(0.05)


# Global instance of the process monitor
# Ensure these are true singletons, even if this file is imported multiple times (e.g., with pytest reloads)
_singleton_init_lock = threading.Lock()

process_monitor = getattr(sys.modules[__name__], "_process_monitor", None)
if process_monitor is None:
    with _singleton_init_lock:
        process_monitor = getattr(sys.modules[__name__], "_process_monitor", None)
        if process_monitor is None:
            process_monitor = ProcessMonitor()
            setattr(sys.modules[__name__], "_process_monitor", process_monitor)

fork_server = getattr(sys.modules[__name__], "_fork_server", None)
if fork_server is None:
    with _singleton_init_lock:
        fork_server = getattr(sys.modules[__name__], "_fork_server", None)
        if fork_server is None:
            fork_server = IPCForkServerClient()
            setattr(sys.modules[__name__], "_fork_server", fork_server)


def execute_code_in_subprocess(
    code: str, timeout: Optional[float] = None, memory_limit_mb: int = 128
) -> Tuple[Any, Optional[Exception]]:
    """
    Executes a Python code string in a memory and time-constrained subprocess.

    This version uses a central monitor thread to poll the child's memory usage
    and execution time. If the child exceeds the limits, it is killed. This
    approach works reliably across different environments (Linux, macOS, Docker)
    without relying on `systemd` or the unreliable `ulimit`.

    Requires the `psutil` library to be installed.

    Args:
        code (str): Python code to execute (should contain a return statement).
        timeout (float, optional): Timeout in seconds.
        memory_limit_mb (int): Memory limit in megabytes.

    Returns:
        A tuple containing the result or an exception.
    """
    with sandbox_process_gate:
        pid = None
        result_path = None
        try:
            # Determine the directory for the temporary file.
            # Use /dev/shm on Linux if available for memory-backed storage.
            temp_dir = None
            if sys.platform == "linux" and os.path.isdir("/dev/shm"):
                temp_dir = "/dev/shm"

            # Create a temporary file to store the result
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, dir=temp_dir) as tmp_file:
                result_path = tmp_file.name

                indented_code = "\n".join("    " + line for line in code.splitlines())
                encoded_user_code = base64.b64encode(
                    indented_code.encode("utf-8")
                ).decode("utf-8")

                runner_script = f"""
import pickle, base64, sys, os
try:
    # This code runs in a forked process.
    encoded_code = {encoded_user_code!r}
    result_path = {result_path!r}
    user_code = base64.b64decode(encoded_code).decode('utf-8')
    exec_globals = {{}}
    full_code = "def user_function():\\n" + user_code
    exec(full_code, exec_globals)
    result = exec_globals['user_function']()
    with open(result_path, 'wb') as f:
        f.write(pickle.dumps({{'result': result }}))
except Exception as e:
    with open(result_path, 'wb') as f:
        f.write(pickle.dumps({{'error': e }}))
finally:
    sys.exit(1)
"""

                try:
                    pid = fork_server.execute(runner_script)

                    # Register the process with the central monitor
                    memory_limit_bytes = memory_limit_mb * 1024 * 1024
                    process_monitor.add_process(pid, memory_limit_bytes, timeout)

                    # Wait for the process to complete. The monitor will kill it if needed.
                    if psutil:
                        try:
                            p = psutil.Process(pid)
                            start_wait = time.time()
                            while p.is_running():
                                if timeout is not None and (
                                    time.time() - start_wait > timeout
                                ):
                                    # Main thread timeout as a fallback
                                    break
                                time.sleep(0.01)
                        except psutil.NoSuchProcess:
                            # Process already finished
                            pass

                    # Check if the monitor thread killed the process
                    info = process_monitor.get_process_info(pid)
                    if info:
                        if info["oom_killed"]:
                            return None, MemoryError(
                                f"Process killed due to excessive memory usage (Limit: ~{memory_limit_mb}MB)."
                            )
                        if info["timeout_killed"]:
                            return None, ExecutionTimeout(
                                timeout if timeout is not None else -1.0
                            )

                    # Read result from temporary file
                    if (
                        not os.path.exists(result_path)
                        or os.path.getsize(result_path) == 0
                    ):
                        # This can happen if the process was killed for other reasons
                        # before it could write the result.
                        return None, Exception(
                            "Result file is empty or missing. Subprocess may have crashed or been killed."
                        )

                    with open(result_path, "rb") as f:
                        try:
                            data = pickle.load(f)
                        except Exception as e:
                            return None, Exception(f"Failed to unpickle result: {e}")
                    if "error" in data:
                        return None, data["error"]
                    elif "result" in data:
                        return data["result"], None
                    else:
                        return None, Exception("Unknown result format in result file.")

                except Exception as e:
                    # This catches errors in the parent process, not the subprocess
                    return None, Exception(f"Subprocess execution failed: {e}")

        finally:
            # Clean up monitoring and temporary files
            if pid:
                process_monitor.remove_process(pid)
            if result_path and os.path.exists(result_path):
                os.remove(result_path)
