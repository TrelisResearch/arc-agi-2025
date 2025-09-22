"""
High-performance, self-contained subprocess executor for Python code
using a fork-server architecture to minimize process startup overhead.
"""

import subprocess
import sys
import pickle
import os
import threading
import time
import atexit
import multiprocessing
import signal
import queue
from multiprocessing.connection import Connection
from typing import Optional, Any, Tuple

from sandbox.subprocess_executor import ExecutionTimeout

# This architecture is based on os.fork() and is not available on Windows.
if sys.platform == "win32":
    raise ImportError("This fork-server executor is not compatible with Windows.")

try:
    import psutil
except ImportError:
    psutil = None

# --- Global state for the fork server process and communication ---
_server_process: Optional[multiprocessing.Process] = None
_main_to_server_conn: Optional[Connection] = None


class ServerProcessError(Exception):
    """Raised when the server process dies unexpectedly."""
    pass


def _monitor_worker(
    task_queue: queue.Queue, result_container: dict
):
    """
    A long-lived worker thread that monitors child processes for memory usage.
    It gets tasks (pid, memory_limit, stop_event) from a queue.
    """
    if not psutil:
        return

    while True:
        try:
            # Block until a monitoring task is available
            task = task_queue.get()
            if task is None: # Shutdown signal
                break

            pid, memory_limit_bytes, stop_event = task
            p = psutil.Process(pid)

            while p.is_running() and not stop_event.is_set():
                try:
                    rss = p.memory_info().rss
                    for child in p.children(recursive=True):
                        rss += child.memory_info().rss

                    if rss > memory_limit_bytes:
                        result_container[pid] = {"oom_killed": True}
                        # Kill the entire process group
                        for child_proc in p.children(recursive=True):
                            child_proc.kill()
                        p.kill()
                        break # Stop monitoring this pid
                except psutil.NoSuchProcess:
                    break # Process finished
                time.sleep(0.02)
        except (psutil.NoSuchProcess, ProcessLookupError):
            # Process may have finished before monitoring even started
            continue
        except Exception:
            # Don't let the monitor thread die
            continue


def _child_executor(code: str, write_pipe: int):
    """
    This function runs in the ephemeral, forked child process.
    It executes the user code and writes the pickled result to a pipe.
    """
    result_payload = None
    try:
        # The user code is wrapped in a function to allow `return` statements.
        indented_code = "\n".join("    " + line for line in code.splitlines())
        full_code = f"def user_function():\n{indented_code}"

        exec_globals = {}
        exec(full_code, exec_globals)
        result = exec_globals['user_function']()
        result_payload = pickle.dumps((result, None))
    except Exception as e:
        # Capture any exception from the user's code
        result_payload = pickle.dumps((None, e))
    finally:
        if result_payload:
            try:
                os.write(write_pipe, result_payload)
            except BrokenPipeError:
                pass
        os.close(write_pipe)
        os._exit(0)


def _handle_timeout_signal(signum, frame):
    """Signal handler that raises an exception when a timeout occurs."""
    raise subprocess.TimeoutExpired(cmd=None, timeout=0)


def _fork_server_main_loop(conn: Connection):
    """
    The main loop for the long-lived server process.
    """

    # Preload these modules.
    import numpy  # noqa: F401
    import scipy  # noqa: F401

    print(f"Fork server started with PID: {os.getpid()}", file=sys.stderr)
    signal.signal(signal.SIGALRM, _handle_timeout_signal)

    # --- Start the single, long-lived monitor thread ---
    monitor_task_queue = queue.Queue()
    monitor_results = {}
    monitor_thread = threading.Thread(
        target=_monitor_worker,
        args=(monitor_task_queue, monitor_results),
        daemon=True
    )
    monitor_thread.start()

    while True:
        try:
            task = conn.recv()
            if task is None:
                monitor_task_queue.put(None) # Signal monitor thread to shut down
                break

            code, timeout, memory_limit_mb = task
            read_fd, write_fd = os.pipe()
            child_pid = os.fork()

            if child_pid == 0:
                # --- CHILD PROCESS ---
                os.close(read_fd)
                conn.close()
                _child_executor(code, write_fd)
            else:
                # --- PARENT PROCESS (The Server) ---
                os.close(write_fd)
                result_payload = None
                exception = None
                stop_monitor_event = threading.Event()

                try:
                    # Submit task to the monitor thread
                    memory_limit_bytes = memory_limit_mb * 1024 * 1024
                    monitor_task_queue.put((child_pid, memory_limit_bytes, stop_monitor_event))
                    
                    # Use setitimer for float-based timeouts
                    if timeout is not None and timeout > 0:
                        signal.setitimer(signal.ITIMER_REAL, timeout)

                    os.waitpid(child_pid, 0) # Block efficiently

                    child_result = monitor_results.pop(child_pid, {})
                    if child_result.get("oom_killed"):
                        exception = MemoryError(f"Process killed due to excessive memory usage (Limit: ~{memory_limit_mb}MB).")
                    else:
                        result_payload = os.read(read_fd, 65536)

                except subprocess.TimeoutExpired:
                    exception = ExecutionTimeout(timeout)
                    if psutil:
                        try:
                            p = psutil.Process(child_pid)
                            for child in p.children(recursive=True):
                                child.kill()
                            p.kill()
                        except psutil.NoSuchProcess:
                            pass
                except Exception as e:
                    exception = e
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0) # Cancel alarm
                    stop_monitor_event.set() # Signal monitor to stop
                    os.close(read_fd)
                    try:
                        os.waitpid(child_pid, os.WNOHANG)
                    except ChildProcessError:
                        pass
                
                if result_payload:
                    conn.send(pickle.loads(result_payload))
                else:
                    conn.send((None, exception))

        except (EOFError, BrokenPipeError):
            break
        except Exception as e:
            print(f"Fork server encountered a critical error: {e}", file=sys.stderr)
            try:
                conn.send((None, e))
            except (EOFError, BrokenPipeError):
                pass
            break

    monitor_thread.join(timeout=0.5)
    print("Fork server shutting down.", file=sys.stderr)


def _start_server():
    """Starts the global fork server process if it's not already running."""
    global _server_process, _main_to_server_conn
    if _server_process is None or not _server_process.is_alive():
        parent_conn, child_conn = multiprocessing.Pipe()
        _server_process = multiprocessing.Process(
            target=_fork_server_main_loop,
            args=(child_conn,),
            daemon=True
        )
        _server_process.start()
        _main_to_server_conn = parent_conn
        child_conn.close()


def _stop_server():
    """Stops the global fork server process."""
    global _server_process, _main_to_server_conn
    if _server_process and _server_process.is_alive():
        try:
            _main_to_server_conn.send(None)
            _main_to_server_conn.close()
        except (BrokenPipeError, EOFError):
            pass
        _server_process.join(timeout=2)
        if _server_process.is_alive():
            _server_process.terminate()
    _server_process = None
    _main_to_server_conn = None


atexit.register(_stop_server)


def execute_code(
    code: str, timeout: Optional[float] = 5.0, memory_limit_mb: int = 256
) -> Tuple[Any, Optional[Exception]]:
    """
    Executes Python code in a sandboxed environment using a high-performance
    fork-server to minimize latency.
    """
    _start_server()

    if not _server_process or not _server_process.is_alive():
        return None, ServerProcessError("Fork server process is not running.")

    try:
        _main_to_server_conn.send((code, timeout, memory_limit_mb))
        result, exception = _main_to_server_conn.recv()
        return result, exception
    except (EOFError, BrokenPipeError):
        _stop_server()
        return None, ServerProcessError("Connection to fork server was lost.")
    except Exception as e:
        return None, e


if __name__ == '__main__':
    print("--- Running Fork Server Executor Tests ---")

    # 1. Test successful execution
    print("\n1. Testing successful execution...")
    code_success = "return [i for i in range(5)]"
    start_time = time.perf_counter()
    result, err = execute_code(code_success)
    end_time = time.perf_counter()
    print(f"Result: {result}, Error: {err}")
    print(f"Time taken (first run, includes server start): {end_time - start_time:.4f}s")
    assert result == [0, 1, 2, 3, 4] and err is None

    # 2. Test subsequent run (should be much faster)
    print("\n2. Testing second run (should be much faster)...")
    total_time = 0
    iterations = 100
    for _ in range(iterations):
        start_time = time.perf_counter()
        result, err = execute_code(code_success)
        end_time = time.perf_counter()
        total_time += (end_time - start_time)
    
    avg_time = (total_time / iterations) * 1000
    print(f"Result: {result}, Error: {err}")
    print(f"Average time over {iterations} iterations: {avg_time:.4f}ms")
    assert result == [0, 1, 2, 3, 4] and err is None

    # 3. Test a standard exception
    print("\n3. Testing code with an exception...")
    code_exception = "a = 1\nb = 0\nreturn a / b"
    result, err = execute_code(code_exception)
    print(f"Result: {result}, Error: {err}")
    assert result is None and isinstance(err, ZeroDivisionError)

    # 4. Test a timeout
    print("\n4. Testing timeout...")
    code_timeout = "import time\ntime.sleep(2)"
    result, err = execute_code(code_timeout, timeout=1.0)
    print(f"Result: {result}, Error: {err}")
    assert result is None and isinstance(err, ExecutionTimeout)

    # 5. Test memory limit (OOM kill)
    if psutil:
        print("\n5. Testing memory limit...")
        code_oom = "' ' * (100 * 1024 * 1024)"
        result, err = execute_code(code_oom, memory_limit_mb=50)
        print(f"Result: {result}, Error: {err}")
        assert result is None and isinstance(err, MemoryError)
    else:
        print("\n5. Skipping memory limit test: psutil not installed.")

    print("\n--- All tests completed ---")

   # 6. Test baseline memory usage of a forked child
    if psutil:
        print("\n6. Testing baseline memory usage...")
        code_mem_check = (
            "import os, psutil\n"
            "p = psutil.Process(os.getpid())\n"
            "return p.memory_info().rss"
        )
        result, err = execute_code(code_mem_check)
        if err is None:
            print(f"Baseline memory usage of child process: {result / (1024*1024):.2f} MB")
        else:
            print(f"Could not measure memory: {err}")
    else:
        print("\n6. Skipping baseline memory test: psutil not installed.")