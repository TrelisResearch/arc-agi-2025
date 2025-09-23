"""
High-performance, thread-safe, self-contained subprocess executor for Python code
using a fork-server architecture to minimize process startup overhead.
"""

import subprocess
import sys
import pickle
import os
import threading
import time
import multiprocessing
import signal
import queue
import uuid
import atexit
from typing import Optional, Any, Tuple, Dict

from sandbox.subprocess_executor import ExecutionTimeout

# This architecture is based on os.fork() and is not available on Windows.
if sys.platform == "win32":
    raise ImportError("This fork-server executor is not compatible with Windows.")

try:
    import psutil
except ImportError:
    psutil = None


class ServerProcessError(Exception):
    """Raised when a server process dies unexpectedly or fails to start."""
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
            task = task_queue.get()
            if task is None: # Shutdown signal
                break

            pid, memory_limit_bytes, stop_event = task
            try:
                p = psutil.Process(pid)
            except psutil.NoSuchProcess:
                continue # Process already finished

            while p.is_running() and not stop_event.is_set():
                try:
                    rss = p.memory_info().rss
                    for child in p.children(recursive=True):
                        rss += child.memory_info().rss

                    if rss > memory_limit_bytes:
                        result_container[pid] = {"oom_killed": True}
                        for child_proc in p.children(recursive=True):
                            child_proc.kill()
                        p.kill()
                        break
                except psutil.NoSuchProcess:
                    break
                time.sleep(0.02)
        except (psutil.NoSuchProcess, ProcessLookupError):
            continue
        except Exception:
            continue


def _child_executor(code: str, write_pipe: int):
    """
    This function runs in the ephemeral, forked child process.
    It executes the user code and writes the pickled result to a pipe.
    """
    result_payload = None
    try:
        indented_code = "\n".join("    " + line for line in code.splitlines())
        full_code = f"def user_function():\n{indented_code}"
        exec_globals = {}
        exec(full_code, exec_globals)
        result = exec_globals['user_function']()
        result_payload = pickle.dumps((result, None))
    except Exception as e:
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


def _fork_server_main_loop(request_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue):
    """
    The main loop for a long-lived server process. It pulls tasks from a shared
    queue and puts results back on another shared queue.
    """
    # Modules like numpy and scipy are pre-loaded by the fork.
    # Re-importing them here is not fork-safe and can lead to hangs.

    print(f"Fork server worker started with PID: {os.getpid()}", file=sys.stderr)
    signal.signal(signal.SIGALRM, _handle_timeout_signal)

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
            task = request_queue.get()
            if task is None:
                monitor_task_queue.put(None)
                break

            task_id, code, timeout, memory_limit_mb = task
            read_fd, write_fd = os.pipe()
            child_pid = os.fork()

            if child_pid == 0:
                os.close(read_fd)
                request_queue.close()
                result_queue.close()
                _child_executor(code, write_fd)
            else:
                os.close(write_fd)
                result_payload = None
                exception = None
                stop_monitor_event = threading.Event()

                try:
                    memory_limit_bytes = memory_limit_mb * 1024 * 1024
                    monitor_task_queue.put((child_pid, memory_limit_bytes, stop_monitor_event))
                    
                    if timeout is not None and timeout > 0:
                        signal.setitimer(signal.ITIMER_REAL, timeout)

                    os.waitpid(child_pid, 0)

                    child_result = monitor_results.pop(child_pid, {})
                    if child_result.get("oom_killed"):
                        exception = MemoryError(f"Process killed due to excessive memory usage (Limit: ~{memory_limit_mb}MB).")
                    else:
                        # Limit read size to avoid blocking forever on large unexpected output
                        result_payload = os.read(read_fd, 256 * 1024 * 1024) # 256MB limit

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
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    stop_monitor_event.set()
                    os.close(read_fd)
                    try:
                        os.waitpid(child_pid, os.WNOHANG)
                    except ChildProcessError:
                        pass
                
                if result_payload:
                    result_queue.put((task_id, *pickle.loads(result_payload)))
                else:
                    result_queue.put((task_id, None, exception))

        except (EOFError, BrokenPipeError):
            break
        except Exception as e:
            print(f"Fork server worker encountered a critical error: {e}", file=sys.stderr)
            break

    monitor_thread.join(timeout=0.5)
    print(f"Fork server worker {os.getpid()} shutting down.", file=sys.stderr)


class ForkServerExecutor:
    """
    A thread-safe executor that runs Python code in sandboxed processes using a
    pool of persistent fork-server workers.
    """
    def __init__(self, max_workers: Optional[int] = None):
        if max_workers is None:
            max_workers = os.cpu_count() or 1
        
        ctx = multiprocessing.get_context('fork')
        self._request_queue = ctx.Queue()
        self._result_queue = ctx.Queue()
        self._workers = []
        for _ in range(max_workers):
            worker_process = ctx.Process(
                target=_fork_server_main_loop,
                args=(self._request_queue, self._result_queue),
                daemon=True
            )
            worker_process.start()
            self._workers.append(worker_process)

        self._pending_tasks: Dict[str, Tuple[Any, threading.Event]] = {}
        self._pending_tasks_lock = threading.Lock()

        self._result_thread = threading.Thread(target=self._result_collector, daemon=True)
        self._result_thread.start()

    def _result_collector(self):
        """
        A dedicated thread that collects results from the result queue and
        notifies the corresponding waiting thread.
        """
        while True:
            try:
                task_id, result, exception = self._result_queue.get()
                with self._pending_tasks_lock:
                    if task_id in self._pending_tasks:
                        result_holder, event = self._pending_tasks.pop(task_id)
                        result_holder['result'] = result
                        result_holder['exception'] = exception
                        event.set()
            except (EOFError, BrokenPipeError):
                # This can happen during shutdown
                break
            except Exception as e:
                print(f"Result collector thread encountered an error: {e}", file=sys.stderr)


    def execute(
        self, code: str, timeout: Optional[float] = 5.0, memory_limit_mb: int = 256
    ) -> Tuple[Any, Optional[Exception]]:
        """
        Executes Python code in a sandboxed environment. This method is thread-safe.
        """
        task_id = str(uuid.uuid4())
        event = threading.Event()
        result_holder: Dict[str, Any] = {}

        with self._pending_tasks_lock:
            self._pending_tasks[task_id] = (result_holder, event)

        self._request_queue.put((task_id, code, timeout, memory_limit_mb))

        # Wait for the result, with an additional grace period on the timeout
        # to account for queueing and IPC delays.
        total_wait_timeout = timeout + 10 if timeout is not None else None
        event.wait(timeout=total_wait_timeout)

        if not result_holder:
            # The task was not processed and the event timed out
            with self._pending_tasks_lock:
                self._pending_tasks.pop(task_id, None) # Clean up
            return None, ExecutionTimeout(timeout)

        return result_holder.get('result'), result_holder.get('exception')

    def shutdown(self):
        """
        Shuts down all worker processes and cleans up resources.
        """
        # Signal all workers to stop
        for _ in self._workers:
            try:
                self._request_queue.put(None)
            except (EOFError, BrokenPipeError):
                pass # Queue might already be closed

        # Wait for workers to terminate
        for worker in self._workers:
            worker.join(timeout=2)
            if worker.is_alive():
                worker.terminate()
        
        self._request_queue.close()
        self._result_queue.close()
        self._result_thread.join(timeout=1)
        print("ForkServerExecutor has been shut down.", file=sys.stderr)


# --- Global Executor Instance ---

# Create a single, process-wide instance of the executor.
# It's configured with half the available CPU cores, with a minimum of 1.
_max_workers = max((os.cpu_count() or 1) // 2, 1)
_global_executor = ForkServerExecutor(max_workers=_max_workers)

# Register a shutdown hook to ensure the executor is cleaned up on exit.
atexit.register(_global_executor.shutdown)


def execute_code(
    code: str, timeout: Optional[float] = 5.0, memory_limit_mb: int = 256
) -> Tuple[Any, Optional[Exception]]:
    """
    Executes Python code in a sandboxed environment using a high-performance,
    thread-safe fork-server pool.
    """
    if not _global_executor:
        # This should not happen in normal operation
        return None, ServerProcessError("Executor is not initialized.")
    return _global_executor.execute(code, timeout, memory_limit_mb)


if __name__ == '__main__':
    import concurrent.futures

    print("--- Running Fork Server Executor Tests ---")
    
    # The executor is now managed automatically by the module.
    # We can directly use the `execute_code` function.

    try:
        # 1. Test successful execution
        print("\n1. Testing successful execution...")
        code_success = "return [i for i in range(5)]"
        start_time = time.perf_counter()
        result, err = execute_code(code_success)
        end_time = time.perf_counter()
        print(f"Result: {result}, Error: {err}")
        print(f"Time taken (first run): {end_time - start_time:.4f}s")
        assert result == [0, 1, 2, 3, 4] and err is None

        # 2. Test parallel execution
        print("\n2. Testing parallel execution with multiple threads...")
        num_parallel_tasks = 20
        code_parallel = "import time; import os; time.sleep(0.5); return os.getpid()"
        
        start_time = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel_tasks) as pool:
            futures = [pool.submit(execute_code, code_parallel, timeout=2) for _ in range(num_parallel_tasks)]
            results = [f.result() for f in futures]
        end_time = time.perf_counter()

        print("Results from parallel execution:")
        for res in results:
            print(f"Result: {res[0]}, Error: {res[1]}")
        pids = {res[0] for res in results if res[1] is None}
        print(f"Executed {num_parallel_tasks} tasks in {end_time - start_time:.4f}s")
        print(f"Got results from {len(pids)} unique worker PIDs: {pids}")
        if _max_workers > 1:
            assert len(pids) > 1, "Should have used multiple worker processes"
        else:
            print(f"Skipping multi-worker assertion as max_workers is {_max_workers}")
        assert len(results) == num_parallel_tasks

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
            # Allocate 150MB of memory as a string
            code_oom = "a = ' ' * (150 * 1024 * 1024); return len(a)"
            result, err = execute_code(code_oom, memory_limit_mb=100)
            print(f"Result: {result}, Error: {err}")
            assert result is None and isinstance(err, MemoryError)
        else:
            print("\n5. Skipping memory limit test: psutil not installed.")

        print("\n--- All tests completed ---")

    finally:
        # No manual shutdown needed, atexit handles it.
        print("\nScript finished. Executor will be shut down automatically by atexit.")
        pass
