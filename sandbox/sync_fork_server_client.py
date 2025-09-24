"""
Synchronous client for the HTTP-based fork server.
"""

import subprocess
import sys
import time
import requests
from typing import Optional
import atexit
import threading
import portpicker


class SyncForkServerClient:
    """
    A synchronous client for the HTTP-based fork server.

    This class manages the lifecycle of the fork server subprocess and provides
    a simple, blocking interface for executing code. It is designed to be
    instantiated once and used throughout the application's life.

    Requires the `requests` library to be installed.
    """

    def __init__(self, port: Optional[int] = None):
        self.port = port if port is not None else portpicker.pick_unused_port()
        self.server_url = f"http://localhost:{self.port}"
        self.server_process: Optional[subprocess.Popen] = None
        self.session: Optional[requests.Session] = None
        self._start_server()
        self._terminate_lock = threading.Lock()
        atexit.register(self.terminate)

    def _start_server(self):
        """Starts the server subprocess and waits for it to be ready by polling."""
        command = [
            sys.executable,
            "-u",
            "-m",
            "sandbox.http_fork_server",
            str(self.port),
        ]

        self.server_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        self.session = requests.Session()

        # Poll the server's root endpoint until it responds with 200 OK.
        max_wait_time = 4  # seconds
        start_time = time.time()

        while time.time() - start_time < max_wait_time:
            if self.server_process.poll() is not None:
                stderr = self.server_process.stderr.read()
                raise RuntimeError(f"Fork server failed to start. Stderr:\n{stderr}")

            try:
                response = self.session.get(f"{self.server_url}/", timeout=0.1)
                if response.status_code == 200:
                    return  # Server is ready
            except requests.ConnectionError:
                # Server is not yet accepting connections, wait and retry.
                time.sleep(0.1)
                continue

        # If the loop finishes, the server failed to start in time.
        # stderr_output = self.server_process.stderr.read()
        self.terminate()
        raise RuntimeError(
            f"Fork server timed out after {max_wait_time} seconds. Stderr:\n"
        )

    def terminate(self):
        """Stops the server process and closes the session. This method is idempotent."""
        if self.session:
            self.session.close()
            self.session = None
        if self.server_process and self.server_process.poll() is None:
            print(f"Terminating fork server process {self.server_process.pid}...")
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
                print(
                    f"Fork server process {self.server_process.pid} terminated successfully."
                )
            except subprocess.TimeoutExpired:
                print(
                    f"Fork server process {self.server_process.pid} did not terminate in time; killing it."
                )
                self.server_process.kill()
            self.server_process = None
        print("Fork server not running, not terminating.")

    def execute(self, code: str) -> int:
        """
        Sends code to the fork server for execution.

        Args:
            code: The Python code string to execute.

        Returns:
            The process ID (PID) of the forked child process executing the code.
        """
        if not self.session:
            raise RuntimeError("Client session not started.")

        try:
            response = self.session.post(
                f"{self.server_url}/execute", json={"code": code}
            )
            response.raise_for_status()
            data = response.json()
            return data["pid"]
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to connect to fork server: {e}") from e


if __name__ == "__main__":
    print("Starting fork server client...")
    # Not specifying a port will cause it to pick a free one.
    client = SyncForkServerClient()
    print(f"Fork server client started on port {client.port}.")

    try:
        code_to_execute = "import time; print('Hello from forked process!'); time.sleep(2); print('Forked process finished.'); import sys; sys.exit(0)"
        print(f"Executing code:\n---\n{code_to_execute}\n---")
        pid = client.execute(code_to_execute)
        print(f"Code execution started in process with PID: {pid}")

        # Wait for the process to finish to see if it hangs
        try:
            import psutil

            p = psutil.Process(pid)
            print(f"Waiting for process {pid} to complete...")
            p.wait(timeout=10)
            print(f"Process {pid} completed successfully.")
        except psutil.NoSuchProcess:
            print(f"Process {pid} finished before we could wait.")
        except psutil.TimeoutExpired:
            print(f"Process {pid} timed out and did not exit.")
            if p.is_running():
                print(f"Process {pid} is still running.")
            else:
                print(f"Process {pid} is no longer running (but timed out?).")

    finally:
        print("Terminating fork server client.")
        client.terminate()
        print("Client terminated.")
