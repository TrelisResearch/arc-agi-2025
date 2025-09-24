"""
A client for the IPC-based fork server.

This class manages a dedicated fork server running in a subprocess, communicating
with it over a unique UNIX domain socket. It provides a simple, thread-safe
method to execute code in an isolated process and get the process ID back.
"""
import subprocess
import socket
import struct
import os
import atexit
import threading
import sys
import time
import traceback

class IPCForkServerClient:
    def __init__(self):
        self.server_process = None
        self.socket_path = None
        self._lock = threading.Lock()
        self._start_server()
        # Ensure the server is shut down cleanly on exit.
        atexit.register(self.shutdown)

    def _start_server(self):
        """Starts the ipc_fork_server in a subprocess."""
        # Use sys.executable to ensure we're using the same Python interpreter.
        command = [sys.executable, "-m", "sandbox.ipc_fork_server"]

        # Start the server process. stdout/stderr are piped to capture output.
        self.server_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # The server's first line of stdout is the unique socket path.
        socket_path = self.server_process.stdout.readline().strip()

        if not socket_path:
            # If we don't get a socket path, the server likely failed to start.
            stderr = self.server_process.stderr.read()
            raise RuntimeError(f"Failed to get socket path from fork server. Error: {stderr}")

        self.socket_path = socket_path

        # Brief wait to ensure the socket is ready before proceeding.
        time.sleep(0.1)

        # Check if the server process died immediately after startup.
        if self.server_process.poll() is not None:
            stderr = self.server_process.stderr.read()
            raise RuntimeError(f"Fork server process died unexpectedly on startup. Error: {stderr}")

        print(f"IPC Fork Server client started, connected to: {self.socket_path}", file=sys.stderr)

    def execute(self, code: str) -> int:
        """
        Executes a string of Python code on the fork server.

        This method is thread-safe.

        Args:
            code: The Python code to execute.

        Returns:
            The process ID (PID) of the forked child process executing the code.
        
        Raises:
            RuntimeError: If the server is not running or communication fails.
        """
        if self.server_process is None or self.server_process.poll() is not None:
            raise RuntimeError("Fork server is not running. It may have crashed or been shut down.")

        # Use a lock to ensure thread-safety for the socket communication.
        with self._lock:
            try:
                # A new socket is created for each request to allow for concurrent calls.
                client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                client_socket.connect(self.socket_path)

                try:
                    # 1. Encode the code string to UTF-8 bytes.
                    encoded_code = code.encode('utf-8')

                    # 2. Pack the length of the code into a 4-byte unsigned int (big-endian).
                    len_data = struct.pack('>I', len(encoded_code))

                    # 3. Send the length, then the code.
                    client_socket.sendall(len_data)
                    client_socket.sendall(encoded_code)

                    # 4. Receive the 4-byte PID from the server.
                    pid_data = client_socket.recv(4)
                    if not pid_data:
                        raise RuntimeError("Did not receive PID from server.")
                    
                    pid = struct.unpack('>I', pid_data)[0]
                    return pid

                finally:
                    client_socket.close()

            except (ConnectionRefusedError, FileNotFoundError):
                 raise RuntimeError(f"Failed to connect to server at {self.socket_path}. The server may have crashed.")
            except Exception as e:
                raise RuntimeError(f"An error occurred during execution: {e}")

    def shutdown(self):
        """Shuts down the fork server process."""
        if self.server_process and self.server_process.poll() is None:
            print(f"Shutting down fork server (PID: {self.server_process.pid})...", file=sys.stderr)
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("Server did not terminate gracefully, killing...", file=sys.stderr)
                self.server_process.kill()
            self.server_process = None
        
        # Clean up the socket file if it still exists.
        if self.socket_path and os.path.exists(self.socket_path):
            try:
                os.remove(self.socket_path)
            except OSError as e:
                print(f"Error removing socket file {self.socket_path}: {e}", file=sys.stderr)

    def __del__(self):
        """Ensures shutdown is called when the object is garbage collected."""
        self.shutdown()

if __name__ == '__main__':
    print("Starting IPC Fork Server client for a demo...")
    client = IPCForkServerClient()

    try:
        print("\n--- Test 1: Simple execution ---")
        code_to_run = "import os; print(f'Hello from process {os.getpid()}');"
        pid = client.execute(code_to_run)
        print(f"Dispatched code to be run in process PID: {pid}")
        # In a real scenario, you would now os.waitpid(pid, 0) or similar.
        time.sleep(0.1) # Give it a moment to print

        print("\n--- Test 2: Execution with a result file ---")
        result_file = "/tmp/ipc_test_result.txt"
        code_with_result = f"""
import os
with open('{result_file}', 'w') as f:
    f.write(f'Result from PID {{os.getpid()}}')
"""
        pid = client.execute(code_with_result)
        print(f"Dispatched code to write to {result_file} in process PID: {pid}")
        
        # Wait for the process to finish
        # Check if the process is still running before waiting on it
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            print(f"Process {pid} is not running (may have already exited).")
        
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                print(f"Result file content: '{f.read()}'")
            os.remove(result_file)
        else:
            print("Result file was not created.")

    except Exception as e:
        print(f"\nAn error occurred during the demo: {e}")
        traceback.print_exc()
    finally:
        print("\nShutting down client and server...")
        client.shutdown()
        print("Demo finished.")
