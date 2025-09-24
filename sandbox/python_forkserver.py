"""
A minimal, thread-safe Python fork server.

This module implements a simple fork server that runs as a separate, clean
process. It is designed to be a faster alternative to `subprocess.Popen` for
creating new Python processes, avoiding the overhead of re-initializing the
Python interpreter for each new process.

The server listens on a Unix domain socket for requests. When a client connects
and sends Python code, the server forks a new child process, executes the code
within that child, and returns the child's PID to the client.

A key feature is the ability to pass file descriptors from the client to the
child process. This allows the client to capture the child's stdout and stderr
from the very beginning of its execution, which is crucial for robust process
monitoring and output handling.

The server is designed to be thread-safe, capable of handling concurrent fork
requests from multiple threads in a client application.

To run the server:
    python -m sandbox.python_forkserver

The client can then be used as follows:
    from sandbox.python_forkserver import ForkServerClient

    client = ForkServerClient()
    client.start_server() # Or start it manually

    # This code will run in a new, forked process
    code_to_run = "import time; print('Hello from child'); time.sleep(2)"

    # The client handles creating pipes for stdout/stderr
    pid, stdout_fd, stderr_fd = client.fork(code_to_run)

    print(f"Forked process with PID: {pid}")

    # Now you can read from the stdout/stderr file descriptors
    # (e.g., using os.fdopen or by integrating with an event loop)

    client.shutdown_server()
"""

import array
import json
import logging
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from typing import Optional, TextIO, Tuple, TextIO

# --- Configuration ---
FORK_SERVER_SOCKET_PATH = "/tmp/python_fork_server.sock"
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# --- Server Implementation ---


class ForkServer:
    """
    The fork server process. Listens on a Unix domain socket for requests.
    """

    def __init__(self, socket_path: str):
        self.socket_path = socket_path
        self.sock: Optional[socket.socket] = None

    def _handle_connection(self, conn: socket.socket):
        """
        Handles a single client connection in its own thread.
        Receives code and file descriptors, forks, and sends back the PID.
        """
        try:
            logging.info(f"Handling new connection from {conn.getpeername()}")
            # Step 1: Receive the data (code and file descriptors)
            # The ancillary data contains the file descriptors.
            fds = array.array("i")
            # Increased buffer size for safety
            msg, ancdata, _, _ = conn.recvmsg(
                4096, socket.CMSG_LEN(fds.itemsize * 2)
            )
            logging.info("Received message and ancillary data.")

            for cmsg_level, cmsg_type, cmsg_data in ancdata:
                if (
                    cmsg_level == socket.SOL_SOCKET
                    and cmsg_type == socket.SCM_RIGHTS
                ):
                    fds.frombytes(cmsg_data[: len(cmsg_data) - (len(cmsg_data) % fds.itemsize)])
            
            payload = json.loads(msg.decode("utf-8"))
            code_to_run = payload["code"]
            stdout_fd, stderr_fd = fds[0], fds[1]
            logging.info(f"Received code, stdout_fd={stdout_fd}, stderr_fd={stderr_fd}")

            # Step 2: Fork the child process
            logging.info("Forking child process...")
            pid = os.fork()

            if pid == 0:  # In the child process
                logging.info("Child process started. Closing inherited resources.")
                # We are now in a fresh process.
                try:
                    # The child must close its copy of the client connection socket.
                    conn.close()

                    # Redirect stdout/stderr to the pipes provided by the client.
                    os.dup2(stdout_fd, sys.stdout.fileno())
                    os.dup2(stderr_fd, sys.stderr.fileno())

                    # Close the original fds as they are now duplicated
                    os.close(stdout_fd)
                    os.close(stderr_fd)

                    # Close the listening socket in the child
                    if self.sock:
                        self.sock.close()

                    # Execute the user's code
                    exec(code_to_run, {"__name__": "__main__"})
                except Exception as e:
                    # If something goes wrong, print to the new stderr and exit
                    print(f"Fork server child execution failed: {e}", file=sys.stderr)
                finally:
                    # Ensure the child process always exits and does not return
                    # to the server's main loop.
                    os._exit(0)

            else:  # In the parent (fork server) process
                logging.info(f"Parent process: Child forked with PID {pid}.")
                # The parent does not need the client's FDs, and should not
                # close them. They are for the child process only. The child
                # is responsible for closing them after dup2.
                # os.close(stdout_fd) # This is incorrect and causes a race.
                # os.close(stderr_fd) # This is incorrect and causes a race.

                # Step 3: Send the new PID back to the client
                logging.info(f"Sending PID {pid} back to client.")
                conn.sendall(struct.pack("!I", pid))
                logging.info("PID sent successfully.")

        except Exception as e:
            logging.error(f"Error handling connection: {e}", exc_info=True)
        finally:
            logging.info("Closing connection.")
            conn.close()

    def run(self):
        """
        Main server loop. Accepts connections and spawns handler threads.
        """
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)

        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(self.socket_path)
        self.sock.listen(100)  # Listen for up to 100 concurrent connections

        logging.info(f"Fork server listening on {self.socket_path}")

        try:
            while True:
                conn, _ = self.sock.accept()
                handler_thread = threading.Thread(
                    target=self._handle_connection, args=(conn,)
                )
                handler_thread.daemon = True
                handler_thread.start()
        finally:
            logging.info("Fork server shutting down.")
            if self.sock:
                self.sock.close()
            if os.path.exists(self.socket_path):
                os.remove(self.socket_path)


# --- Client Implementation ---


class ForkServerClient:
    """
    A thread-safe client for the fork server.
    """

    def __init__(self, socket_path: str = FORK_SERVER_SOCKET_PATH):
        self.socket_path = socket_path
        self._server_process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()  # To ensure thread-safe fork requests
        self._stdout_log: Optional[TextIO] = None
        self._stderr_log: Optional[TextIO] = None

    def start_server(self, timeout: float = 5.0) -> bool:
        """
        Starts the fork server as a separate process.
        Returns True if the server started successfully.
        """
        if self.is_server_running():
            logging.info("Server is already running.")
            return True

        # The server process now handles its own logging.
        # We just need to start it.
        logging.info("Starting fork server process...")
        command = [sys.executable, "-m", "sandbox.python_forkserver"]
        self._server_process = subprocess.Popen(
            command,
            # Detach from parent's stdio
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for the socket file to be created
        start_time = time.monotonic()
        while not os.path.exists(self.socket_path):
            if self._server_process.poll() is not None:
                logging.error(
                    "Fork server process terminated prematurely. "
                    "Check server logs in /tmp/trelis_arc_logs/"
                )
                return False
            if time.monotonic() - start_time > timeout:
                logging.error("Fork server failed to start in time.")
                self.shutdown_server()
                return False
            time.sleep(0.05)

        logging.info(f"Fork server started with PID {self._server_process.pid}")
        return True

    def shutdown_server(self):
        """
        Shuts down the fork server process.
        """
        if self._server_process:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
            self._server_process = None
            logging.info("Fork server shut down.")

        # Clean up socket file if it's left over
        if os.path.exists(self.socket_path):
            try:
                os.remove(self.socket_path)
            except OSError:
                pass

    def is_server_running(self) -> bool:
        """
        Checks if the server socket is available.
        """
        if os.path.exists(self.socket_path):
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                    s.connect(self.socket_path)
                return True
            except (ConnectionRefusedError, FileNotFoundError):
                return False
        return False

    def fork(self, code_to_run: str) -> Tuple[int, int, int]:
        """
        Requests the server to fork a new process.

        Args:
            code_to_run: The Python code to execute in the child.

        Returns:
            A tuple of (pid, stdout_read_fd, stderr_read_fd).
            The caller is responsible for closing the file descriptors.
        """
        with self._lock:
            # Step 1: Create pipes for stdout and stderr
            stdout_r, stdout_w = os.pipe()
            stderr_r, stderr_w = os.pipe()

            conn = None
            try:
                # Step 2: Connect to the server
                conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                conn.connect(self.socket_path)

                # Step 3: Send code and file descriptors
                payload = json.dumps({"code": code_to_run}).encode("utf-8")
                fds = array.array("i", [stdout_w, stderr_w])
                conn.sendmsg([payload], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds)])

                # The write ends of the pipes are now in the hands of the server/child,
                # so we can close them in this process.
                os.close(stdout_w)
                os.close(stderr_w)
                # Nullify to prevent double-closing in the finally block
                stdout_w = -1
                stderr_w = -1


                # Step 4: Receive the PID from the server
                response = conn.recv(struct.calcsize("!I"))
                if not response:
                    raise ConnectionAbortedError("Server closed connection unexpectedly.")
                
                pid = struct.unpack("!I", response)[0]

                return pid, stdout_r, stderr_r

            except Exception as e:
                # If something goes wrong, we still own the read ends, so close them.
                os.close(stdout_r)
                os.close(stderr_r)
                logging.error(f"Failed to fork: {e}")
                raise
            finally:
                if conn:
                    conn.close()
                # Clean up write ends only if they haven't been successfully closed.
                if stdout_w != -1:
                    os.close(stdout_w)
                if stderr_w != -1:
                    os.close(stderr_w)


# --- Main Execution Block ---

if __name__ == "__main__":
    # Add a file handler to the root logger for the server process
    log_dir = "/tmp/trelis_arc_logs"
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, "forkserver_main.log"))
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().setLevel(logging.INFO)

    logging.info("Starting fork server...")
    server = ForkServer(FORK_SERVER_SOCKET_PATH)
    server.run()
