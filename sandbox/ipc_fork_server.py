"""
A simple IPC-based fork server for executing Python code in isolated processes.

This server uses a UNIX domain socket for communication and provides a lightweight
and fast way to execute arbitrary code without the overhead of starting a new
Python interpreter for each execution.

It is designed to be used with an external process monitor.

The communication protocol is as follows:
1. The client connects to the server's UNIX domain socket.
2. The client sends a 4-byte unsigned integer (big-endian) representing the
   length of the code to be executed.
3. The client sends the UTF-8 encoded code.
4. The server forks a new process to execute the code.
5. The server sends back a 4-byte unsigned integer (big-endian) representing
   the process ID (PID) of the child process.
6. The connection is closed.
"""
import socket
import os
import sys
import signal
import struct
import threading
import uuid

def handle_client(client_socket):
    try:
        # Read the length of the code
        len_data = client_socket.recv(4)
        if not len_data:
            return
        code_len = struct.unpack('>I', len_data)[0]

        # Read the code
        code = b''
        while len(code) < code_len:
            part = client_socket.recv(code_len - len(code))
            if not part:
                # Connection closed prematurely
                return
            code += part
        code = code.decode('utf-8')

        # The core of the fork server.
        pid = os.fork()

        if pid == 0:
            # --- Child Process ---
            # Detach from the parent's session to ensure clean exit
            os.setsid()

            # Close inherited sockets
            client_socket.close()
            # The server socket is not available here to be closed, which is fine.

            # Execute the provided code directly.
            # The code is responsible for its own execution logic,
            # including handling results and errors.
            try:
                exec(code)
            finally:
                # Ensure the child process always exits immediately.
                os._exit(0)
        else:
            # --- Parent Process ---
            # Respond to the client immediately with the PID.
            pid_data = struct.pack('>I', pid)
            client_socket.sendall(pid_data)

    except Exception as e:
        # Use stderr for server-side errors to not pollute stdout
        print(f"Error handling client: {e}", file=sys.stderr)
    finally:
        client_socket.close()

def run_server(socket_path=None):
    # Ignore SIGCHLD to prevent zombie processes. The kernel will auto-reap them.
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)

    if socket_path is None:
        socket_path = f"/tmp/fork_server_{uuid.uuid4()}.sock"

    if os.path.exists(socket_path):
        os.remove(socket_path)

    server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_socket.bind(socket_path)
    server_socket.listen(128) # Allow a backlog of connections

    # Print the socket path to stdout so the parent process can capture it.
    print(socket_path)
    sys.stdout.flush()

    try:
        while True:
            client_socket, _ = server_socket.accept()
            # Handle each client in a new thread to support multiplexing
            client_thread = threading.Thread(target=handle_client, args=(client_socket,))
            client_thread.daemon = True
            client_thread.start()
    finally:
        server_socket.close()
        if os.path.exists(socket_path):
            os.remove(socket_path)
        # Use stderr for server-side messages
        print("Server shut down.", file=sys.stderr)


if __name__ == "__main__":
    # To run this server: python -m sandbox.ipc_fork_server [socket_path]
    # If socket_path is not provided, a unique one will be generated and printed to stdout.
    run_server(sys.argv[1] if len(sys.argv) > 1 else None)
