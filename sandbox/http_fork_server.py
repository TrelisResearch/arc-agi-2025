"""
A simple HTTP-based fork server for executing Python code in isolated processes.

This server provides a lightweight and fast way to execute arbitrary code
without the overhead of starting a new Python interpreter for each execution.

It is designed to be used with an external process monitor.

Endpoints:
  - POST /execute:
    - Expects a JSON body: {"code": "..."}
    - Forks a new process to execute the code.
    - Immediately returns a JSON response: {"pid": ...}
    - The client is responsible for monitoring the PID and handling the process's
      output, which is expected to be managed via mechanisms embedded in the
      provided code (e.g., writing to a file path passed within the code).
"""
import http.server
import socketserver
import json
import os
import sys
import signal

class HTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'status': 'ok'}).encode('utf-8'))

    def do_POST(self):
        if self.path == '/execute':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                payload = json.loads(post_data)
                code = payload['code']

                # The core of the fork server.
                pid = os.fork()

                if pid == 0:
                    # --- Child Process ---
                    # Detach from the parent's session to ensure clean exit
                    os.setsid()

                    # Close the inherited server socket
                    self.server.socket.close()

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
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    response = {'pid': pid}
                    self.wfile.write(json.dumps(response).encode('utf-8'))

            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                error_response = {'error': f'Internal server error: {e}'}
                self.wfile.write(json.dumps(error_response).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')

    def log_message(self, format, *args):
        # Suppress logging to keep stdout clean
        return

def run_server(port=8765):
    # Ignore SIGCHLD to prevent zombie processes. The kernel will auto-reap them.
    signal.signal(signal.SIGCHLD, signal.SIG_IGN)
    with socketserver.TCPServer(("", port), HTTPRequestHandler) as httpd:
        print(f"Forking HTTP server started on port {port}...")
        httpd.serve_forever()

if __name__ == "__main__":
    # To run this server: python -m sandbox.http_fork_server <port>
    run_server(int(sys.argv[1]) if len(sys.argv) > 1 else 8765)
