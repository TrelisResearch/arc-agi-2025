from sandbox.python_forkserver import ForkServerClient
import os

# Create a client
client = ForkServerClient()
client.start_server()

# You can start the server programmatically if you prefer
# client.start_server()

# Define the code to run
code = """
import sys
import time
print("Hello from the forked process!")

print("This is going to stderr.", file=sys.stderr)
"""

# Fork the process
pid, stdout_fd, stderr_fd = client.fork(code)
print(f"Started process with PID: {pid}")

# Now you can read the output using the file descriptors.
# It's best to use non-blocking reads or a selector in a real app.
with os.fdopen(stdout_fd, 'r') as stdout_stream:
    print("Reading from child stdout:")
    print(stdout_stream.read())

with os.fdopen(stderr_fd, 'r') as stderr_stream:
    print("Reading from child stderr:")
    print(stderr_stream.read())

# Don't forget to manage the process lifecycle (e.g., os.waitpid)
try:
    os.waitpid(pid, 0)
except ChildProcessError:
    # The child process has already exited
    pass

# When you're all done
# client.shutdown_server()