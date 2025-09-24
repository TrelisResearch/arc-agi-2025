"""
Client for the HTTP-based fork server.
"""
import asyncio
import subprocess
import sys
import json
import aiohttp
import os
from typing import Optional

class ForkServerClient:
    """
    A client for the HTTP-based fork server.

    This class manages the lifecycle of the fork server subprocess and provides
    an async interface for executing code. It is designed to be used as an
    async context manager.

    Requires the `aiohttp` library to be installed.

    Example:
        async with ForkServerClient() as client:
            pid = await client.execute("print('hello from a fork')")
            # The client is now responsible for monitoring this PID.
            print(f"Got PID: {pid}")
    """

    def __init__(self, port=8765):
        self.port = port
        self.server_url = f"http://localhost:{self.port}"
        self.server_process: Optional[subprocess.Popen] = None
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Starts the server subprocess and waits for it to be ready."""
        # Command to run the server module. Using -u for unbuffered output.
        command = [sys.executable, "-u", "-m", "sandbox.http_fork_server"]
        
        self.server_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # This is a simple readiness check.
        try:
            await self._await_server_ready()
        except Exception as e:
            self.terminate() # Clean up if readiness check fails
            raise e

        self.session = aiohttp.ClientSession()
        return self

    async def _await_server_ready(self):
        """Asynchronously waits for the server to be ready by polling its root endpoint."""
        # Create a temporary session for the readiness check.
        async with aiohttp.ClientSession() as session:
            for _ in range(40):  # Poll for up to 4 seconds
                if self.server_process.poll() is not None:
                    stderr = self.server_process.stderr.read()
                    raise RuntimeError(
                        f"Fork server failed to start. Stderr:\n{stderr}"
                    )
                try:
                    async with session.get(f"{self.server_url}/") as response:
                        if response.status == 200:
                            return
                except aiohttp.ClientConnectorError:
                    # Server not yet available, wait and retry
                    await asyncio.sleep(0.1)
                    continue
        
        # If the loop finishes, the server is not ready.
        stderr_output = self.server_process.stderr.read()
        raise RuntimeError(f"Fork server timed out. Stderr:\n{stderr_output}")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Terminates the server subprocess and closes the session."""
        if self.session:
            await self.session.close()
        self.terminate()

    def terminate(self):
        """Helper to stop the server process."""
        if self.server_process:
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_process.kill()

    async def execute(self, code: str) -> int:
        """
        Sends code to the fork server for execution.

        Args:
            code: The Python code string to execute.

        Returns:
            The process ID (PID) of the forked child process executing the code.
        """
        if not self.session:
            raise RuntimeError("Client session not started. Use within an 'async with' block.")

        try:
            async with self.session.post(f"{self.server_url}/execute", json={"code": code}) as response:
                response.raise_for_status()
                data = await response.json()
                return data['pid']
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Failed to connect to fork server: {e}") from e

async def main():
    """Example usage of the ForkServerClient."""
    print("Starting ForkServerClient example...")
    try:
        async with ForkServerClient() as client:
            print("Client started. Executing code...")
            pid = await client.execute(
                "import time; print('Child process started'); time.sleep(2); print('Child process finished')"
            )
            print(f"Code execution started in process with PID: {pid}")
            
            # For this example, we'll just wait a bit for it to finish.
            await asyncio.sleep(2.5)
            print("Example finished.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
