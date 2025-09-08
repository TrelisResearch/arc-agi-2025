"""
Docker-based Python code executor with isolated environment.
"""

import time
import pickle
import base64
import random
from docker import DockerClient
import requests
import atexit
import threading
from typing import Any, Optional, Tuple
from pathlib import Path

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

from .base_executor import BaseExecutor


class DockerSandboxExecutor(BaseExecutor):
    """
    Executor that runs Python code in a Docker container for isolation.
    """
    
    def __init__(self, image_name: str = "python-sandbox:v3", container_port: int = 7934):
        """
        Initialize the Docker sandbox executor.
        
        Args:
            image_name: Name of the Docker image to use
            container_port: Port to expose on the container
        """
        if not DOCKER_AVAILABLE:
            raise ImportError("Docker Python client not available. Install with: pip install docker")
            
        self.image_name = image_name
        self.container_port = container_port
        self.host_port: Optional[int] = None
        self.container: Optional[docker.models.containers.Container] = None
        self.client: Optional[DockerClient] = None
        self._setup_done = False
        self._shutdown_lock = threading.Lock()
        
        # Register cleanup on exit
        atexit.register(self._cleanup_on_exit)
    
    def setup(self) -> None:
        """Set up the Docker container and build the image if needed."""
        if self._setup_done:
            return
            
        try:
            self.client = docker.from_env()
            
            # Build the Docker image if it doesn't exist
            self._build_image_if_needed()
            
            # Start the container
            self._start_container()
            
            # Wait for the container to be ready
            self._wait_for_container_ready()
            
            self._setup_done = True
            
        except Exception as e:
            self.cleanup()
            raise Exception(f"Failed to set up Docker sandbox: {e}")
    
    def _build_image_if_needed(self) -> None:
        """Build the Docker image if it doesn't exist."""
        if not self.client:
            raise Exception("Docker client not initialized")
            
        try:
            self.client.images.get(self.image_name)
            # Image already exists - no need to print
        except docker.errors.ImageNotFound:
            # Only print when we need to build
            print(f"Building Docker image '{self.image_name}'...")
            
            # Get the directory containing this script
            script_dir = Path(__file__).parent
            
            # Build the image
            image, logs = self.client.images.build(
                path=str(script_dir),
                tag=self.image_name,
                rm=True
            )
            
            print(f"Successfully built Docker image '{self.image_name}'")
    
    def _start_container(self) -> None:
        """Start the Docker container."""
        if not self.client:
            raise Exception("Docker client not initialized")
            
        try:
            # Find an available port
            self.host_port = self._find_available_port()
            
            # Start the container with unique name (timestamp + random to prevent conflicts)
            unique_id = f"{int(time.time())}_{random.randint(10000, 99999)}"
            self.container = self.client.containers.run(
                self.image_name,
                ports={f'{self.container_port}/tcp': self.host_port},
                detach=True,
                remove=True,  # Auto-remove when stopped
                name=f"python-sandbox-{unique_id}"
            )
            
            # Container started successfully - no need to print
            
        except Exception as e:
            raise Exception(f"Failed to start Docker container: {e}")
    
    def _find_available_port(self) -> int:
        """Find an available port on the host."""
        import socket
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    def _wait_for_container_ready(self, max_wait: int = 30) -> None:
        """Wait for the container to be ready to accept requests."""
        base_url = f"http://localhost:{self.host_port}"
        
        for i in range(max_wait):
            try:
                response = requests.get(f"{base_url}/health", timeout=1)
                if response.status_code == 200:
                    # Container is ready - no need to print
                    return
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(1)
        
        raise Exception(f"Container did not become ready within {max_wait} seconds")
    
    def execute_code(self, code: str, timeout: Optional[float] = 5) -> Tuple[Any, Optional[Exception]]:
        """
        Execute Python code in the Docker container.
        
        Args:
            code (str): Python code to execute (should contain a return statement).
            timeout (float, optional): Timeout in seconds.
        
        Returns:
            Tuple[Any, Optional[Exception]]: A tuple containing:
                - The result of the code execution (None if there was an error)
                - An exception if an error occurred (None if successful)
        """
        if not self._setup_done:
            self.setup()
        
        if not self.container:
            return None, Exception("Docker container is not available")
        
        try:
            # Check if container is still running
            self.container.reload()
            if self.container.status != 'running':
                return None, Exception("Docker container is not running")
            
            # Make request to the container
            base_url = f"http://localhost:{self.host_port}"
            request_data = {
                "code": code,
                "timeout": timeout
            }
            
            response = requests.post(
                f"{base_url}/execute",
                json=request_data,
                timeout=timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"HTTP error {response.status_code}: {response.text}")
            
            result_data = response.json()
            
            if result_data["success"]:
                # Deserialize the result
                try:
                    decoded_result = base64.b64decode(result_data["result"])
                    deserialized_result = pickle.loads(decoded_result)
                    return deserialized_result, None
                except Exception as e:
                    return None, Exception(f"Failed to deserialize result: {e}")
            else:
                # Return the error from the container
                error_msg = result_data.get("error", "Unknown error")
                error_type = result_data.get("error_type", "Exception")
                return None, Exception(f"{error_type}: {error_msg}")
                
        except requests.exceptions.Timeout:
            raise Exception(f"Request timed out after {timeout} seconds")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {e}")
        except Exception as e:
            raise Exception(f"Execution failed: {e}")
    
    def cleanup(self) -> None:
        """Clean up the Docker container and resources."""
        with self._shutdown_lock:
            if self.container:
                try:
                    self.container.stop(timeout=5)
                    # Explicitly remove container to ensure cleanup (even though remove=True should handle this)
                    try:
                        self.container.remove(force=True)
                    except Exception as remove_e:
                        # Container might already be auto-removed, which is fine
                        pass
                except Exception as e:
                    print(f"Error stopping container: {e}")
                finally:
                    self.container = None
            
            if self.client:
                try:
                    self.client.close()
                except Exception as e:
                    print(f"Error closing Docker client: {e}")
                finally:
                    self.client = None
    
    def _cleanup_on_exit(self) -> None:
        """Cleanup function called on process exit."""
        self.cleanup()