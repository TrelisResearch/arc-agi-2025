"""
Simple, self-contained subprocess executor for Python code.
No dependencies on base classes or complex imports.
"""
import subprocess
import sys
import pickle
import base64
import os
import shutil
from typing import Optional, Any, Tuple

# Set the memory limit for the subprocess in Megabytes.
MEMORY_LIMIT_MB = 256

def execute_code_in_subprocess(
    code: str, timeout: Optional[float] = None
) -> Tuple[Any, Optional[Exception]]:
    """
    Executes a Python code string in a memory and time-constrained subprocess.

    This function uses the best available mechanism for the host OS:
    - On Linux with systemd: Uses `systemd-run` for strict, cgroup-based limits.
    - On other Linux systems & macOS: Falls back to using `ulimit -m`, a
      "best-effort" approach that works in environments like GitHub Actions CI.
    - Other OS: Not supported.

    Args:
        code (str): Python code to execute (should contain a return statement).
        timeout (float, optional): Timeout in seconds.

    Returns:
        A tuple containing the result or an exception.
    """
    # This Python script will be run by the sandboxing command. It deserializes the
    # user code and executes it, returning the result or error.
    runner_script = """
import pickle, base64, sys
try:
    encoded_code = sys.argv[1]
    user_code = base64.b64decode(encoded_code).decode('utf-8')
    # Dynamically create the function and execute it
    exec_globals = {}
    full_code = "def user_function():\\n" + user_code
    exec(full_code, exec_globals)
    result = exec_globals['user_function']()
    serialized = base64.b64encode(pickle.dumps(result)).decode('utf-8')
    print(f"RESULT_START{serialized}RESULT_END")
except Exception as e:
    serialized_error = base64.b64encode(pickle.dumps(e)).decode('utf-8')
    print(f"ERROR_START{serialized_error}ERROR_END", file=sys.stderr)
    sys.exit(1)
"""

    try:
        # 1. Encode the user's code to be passed as a safe shell argument
        indented_code = "\n".join("    " + line for line in code.splitlines())
        encoded_user_code = base64.b64encode(indented_code.encode('utf-8')).decode('utf-8')
        
        # --- OS-Specific Sandbox Implementation ---
        
        #
        # ** PREFERRED: Linux with systemd (cgroups) **
        #
        if sys.platform == "linux" and shutil.which("systemd-run"):
            command_args = [
                'systemd-run',
                '--scope',
                '--user',
                '-p', f'MemoryMax={MEMORY_LIMIT_MB}M',
                sys.executable,
                '-c',
                runner_script,
                encoded_user_code
            ]
        #
        # ** FALLBACK: Linux without systemd, or macOS (ulimit) **
        # This branch will be used by GitHub Actions CI runners.
        #
        elif sys.platform in ["linux", "darwin"]:
            memory_limit_kb = MEMORY_LIMIT_MB * 1024
            encoded_runner_script = base64.b64encode(runner_script.encode('utf-8')).decode('utf-8')
            
            # Note: `base64 -d` is standard on Linux and macOS.
            ulimit_command = f"""
ulimit -m {memory_limit_kb};
runner_script=$(echo {encoded_runner_script} | base64 -d);
exec {sys.executable} -c "$runner_script" {encoded_user_code}
"""
            command_args = ['sh', '-c', ulimit_command]
        #
        # ** OTHER OS - NOT SUPPORTED **
        #
        else:
            return None, NotImplementedError(f"This function is not implemented for OS '{sys.platform}'.")


        # 3. Execute the command.
        proc = subprocess.run(
            command_args,
            capture_output=True, text=True, timeout=timeout
        )

        stdout, stderr = proc.stdout, proc.stderr

        # Check for OOM killer messages, which vary by platform.
        if proc.returncode != 0 and (
            "oom-kill" in stderr or # systemd
            "Failed to allocate memory" in stderr or # systemd
            "Killed" in stderr or # ulimit
            "Cannot allocate memory" in stderr # ulimit
        ):
             return None, MemoryError(f"Process killed due to excessive memory usage (Limit: ~{MEMORY_LIMIT_MB}MB).")

        if "ERROR_START" in stderr:
            start_marker, end_marker = "ERROR_START", "ERROR_END"
            start_idx = stderr.find(start_marker) + len(start_marker)
            end_idx = stderr.find(end_marker)
            serialized_error = stderr[start_idx:end_idx]
            return None, pickle.loads(base64.b64decode(serialized_error))
        elif "RESULT_START" in stdout:
            start_marker, end_marker = "RESULT_START", "RESULT_END"
            start_idx = stdout.find(start_marker) + len(start_marker)
            end_idx = stdout.find(end_marker)
            serialized_result = stdout[start_idx:end_idx]
            return pickle.loads(base64.b64decode(serialized_result)), None
        else:
            # Catchall for other systemd errors (e.g., can't create scope)
            return None, Exception(f"An unknown error occurred. Stderr: {stderr}")

    except subprocess.TimeoutExpired:
        return None, Exception(f"Code execution timed out after {timeout} seconds.")
    except Exception as e:
        return None, Exception(f"Subprocess execution failed: {e}")

