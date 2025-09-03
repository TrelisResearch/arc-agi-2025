import os
import sys
import threading
import time


def _force_exit_after(status: int, timeout_seconds: int):
    """
    After a delay, force the program to exit.
    To be run in a daemon thread.
    """
    time.sleep(timeout_seconds)
    print(
        f"Graceful exit failed. Timeout of {timeout_seconds}s reached. Forcing exit now."
    )
    os._exit(status)  # Exit with a non-zero status code to indicate failure


def ensure_system_exit(status: int, graceful_timeout=10):
    print(
        f"Graceful exit starting, process will be force terminated in {graceful_timeout} seconds."
    )

    watchdog = threading.Thread(
        target=_force_exit_after,
        args=(
            status,
            graceful_timeout,
        ),
        daemon=True,
    )
    watchdog.start()

    sys.exit(status)
