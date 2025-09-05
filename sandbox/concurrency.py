import threading


class ConcurrencyGate:
    """
    A thread-safe context manager to limit the number of threads
    concurrently executing a block of code.
    """

    def __init__(self, max_concurrent: int):
        """
        Initializes the gate with a maximum number of concurrent threads.

        Args:
            max_concurrent: The maximum number of threads allowed to be
                            "inside" the gate at any one time.
        """
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be greater than 0")
        # A BoundedSemaphore is a good choice as it prevents bugs where release()
        # is called more times than acquire().
        self.semaphore = threading.BoundedSemaphore(max_concurrent)
        self.max_concurrent = max_concurrent

    def __enter__(self):
        """
        Acquires the semaphore, blocking if the limit is reached.
        This is called at the start of the 'with' block.
        """
        self.semaphore.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Releases the semaphore, allowing another waiting thread to enter.
        This is called at the end of the 'with' block.
        """
        self.semaphore.release()
