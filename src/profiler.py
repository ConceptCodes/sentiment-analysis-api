import time
import logging
from functools import wraps

def profile(func):
    """Decorator to measure and log the execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logging.info(f"Function '{func.__name__}' executed in {elapsed:.4f} seconds.")
        return result
    return wrapper

