import errno
import os
import signal
import functools


# adapted from https://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish
def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                results = func(*args, **kwargs)
            finally:
                signal.alarm(0)

            return results

        return wrapper

    return decorator
