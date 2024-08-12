import errno
import os
import signal
import functools
import time


# adapted from https://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish
def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            delta = seconds
            try:
                t1 = time.time()
                results = func(*args, **kwargs)
                t2 = time.time()
                delta = t2 - t1
            except KeyError:
                pass
            finally:
                signal.alarm(0)

            return results, delta

        return wrapper

    return decorator
