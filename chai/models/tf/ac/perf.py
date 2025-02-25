import time

def timeit(func):
    def timed(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
            print('%r  %2.2f ms' % (func.__name__, (te - ts) * 1000))
        return result
    return timed