import functools


def countcalls(fn):
    @functools.wraps(fn)
    def wrapped(*args):
        wrapped.num_calls += 1
        return fn(*args)

    wrapped.num_calls = 0
    return wrapped
