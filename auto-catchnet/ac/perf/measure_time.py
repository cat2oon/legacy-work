import time


class MeasureTime:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        start_time = time.perf_counter()
        cpu_start_time = time.process_time()

        ret = self.func(*args, **kwargs)

        time_elapsed = time.perf_counter() - start_time
        cpu_time_elapsed = time.process_time() - cpu_start_time

        print("elapsed time: {:.1f} (s)".format(time_elapsed))
        print("cpu elapsed time: {:.1f} (s)".format(cpu_time_elapsed))

        return ret
