import time
from keras.callbacks import Callback


class TimeHistory(Callback):
    def __init__(self):
        self.times = None
        self.last_epoch_start = None

    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.last_epoch_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.last_epoch_start)

    def on_train_end(self, logs={}):
        print('\n==> Total time elapsed: ', self.total_time())

    def total_time(self):
        return sum(self.times)
