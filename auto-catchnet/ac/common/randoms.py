import numpy as np


class RandomSeed:
    def __init__(self):
        self.numeric_seed = 860515
        self.alpha_numeric_seed = "860515"

    def set_numpy_seed(self):
        np.random.seed(self.numeric_seed)

    def alpha_numeric(self):
        return self.alpha_numeric_seed
