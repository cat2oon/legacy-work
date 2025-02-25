

class ModelOutputs:
    def __init__(self):
        self.bag = {}

    def __getitem__(self, key):
        return self.bag[key]

    def __setitem__(self, key, value):
        self.bag[key] = value

