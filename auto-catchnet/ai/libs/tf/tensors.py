
class Tensors:
    def __init__(self):
        self.bag = {}
        self.ordered = []

    def init(self):
        self.bag = {}
        self.ordered = []

    def __getitem__(self, key):
        return self.bag[key]

    def __setitem__(self, key, value):
        self.bag[key] = value
        self.ordered.append(value)

    def get_tensors(self):
        return self.ordered
