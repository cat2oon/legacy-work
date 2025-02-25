class ReferenceDecorator:
    def __init__(self, func, link):
        self.func = func
        self.link = link

    def __call__(self, *args, **kwargs):
        # print("'{}' is referred from {}".format(self.func.__name__, self.link))
        return self.func(*args, **kwargs)