class DeprecatedDecorator:
    def __init__(self, f):
        self.func = f

    def __call__(self, *args, **kwargs):
        print("this function is deprecated")
        self.func(*args, **kwargs)