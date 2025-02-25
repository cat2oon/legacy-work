def make_counter():
    def counter():
        counter.count = counter.count + 1

    counter.count = 0
    return counter


def get_field_key_val_pairs(obj):
    return [(key, getattr(obj, key)) for key in obj.__dict__.keys()]

