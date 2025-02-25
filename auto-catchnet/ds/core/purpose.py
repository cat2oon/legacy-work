import enum


class Purpose(enum.Enum):

    ALL = "all"
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"

    def for_all(self):
        return self is Purpose.ALL

    def for_train(self):
        return self is Purpose.TRAIN

    def for_valid(self):
        return self is Purpose.VALID

    def for_test(self):
        return self is Purpose.VALID

