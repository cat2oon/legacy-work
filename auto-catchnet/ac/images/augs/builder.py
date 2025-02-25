from imgaug import augmenters as iaa


class AugBuilder:
    def __init__(self):
        self.pipeline = []
        self.phasor = None

    def build(self):
        return iaa.Sequential(self.pipeline)

    def add(self, aug):
        aug = self.apply_phasor(aug)
        self.pipeline.append(aug)
        return self

    def apply_phasor(self, aug):
        if self.phasor is None:
            return aug
        aug = self.phasor(aug)
        self.phasor = None
        return aug

    def sometimes(self, percent):
        self.phasor = lambda aug: iaa.Sometimes(percent, aug)
        return self

    """
     Operator(s) 
    """

    def translator(self, x, y):
        return self.add(iaa.Affine(translate_percent={
            "x": (-x, x), "y": (-y, y)
        }))

    def scalar(self, ratio_range):
        return self.add(iaa.Affine(scale=ratio_range))

    def rotator(self, degree):
        return self.add(iaa.Affine(rotate=(-degree, degree)))

    def blur(self):
        aug = iaa.SomeOf((1, 2), [
            iaa.Noop(),
            iaa.AverageBlur(k=(1, 7)),
            iaa.GaussianBlur(sigma=(0.1, 2.0)),
            iaa.MotionBlur(k=(6, 11), angle=(-20, 20)),
            iaa.Multiply((0.5, 1.3))
        ], random_order=True)
        return self.add(aug)

    def noisy(self):
        aug = iaa.SomeOf((1, 2), [
            iaa.Noop(),
            iaa.ChannelShuffle(1.0),
            iaa.Add((-20, 20)),
            iaa.AdditiveGaussianNoise(scale=0.05 * 255),
            iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),
        ], random_order=True)
        return self.add(aug)

    def drop(self):
        aug = iaa.SomeOf((1, 3), [
            iaa.Noop(),
            iaa.Dropout(p=(0, 0.05)),
            iaa.CoarseDropout((0.01, 0.10), size_percent=(0.01, 0.3), per_channel=True),
            iaa.CoarseDropout((0.01, 0.10), size_percent=(0.01, 0.3)),
        ], random_order=True)
        return self.add(aug)

