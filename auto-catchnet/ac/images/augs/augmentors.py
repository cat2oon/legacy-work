from imgaug import augmenters as iaa

from ac.images.augs.builder import AugBuilder


def make_basic():
    builder = AugBuilder()
    builder \
        .sometimes(0.38) \
        .translator(0.15, 0.15) \
        .sometimes(0.35) \
        .scalar((0.7, 1.25)) \
        .sometimes(0.3) \
        .blur() \
        .sometimes(0.3) \
        .noisy() \
        .sometimes(0.3) \
        .drop()
    # .sometimes(0.3) \
    #     .rotator(15) \

    return builder.build()


def for_iris_contour():
    builder = AugBuilder()
    builder \
        .sometimes(0.4) \
        .blur() \
        .sometimes(0.5) \
        .noisy() \
        .sometimes(0.4) \
        .drop()

    return builder.build()


