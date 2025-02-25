def set_random_seed(seed_num):
    from numpy.random import seed
    from tensorflow import set_random_seed
    seed(seed_num)
    set_random_seed(seed_num)


def set_nhwc_order():
    import keras.backend as K
    K.set_image_dim_ordering('tf')


