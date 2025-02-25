from keras.callbacks import EarlyStopping


def early_stopper(monitor='val_loss',
                  min_delta=0,
                  patience=5,
                  mode='auto',
                  restore_best_weights=True):

    stopper = EarlyStopping(monitor=monitor,
                            min_delta=min_delta,
                            patience=patience,
                            mode=mode,
                            baseline=None,
                            restore_best_weights=restore_best_weights)
    return stopper
