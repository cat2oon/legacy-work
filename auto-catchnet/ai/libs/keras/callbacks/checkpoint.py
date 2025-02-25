from keras.callbacks import ModelCheckpoint


def model_checkpoint(checkpoint_path,
                     save_best_only=True,
                     save_weights_only=True,
                     mode='auto',
                     period=1):

    mc = ModelCheckpoint(checkpoint_path,
                         monitor='val_loss',
                         verbose=0,
                         save_best_only=save_best_only,
                         save_weights_only=save_weights_only,
                         mode=mode,
                         period=period)
    return mc
