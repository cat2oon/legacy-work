from keras.callbacks import TensorBoard


def make_tensor_board(log_dir='./graph'):
    tb = TensorBoard(log_dir=log_dir,
                     histogram_freq=1,
                     write_graph=True,
                     write_images=True)
    return tb

