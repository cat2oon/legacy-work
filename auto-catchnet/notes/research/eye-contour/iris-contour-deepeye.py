import sys
import keras
import warnings
import numpy as np
             
sys.path.append("../../../")

from keras import backend as K
from ds.icontour.npz.gen import IrisContourGenerator
from ds.icontour.npz.gen import Purpose
from ac.visualizer.plotter import *
from ai.model.iris.deepeye.iris import make_deep_eye_net
from ai.libs.keras.callbacks.history import TimeHistory
from ai.libs.keras.callbacks.stopper import EarlyStopping
from ai.libs.keras.callbacks.tensorboard import TensorBoard
from ai.libs.keras.callbacks.checkpoint import model_checkpoint

warnings.filterwarnings("ignore")
np.set_printoptions(precision=6, suppress=True)

INPUT_SHAPE=(64, 64, 3)
im = make_deep_eye_net(input_shape=INPUT_SHAPE, num_filters=16, aspp_deep=2, num_classes=2)
im.summary()


chk_path = "/home/chy/archive-model/incubator/iris-contour/deepeye-{epoch:02d}-{val_loss:.7f}.hdf5"

history = TimeHistory()
checkpoint = model_checkpoint(chk_path)
tensorboard = TensorBoard()
stopper = EarlyStopping(monitor='val_loss', 
                        min_delta=0, 
                        patience=5, 
                        verbose=0, 
                        mode='auto', 
                        baseline=None, 
                        restore_best_weights=True)

callbacks = [history, checkpoint, stopper, tensorboard]


EXP_CODE = "deepeye"
NUM_EPOCH = 20
BATCH_SIZE = 64


model_json = im.to_json()
with open("./iris-{}.json".format(EXP_CODE), "w") as json_file : 
    json_file.write(model_json)


# npz_path = "/home/chy/archive-data/processed/iris-contour-npz/unity"
# npz_path = "/home/chy/archive-data/processed/iris-contour-npz/unity-partial"
npz_path = "/home/chy/archive-data/processed/iris-contour-npz/vc"
# npz_path = "/home/chy/archive-data/processed/iris-contour-npz/vc-partial"

gen_train = IrisContourGenerator(npz_base_path=npz_path,
                                 out_shape=(64, 64),
                                 batch_size=BATCH_SIZE, 
                                 purpose=Purpose.TRAIN,
                                 is_ellipse_mode=False,
                                 use_softmax_pred=True,
                                 use_aug=True)

gen_valid = IrisContourGenerator(npz_base_path=npz_path, 
                                 out_shape=(64, 64),
                                 batch_size=BATCH_SIZE, 
                                 purpose=Purpose.VALID,
                                 is_ellipse_mode=False,
                                 use_softmax_pred=True,
                                 use_aug=False)


im.fit_generator(generator=gen_train,
                 validation_data=gen_valid,
                 callbacks=callbacks,
                 epochs=NUM_EPOCH,
                 workers=16, 
                 use_multiprocessing=True,
                 shuffle=True)


gen_valid = IrisContourGenerator(npz_base_path=npz_path, 
                                  batch_size=BATCH_SIZE, 
                                  purpose=Purpose.TEST,
                                  is_ellipse_mode=False,
                                  use_aug=True)

