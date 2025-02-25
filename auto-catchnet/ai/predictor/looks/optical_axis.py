import numpy as np

from ac.images.filters.filters import color_to_grey
from ai.predictor.cores.outputs import ModelOutputs
from ai.predictor.cores.predictor import Predictor
from al.optics.vector import Vector3D


class OpticalAxisPredictor(Predictor):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.image_shape = (56, 112, 1)

    def load_model(self, model_path, weight_path):
        super().load_model(model_path, weight_path)

    def preprocess(self, inputs):
        return inputs

    def predict_img(self, img, head_pose) -> Vector3D:
        img = color_to_grey(img)
        img = np.expand_dims(img, axis=0)
        img = np.expand_dims(img, axis=3)
        head_pose = np.expand_dims(head_pose, axis=0)
        batch_input = [img, head_pose]

        vec = self.net.predict(batch_input, batch_size=1)
        return Vector3D.infer_unit(vec[0])

    def predict(self, inputs):
        batch = self.preprocess(inputs)
        preds = self.net.predict(batch, batch_size=inputs.num_items())

        outs = ModelOutputs()
        outs['preds'] = preds

        return outs

