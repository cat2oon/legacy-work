import cv2
import numpy as np

from keras.models import model_from_json

from ai.predictor.cores.outputs import ModelOutputs
from ai.predictor.cores.predictor import Predictor
from al.optics.vector import Vector3D


class NullPredictor(Predictor):
    def __init__(self):
        super(self.__class__, self).__init__()

    def load_model(self, model_path, weight_path):
        pass

    def preprocess(self, inputs):
        pass

    def predict(self, inputs):
        return ModelOutputs()

