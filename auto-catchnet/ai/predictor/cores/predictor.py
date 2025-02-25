from abc import ABC, abstractmethod

from ai.libs.keras.models.loader import KerasLoader
from ai.predictor.cores.inputs import ModelInputs
from ai.predictor.cores.outputs import ModelOutputs


class Predictor(ABC):
    def __init__(self):
        self.net = None
        self.image_shape = None

    def get_required_image_shape(self):
        return self.image_shape

    @abstractmethod
    def load_model(self, model_path, weight_path):
        self.net = KerasLoader.load_model(model_path, weight_path)

    @abstractmethod
    def preprocess(self, inputs: ModelInputs):
        pass

    @abstractmethod
    def predict(self, inputs: ModelInputs) -> ModelOutputs:
        return ModelOutputs()

