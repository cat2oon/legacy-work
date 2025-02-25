from abc import ABC, abstractmethod

from ai.libs.keras.models.loader import KerasLoader
from ai.predictor.cores.inputs import ModelInputs
from ai.predictor.cores.outputs import ModelOutputs


# TODO
# extract model vector
#   - from look-vec predictor
#   - from head-pose predictor

# target
# dense (여기서 잘되면 real 이미지로 look-vec predict 자체가 잘되는 것)
# conv activation


# Candidate #1
# out_relu (ReLU)                 (None, 2, 4, 1280)   0           Conv_1_bn[0][0]

# Custom Candidate #2
# concatenate_1 (Concatenate)     (None, 515)          0           input_2[0][0]


class FeatureExtractor(ABC):
    def __init__(self):
        self.net = None

    @abstractmethod
    def load_model(self, model_path, weight_path):
        self.net = KerasLoader.load_model(model_path, weight_path)

    @abstractmethod
    def custom_model(self):
        pass

    @abstractmethod
    def preprocess(self, inputs: ModelInputs):
        pass

    @abstractmethod
    def extract(self, inputs: ModelInputs) -> ModelOutputs:
        pass
