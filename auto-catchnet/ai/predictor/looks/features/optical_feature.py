from keras.engine.training import Model

from ai.predictor.cores.extractor import FeatureExtractor
from ai.predictor.cores.inputs import ModelInputs
from ai.predictor.cores.outputs import ModelOutputs


class OpticalAxisFeatureExtractor(FeatureExtractor):
    def load_model(self, model_path, weight_path):
        super().load_model(model_path, weight_path)

    def custom_model(self):
        model = self.net
        model.layers.pop()
        model.layers.pop()
        model.layers.pop()

        extractor_model = Model()
        self.net = extractor_model

    def preprocess(self, inputs: ModelInputs):
        return inputs

    def extract(self, inputs: ModelInputs) -> ModelOutputs:
        ins = self.preprocess(inputs)
        preds = self.net.predict(ins, batch_size=inputs.num_items())
        outs = ModelOutputs()
        return outs


