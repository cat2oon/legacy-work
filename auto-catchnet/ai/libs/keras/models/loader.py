from keras.engine.saving import model_from_json


class KerasLoader:
    def __init__(self):
        pass

    @staticmethod
    def load_model(model_path, weight_path):
        with open(model_path, "r") as json_file:
            model_json = json_file.read()
            net = model_from_json(model_json)
        net.load_weights(weight_path)

        return net


