import glob
import os

from ai.nas.params.model import ModelParams


def to_dataset(tfrecord_files, batch_size=32, num_epochs=1):
    from ds.core.tf.tfrecords import to_dataset as dataset
    from ds.everyone.tfrecord.decoders import deserialize_example
    return dataset(tfrecord_files, deserialize_example, batch_size=batch_size, num_epochs=num_epochs)


class ModelInputs:

    def __init__(self, model_params: ModelParams):
        self.val_iter = None
        self.test_iter = None
        self.train_iter = None
        self.inputs_bags = {}

        self.num_epochs = None
        self.train_batch_size = None
        self.eval_batch_size = None

        self.num_train_examples = None
        self.num_valid_examples = None
        self.num_test_examples = None

        self.from_model_params(model_params)

    def prepare(self, purpose="train"):
        purpose = purpose.upper()
        iterator = self.select_iterator(purpose)
        batch = iterator.get_next()

        # TODO : tfexample deserializer에 정의된 것 사용
        self.inputs_bags[purpose] = {
            "uid": batch[0],
            "frame": batch[1],
            "face": batch[2],
            "left_eye": batch[3],
            "right_eye": batch[4],
            "orientation": batch[5],
            "cam_x": batch[6],
            "cam_y": batch[7],
            "cam_to_x": batch[8],
            "cam_to_y": batch[9],
            "candide": batch[10],
        }

        return batch

    def get_inputs_dict(self, purpose):
        return self.inputs_bags[purpose.upper()]

    def select_iterator(self, purpose):
        if purpose == "TRAIN":
            return self.train_iter
        if purpose == "VALID":
            return self.val_iter
        if purpose == "TEST":
            return self.test_iter

    def get_initializers(self):
        return [
            self.train_iter.initializer,
            self.val_iter.initializer,
            self.test_iter.initializer
        ]

    def select_tfrecord(self, tfrecord_files, name="full"):
        if name == "full":
            for_train = tfrecord_files[0:150]
            for_valid = tfrecord_files[150:175]
            for_test = tfrecord_files[175:185]
        elif name == "slow":
            for_train = tfrecord_files[0:50]
            for_valid = tfrecord_files[150:175]
            for_test = tfrecord_files[175:185]
        elif name == "mid":
            for_train = tfrecord_files[0:25]
            for_valid = tfrecord_files[160:175]
            for_test = tfrecord_files[175:185]
        elif name == "fast":
            for_train = tfrecord_files[50:60]
            for_valid = tfrecord_files[160:170]
            for_test = tfrecord_files[175:180]
        elif name == "single":
            for_train = tfrecord_files[0:1]
            for_valid = tfrecord_files[1:2]
            for_test = tfrecord_files[2:3]
        else:
            for_train = tfrecord_files[0:10]
            for_valid = tfrecord_files[150:160]
            for_test = tfrecord_files[160:170]

        return for_train, for_valid, for_test

    def from_model_params(self, params: ModelParams):
        dataset_mode = params.dataset_mode
        tfrecord_files = glob.glob(os.path.join(params.data_path, '*.tfr'))
        for_train, for_valid, for_test = self.select_tfrecord(tfrecord_files, dataset_mode)

        self.num_epochs = num_epochs = params.num_epochs
        self.train_batch_size = params.batch_size
        self.eval_batch_size = params.eval_batch_size

        self.train_iter = to_dataset(for_train, batch_size=self.train_batch_size, num_epochs=num_epochs)
        self.val_iter = to_dataset(for_valid, batch_size=self.eval_batch_size, num_epochs=num_epochs)
        self.test_iter = to_dataset(for_test, batch_size=self.eval_batch_size, num_epochs=num_epochs)

        # NOTE: gzip tfexample 카운팅이 매우 느리므로 직접 수량 계산
        # self.num_train_examples = count_tfexamples(for_train)
        self.num_train_examples = len(for_train) * 8192
        self.num_valid_examples = len(for_valid) * 8192
        self.num_test_examples = len(for_test) * 8192

    """ miscellaneous """

    def get_num_train_batches(self):
        return self.num_train_examples // self.train_batch_size

    def get_num_val_batches(self):
        return self.num_valid_examples // self.eval_batch_size

    def get_num_test_batches(self):
        return self.num_test_examples // self.eval_batch_size
