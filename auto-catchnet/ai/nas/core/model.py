import sys
from abc import *

from ac.common.prints import console_print
from ai.nas.everyone.micro.losses import *
from ai.nas.everyone.micro.micro_controller import MicroController
from ai.nas.everyone.micro.model_inputs import ModelInputs
from ai.nas.layer.fixed import fixed_layer
from ai.nas.layer.nas import nas_layer, factorized_reduction
from ai.nas.params.model import ModelParams
from ai.nas.utils.tfs import get_width
from ai.libs.tf.ops.custom.fcs import fully_connect
from ai.libs.tf.ops.convs import conv_1_f, conv, apply_drop_path
from ai.libs.tf.ops.poolings import avg_pooling, global_avg_pool
from ai.libs.tf.setup.inits import get_or_make_global_step
from ai.libs.tf.utils import *


class Model(ABC):

    def __init__(self, inputs: ModelInputs, params: ModelParams):
        console_print("Build models {}".format(params.name))

        self.params = params
        self.inputs = inputs
        self.name = params.name

        self.pool_layers = None
        self.aux_head_indices = []
        self.global_step = get_or_make_global_step()

        self.lr = None
        self.loss = None
        self.rl_loss = None
        self.test_loss = None
        self.valid_loss = None
        self.aux_loss = None
        self.aux_preds = None

        """ arc-seq for sample """
        self.conv_sample_arc = None
        self.reduce_sample_arc = None

        self.train_op = None
        self.grad_norm = None
        self.optimizer = None
        self.num_train_steps = None  # total steps
        self.num_train_batches = None  # per epoch

    @abstractmethod
    def model(self, input_tensors: dict, is_training, reuse=None):
        if self.is_nas_mode():
            is_training = True

        num_layers = self.params.num_layers
        out_filters = self.params.out_filters

        with tf.variable_scope(self.name, reuse=reuse):
            """ Stem Conv """
            stem_layer = self.build_stem_layer(input_tensors, out_filters, is_training)

            """ Auto Architecture Layers """
            layers = stem_layer
            for layer_id in range(num_layers + 2):
                with tf.variable_scope("layer_{0}".format(layer_id)):
                    layers, out_filters = self.build_layer(layer_id, layers, out_filters, is_training)
                if self.is_aux_head_layer(layer_id, is_training):
                    self.build_aux_heads(layer_id, layers[-1])

            """ Customized Final Layer(s) """
            return self.build_final_layer(layers[-1], input_tensors, is_training)

    @abstractmethod
    def build_stem_layer(self, input_tensors, out_filters, is_training):
        raise NotImplementedError("abstract method")

    @abstractmethod
    def build_final_layer(self, inputs, out_filters, is_training):
        raise NotImplementedError("abstract method")

    """ no override """

    def build_aux_heads(self, layer_id, x):
        print("Using aux_head at layer {0}".format(layer_id))
        with tf.variable_scope("aux_head"):
            aux_logits = tf.nn.relu(x)
            aux_logits = avg_pooling(aux_logits, 5, 3)
            with tf.variable_scope("proj"):
                aux_logits = conv_1_f(aux_logits, 128, True)
            with tf.variable_scope("avg_pool"):
                hw = get_width(aux_logits)
                aux_logits = conv(aux_logits, hw, 768, True)
            with tf.variable_scope("fc"):
                aux_logits = global_avg_pool(aux_logits)
                pred_x = fully_connect(aux_logits, 1, name="aux_pred_x")
                pred_y = fully_connect(aux_logits, 1, name="aux_pred_y")
                pred_x = tf.squeeze(pred_x)
                pred_y = tf.squeeze(pred_y)
                self.aux_preds = (pred_x, pred_y)

        aux_head_variables = get_aux_head_vars(self.name)
        num_aux_vars = count_model_params(aux_head_variables)
        console_print("Aux head uses {0} params".format(num_aux_vars))

    """ layer builders"""

    def build_layer(self, layer_id, layers, out_filters, is_training):
        num_cells = self.params.num_cells

        if self.is_pool_layer(layer_id):
            x, layers, out_filters = self.build_reduce_layer(layer_id, layers, num_cells, out_filters, is_training)
            print("Pool Layer {0:>4d}: {1}".format(layer_id, x))
        else:
            x, layers, out_filters = self.build_conv_layer(layer_id, layers, num_cells, out_filters, is_training)
            print("Conv Layer {0:>4d}: {1}".format(layer_id, x))

        layers = [layers[-1], x]
        return layers, out_filters

    def build_conv_layer(self, layer_id, layers, num_cells, out_filters, is_training):
        conv_arc = self.get_conv_arc()

        if self.is_nas_mode():
            x = nas_layer(layers, conv_arc, out_filters, num_cells)
        else:
            drop_path_apply = self.make_drop_path_applier()
            x = fixed_layer(layer_id, layers, conv_arc, out_filters, 1, is_training, num_cells, drop_path_apply)

        return x, layers, out_filters

    def build_reduce_layer(self, layer_id, layers, num_cells, out_filters, is_training):
        out_filters *= 2
        reduce_arc = self.get_reduce_arc()

        if self.is_nas_mode():
            x = factorized_reduction(layers[-1], out_filters, 2, is_training)
            layers = [layers[-1], x]
            x = nas_layer(layers, reduce_arc, out_filters, num_cells)
        else:
            drop_path_apply = self.make_drop_path_applier()
            x = fixed_layer(layer_id, layers, reduce_arc, out_filters, 2, is_training, num_cells, drop_path_apply)

        return x, layers, out_filters

    def make_drop_path_applier(self):
        global_step = self.global_step
        num_layers = self.params.num_layers
        num_train_step = self.num_train_steps
        drop_path_keep_prob = self.params.drop_path_keep_prob

        def drop_path_applier(x, layer_id, op_id, is_training):
            if op_id in [0, 1, 2, 3] and drop_path_keep_prob is not None and is_training:
                x = apply_drop_path(x, layer_id, drop_path_keep_prob, num_layers, global_step, num_train_step)
            return x

        return drop_path_applier

    """ method """

    def get_model_ops(self):
        model_ops = {
            "lr": self.lr,
            "loss": self.loss,
            "train_op": self.train_op,
            "grad_norm": self.grad_norm,
            "optimizer": self.optimizer,
            "global_step": self.global_step,
            "num_train_batches": self.num_train_batches,
        }
        return model_ops

    def prepare_dataset(self):
        print("Build data ops")

        self.num_train_batches = self.inputs.get_num_train_batches()
        self.num_train_steps = self.params.num_epochs * self.num_train_batches

        self.inputs.prepare("train")
        self.inputs.prepare("valid")
        self.inputs.prepare("test")
        # TODO: RL 따로 iterator 할당하기 (랜덤 선정)

    def connect_controller(self, controller: MicroController):
        if self.is_nas_mode():
            self.conv_sample_arc, self.reduce_sample_arc = controller.get_sample_arc()
            return

        num_cells = self.params.num_cells
        fixed_arc = np.array([int(x) for x in self.params.fixed_arc.split(" ") if x])
        self.conv_sample_arc = fixed_arc[:4 * num_cells]
        self.reduce_sample_arc = fixed_arc[4 * num_cells:]
        console_print("Using FIXED ARC\nconv-arc:\t{}\nreduce-arc:\t{}".format(
            self.conv_sample_arc, self.reduce_sample_arc))

    def eval_once(self, sess, eval_set, feed_dict=None):
        assert self.global_step is not None

        global_step = sess.run(self.global_step)
        print("Eval at step: {}".format(global_step))

        if eval_set == "valid":
            num_batches = self.inputs.get_num_val_batches()
            loss_op = self.valid_loss
        else:
            num_batches = self.inputs.get_num_test_batches()
            loss_op = self.test_loss

        # TODO : Peak, Min, Max, dist trace
        # TODO : trade-off (variance, bias) ==> bias 우세하게
        print("eval num batches {}".format(num_batches))
        acc_loss, total_eval_count = 0.0, 0
        for batch_id in range(num_batches):
            loss = sess.run(loss_op, feed_dict=feed_dict)
            acc_loss += loss
            total_eval_count += 1
            sys.stdout.write("\nloss: {:<2f}, total_count: {:>2d}".format(loss, total_eval_count))
        console_print("{}_loss : {:<6.4f}".format(eval_set, float(acc_loss) / total_eval_count))

    """ define run ops """

    @abstractmethod
    def create_trainer(self):
        raise NotImplementedError("abstract method")

    def build_train(self):
        console_print("Build train graph")
        trainer_outs = self.create_trainer()

        self.lr = trainer_outs["lr"]
        self.loss = trainer_outs["loss"]
        self.train_op = trainer_outs["train_op"]
        self.grad_norm = trainer_outs["grad_norm"]
        self.optimizer = trainer_outs["optimizer"]

    def build_valid(self):
        console_print("Build valid graph")

        inputs = self.inputs.get_inputs_dict("valid")
        preds = self.model(inputs, False, reuse=True)
        self.valid_loss = get_dist_loss(inputs, preds)

    def build_test(self):
        console_print("Build test graph")

        inputs = self.inputs.get_inputs_dict("test")
        preds = self.model(inputs, False, reuse=True)
        self.test_loss = get_dist_loss(inputs, preds)

    def build_valid_rl(self):
        console_print("Build valid graph on shuffled data")

        is_training = True  # TODO: 현재 shuffle 모드가 아니라서 애매
        inputs = self.inputs.get_inputs_dict("test")
        preds = self.model(inputs, is_training, reuse=True)
        self.rl_loss = get_dist_loss(inputs, preds)

    @abstractmethod
    def get_rl_reward(self):
        raise NotImplementedError("abstract method")

    """ miscellaneous """

    @abstractmethod
    def get_pool_layers(self, ratio=3):
        if self.params.pool_layers is not None:
            pool_distance = 0
            pool_layers = self.params.pool_layers
            pool_layers = [int(v) for v in pool_layers.split()]
        elif self.params.num_layers >= 7:
            pool_distance = self.params.num_layers // ratio
            pool_layers = [pool_distance, 2 * pool_distance + 1, 3 * pool_distance + 1]
        else:
            pool_distance = self.params.num_layers // ratio
            pool_layers = [pool_distance, 2 * pool_distance + 1]

        print("WARNING: Customized pooling layer distance. Set num layers carefully.")
        print("pool distance: {}, pool layers:{}".format(pool_distance, pool_layers))
        return pool_layers

    def get_aux_head_index(self):
        return [self.pool_layers[-1] + 1] if self.params.use_aux_heads else []

    def is_pool_layer(self, layer_id):
        return layer_id in self.pool_layers

    def get_conv_arc(self):
        return self.conv_sample_arc

    def get_reduce_arc(self):
        return self.reduce_sample_arc

    def is_fixed_arc(self):
        return self.params.fixed_arc is not None

    def is_nas_mode(self):
        return self.params.fixed_arc is None

    def use_aux_heads(self):
        return self.params.use_aux_heads is True

    def is_aux_head_layer(self, layer_id, is_training):
        return layer_id in self.aux_head_indices and is_training
