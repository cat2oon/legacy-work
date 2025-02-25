from ac.common.flags import is_debug_mode
from ac.common.prints import console_print
from ai.nas.core.model import Model
from ai.nas.everyone.micro.losses import get_dist_loss
from ai.nas.everyone.micro.model_inputs import ModelInputs
from ai.nas.everyone.micro.train_ops import TrainOps
from ai.nas.params.model import ModelParams
from ai.libs.tf.ops.custom.fcs import *
from ai.libs.tf.ops.monitors import make_loss_monitor
from ai.libs.tf.ops.poolings import *
from ai.libs.tf.setup.inits import *
from ai.libs.tf.utils import get_trainable_vars, count_model_params

l2_reg = tf.keras.regularizers.l2


class MicroChild(Model):

    def __init__(self, inputs: ModelInputs, params: ModelParams):
        super(self.__class__, self).__init__(inputs, params)

        self.keep_prob = self.params.keep_prob
        self.pool_layers = self.get_pool_layers()
        self.aux_head_indices = self.get_aux_head_index()

        if self.params.drop_path_keep_prob is not None:
            assert self.params.num_epochs is not None, "Need num_epochs to drop_path"

    def build_final_layer(self, x, inputs, is_training):
        console_print("shape of input before final layer: {}".format(x.shape))

        fv = tf.nn.relu(x)
        fv = global_avg_pool(fv)
        print("gap {}".format(fv.shape))

        # dfv = dynamic_branch(fv, inputs)
        dfv = uni_fc(fv)

        orient = tf.cast(inputs["orientation"], tf.float32)
        cam_to_x = inputs["cam_to_x"]
        cam_to_y = inputs["cam_to_y"]
        candide = inputs["candide"]

        orient = tf.expand_dims(orient, 1)
        cam_to_x = tf.expand_dims(cam_to_x, 1)
        cam_to_y = tf.expand_dims(cam_to_y, 1)
        candide = tf.reshape(candide, [-1, 20])

        fb = tf.concat([dfv, cam_to_x, cam_to_y, candide, orient], axis=1)
        fb = fully_connect(fb, 64, name="fx-01")
        preds = fully_connect(fb, 2, name="preds")
        fx, fy = tf.split(preds, num_or_size_splits=2, axis=1)
        pred_x = tf.squeeze(fx)
        pred_y = tf.squeeze(fy)

        return pred_x, pred_y

    def model(self, input_tensors, is_training, reuse=False):
        model_outs = super().model(input_tensors, is_training, reuse)
        return model_outs

    def build_stem_layer(self, input_tensors, out_filters, is_training):
        x = stem_conv(input_tensors["left_eye"], out_filters, is_training, ch=1, filter_size=1, stride=1, suffix="le")
        y = stem_conv(input_tensors["right_eye"], out_filters, is_training, ch=1, filter_size=1, stride=1, suffix="re")
        return [x, y]

    def create_trainer(self):
        inputs = self.inputs.get_inputs_dict("train")
        preds = self.model(inputs, is_training=True)

        dist_loss = get_dist_loss(inputs, preds)
        train_loss = dist_loss

        if is_debug_mode():
            dist_loss = make_loss_monitor(dist_loss, preds, inputs, self.get_conv_arc(),
                                          threshold=self.params.monitor_thresholds)

        aux_loss = None
        if self.use_aux_heads():
            aux_loss = get_dist_loss(inputs, self.aux_preds)
            train_loss += 0.4 * aux_loss

        tf_variables = get_trainable_vars(self.name)
        num_vars = count_model_params(tf_variables)
        console_print("Model has {:,} params".format(num_vars))

        train_op_params = self.params.get_for_train_op(self.num_train_batches)
        train_op_params.pretty_print()
        train_ops = TrainOps(train_loss, tf_variables, self.global_step, train_op_params)

        trainer_outs = {
            "lr": train_ops.learning_rate,
            "loss": dist_loss,
            "aux_loss": aux_loss,
            "train_op": train_ops.train_op,
            "grad_norm": train_ops.grad_norm,
            "optimizer": train_ops.optimizer,
        }

        return trainer_outs

    def get_rl_reward(self):
        # TODO : train step 고려한 reward 설계 필요
        loss = self.rl_loss

        def compute_reward(tf_loss, comp=32.0):
            limit = tf.constant(9.0, dtype=tf.float32)
            max_reward = tf.constant(comp, dtype=tf.float32)
            # r = 1.0 / tf_loss     # 1/loss는 변별력이 낮음
            # r = 4 * r
            # r = tf.maths.minimum(max_reward, r)
            r = 1.0 / (tf_loss ** 2)
            r = comp * r
            r = tf.math.minimum(max_reward, r)
            return r

        # reward limit cut을 적용하는 경우 초반에 ctrl 학습이 너무 더딤
        # reward = tf.cond(tf.greater(loss, limit),
        #                  lambda: tf.constant(0, dtype=tf.float32),
        #                  lambda: compute_reward(loss))
        reward = compute_reward(loss)

        return reward, loss

    def get_pool_layers(self, ratio=3):
        return super().get_pool_layers(ratio)
